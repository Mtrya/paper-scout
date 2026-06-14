"""
RepWAM core-mechanism probe.

This script implements a minimal, runnable version of the two most distinctive
pieces of RepWAM:

1. Representation Visual-Action Tokenizer (RepViTok)
   - a tiny ViT autoencoder with reconstruction + perceptual-style + alignment losses
   - a Latent Action Tokenizer (LAT) that couples an Inverse Dynamics Model (IDM)
     with a Forward Dynamics Model (FDM) producing a transport operator K and a
     residual delta.

2. Causal World Action Model (WAM) training step
   - chunk-based causal sequence construction
   - conditional flow-matching loss over visual-action chunks

The probe runs entirely on CPU with synthetic moving-blob videos so the math can
be exercised without needing the (unreleased) official weights or datasets.
"""

import math
import os
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Synthetic data: a Gaussian blob translating on a 2-D grid.
# We encode each frame as a flattened spatial grid (L = H*W) so the LAT can
# operate on the same tensor layout described in the paper: z_t \in R^{L x d_v}.
# ---------------------------------------------------------------------------

def make_blob_video(
    n_frames: int = 17,
    height: int = 16,
    width: int = 16,
    d_v: int = 8,
    motion: str = "horizontal",
    seed: int = None,
) -> np.ndarray:
    """Return one video as (T, L, d_v) with L=H*W."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    y, x = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing="ij")
    frames = []
    for t in range(n_frames):
        alpha = t / max(1, n_frames - 1)
        if motion == "horizontal":
            cx, cy = -0.8 + 1.6 * alpha, 0.0
        elif motion == "vertical":
            cx, cy = 0.0, -0.8 + 1.6 * alpha
        elif motion == "diagonal":
            cx, cy = -0.6 + 1.2 * alpha, -0.6 + 1.2 * alpha
        else:
            cx, cy = 0.0, 0.0
        blob = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) * 18.0)
        # Add d_v channels by stacking independent spatial features.
        frame = np.stack([blob * (0.8 + 0.4 * rng.rand()) for _ in range(d_v)], axis=-1)
        frames.append(frame.reshape(height * width, d_v))
    return np.asarray(frames, dtype=np.float32)


def make_dataset(
    n_videos: int = 256,
    n_frames: int = 17,
    height: int = 16,
    width: int = 16,
    d_v: int = 8,
) -> torch.Tensor:
    motions = ["horizontal", "vertical", "diagonal"]
    videos = []
    for i in range(n_videos):
        motion = motions[i % len(motions)]
        videos.append(make_blob_video(n_frames, height, width, d_v, motion, seed=i))
    return torch.from_numpy(np.stack(videos, axis=0))  # (N, T, L, d_v)


# ---------------------------------------------------------------------------
# 1a. Toy visual tokenizer (ViT autoencoder).
# The paper's encoder is a 12-layer Transformer with hidden dim 768; here we
# use a tiny 2-layer Transformer so the probe trains in seconds on CPU.
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, L, d_v) -> (B, L, hidden_dim)
        return self.proj(z)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), x, x, attn_mask=mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ToyVisualTokenizer(nn.Module):
    """Encoder + decoder. Latents keep the same spatial layout as input."""

    def __init__(self, d_v: int, hidden_dim: int = 48, n_layers: int = 2):
        super().__init__()
        self.enc_proj = PatchEmbed(d_v, hidden_dim)
        self.encoder = nn.ModuleList([TransformerBlock(hidden_dim) for _ in range(n_layers)])
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, d_v),
            nn.LayerNorm(d_v),
        )
        self.dec_proj = PatchEmbed(d_v, hidden_dim)
        self.decoder = nn.ModuleList([TransformerBlock(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, d_v)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_v)
        h = self.enc_proj(x)
        for blk in self.encoder:
            h = blk(h)
        return self.latent_proj(h)  # (B, L, d_v)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_proj(z)
        for blk in self.decoder:
            h = blk(h)
        return self.out_proj(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


# ---------------------------------------------------------------------------
# 1b. Latent Action Tokenizer (LAT) with IDM + FDM (transport + residual).
#
# Paper equations (3) and (4):
#   l_t      = q_phi(z_t, z_{t+1})
#   K_t, d_t = f_psi(z_t, l_t)
#   \hat z_{t+1} = K_t z_t + d_t
#
# K_t is an L x L soft transport matrix (left-multiplied along the spatial
# token dimension). d_t is an L x d_v residual.
# ---------------------------------------------------------------------------

class IDM(nn.Module):
    """Inverse dynamics model: maps (z_t, z_{t+1}) -> l_t."""

    def __init__(self, d_v: int, d_l: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        dims = [2 * d_v] + [hidden_dim] * n_layers + [d_l]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        # Pool over spatial tokens to prevent content leakage (the paper says d_l << d_v).
        z_t_pooled = z_t.mean(dim=1)
        z_tp1_pooled = z_tp1.mean(dim=1)
        return self.net(torch.cat([z_t_pooled, z_tp1_pooled], dim=-1))


class FDM(nn.Module):
    """Forward dynamics model: maps (z_t, l_t) -> (K_t, delta_t).

    Implements the paper's transport operator directly: K_t in R^{L x L} and
    z_hat_{t+1} = K_t z_t + delta_t is a left matrix multiply over the spatial
    token dimension.
    """

    def __init__(self, d_v: int, d_l: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.d_v = d_v
        # Concatenate per-token z_t with broadcast latent action l_t.
        dims = [d_v + d_l] + [hidden_dim] * n_layers
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
        self.backbone = nn.Sequential(*layers)
        # K_t logits for each (target) token over all source positions.
        self.k_head = nn.Linear(hidden_dim, 1)  # placeholder; replaced in forward
        self.delta_head = nn.Linear(hidden_dim, d_v)
        self._built = False

    def _build_k_head(self, L: int):
        if not self._built or self.k_head.out_features != L:
            self.k_head = nn.Linear(self.backbone[-2].out_features, L).to(
                next(self.backbone.parameters()).device
            )
            self._built = True

    def forward(self, z_t: torch.Tensor, l_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z_t: (B, L, d_v); l_t: (B, d_l)
        L = z_t.size(1)
        self._build_k_head(L)
        l_broadcast = l_t[:, None, :].expand(-1, L, -1)
        h = self.backbone(torch.cat([z_t, l_broadcast], dim=-1))  # (B, L, H)
        k_logits = self.k_head(h)  # (B, L, L)
        K_t = F.softmax(k_logits, dim=-1)  # rows = target tokens, cols = source tokens
        delta_t = self.delta_head(h)  # (B, L, d_v)
        # Left multiply: (B, L, L) @ (B, L, d_v) -> (B, L, d_v)
        z_tp1_hat = torch.bmm(K_t, z_t) + delta_t
        return K_t, delta_t, z_tp1_hat


class LatentActionTokenizer(nn.Module):
    def __init__(self, d_v: int, d_l: int, hidden_dim: int = 128):
        super().__init__()
        self.idm = IDM(d_v, d_l, hidden_dim)
        self.fdm = FDM(d_v, d_l, hidden_dim)

    def forward(
        self, z_t: torch.Tensor, z_tp1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        l_t = self.idm(z_t, z_tp1)
        K_t, delta_t, z_tp1_hat = self.fdm(z_t, l_t)
        return l_t, K_t, delta_t, z_tp1_hat

    def predict_next(self, z_t: torch.Tensor, l_t: torch.Tensor) -> torch.Tensor:
        _, _, z_tp1_hat = self.fdm(z_t, l_t)
        return z_tp1_hat


# ---------------------------------------------------------------------------
# 1c. Losses for the tokenizer stage.
# Paper Eq. (1): L_rec  = l1 + L_perc + L_gan
# Paper Eq. (2): L_align = ||avg(W_align z) - avg(G(o))||^2
# Paper Eq. (4): L_fwd + L_cons
# ---------------------------------------------------------------------------

def visual_tokenizer_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    z: torch.Tensor,
    teacher: torch.Tensor,
    lambda_l1: float = 1.0,
    lambda_align: float = 1.0,
) -> dict:
    """Return reconstruction + semantic-alignment losses (paper Eq. 1-2)."""
    l1 = F.l1_loss(x_hat, x)
    # A tiny perceptual proxy: L2 on a random feature projection.
    feat_x = x.reshape(x.size(0), -1)
    feat_hat = x_hat.reshape(x_hat.size(0), -1)
    perc = F.mse_loss(feat_x, feat_hat)
    # Semantic alignment with a dummy frozen teacher (here just the pooled input).
    z_pooled = z.mean(dim=1)
    t_pooled = teacher.mean(dim=1)
    align = F.mse_loss(z_pooled, t_pooled)
    total = lambda_l1 * l1 + lambda_align * align + 0.1 * perc
    return {"total": total, "l1": l1.item(), "align": align.item(), "perc": perc.item()}


def latent_action_loss(
    lat: LatentActionTokenizer,
    z_seq: torch.Tensor,
) -> dict:
    """Forward + backward consistency on latent frame pairs (paper Eq. 4)."""
    Tp = z_seq.size(1)
    fwd_loss = 0.0
    cons_loss = 0.0
    for t in range(Tp - 1):
        z_t, z_tp1 = z_seq[:, t], z_seq[:, t + 1]
        _, _, _, z_tp1_hat = lat(z_t, z_tp1)
        fwd_loss = fwd_loss + F.mse_loss(z_tp1_hat, z_tp1)
        # Backward: swap the pair and try to recover z_t.
        _, _, _, z_t_hat = lat(z_tp1, z_t)
        cons_loss = cons_loss + F.mse_loss(z_t_hat, z_t)
    return {
        "total": fwd_loss + cons_loss,
        "fwd": fwd_loss.item(),
        "cons": cons_loss.item(),
    }


# ---------------------------------------------------------------------------
# 2. Toy causal WAM training step.
# Paper Eq. (5): chunk u_{t:t+k} = [z_{t:t+k}, l_{t:t+k-1}]
# Paper Eq. (6-7): conditional flow matching on visual + action components.
# ---------------------------------------------------------------------------

class ToyCausalWAM(nn.Module):
    """Minimal causal transformer that denoises visual-action chunks."""

    def __init__(self, d_v: int, d_l: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.d_v = d_v
        self.d_l = d_l
        # Each chunk token is either a visual token (d_v) or an action token (d_l).
        # We project both to a common hidden dim.
        self.v_proj = nn.Linear(d_v, hidden_dim)
        self.a_proj = nn.Linear(d_l, hidden_dim)
        self.time_emb = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(hidden_dim) for _ in range(n_layers)])
        self.v_out = nn.Linear(hidden_dim, d_v)
        self.a_out = nn.Linear(hidden_dim, d_l)

    def _build_causal_mask(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(n, n, device=device) * float("-inf"), diagonal=1)

    def forward(
        self,
        x_v: torch.Tensor,
        x_a: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_v: (B, k, L, d_v) noisy visual tokens inside the chunk.
        x_a: (B, k-1, d_l) noisy action tokens inside the chunk.
        alpha: (B, 1) flow time, broadcast to every token.
        """
        B, k, L, _ = x_v.shape
        v_tokens = self.v_proj(x_v).reshape(B, k * L, -1)
        a_tokens = self.a_proj(x_a)  # (B, k-1, H)
        tokens = torch.cat([v_tokens, a_tokens], dim=1)  # (B, k*L + k-1, H)
        t_emb = self.time_emb(alpha.reshape(-1, 1)[:, None, :])  # (B, 1, H)
        tokens = tokens + t_emb
        mask = self._build_causal_mask(tokens.size(1), tokens.device)
        for layer in self.layers:
            tokens = layer(tokens, mask=mask)
        v_len = k * L
        v_out = self.v_out(tokens[:, :v_len]).reshape(B, k, L, self.d_v)
        a_out = self.a_out(tokens[:, v_len:])
        return v_out, a_out


def wam_flow_matching_loss(
    model: ToyCausalWAM,
    z_chunk: torch.Tensor,
    l_chunk: torch.Tensor,
    lambda_a: float = 1.0,
) -> dict:
    """One training step of conditional flow matching (paper Eq. 6-7)."""
    B, k, L, d_v = z_chunk.shape
    _, k_a, d_l = l_chunk.shape
    assert k_a == k - 1

    eps_v = torch.randn_like(z_chunk)
    eps_a = torch.randn_like(l_chunk)
    alpha = torch.rand(B, 1, 1, 1, device=z_chunk.device)
    x_alpha_v = (1 - alpha) * eps_v + alpha * z_chunk
    dot_x_v = z_chunk - eps_v

    alpha_a = alpha.squeeze(-1)  # (B, 1, 1)
    x_alpha_a = (1 - alpha_a) * eps_a + alpha_a * l_chunk
    dot_x_a = l_chunk - eps_a

    pred_v, pred_a = model(x_alpha_v, x_alpha_a, alpha_a)
    loss_v = F.mse_loss(pred_v, dot_x_v)
    loss_a = F.mse_loss(pred_a, dot_x_a)
    total = loss_v + lambda_a * loss_a
    return {"total": total, "v": loss_v.item(), "a": loss_a.item()}


# ---------------------------------------------------------------------------
# Training loop for the probe.
# ---------------------------------------------------------------------------

def train_tokenizer(
    videos: torch.Tensor,
    d_l: int = 4,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[ToyVisualTokenizer, LatentActionTokenizer, dict]:
    videos = videos.to(device)
    B, T, L, d_v = videos.shape

    vit = ToyVisualTokenizer(d_v).to(device)
    lat = LatentActionTokenizer(d_v, d_l).to(device)
    opt = torch.optim.Adam(list(vit.parameters()) + list(lat.parameters()), lr=lr)

    # The paper temporally patchifies with a factor of 4. We simulate that by
    # downsampling the temporal dimension: T' = 1 + (T-1)//4.
    Tp = 1 + (T - 1) // 4
    idx = torch.linspace(0, T - 1, Tp).long()

    losses_history = {"vis": [], "lat": []}
    for epoch in range(epochs):
        perm = torch.randperm(B)
        epoch_vis, epoch_lat = 0.0, 0.0
        n_batches = 0
        for i in range(0, B, batch_size):
            b_idx = perm[i : i + batch_size]
            x = videos[b_idx][:, idx]  # (B, Tp, L, d_v)
            Bb = x.size(0)
            x_flat = x.reshape(Bb * Tp, L, d_v)

            z, x_hat = vit(x_flat)
            vis_loss = visual_tokenizer_loss(
                x_flat, x_hat, z, teacher=x_flat, lambda_l1=1.0, lambda_align=0.5
            )

            z_seq = z.reshape(Bb, Tp, L, d_v)
            lat_loss = latent_action_loss(lat, z_seq)

            loss = vis_loss["total"] + lat_loss["total"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(vit.parameters()) + list(lat.parameters()), 1.0
            )
            opt.step()

            epoch_vis += vis_loss["total"].item()
            epoch_lat += lat_loss["total"].item()
            n_batches += 1

        losses_history["vis"].append(epoch_vis / n_batches)
        losses_history["lat"].append(epoch_lat / n_batches)
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d}: vis_loss={losses_history['vis'][-1]:.4f} "
                f"lat_loss={losses_history['lat'][-1]:.4f}"
            )

    return vit, lat, losses_history


def train_wam(
    vit: ToyVisualTokenizer,
    lat: LatentActionTokenizer,
    videos: torch.Tensor,
    epochs: int = 40,
    batch_size: int = 32,
    chunk_size: int = 4,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[ToyCausalWAM, dict]:
    videos = videos.to(device)
    B, T, L, d_v = videos.shape
    d_l = lat.idm.net[-1].out_features
    Tp = 1 + (T - 1) // 4
    idx = torch.linspace(0, T - 1, Tp).long()

    wam = ToyCausalWAM(d_v, d_l).to(device)
    opt = torch.optim.Adam(wam.parameters(), lr=lr)

    # Pre-compute latents with frozen tokenizer.
    vit.eval()
    lat.eval()
    with torch.no_grad():
        x = videos[:, idx]  # (B, Tp, L, d_v)
        z_seq = vit.encode(x.reshape(B * Tp, L, d_v)).reshape(B, Tp, L, d_v)
        l_seq = []
        for t in range(Tp - 1):
            l_t = lat.idm(z_seq[:, t], z_seq[:, t + 1])
            l_seq.append(l_t)
        l_seq = torch.stack(l_seq, dim=1)  # (B, Tp-1, d_l)

    losses_history = []
    for epoch in range(epochs):
        perm = torch.randperm(B)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, B, batch_size):
            b_idx = perm[i : i + batch_size]
            z = z_seq[b_idx]
            l = l_seq[b_idx]
            Bb = z.size(0)
            # Random chunk start.
            max_start = max(1, Tp - chunk_size)
            starts = torch.randint(0, max_start, (Bb,))
            z_chunks, l_chunks = [], []
            for j, s in enumerate(starts):
                s = int(s)
                z_chunks.append(z[j, s : s + chunk_size])
                l_chunks.append(l[j, s : s + chunk_size - 1])
            z_chunk = torch.stack(z_chunks, dim=0)
            l_chunk = torch.stack(l_chunks, dim=0)

            fm = wam_flow_matching_loss(wam, z_chunk, l_chunk, lambda_a=1.0)
            opt.zero_grad()
            fm["total"].backward()
            torch.nn.utils.clip_grad_norm_(wam.parameters(), 1.0)
            opt.step()

            epoch_loss += fm["total"].item()
            n_batches += 1

        losses_history.append(epoch_loss / n_batches)
        if (epoch + 1) % 10 == 0:
            print(f"WAM epoch {epoch+1:03d}: fm_loss={losses_history[-1]:.4f}")

    return wam, {"fm": losses_history}


# ---------------------------------------------------------------------------
# Visualization and diagnostics.
# ---------------------------------------------------------------------------

def visualize_transport(
    lat: LatentActionTokenizer,
    videos: torch.Tensor,
    vit: ToyVisualTokenizer,
    out_path: str,
    device: str = "cpu",
):
    videos = videos.to(device)
    T, L, d_v = videos.shape[1:]
    Tp = 1 + (T - 1) // 4
    idx = torch.linspace(0, T - 1, Tp).long()
    x = videos[:4, idx].reshape(4 * Tp, L, d_v)

    vit.eval()
    lat.eval()
    with torch.no_grad():
        z = vit.encode(x).reshape(4, Tp, L, d_v)
        _, K, delta, z_hat = lat(z[:, 0], z[:, 1])

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        k = K[i].cpu().numpy()  # (L, L)
        axes[0, i].imshow(k, cmap="viridis", aspect="auto")
        axes[0, i].set_title(f"transport K (sample {i})")
        axes[0, i].axis("off")

        err = (z_hat - z[:, 1])[i].cpu().numpy()
        err_norm = np.linalg.norm(err, axis=-1).reshape(int(np.sqrt(L)), -1)
        axes[1, i].imshow(err_norm, cmap="inferno")
        axes[1, i].set_title(f"residual norm (sample {i})")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved transport visualization to {out_path}")


def plot_losses(losses: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(losses["vis"], label="visual tokenizer")
    axes[0].plot(losses["lat"], label="LAT")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Tokenizer training losses")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(losses["wam"], label="flow matching")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_title("WAM flow-matching loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved loss curves to {out_path}")


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

def main():
    out_dir = os.environ.get("REPWAM_PROBE_OUT", ".")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Generating synthetic moving-blob dataset...")
    videos = make_dataset(n_videos=128, n_frames=17, height=8, width=8, d_v=8)
    print(f"Dataset shape: {videos.shape}")

    print("\nTraining RepViTok + LAT...")
    vit, lat, tok_losses = train_tokenizer(
        videos, d_l=4, epochs=30, batch_size=32, lr=1e-3, device=device
    )

    print("\nTraining toy causal WAM...")
    wam, wam_losses = train_wam(
        vit, lat, videos, epochs=15, batch_size=32, chunk_size=4, lr=1e-3, device=device
    )

    combined = {"vis": tok_losses["vis"], "lat": tok_losses["lat"], "wam": wam_losses["fm"]}
    plot_losses(combined, os.path.join(out_dir, "repwam_probe_losses.png"))
    visualize_transport(
        lat, videos, vit, os.path.join(out_dir, "repwam_probe_transport.png"), device=device
    )

    # Final sanity numbers.
    vit.eval()
    lat.eval()
    with torch.no_grad():
        T, L, d_v = videos.shape[1:]
        Tp = 1 + (T - 1) // 4
        idx = torch.linspace(0, T - 1, Tp).long()
        x = videos[:32, idx].reshape(32 * Tp, L, d_v).to(device)
        z = vit.encode(x).reshape(32, Tp, L, d_v)
        l_test = []
        for t in range(Tp - 1):
            _, _, _, z_hat = lat(z[:, t], z[:, t + 1])
            l_test.append(F.mse_loss(z_hat, z[:, t + 1]).item())
        print("\nFinal LAT forward MSE per transition:", [f"{v:.4f}" for v in l_test])
        print(f"Mean forward MSE: {np.mean(l_test):.4f}")

    # Save a compact model summary.
    summary_path = os.path.join(out_dir, "repwam_probe_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Toy RepWAM probe\n")
        f.write(f"Visual tokenizer params: {sum(p.numel() for p in vit.parameters())}\n")
        f.write(f"LAT params: {sum(p.numel() for p in lat.parameters())}\n")
        f.write(f"WAM params: {sum(p.numel() for p in wam.parameters())}\n")
        f.write(f"Mean LAT forward MSE: {np.mean(l_test):.4f}\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
