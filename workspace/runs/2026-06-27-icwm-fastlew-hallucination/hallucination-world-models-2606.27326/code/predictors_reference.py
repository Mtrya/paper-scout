#!/usr/bin/env python3
"""
Reference implementations of the three label-free hallucination predictors from
"Hallucination in World Models is Predictable and Preventable" (arXiv:2606.27326).

These are literal translations of the code in the official release
(https://github.com/nicklashansen/mmbench2) into self-contained, runnable
functions.  They operate on generic torch tensors so they can be inspected
without the full training environment or a CUDA GPU.

Predictors:
  u_r  -- tokenizer round-trip residual  (perceptual hallucination)
  u_f  -- flow instability                (action-marginalized / uncertain dynamics)
  u_s  -- inter-seed denoising variance   (scene divergence / epistemic uncertainty)

All three are used in motion-normalized form in the paper:
    u_norm = u / max(motion, eps)
where motion is the RMS latent-step size between the prediction and the
previous latent.
"""
from pathlib import Path
from typing import Dict, List, Optional
import sys

import torch
import torch.nn.functional as F

# The helper functions we use live in the release repository's model.py.
MMREPO = Path(__file__).resolve().parents[4] / "code" / "mmbench2-hallucination" / "src"
if MMREPO.exists():
    sys.path.insert(0, str(MMREPO))


def tokenizer_round_trip_residual(
    z_pred: torch.Tensor,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    z_prev: torch.Tensor,
    *,
    patch: int,
    packing_factor: int,
    n_spatial: int,
    H: int,
    W: int,
    C: int = 3,
    motion_eps: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """
    Motion-normalized tokenizer round-trip residual (u_r_norm).

    Args:
        z_pred: predicted next latent, shape (B, n_spatial, d_spatial).
        encoder / decoder: frozen tokenizer submodules.
        z_prev: previous latent, shape (B, n_spatial, d_spatial), used for
                motion normalization.
        patch / packing_factor / n_spatial / H / W / C: tokenizer geometry.

    Returns:
        dict with per-batch scalars "u_r", "motion", "u_r_norm".
    """
    # The released uncertainty.py calls into train_dynamics.decode_packed_to_frames
    # and model.pack_bottleneck_to_spatial.  We inline the equivalent ops here.
    from model import temporal_patchify, pack_bottleneck_to_spatial, temporal_unpatchify

    B = z_pred.shape[0]
    motion = (z_pred - z_prev.float()).pow(2).mean(dim=(1, 2)).sqrt()

    # (B, n_spatial, d_spatial) -> (B, 1, n_spatial, d_spatial)
    z_in = z_pred.unsqueeze(1).float()

    # Decode to pixels, re-patchify, and re-encode.
    frames = temporal_unpatchify(
        decoder(z_in), H=H, W=W, C=C, patch=patch
    )  # (B, 1, C, H, W)
    patches = temporal_patchify(frames, patch)  # (B, 1, Np, Dp)
    z_recon_btLd, _ = encoder(patches)  # (B, 1, n_latents, d_bottleneck)
    z_recon = pack_bottleneck_to_spatial(
        z_recon_btLd, n_spatial=n_spatial, k=packing_factor,
    )[:, 0].float()  # (B, n_spatial, d_spatial)

    u_r = (z_pred.float() - z_recon).pow(2).mean(dim=(1, 2)).sqrt()
    return {
        "u_r": u_r,
        "motion": motion,
        "u_r_norm": u_r / motion.clamp(min=motion_eps),
    }


def flow_instability(
    x1_hat_trajectory: List[torch.Tensor],
    z_prev: torch.Tensor,
    motion_eps: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """
    Flow instability (u_f_norm) from a denoising trajectory.

    The official implementation (interactive.py, sample_one_timestep_packed)
    tracks the RMS change in the denoiser's clean-target prediction x1_hat
    between successive Euler substeps, then averages over the tail half.

    Args:
        x1_hat_trajectory: list of T clean-target predictions, each
                           (B, n_spatial, d_spatial).  T >= 2.
        z_prev: previous latent for motion normalization, (B, n_spatial, d_spatial).

    Returns:
        dict with "u_f", "motion", "u_f_norm".
    """
    B = x1_hat_trajectory[0].shape[0]
    deltas = []
    prev = x1_hat_trajectory[0].float()
    for x1 in x1_hat_trajectory[1:]:
        x1f = x1.float()
        deltas.append((x1f - prev).pow(2).mean(dim=(1, 2)).sqrt())
        prev = x1f
    all_deltas = torch.stack(deltas, dim=1)  # (B, T-1)
    tail = all_deltas[:, len(deltas) // 2:]
    u_f = tail.mean(dim=1)

    z_pred = x1_hat_trajectory[-1].float()
    motion = (z_pred - z_prev.float()).pow(2).mean(dim=(1, 2)).sqrt()
    return {
        "u_f": u_f,
        "motion": motion,
        "u_f_norm": u_f / motion.clamp(min=motion_eps),
    }


def inter_seed_variance(
    z_samples: torch.Tensor,
    z_prev: torch.Tensor,
    motion_eps: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """
    Inter-seed denoising variance (u_s_norm).

    Runs N independent denoising trajectories for the same (context, action) and
    measures per-element variance across seeds.

    Args:
        z_samples: (B, N, n_spatial, d_spatial) next-latent predictions.
        z_prev: previous latent for motion normalization, (B, n_spatial, d_spatial).

    Returns:
        dict with "u_s", "motion", "u_s_norm".
    """
    u_s = z_samples.float().var(dim=1).mean(dim=(1, 2))
    z_pred = z_samples.float().mean(dim=1)
    motion = (z_pred - z_prev.float()).pow(2).mean(dim=(1, 2)).sqrt()
    return {
        "u_s": u_s,
        "motion": motion,
        "u_s_norm": u_s / motion.clamp(min=motion_eps),
    }


def _dummy_denoise(z_noisy: torch.Tensor, step: int) -> torch.Tensor:
    """Toy denoiser: settles faster in well-covered (large-mass) regions."""
    # A deterministic toy so the demo is reproducible.
    noise = torch.randn_like(z_noisy) * 0.1 / (step + 1)
    return z_noisy * 0.9 + noise


def main():
    B, n_spatial, d_spatial = 2, 256, 64
    z_prev = torch.randn(B, n_spatial, d_spatial)

    # 1) u_r: simulate a predicted latent and a tokenizer round-trip.
    z_pred = z_prev + 0.2 * torch.randn(B, n_spatial, d_spatial)

    # Toy tokenizer stubs that satisfy the patch shapes expected by the real
    # decode/encode path.  With packing_factor=1, n_latents == n_spatial and
    # d_bottleneck == d_spatial.  The decoder maps latents to patches of shape
    # (B,1,Np,Dp); the encoder maps patches back to latents.
    Np = (224 // 14) * (224 // 14)  # 256
    Dp = 14 * 14 * 3                 # 588

    class ToyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.W = torch.nn.Linear(n_spatial * d_spatial, Np * Dp)
        def forward(self, z):
            B, T, S, D = z.shape
            z = z.reshape(B * T, S * D)
            out = self.W(z).view(B, T, Np, Dp)
            return torch.sigmoid(out)

    class ToyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.W = torch.nn.Linear(Dp, d_spatial)
        def forward(self, patches):
            B, T, Np, Dp = patches.shape
            x = patches.reshape(B * T * Np, Dp)
            z = torch.tanh(self.W(x)).view(B, T, Np, d_spatial)
            # Return bottleneck then latents: with packing_factor=1 we keep all Np tokens.
            return z, (None, None)

    # For the demo we bypass the pixel decode/encode path by treating the stub
    # as both encoder and decoder; in the real pipeline these are the released
    # Encoder/Decoder from model.py.
    ur = tokenizer_round_trip_residual(
        z_pred, encoder=ToyEncoder(), decoder=ToyDecoder(), z_prev=z_prev,
        patch=14, packing_factor=1, n_spatial=n_spatial, H=224, W=224,
    )
    print("u_r:", {k: v.tolist() for k, v in ur.items()})

    # 2) u_f: simulate a denoising trajectory with late-step oscillation.
    T = 8
    x1_traj = []
    z = torch.randn(B, n_spatial, d_spatial)
    for i in range(T):
        z = _dummy_denoise(z, i)
        x1_traj.append(z)
    uf = flow_instability(x1_traj, z_prev)
    print("u_f:", {k: v.tolist() for k, v in uf.items()})

    # 3) u_s: simulate N=4 seed predictions.
    N = 4
    z_samples = z_pred.unsqueeze(1) + 0.05 * torch.randn(B, N, n_spatial, d_spatial)
    us = inter_seed_variance(z_samples, z_prev)
    print("u_s:", {k: v.tolist() for k, v in us.items()})


if __name__ == "__main__":
    main()
