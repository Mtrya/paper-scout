"""
Next Forcing: Multi-Chunk Prediction (MCP) Module — PyTorch Sketch

Derived from the paper:
  "Next Forcing: Causal World Modeling with Multi-Chunk Prediction"
  Gangwei Xu et al. (arXiv:2606.11187)

This is a compact, self-contained sketch of the MCP architecture layered on top
of a standard transformer video diffusion backbone (e.g. Wan2.2, 30 layers).
It is meant to clarify the mechanism, not to be a production training script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MCPConfig:
    # Backbone
    hidden_dim: int = 1536
    n_layers: int = 30
    # MCP design (from paper Sec 4.3 / 5.1)
    mcp_depths: Tuple[int, ...] = (1, 2, 3)          # next1, next2, next3
    mcp_layers: Tuple[int, ...] = (4, 12, 20, 30)    # feature extraction layers
    mcp_blocks_per_depth: int = 3
    mcp_timestep_shift: float = 10.0                   # s_mcp
    main_timestep_shift: float = 5.0                   # s_main
    # Loss weights (from paper Sec 5.1)
    loss_weights: Tuple[float, ...] = (0.5, 0.2, 0.1)
    # Inference
    parallel_mode: bool = False                        # keep MCP at inference?


# ---------------------------------------------------------------------------
# Timestep shift (Appendix C, Eq. 14)
# ---------------------------------------------------------------------------

def build_shifted_schedule(T: int = 1000, sigma_min: float = 0.0,
                           sigma_max: float = 1.0, shift: float = 5.0) -> torch.Tensor:
    """Build a shifted noise-level schedule."""
    sigma = torch.linspace(sigma_min, sigma_max, T)
    sigma_tilde = (shift * sigma) / (1.0 + (shift - 1.0) * sigma)
    return sigma_tilde


def sample_timestep(shifted_schedule: torch.Tensor, batch_size: int,
                    device: torch.device) -> torch.Tensor:
    """Uniform index sample → look up shifted noise level."""
    T = shifted_schedule.numel()
    ids = torch.randint(0, T, (batch_size,), device=device)
    return shifted_schedule[ids]


# ---------------------------------------------------------------------------
# Multi-Layer Feature Fusion (Sec 4.3, Eq. 7)
# ---------------------------------------------------------------------------

class FeatureFusion(nn.Module):
    """
    Collects hidden states from K intermediate backbone layers,
    concatenates them, and compresses back to hidden_dim via a 2-layer MLP.
    """
    def __init__(self, hidden_dim: int, source_layers: Tuple[int, ...]):
        super().__init__()
        self.source_layers = source_layers
        in_dim = hidden_dim * len(source_layers)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # hidden_states[k]: [B, N, D] from backbone layer source_layers[k]
        selected = [hidden_states[i - 1] for i in self.source_layers]
        fused = torch.cat(selected, dim=-1)          # [B, N, K*D]
        return self.mlp(fused)                       # [B, N, D]


# ---------------------------------------------------------------------------
# Causal MCP Chain (Sec 4.3, Eq. 8)
# ---------------------------------------------------------------------------

class MCPDepth(nn.Module):
    """
    One depth in the causal MCP chain.
    Input:  previous-depth hidden state + embedded noisy target
    Output: velocity prediction + hidden state for next depth
    """
    def __init__(self, hidden_dim: int, num_blocks: int):
        super().__init__()
        # Projection W_k that fuses previous depth + embedded target
        self.fuse_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        # Lightweight transformer blocks (paper uses 3)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim) for _ in range(num_blocks)
        ])
        # Flow-matching velocity head
        self.velocity_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_prev: torch.Tensor, x_embed: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_prev: [B, N, D]   from previous depth (or fused features for k=1)
        # x_embed: [B, N, D]  patch-embedded noisy shifted target
        z = torch.cat([h_prev, x_embed], dim=-1)     # [B, N, 2D]
        h = self.fuse_proj(z)                        # [B, N, D]
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask)
        v = self.velocity_head(h)                    # predicted velocity
        return v, h                                  # h feeds next depth


class TransformerBlock(nn.Module):
    """Minimal transformer block for sketch purposes."""
    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                          attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full Next-Forcing Training Wrapper
# ---------------------------------------------------------------------------

class NextForcingTrainer(nn.Module):
    """
    Wraps a main video diffusion backbone with MCP auxiliary modules.

    During training:
      - Main model denoises current chunk (standard teacher forcing).
      - MCP modules simultaneously denoise next1/next2/next3 chunks.
      - Loss = L_video + L_action + Σ w_k * L_mcp^k

    During inference:
      - Zero-overhead: drop MCP, use main model only.
      - Parallel: keep depth-1 MCP to predict next chunk alongside current one.
    """
    def __init__(self, backbone: nn.Module, patch_embed: nn.Module, config: MCPConfig):
        super().__init__()
        self.backbone = backbone          # e.g. 30-layer Wan2.2 Transformer
        self.patch_embed = patch_embed    # shared patch embed for main + MCP
        self.cfg = config

        self.fusion = FeatureFusion(config.hidden_dim, config.mcp_layers)
        self.mcp_modules = nn.ModuleList([
            MCPDepth(config.hidden_dim, config.mcp_blocks_per_depth)
            for _ in config.mcp_depths
        ])

        # Pre-compute shifted schedules
        self.register_buffer("schedule_main",
            build_shifted_schedule(shift=config.main_timestep_shift))
        self.register_buffer("schedule_mcp",
            build_shifted_schedule(shift=config.mcp_timestep_shift))

    def forward(
        self,
        x_t: torch.Tensor,                # noisy current chunk  [B, C, M, H, W]
        x0_gt: torch.Tensor,              # clean full video     [B, C, F, H, W]
        chunk_idx: int,
        context: torch.Tensor | None = None,
    ) -> dict:
        """
        Training forward. Returns dict of losses.
        """
        B, C, F, H, W = x0_gt.shape
        N = x_t.shape[1] if x_t.dim() == 3 else x_t.numel() // B  # latent tokens
        device = x_t.device

        # ---- Main model forward (current chunk) ----
        # We assume backbone can return intermediate hidden states when requested.
        hidden_states, v_main = self._main_forward(x_t, context, return_all_layers=True)

        # Flow-matching target for main model
        # (In practice, ε and t are sampled outside; here we keep the interface simple.)
        # L_video = ||v_main - (ε - x0_current)||^2

        # ---- MCP: multi-layer feature fusion ----
        h_fuse = self.fusion(hidden_states)   # [B, N, D]

        # ---- MCP: causal chain over depths ----
        mcp_losses = []
        h_prev = h_fuse
        for k, mcp in enumerate(self.mcp_modules, start=1):
            # 1. Temporal chunk shift (Eq. 4)
            shifted = self._shift_chunk(x0_gt, chunk_idx, k)   # [B, C, M, H, W]

            # 2. Independent noise injection (Eq. 5)
            t_k = sample_timestep(self.schedule_mcp, B, device)
            eps_k = torch.randn_like(shifted)
            x_tk = (1.0 - t_k.view(B, 1, 1, 1, 1)) * shifted + t_k.view(B, 1, 1, 1, 1) * eps_k

            # 3. Embed noisy target
            x_embed = self.patch_embed(x_tk)   # [B, N, D]

            # 4. RoPE position encoding shift (Eq. 6)
            # In practice: modify RoPE frequency computation by +k chunk positions.
            # Here we note it conceptually:
            #   rope(x_tk, pos=i+k) instead of rope(x_tk, pos=i)

            # 5. MCP depth forward
            v_k, h_prev = mcp(h_prev, x_embed, attn_mask=None)

            # 6. MCP flow-matching loss (Eq. 12)
            target = eps_k.flatten(2).transpose(1, 2)   # [B, N, C]  (simplified)
            x0_flat = shifted.flatten(2).transpose(1, 2)
            target_velocity = target - x0_flat
            L_k = F.mse_loss(v_k, target_velocity)
            mcp_losses.append(L_k)

        # ---- Total loss (Eq. 13) ----
        total_mcp_loss = sum(
            w * L for w, L in zip(self.cfg.loss_weights, mcp_losses)
        )

        return {
            "main_loss": torch.tensor(0.0, device=device),  # placeholder
            "mcp_losses": mcp_losses,
            "total_mcp_loss": total_mcp_loss,
        }

    # -----------------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def infer_standard(self, x_t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Zero-overhead inference: drop MCP, main model only."""
        _, v = self._main_forward(x_t, context, return_all_layers=False)
        return v

    @torch.no_grad()
    def infer_parallel(self, x_t: torch.Tensor, context: torch.Tensor,
                       h_fuse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel chunk generation: main model predicts current chunk,
        depth-1 MCP predicts next chunk simultaneously.
        Returns (v_current, v_next1).
        """
        hidden_states, v_current = self._main_forward(x_t, context, return_all_layers=True)
        h_fuse = self.fusion(hidden_states)

        # Next1 chunk: we need a noisy embedding of the *next* chunk target.
        # At inference we start from pure noise for the next chunk.
        x_next_noisy = torch.randn_like(x_t)
        x_embed = self.patch_embed(x_next_noisy)
        v_next1, _ = self.mcp_modules[0](h_fuse, x_embed)
        return v_current, v_next1

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _main_forward(self, x_t: torch.Tensor, context: torch.Tensor | None,
                      return_all_layers: bool = True):
        """
        Placeholder for the main backbone forward.
        In practice, this is the 30-layer Wan2.2 Transformer from LingBot-VA.
        When return_all_layers=True, returns (list_of_hiddens, velocity).
        """
        raise NotImplementedError("Plug in the actual LingBot-VA / Wan2.2 backbone here.")

    def _shift_chunk(self, x0: torch.Tensor, i: int, k: int) -> torch.Tensor:
        """Temporal chunk shift (Eq. 4): advance by k chunks, replicate last chunk for padding."""
        _, C, F, H, W = x0.shape
        M = F  # simplified: treat each chunk as one step
        idx = min(i + k, F - 1)
        return x0[:, :, idx:idx + 1, :, :].expand(-1, -1, M, -1, -1)


# ---------------------------------------------------------------------------
# Self-contained runnable sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = MCPConfig()
    print("Config:", cfg)

    # Shift schedule sanity check
    sched = build_shifted_schedule(shift=10.0)
    print("Shift schedule mean:", sched.mean().item(), "(should be > 0.5 for s=10)")

    # Feature fusion shape test
    B, N, D = 2, 64, cfg.hidden_dim
    hidden_states = [torch.randn(B, N, D) for _ in cfg.mcp_layers]
    fusion = FeatureFusion(D, cfg.mcp_layers)
    out = fusion(hidden_states)
    assert out.shape == (B, N, D), out.shape
    print("FeatureFusion output shape:", out.shape)

    # MCP depth shape test
    mcp = MCPDepth(D, cfg.mcp_blocks_per_depth)
    h_prev = torch.randn(B, N, D)
    x_embed = torch.randn(B, N, D)
    v, h_next = mcp(h_prev, x_embed)
    assert v.shape == (B, N, D)
    assert h_next.shape == (B, N, D)
    print("MCPDepth output shapes:", v.shape, h_next.shape)

    print("All sanity checks passed.")
