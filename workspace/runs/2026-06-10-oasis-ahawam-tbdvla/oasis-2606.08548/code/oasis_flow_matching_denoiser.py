"""
Reconstruction of the OASIS Flow-Matching denoiser based on Sec. 3.3 of the paper.

Key design choices from the paper:
- Transformer-based denoiser with action chunking (F=32 frames)
- Inputs: frozen CLIP text encoder, frozen DINOv2 visual encoder (3-view images),
  MLP-encoded proprioception over H=2 frames
- Output: whole-body reference motion m_{t:t+F} in R^{F x 67}
- Trained with conditional flow matching (Eq. 2) using linear interpolation
- Inference: Euler solver with 10 denoising steps

This is a faithful architectural reconstruction; exact hyperparameters
(hidden dim, number of layers) are not specified in the paper and are
inferred from comparable systems (e.g., GR00T N1, Diffusion Policy).
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Timestep embedding used in diffusion/flow models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TransformerDenoiser(nn.Module):
    """
    Reconstructed OASIS high-level planner.

    Architecture:
    1. Encode condition tokens (text + vision + proprioception) -> condition sequence c
    2. Concatenate noisy action chunk a_tau as "query" tokens
    3. Cross-attention Transformer layers predict velocity field v_theta(a_tau, tau, c)
    """

    def __init__(
        self,
        action_dim: int = 67,
        chunk_size: int = 32,
        text_embed_dim: int = 512,   # CLIP text encoder output
        vision_embed_dim: int = 768,  # DINOv2-B output (ViT-B/14 -> 768)
        proprio_dim: int = 67 * 2,   # H=2 frames of reference motion commands
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Proprioception MLP (Sec. 3.3.1: "encoded by an MLP")
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project frozen encoders into hidden_dim
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_embed_dim, hidden_dim)

        # Action chunk embedding (noisy input a_tau)
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Transformer layers: action tokens attend to condition tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: predict velocity field
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        a_tau: torch.Tensor,          # (B, F, action_dim) noisy action chunk
        tau: torch.Tensor,            # (B,) noise level in [0, 1]
        text_embed: torch.Tensor,     # (B, text_embed_dim) from frozen CLIP
        vision_embed: torch.Tensor,   # (B, 3, vision_embed_dim) from frozen DINOv2
        proprio: torch.Tensor,        # (B, H * action_dim) reference motion history
    ) -> torch.Tensor:
        B = a_tau.size(0)

        # 1. Build condition token sequence c
        time_tokens = self.time_embed(tau).unsqueeze(1)           # (B, 1, D)
        text_tokens = self.text_proj(text_embed).unsqueeze(1)     # (B, 1, D)
        vision_tokens = self.vision_proj(vision_embed)            # (B, 3, D)
        proprio_tokens = self.proprio_encoder(proprio).unsqueeze(1)  # (B, 1, D)

        cond_tokens = torch.cat([time_tokens, text_tokens, vision_tokens, proprio_tokens], dim=1)
        # cond_tokens: (B, 1 + 1 + 3 + 1 = 6, D)

        # 2. Embed noisy actions
        action_tokens = self.action_embed(a_tau)  # (B, F, D)

        # 3. Concatenate and pass through Transformer
        tokens = torch.cat([cond_tokens, action_tokens], dim=1)  # (B, 6 + F, D)
        tokens = self.transformer(tokens)

        # 4. Extract action token outputs and predict velocity
        action_out = tokens[:, -self.chunk_size:, :]  # (B, F, D)
        velocity = self.velocity_head(action_out)     # (B, F, action_dim)
        return velocity


def flow_matching_loss(
    denoiser: TransformerDenoiser,
    text_embed: torch.Tensor,
    vision_embed: torch.Tensor,
    proprio: torch.Tensor,
    a_0: torch.Tensor,      # ground-truth action chunk
) -> torch.Tensor:
    """
    Eq. 2 from the paper:
        L_FM = E_{tau, a_0, a_1} [ || v_theta(a_tau, tau, c) - (a_1 - a_0) ||^2 ]
    where a_tau = (1 - tau) * a_0 + tau * a_1,  a_1 ~ N(0, I)
    """
    B, F, D = a_0.shape
    a_1 = torch.randn_like(a_0)
    tau = torch.rand(B, device=a_0.device)

    # Linear interpolation (constant-velocity path)
    tau_expanded = tau.view(B, 1, 1)
    a_tau = (1 - tau_expanded) * a_0 + tau_expanded * a_1

    # Predict velocity field
    v_pred = denoiser(a_tau, tau, text_embed, vision_embed, proprio)

    # Target is constant velocity: a_1 - a_0
    target = a_1 - a_0
    loss = F.mse_loss(v_pred, target)
    return loss


def euler_sampler(
    denoiser: TransformerDenoiser,
    text_embed: torch.Tensor,
    vision_embed: torch.Tensor,
    proprio: torch.Tensor,
    num_steps: int = 10,
    action_dim: int = 67,
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    Inference: integrate learned velocity field with Euler solver.
    Paper specifies 10 denoising steps.
    """
    B = text_embed.size(0)
    device = text_embed.device

    # Start from Gaussian prior a_1 ~ N(0, I)
    a = torch.randn(B, chunk_size, action_dim, device=device)
    dt = -1.0 / num_steps  # integrate from tau=1 to tau=0

    for i in range(num_steps):
        tau = torch.ones(B, device=device) * (1.0 + i * dt)
        v = denoiser(a, tau, text_embed, vision_embed, proprio)
        a = a + dt * v

    return a  # final denoised action chunk a_0


if __name__ == "__main__":
    # Smoke test
    B = 4
    model = TransformerDenoiser()
    text = torch.randn(B, 512)
    vision = torch.randn(B, 3, 768)
    proprio = torch.randn(B, 67 * 2)
    a_0 = torch.randn(B, 32, 67)

    loss = flow_matching_loss(model, text, vision, proprio, a_0)
    print(f"Loss: {loss.item():.4f}")

    with torch.no_grad():
        pred = euler_sampler(model, text, vision, proprio)
    print(f"Sampled action shape: {pred.shape}")
