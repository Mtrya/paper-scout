"""
PyTorch reconstruction of the AHA-WAM layerwise joint attention (Eq. 3).

In AHA-WAM, the action DiT does not attend to dense visual tokens directly.
Instead, at each layer ℓ, the action branch's queries attend to a concatenation
of:
  - its own action keys/values (local action context)
  - the OVCR-adapted planner keys/values (long-horizon world context)

This preserves WAM-style interaction between visual dynamics and action
generation while amortizing expensive video-DiT computation across multiple
high-frequency action updates.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def layerwise_joint_attention(
    q_action: torch.Tensor,          # [B, S_a, d]
    k_action: torch.Tensor,          # [B, S_a, d]
    v_action: torch.Tensor,          # [B, S_a, d]
    k_planner: torch.Tensor,         # [B, S_p, d]  -- OVCR-adapted K̃
    v_planner: torch.Tensor,         # [B, S_p, d]  -- OVCR-adapted Ṽ
    num_heads: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Eq. 3 from AHA-WAM:

        H̄_t^{a,ℓ} = Attn( Q_t^{a,ℓ}, [K_t^{a,ℓ} ; K̃_t^{p,ℓ}],
                                      [V_t^{a,ℓ} ; Ṽ_t^{p,ℓ}] )

    Args:
        q_action: Action queries [B, S_a, d]
        k_action: Action keys [B, S_a, d]
        v_action: Action values [B, S_a, d]
        k_planner: Adapted planner keys [B, S_p, d]
        v_planner: Adapted planner values [B, S_p, d]
        num_heads: Number of attention heads.
        attention_mask: Optional mask [B, S_a, S_a + S_p] or broadcastable.

    Returns:
        Planner-conditioned action hidden state [B, S_a, d]
    """
    B, S_a, d = q_action.shape
    S_p = k_planner.shape[1]
    head_dim = d // num_heads

    # Concatenate action and planner keys/values.
    k_cat = torch.cat([k_action, k_planner], dim=1)   # [B, S_a + S_p, d]
    v_cat = torch.cat([v_action, v_planner], dim=1)   # [B, S_a + S_p, d]

    # Reshape to multi-head: [B, num_heads, seq_len, head_dim]
    def reshape(t, seq_len):
        return t.view(B, seq_len, num_heads, head_dim).transpose(1, 2)

    q = reshape(q_action, S_a)        # [B, H, S_a, Dh]
    k = reshape(k_cat, S_a + S_p)     # [B, H, S_a+S_p, Dh]
    v = reshape(v_cat, S_a + S_p)     # [B, H, S_a+S_p, Dh]

    # Scaled dot-product attention.
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, H, S_a, S_a+S_p]

    if attention_mask is not None:
        # Assume mask is True for positions to attend to.
        scores = scores.masked_fill(~attention_mask.unsqueeze(1), float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)  # [B, H, S_a, Dh]

    # Merge heads.
    out = out.transpose(1, 2).contiguous().view(B, S_a, d)
    return out


class AsyncDualDiTBlock(nn.Module):
    """
    A single transformer block in the asynchronous dual-DiT architecture.

    The video planner block runs infrequently and produces K/V context that is
    cached. The action executor block runs frequently; at each denoising step
    it consumes the OVCR-adapted planner context via layerwise joint attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Action self-attention projections.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # FFN.
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model, eps=eps),
            nn.Linear(d_model, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, d_model),
        )

        # Modulation (simplified; real DiT uses adaptive layer norm).
        self.norm1 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.modulation = nn.Parameter(torch.randn(1, 6, d_model) / d_model ** 0.5)

    def forward(
        self,
        action_tokens: torch.Tensor,          # [B, S_a, d]
        t_mod: torch.Tensor,                  # [B, 6, d]  time modulation
        adapted_planner_kv: Optional[Dict[str, torch.Tensor]] = None,
        action_freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            action_tokens: Noisy action tokens at this layer.
            t_mod: Time/step modulation for AdaLN.
            adapted_planner_kv: Dict with "k" and "v" from OVCR, or None for
                the first layer (where pre_dit already handled it).
            action_freqs: RoPE frequencies for action tokens.

        Returns:
            Updated action tokens [B, S_a, d]
        """
        # AdaLN modulation.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            (self.modulation + t_mod.unsqueeze(1)).chunk(6, dim=1)
        )
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        shift_mlp = shift_mlp.squeeze(1)
        scale_mlp = scale_mlp.squeeze(1)
        gate_mlp = gate_mlp.squeeze(1)

        # Pre-norm + modulation.
        x = action_tokens
        attn_input = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        # Action Q/K/V.
        q = self.q_proj(attn_input)
        k = self.k_proj(attn_input)
        v = self.v_proj(attn_input)

        # Apply RoPE if provided (simplified).
        if action_freqs is not None:
            # action_freqs shape: [S_a, 1, head_dim]
            # Real implementation uses complex rotations; here we just add for demo.
            q = q + action_freqs.view(1, -1, self.d_model).expand(q.shape[0], -1, -1)
            k = k + action_freqs.view(1, -1, self.d_model).expand(k.shape[0], -1, -1)

        if adapted_planner_kv is not None:
            # Layerwise joint attention with planner context.
            mixed = layerwise_joint_attention(
                q_action=q,
                k_action=k,
                v_action=v,
                k_planner=adapted_planner_kv["k"],
                v_planner=adapted_planner_kv["v"],
                num_heads=self.num_heads,
            )
        else:
            # Pure action self-attention (fallback).
            mixed = layerwise_joint_attention(
                q_action=q,
                k_action=k,
                v_action=v,
                k_planner=torch.zeros_like(k),
                v_planner=torch.zeros_like(v),
                num_heads=self.num_heads,
            )

        # Residual + gate.
        x = x + gate_msa.unsqueeze(1) * self.o_proj(mixed)

        # FFN.
        mlp_input = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(mlp_input)
        return x


if __name__ == "__main__":
    B, S_a, S_p, d = 2, 16, 256, 1024
    num_heads = 16

    block = AsyncDualDiTBlock(d_model=d, num_heads=num_heads, ffn_dim=d * 4)
    action_tokens = torch.randn(B, S_a, d)
    t_mod = torch.randn(B, 6, d)
    adapted_kv = {
        "k": torch.randn(B, S_p, d),
        "v": torch.randn(B, S_p, d),
    }

    out = block(action_tokens, t_mod, adapted_kv)
    assert out.shape == (B, S_a, d)
    print("Layerwise joint attention reconstruction sanity check passed.")
