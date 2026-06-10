"""
PyTorch reconstruction of Observation-Guided Video-Context Routing (OVCR).

Based on Eqs. 8-10 in AHA-WAM (arXiv:2606.09811).

OVCR turns static planner-context caching into observation-conditioned retrieval
and adaptation. For each action chunk, it:
  1. Constructs routing queries from the latest visual observation (Eq. 8)
  2. Reads planner features via attention (Eq. 9, first half)
  3. Predicts residual K/V updates with a lightweight router (Eq. 9, second half)
  4. Applies gated residual updates to the cached planner K/V (Eq. 10)

The adapted context is then consumed by the action DiT through layerwise joint
attention (Eq. 3; see layerwise_joint_attention.py).
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationGuidedVideoContextRouter(nn.Module):
    """
    Per-layer OVCR module that adapts cached planner K/V context to the latest
    visual observation.

    Args:
        d_model: Hidden dimension of the transformer layer (must match both
            video planner and action DiT at this layer).
        num_routing_queries: Number of learnable base queries Q. Paper uses 32.
        num_heads: Number of attention heads.
        eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        d_model: int,
        num_routing_queries: int = 32,
        num_heads: int = 24,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_routing_queries = num_routing_queries
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # ------------------------------------------------------------------
        # Eq. 8: construct observation-guided routing queries
        # ------------------------------------------------------------------
        # Learnable base queries B ∈ R^{Q x d}
        self.base_queries = nn.Parameter(torch.randn(num_routing_queries, d_model) * 0.02)

        # Lightweight visual projection f_v for observation tokens.
        # In the paper, visual observations are encoded by the pretrained VAE;
        # here we assume observation tokens are already in model hidden dim.
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(d_model, eps=eps),
            nn.Linear(d_model, d_model),
        )

        # ------------------------------------------------------------------
        # Eq. 9: read planner features + predict residual K/V updates
        # ------------------------------------------------------------------
        # The read features R_t^ℓ are produced by cross-attention from routing
        # queries to planner K/V. We then concatenate R with the routing queries
        # and feed through a lightweight MLP to predict ΔK and ΔV.
        self.router_mlp = nn.Sequential(
            nn.LayerNorm(d_model, eps=eps),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model * 4, d_model * 2),  # outputs ΔK and ΔV
        )

        # ------------------------------------------------------------------
        # Eq. 10: learned gate α_t^ℓ
        # ------------------------------------------------------------------
        # A scalar gate per routing query, averaged to a layer gate.
        # We implement it as a small projection from the concatenated
        # [read_features; routing_queries] pooled representation.
        self.gate_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2, eps=eps),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def _attention_pool(
        self,
        queries: torch.Tensor,          # [B, Q, d]
        keys: torch.Tensor,             # [B, S, d]
        values: torch.Tensor,           # [B, S, d]
    ) -> torch.Tensor:
        """Scaled dot-product attention without multi-head split.

        Used for Eq. 8 (observation pooling) and Eq. 9 (planner read).
        """
        d = queries.shape[-1]
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, values)
        return out

    def forward(
        self,
        observation_visual_tokens: torch.Tensor,   # [B, N_vis, d]
        planner_k: torch.Tensor,                     # [B, S_p, d]  cached K
        planner_v: torch.Tensor,                     # [B, S_p, d]  cached V
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            observation_visual_tokens: Current visual observation tokens.
                In AHA-WAM these come from the VAE-encoded latest frame.
            planner_k: Cached video-planner key context from the latest refresh.
            planner_v: Cached video-planner value context from the latest refresh.

        Returns:
            adapted_k: Observation-adapted planner keys [B, S_p, d]
            adapted_v: Observation-adapted planner values [B, S_p, d]
        """
        B = observation_visual_tokens.shape[0]

        # ------------------------------------------------------------------
        # Eq. 8: Z_t^q = Attn(B, f_v(X_t^v), f_v(X_t^v))
        # ------------------------------------------------------------------
        visual_proj = self.visual_proj(observation_visual_tokens)  # [B, N_vis, d]
        base_q = self.base_queries.unsqueeze(0).expand(B, -1, -1)   # [B, Q, d]
        routing_queries = self._attention_pool(
            queries=base_q,
            keys=visual_proj,
            values=visual_proj,
        )  # [B, Q, d]

        # ------------------------------------------------------------------
        # Eq. 9: R_t^ℓ = Attn(Z_t^{q,ℓ}, K^{p}, V^{p})
        #         (ΔK, ΔV) = g_ψ(R, Z^q)
        # ------------------------------------------------------------------
        read_features = self._attention_pool(
            queries=routing_queries,
            keys=planner_k,
            values=planner_v,
        )  # [B, Q, d]

        # Concatenate read features with routing queries for router input.
        router_input = torch.cat([read_features, routing_queries], dim=-1)  # [B, Q, 2d]

        # Pool across queries to get a single vector per batch item.
        pooled = router_input.mean(dim=1)  # [B, 2d]

        # Predict residual updates. The paper says g_ψ^ℓ is a lightweight
        # layerwise router; we produce per-token residuals by broadcasting.
        gate = self.gate_proj(pooled)  # [B, 1]

        # The router outputs ΔK and ΔV. We expand to planner sequence length.
        residuals = self.router_mlp(pooled)  # [B, 2d]
        delta_k, delta_v = residuals.chunk(2, dim=-1)  # each [B, d]

        # Broadcast to planner sequence length.
        delta_k = delta_k.unsqueeze(1).expand(-1, planner_k.shape[1], -1)  # [B, S_p, d]
        delta_v = delta_v.unsqueeze(1).expand(-1, planner_v.shape[1], -1)  # [B, S_p, d]

        # ------------------------------------------------------------------
        # Eq. 10: gated residual update
        #   K̃ = K + α ΔK,   Ṽ = V + α ΔV
        # ------------------------------------------------------------------
        gate = gate.unsqueeze(-1)  # [B, 1, 1]
        adapted_k = planner_k + gate * delta_k
        adapted_v = planner_v + gate * delta_v

        return adapted_k, adapted_v


class OVCRCoupling(nn.Module):
    """
    Full layerwise OVCR coupling: one router per transformer layer.

    This is what AHA-WAM actually instantiates: for each of the L layers in the
    dual-DiT stack, there is an independent OVCR module with its own base
    queries, visual projection, and router MLP. Total OVCR parameters are
    reported as part of the 1.22B memory+routing parameters.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_routing_queries: int = 32,
        num_heads: int = 24,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.routers = nn.ModuleList([
            ObservationGuidedVideoContextRouter(
                d_model=d_model,
                num_routing_queries=num_routing_queries,
                num_heads=num_heads,
                eps=eps,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        observation_visual_tokens: torch.Tensor,
        planner_context: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            observation_visual_tokens: [B, N_vis, d]
            planner_context: List of L dicts, each with keys "k", "v".
                These are the cached planner K/V from the latest video-DiT refresh.

        Returns:
            adapted_context: List of L dicts with adapted "k", "v".
        """
        adapted = []
        for layer_idx, layer_cache in enumerate(planner_context):
            k_adapted, v_adapted = self.routers[layer_idx](
                observation_visual_tokens=observation_visual_tokens,
                planner_k=layer_cache["k"],
                planner_v=layer_cache["v"],
            )
            adapted.append({"k": k_adapted, "v": v_adapted})
        return adapted


# ---------------------------------------------------------------------------
# Sanity-check script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, L, S_p, N_vis, d = 2, 30, 256, 64, 1024
    num_routing_queries = 32

    # Simulate planner context from a video-DiT prefill.
    planner_context = [
        {"k": torch.randn(B, S_p, d), "v": torch.randn(B, S_p, d)}
        for _ in range(L)
    ]

    # Simulate current visual observation tokens (post-VAE).
    obs_tokens = torch.randn(B, N_vis, d)

    ovcr = OVCRCoupling(
        num_layers=L,
        d_model=d,
        num_routing_queries=num_routing_queries,
        num_heads=24,
    )

    adapted_context = ovcr(obs_tokens, planner_context)
    assert len(adapted_context) == L
    assert adapted_context[0]["k"].shape == (B, S_p, d)
    assert adapted_context[0]["v"].shape == (B, S_p, d)

    total_params = sum(p.numel() for p in ovcr.parameters())
    print(f"OVCR parameters: {total_params / 1e6:.2f}M")
    print("OVCR reconstruction sanity check passed.")
