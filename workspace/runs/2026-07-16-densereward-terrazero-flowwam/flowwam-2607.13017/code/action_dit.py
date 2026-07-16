"""
Action expert with per-layer video-feature conditioning (IDM).

Each action-DiT block cross-attends to one layer of the dual-stream video DiT's
hidden states (rgb + flow, concatenated along the token axis, dim = video hidden
dim). Those per-layer features are captured while the video is denoised, so the
action expert reads a full stack of video representations rather than a single
pooled latent.

Configurable heads:
    - pred_target="velocity": the head predicts the flow-matching velocity
      (noise - x0). "x0" makes the head predict the clean action and converts
      to velocity at inference.
    - use_rope=True: 1D rotary position embedding on the action self-attention.
      Set False for an additive learnable ``pos_embed``.
    - proprio_mode="text": the proprio (qpos) token is projected to the text
      dim and appended to the text context, entering through the text
      cross-attention. "state_token" prepends it to the action sequence.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def sinusoidal_embedding_1d(dim, position):
    position = position.reshape(-1)
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64,
                          device=position.device).div(dim // 2),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_rope_1d(head_dim: int, max_len: int, theta: float = 10000.0):
    """1D rotary frequencies (complex), shape [max_len, head_dim // 2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape [B, n_heads, S, head_dim].

    ``freqs_cis`` is complex, shape [S, head_dim // 2].
    """
    b, n, s, d = x.shape
    xc = torch.view_as_complex(x.float().reshape(b, n, s, d // 2, 2))
    fc = freqs_cis[:s].to(x.device).view(1, 1, s, d // 2)
    xo = torch.view_as_real(xc * fc).reshape(b, n, s, d)
    return xo.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        normed = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return normed.to(dtype) * self.weight


class ActionSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = rearrange(self.norm_q(self.q(x)), "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(self.norm_k(self.k(x)), "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(self.v(x), "b s (n d) -> b n s d", n=self.num_heads)
        if rope is not None:
            q = apply_rotary(q, rope)
            k = apply_rotary(k, rope)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b n s d -> b s (n d)")
        return self.o(out)


class ActionVideoCrossAttention(nn.Module):
    """Cross-attention from action tokens (dim) to video features (kv_dim)."""

    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.norm_q(self.q(x)), "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(self.norm_k(self.k(ctx)), "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(self.v(ctx), "b s (n d) -> b n s d", n=self.num_heads)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b n s d -> b s (n d)")
        return self.o(out)


class ActionTextCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.norm_q(self.q(x)), "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(self.norm_k(self.k(ctx)), "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(self.v(ctx), "b s (n d) -> b n s d", n=self.num_heads)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b n s d -> b s (n d)")
        return self.o(out)


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ActionDiTBlockIDM(nn.Module):
    """AdaLN self-attn (+RoPE) + cross-attn(video feats) + cross-attn(text) + FFN."""

    def __init__(self, dim: int, video_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.self_attn = ActionSelfAttention(dim, num_heads, eps)
        self.cross_attn = ActionVideoCrossAttention(dim, video_dim, num_heads, eps)
        self.cross_attn_text = ActionTextCrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.norm_text = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
        self,
        x: torch.Tensor,
        video_ctx: torch.Tensor,
        t_mod: torch.Tensor,
        text_ctx: torch.Tensor = None,
        rope: torch.Tensor = None,
    ) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        h = _modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa * self.self_attn(h, rope=rope)
        x = x + self.cross_attn(self.norm3(x), video_ctx)
        if text_ctx is not None:
            x = x + self.cross_attn_text(self.norm_text(x), text_ctx)
        h = _modulate(self.norm2(x), shift_ff, scale_ff)
        x = x + gate_ff * self.ffn(h)
        return x


class ActionHead(nn.Module):
    def __init__(self, dim: int, action_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, action_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = (
            self.modulation.to(dtype=t_emb.dtype, device=t_emb.device) + t_emb
        ).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class ActionExpertIDM(nn.Module):
    """Action expert conditioned on the video DiT's per-layer features.

    Args:
        dim:               Action hidden dim.
        video_dim:         Video-DiT hidden dim (e.g. 3072 for Wan2.2-5B).
        num_heads:         Action attention heads.
        num_layers:        Action depth (each block cross-attends one video layer).
        ffn_dim:           FFN intermediate dim.
        freq_dim:          Sinusoidal embedding dim.
        action_dim:        Output action dim.
        max_action_len:    Max action sequence length.
        text_context_dim:  T5 text dim (4096). 0 disables text cross-attn.
        joint_state_dim:   Proprio (qpos) dim. 0 disables proprio.
        pred_target:       "velocity" or "x0".
        use_rope:          1D RoPE on action self-attn vs learnable additive
                           pos_embed.
        proprio_mode:      "text" (append proprio to text context) or
                           "state_token" (prepend to action sequence).
        eps:               LayerNorm epsilon.
    """

    def __init__(
        self,
        dim: int = 1024,
        video_dim: int = 3072,
        num_heads: int = 16,
        num_layers: int = 30,
        ffn_dim: int = 4096,
        freq_dim: int = 256,
        action_dim: int = 14,
        max_action_len: int = 49,
        text_context_dim: int = 0,
        joint_state_dim: int = 0,
        pred_target: str = "velocity",
        use_rope: bool = True,
        proprio_mode: str = "text",
        eps: float = 1e-6,
    ):
        super().__init__()
        assert pred_target in ("velocity", "x0"), pred_target
        assert proprio_mode in ("text", "state_token"), proprio_mode
        self.dim = dim
        self.video_dim = video_dim
        self.num_heads = num_heads
        self.freq_dim = freq_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.max_action_len = max_action_len
        self.text_context_dim = text_context_dim
        self.joint_state_dim = joint_state_dim
        self.pred_target = pred_target
        self.use_rope = use_rope
        self.proprio_mode = proprio_mode

        self.action_embed = nn.Linear(action_dim, dim)
        if use_rope:
            # The complex RoPE table is deliberately NOT stored as a buffer:
            # module.to(bfloat16) would cast complex->real and destroy the
            # rotary phases. It is (cheaply) recomputed on the fly in forward.
            self.pos_embed = None
            self.rope_head_dim = dim // num_heads
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, max_action_len, dim) * 0.02)
            self.rope_head_dim = None

        # --- Proprio ---
        self.joint_state_encoder = None
        self.state_pos_embed = None
        self.proprio_to_text = None
        if joint_state_dim > 0:
            if proprio_mode == "text":
                assert text_context_dim > 0, (
                    "proprio_mode='text' requires text_context_dim > 0"
                )
                self.proprio_to_text = nn.Linear(joint_state_dim, text_context_dim)
            else:
                self.joint_state_encoder = nn.Linear(joint_state_dim, dim)
                self.state_pos_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # Cond noise-level embedding added to video features (zero-init).
        self.cond_ts_embedding = nn.Sequential(
            nn.Linear(freq_dim, video_dim),
            nn.SiLU(),
            nn.Linear(video_dim, video_dim),
        )
        nn.init.zeros_(self.cond_ts_embedding[-1].weight)
        nn.init.zeros_(self.cond_ts_embedding[-1].bias)

        if text_context_dim > 0:
            self.text_proj = nn.Sequential(
                nn.LayerNorm(text_context_dim),
                nn.Linear(text_context_dim, dim),
            )
        else:
            self.text_proj = None

        self.blocks = nn.ModuleList([
            ActionDiTBlockIDM(dim, video_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])

        self.head = ActionHead(dim, action_dim, eps)

    def _layer_map(self, num_video_layers: int) -> List[int]:
        if self.num_layers == 1:
            return [num_video_layers - 1]
        return [
            round(i * (num_video_layers - 1) / (self.num_layers - 1))
            for i in range(self.num_layers)
        ]

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        video_layer_feats: List[torch.Tensor],
        text_context: torch.Tensor = None,
        joint_state: torch.Tensor = None,
        sigma: torch.Tensor = None,
        cond_timestep: torch.Tensor = None,
        return_x0: bool = False,
    ) -> torch.Tensor:
        """Returns velocity (pred_target='velocity'), or x0 (pred_target='x0'
        with return_x0=True) / velocity (x0 with return_x0=False).
        """
        N = noisy_actions.shape[1]
        if len(video_layer_feats) == 0:
            raise ValueError("video_layer_feats must be a non-empty list")
        Lv = len(video_layer_feats)
        layer_map = self._layer_map(Lv)
        B = noisy_actions.shape[0]

        # Cond noise-level embedding (added to every video feat token).
        cs_emb = None
        if cond_timestep is not None:
            if cond_timestep.dim() == 0:
                cond_timestep = cond_timestep.expand(B)
            cs_emb = self.cond_ts_embedding(
                sinusoidal_embedding_1d(self.freq_dim, cond_timestep).to(
                    video_layer_feats[0].dtype)
            )[:, None, :]

        # Text context (+ optional appended proprio token).
        text_ctx = None
        if text_context is not None and self.text_proj is not None:
            ctx = text_context
            if self.proprio_to_text is not None and joint_state is not None:
                proprio_tok = self.proprio_to_text(
                    joint_state.to(dtype=ctx.dtype)
                ).unsqueeze(1)  # (B, 1, text_dim)
                ctx = torch.cat([ctx, proprio_tok], dim=1)
            text_ctx = self.text_proj(ctx)

        x = self.action_embed(noisy_actions)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :N]

        # Optional prepended proprio state token.
        has_state_token = False
        if self.joint_state_encoder is not None:
            assert joint_state is not None
            state_tok = self.joint_state_encoder(
                joint_state.to(dtype=x.dtype)).unsqueeze(1)
            state_tok = state_tok + self.state_pos_embed.to(dtype=x.dtype, device=x.device)
            x = torch.cat([state_tok, x], dim=1)
            has_state_token = True

        rope = None
        if self.use_rope:
            rope = precompute_rope_1d(self.rope_head_dim, x.shape[1]).to(x.device)

        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        t_head = t_mod[:, :2].reshape(t_mod.shape[0], 2, self.dim)

        for i, block in enumerate(self.blocks):
            vfeat = video_layer_feats[layer_map[i]]
            if cs_emb is not None:
                vfeat = vfeat + cs_emb.to(dtype=vfeat.dtype)
            x = block(x, vfeat, t_mod, text_ctx, rope=rope)

        if has_state_token:
            x = x[:, 1:]

        head_out = self.head(x, t_head)

        if self.pred_target == "velocity":
            return head_out

        # pred_target == "x0"
        if return_x0:
            return head_out
        assert sigma is not None, "sigma required for x0->velocity conversion"
        sig = sigma.to(device=head_out.device, dtype=head_out.dtype)
        while sig.dim() < head_out.dim():
            sig = sig.unsqueeze(-1)
        sig = sig.clamp(min=1e-6)
        return (noisy_actions - head_out) / sig

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_action_expert_idm(
    action_dim: int = 14,
    num_frames: int = 49,
    video_dim: int = 3072,
    text_context_dim: int = 0,
    joint_state_dim: int = 0,
    num_layers: int = 30,
    dim: int = 1024,
    num_heads: int = 16,
    ffn_dim: int = 4096,
    pred_target: str = "velocity",
    use_rope: bool = True,
    proprio_mode: str = "text",
) -> ActionExpertIDM:
    """Factory. Defaults: velocity head, RoPE, proprio appended to text."""
    model = ActionExpertIDM(
        dim=dim,
        video_dim=video_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        freq_dim=256,
        action_dim=action_dim,
        max_action_len=num_frames,
        text_context_dim=text_context_dim,
        joint_state_dim=joint_state_dim,
        pred_target=pred_target,
        use_rope=use_rope,
        proprio_mode=proprio_mode,
    )
    print(
        f"[ActionExpertIDM] Built: {model.param_count() / 1e6:.1f}M params, "
        f"dim={dim}, num_layers={num_layers}, video_dim={video_dim}, "
        f"action_dim={action_dim}, text_context_dim={text_context_dim}, "
        f"joint_state_dim={joint_state_dim}, max_action_len={num_frames}, "
        f"pred_target={pred_target}, use_rope={use_rope}, proprio_mode={proprio_mode}"
    )
    return model
