"""
Curated excerpt from facebookresearch/vjepa2 src/models/ac_predictor.py
V-JEPA 2-AC action-conditioned predictor architecture used as Foresight's backbone.

Key facts from the paper & code:
- Encoder is FROZEN; only the predictor is trained on robot rollouts.
- predictor_embed maps encoder tokens (ViT-Giant, 1408-d) to predictor dim 1024.
- action_encoder / state_encoder map 7-DoF action/state deltas to predictor dim.
- 24 transformer layers, 16 heads, frame-causal attention, RoPE positional encoding.
- Output is projected back to encoder embedding dim (1408) for latent prediction.
"""

import torch
import torch.nn as nn


class VisionTransformerPredictorAC(nn.Module):
    """Action Conditioned Vision Transformer Predictor (V-JEPA 2-AC)."""

    def __init__(
        self,
        img_size=(256, 256),
        patch_size=16,
        num_frames=8,
        tubelet_size=2,
        embed_dim=1408,          # ViT-Giant encoder output dim
        predictor_embed_dim=1024, # Internal predictor dim
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        is_frame_causal=True,
        use_rope=True,
        action_embed_dim=7,      # EEF pose deltas (x,y,z,r,p,y,gripper)
        use_extrinsics=False,
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics

        # Map frozen encoder tokens into predictor space
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Encode the action chunk and current state into conditioning tokens
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim, bias=True)

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_height = img_size[0] // patch_size   # 16 for 256x256 / 16
        self.grid_width = img_size[1] // patch_size    # 16

        # 24 RoPE attention blocks (full config in configs/train/vitg16/droid-256px-8f.yaml)
        self.predictor_blocks = nn.ModuleList([
            ACBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_rope=use_rope,
                grid_size=self.grid_height,
            )
            for _ in range(depth)
        ])

        # Project back to encoder embedding dimension -> z_t^p
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

    def forward(self, x, actions, states, extrinsics=None):
        """
        Args:
            x: [B, T*H*W, embed_dim] hidden latent tokens z_t^h from frozen encoder.
            actions: [B, T-1, action_dim] policy action chunk A_t.
            states: [B, T, action_dim] current states (pose + gripper).
        Returns:
            [B, T*H*W, embed_dim] predicted latent tokens z_t^p.
        """
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)

        # Interleave action/state tokens in front of every frame's patch tokens
        s = self.state_encoder(states).unsqueeze(2)
        a = self.action_encoder(actions).unsqueeze(2)
        x = x.view(B, T, self.grid_height * self.grid_width, D)

        cond_tokens = 3 if self.use_extrinsics else 2
        if self.use_extrinsics:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, s, e, x], dim=2).flatten(1, 2)
        else:
            x = torch.cat([a, s, x], dim=2).flatten(1, 2)

        # Frame-causal attention mask: each frame sees itself and past frames
        attn_mask = self.attn_mask[:x.size(1), :x.size(1)].to(x.device)

        for blk in self.predictor_blocks:
            x = blk(
                x,
                mask=None,
                attn_mask=attn_mask,
                T=T,
                H=self.grid_height,
                W=self.grid_width,
                action_tokens=cond_tokens,
            )

        # Strip conditioning tokens and project back to encoder dim
        x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)
        x = x[:, :, cond_tokens:, :].flatten(1, 2)
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x


def vit_ac_predictor(**kwargs):
    return VisionTransformerPredictorAC(
        mlp_ratio=4, qkv_bias=True,
        norm_layer=lambda: nn.LayerNorm(1024, eps=1e-6),
        **kwargs,
    )


# Paper configuration match (Appendix 7):
#   ViT-Giant encoder frozen (1,012 M params), 1408-d patch tokens
#   24-layer predictor, 1024-d embed, 16 heads, frame-causal=True
#   Input: sliding window of 8 frames, 16x16 patches -> 256 tokens/frame
#   Mean-pooled over 256 spatial patches -> 1408-d latent per timestep
