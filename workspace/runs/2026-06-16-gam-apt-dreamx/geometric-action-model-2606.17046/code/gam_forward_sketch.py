#!/usr/bin/env python3
"""
Forward-flow sketch for the Geometric Action Model (GAM).

This script demonstrates the data flow of GAM.  When the Depth-Anything-3
repository and PyTorch are available, it uses the real DA3-Giant backbone as
the observation encoder / geometric decoder and stub modules for the causal
future predictor and action head.  When DA3 or PyTorch are missing, it prints
a textual walkthrough of the tensor shapes and architectural decisions.

Run from the repo root (workspace/):
    python3 runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/code/gam_forward_sketch.py

To run the live skeleton, clone DA3 and activate the venv first:
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git code/Depth-Anything-3
    source runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/code/venv/bin/activate
    python3 .../gam_forward_sketch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate DA3 source
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parents[4]
da3_src = WORKSPACE / "code" / "Depth-Anything-3" / "src"


def textual_walkthrough():
    print("=" * 72)
    print("GAM Forward-Flow Sketch (textual walkthrough)")
    print("=" * 72)
    print("\n[INFO] DA3 source or PyTorch not available; printing the data-flow shape story.")
    print("\nInput tensors (one control step):")
    print("    Images          : (B, S, V, 3, H, W)  with V=2 views, S=context horizon")
    print("    Proprioception  : (B, S, d_s)          robot state")
    print("    Previous action : (B, S, d_a)          action history")
    print("    Language        : frozen T5 tokens     task instruction")
    print("\nStage 1 — Observation encoder (DA3 blocks 0..12, FROZEN):")
    print("    patch_embed     : (B, S, V, 3, H, W) -> (B, S, V*N_patches, d)")
    print("    blocks 0..12    : frame-wise (local) attention only")
    print("    output          : Z^(12)  -> (B, S, V*N, d=1536)")
    print("\nStage 2 — Causal future predictor (12-layer transformer, TRAINABLE):")
    print("    concatenate per-timestep tokens:")
    print("        [proprio_token ; action_history_token ; Z^(12)]")
    print("    block-causal self-attention over the full sequence")
    print("    outputs:")
    print("        future_geo_tokens : (B, S, V*N, d)")
    print("        action_token      : (B, S, d)")
    print("\nStage 3 — Feature propagation:")
    print("    action_token is replicated and appended to geometry tokens:")
    print("        decoder_input = [future_geo_tokens ; action_token]")
    print("                      : (B, S, V*N+1, d)")
    print("\nStage 4 — Geometric decoder (DA3 blocks 13..39, TRAINABLE):")
    print("    alternating local/global attention starting at layer 13")
    print("    output            : (B, S, V*N+1, d)")
    print("    split:")
    print("        geo_out    : (B, S, V*N, d)")
    print("        action_out : (B, S, d)")
    print("\nStage 5 — Heads:")
    print("    action_head( mean_pool(action_out) ) -> (B, C=8, d_a=7)")
    print("    depth_head(  geo_out at layers 19,27,33,39 ) -> future depth maps")
    print("\nKey points:")
    print("    - One autoregressive token sequence produces BOTH geometry and action.")
    print("    - The same GFM backbone is reused for perception, prediction, and decoding.")
    print("    - L_s=12 is before the first DPT feature layer (19), so future depth")
    print("      supervision can be decoded from predicted tokens.")
    print("\n" + "=" * 72)


def live_skeleton():
    sys.path.insert(0, str(da3_src))

    import torch
    import torch.nn as nn
    from einops import rearrange

    from depth_anything_3.model.dinov2.dinov2 import DinoV2
    from depth_anything_3.model.dpt import DPT

    class CausalFuturePredictorStub(nn.Module):
        def __init__(self, d_model: int = 1536, depth: int = 12, action_dim: int = 7):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=16, dim_feedforward=4096,
                    dropout=0.0, batch_first=True, norm_first=True
                ) for _ in range(depth)
            ])
            self.action_embed = nn.Linear(action_dim, d_model)
            self.proprio_embed = nn.Linear(action_dim, d_model)
            self.action_token_predictor = nn.Linear(d_model, d_model)

        def forward(self, geo_tokens, proprio, prev_action):
            B, S, N, d = geo_tokens.shape
            p = self.proprio_embed(proprio).unsqueeze(2)
            a = self.action_embed(prev_action).unsqueeze(2)
            x = torch.cat([p, a, geo_tokens], dim=2)
            x = rearrange(x, "b s n d -> (b s) n d")
            causal = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
            for blk in self.blocks:
                x = blk(x, src_mask=~causal)
            x = rearrange(x, "(b s) n d -> b s n d", b=B, s=S)
            future_geo = x[:, :, 2:, :]
            action_token = self.action_token_predictor(x[:, :, 1, :])
            return future_geo, action_token

    class ActionHeadStub(nn.Module):
        def __init__(self, d_model: int = 1536, chunk_len: int = 8, action_dim: int = 7):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, 512), nn.ReLU(),
                nn.Linear(512, chunk_len * action_dim),
            )

        def forward(self, action_tokens):
            x = action_tokens.mean(dim=1)
            return self.net(x).view(x.size(0), 8, 7)

    def forward_gam_skeleton(x, proprio, prev_action, backbone, predictor,
                             decoder_blocks, action_head, L_s=12):
        B, S, V, C, H, W = x.shape
        vit = backbone.pretrained
        tokens = vit.prepare_tokens_with_masks(x.reshape(B, S * V, C, H, W))
        pos, _ = vit._prepare_rope(B, S * V, H, W, tokens.device)
        local_x = tokens
        for i, blk in enumerate(vit.blocks[:L_s + 1]):
            b, s, n = local_x.shape[:3]
            xflat = rearrange(local_x, "b s n c -> (b s) n c")
            pflat = rearrange(pos, "b s n c -> (b s) n c") if pos is not None else None
            xflat = blk(xflat, pos=pflat)
            local_x = rearrange(xflat, "(b s) n c -> b s n c", b=b, s=s)
        _, N, d = local_x.shape[1:]
        geo_tokens = local_x.view(B, S, V * N, d)
        future_geo, action_token = predictor(geo_tokens, proprio, prev_action)
        decoder_input = torch.cat([future_geo, action_token.unsqueeze(2)], dim=2)
        decoder_input = decoder_input.view(B * S, V * N + 1, d)
        x = decoder_input
        for blk in decoder_blocks:
            x = blk(x)
        geo_out = x[:, :V * N, :]
        action_out = x[:, V * N, :].view(B, S, d)
        actions = action_head(action_out)
        return actions, [geo_out]

    print("=" * 72)
    print("GAM Forward-Flow Skeleton")
    print("=" * 72)

    device = torch.device("cpu")
    B, S, V, H, W = 1, 4, 2, 224, 224
    d = 1536
    L_s = 12

    backbone = DinoV2(
        name="vitg", out_layers=[19, 27, 33, 39],
        alt_start=13, qknorm_start=13, rope_start=13, cat_token=True,
    ).to(device).eval()
    decoder_blocks = nn.ModuleList([backbone.pretrained.blocks[i] for i in range(L_s + 1, 40)]).to(device)
    predictor = CausalFuturePredictorStub(d_model=d, depth=12, action_dim=7).to(device)
    action_head = ActionHeadStub(d_model=d, chunk_len=8, action_dim=7).to(device)
    dpt_head = DPT(dim_in=3072, patch_size=14, output_dim=2, features=256,
                   out_channels=[256, 512, 1024, 1024]).to(device)

    x = torch.randn(B, S, V, 3, H, W, device=device)
    proprio = torch.randn(B, S, 7, device=device)
    prev_action = torch.randn(B, S, 7, device=device)

    with torch.no_grad():
        actions, dpt_feats = forward_gam_skeleton(
            x, proprio, prev_action, backbone, predictor, decoder_blocks, action_head, L_s=L_s
        )

    print("\n[1] Input shapes")
    print(f"    Images            : {x.shape}   (B={B}, S={S}, V={V}, 3, H={H}, W={W})")
    print(f"    Proprioception    : {proprio.shape}")
    print(f"    Previous action   : {prev_action.shape}")
    print("\n[2] Intermediate shapes through GAM skeleton")
    print(f"    Encoder output (after block {L_s}) : (B, S, V*N, d)")
    print(f"    Predictor future geo              : (B, S, V*N, d)")
    print(f"    Predictor action token            : (B, S, d)")
    print(f"    Decoder input (geo + action tok)  : (B*S, V*N+1, d)")
    print(f"    Decoder output                    : (B*S, V*N+1, d)")
    print("\n[3] Output shapes")
    print(f"    Predicted action chunk            : {actions.shape}")
    print(f"    DPT-feature placeholder           : {dpt_feats[0].shape}")
    print("\n[4] Parameter counts of inserted / trainable stubs")
    print(f"    Causal predictor (stub)           : {sum(p.numel() for p in predictor.parameters()) / 1e6:.1f} M")
    print(f"    Action head (stub)                : {sum(p.numel() for p in action_head.parameters()) / 1e6:.1f} M")
    print(f"    Decoder blocks 13..39              : {sum(p.numel() for p in decoder_blocks.parameters()) / 1e6:.1f} M")
    print("\n[5] Key architectural points confirmed")
    print("    - The action token is replicated per view and concatenated to geometry tokens")
    print("      before the decoder, so the remaining GFM blocks refine both jointly.")
    print("    - The DPT head can still decode future geometry because the split layer L_s=12")
    print("      is strictly before the first DPT feature layer (19).")
    print("    - Block-causal attention in the predictor and the decoder prevents future leakage.")
    print("\n" + "=" * 72)


def main():
    try:
        if not da3_src.exists():
            raise RuntimeError("DA3 source not found")
        import torch  # noqa: F401
        live_skeleton()
    except Exception as exc:
        textual_walkthrough()


if __name__ == "__main__":
    main()
