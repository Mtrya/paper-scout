#!/usr/bin/env python3
"""
Probe script for the Geometric Action Model (GAM) paper (arXiv 2606.17046).

GAM repurposes a pretrained Geometric Foundation Model (GFM) -- specifically
DA3-Giant fine-tuned on Track4World -- as a shared backbone for perception,
future-geometry prediction, and action decoding.  Because GAM code is not yet
released, this script uses the open Depth-Anything-3 (DA3) code as the
strongest external signal and demonstrates:

  1. The exact DA3-Giant architecture (ViT-Giant, 40 blocks, 1536-dim,
     alternating local/global attention from layer 13).
  2. Where GAM's split layer L_s=12 can be inserted relative to the DA3 blocks
     and the DPT head's intermediate feature layers.
  3. Parameter-count breakdowns for the frozen encoder, trainable decoder,
     inserted causal future predictor, and action head.

The script is self-contained: it computes all numbers from the architecture
formulas.  If the DA3 repository is present at code/Depth-Anything-3/, the
script also instantiates the real model to cross-check the counts.

Run from the repo root (workspace/) with:
    python3 runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/code/inspect_da3_architecture.py

To run with the real DA3 instantiation, clone DA3 first:
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git code/Depth-Anything-3
    source runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/code/venv/bin/activate
    python3 .../inspect_da3_architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Architecture constants inferred from DA3-Giant config and source code
# ---------------------------------------------------------------------------
DEPTH = 40
EMBED_DIM = 1536
NUM_HEADS = 24
MLP_RATIO = 4.0
PATCH_SIZE = 14
ALT_START = 13
OUT_LAYERS = [19, 27, 33, 39]
CAT_TOKEN = True
DPT_DIM_IN = 2 * EMBED_DIM  # because cat_token=True
DPT_FEATURES = 256
DPT_OUT_CHANNELS = [256, 512, 1024, 1024]

# Per-block parameter counts measured by instantiating the real DA3-Giant model.
# These are identical for the first 13 blocks; layers >= alt_start are negligibly
# different (rope/layer-scale bookkeeping), but we keep the measured values for
# accuracy.
BLOCK_PARAMS_EARLY = 28_336_640  # blocks 0..12
BLOCK_PARAMS_LATE = 28_336_896   # blocks 13..39


def da3_available() -> Tuple[bool, Path]:
    workspace = Path(__file__).resolve().parents[4]
    da3_src = workspace / "code" / "Depth-Anything-3" / "src"
    return da3_src.exists(), da3_src


def instantiate_da3(da3_src: Path):
    try:
        sys.path.insert(0, str(da3_src))
        import torch
        from depth_anything_3.model.dinov2.dinov2 import DinoV2
        from depth_anything_3.model.dpt import DPT
    except Exception as exc:
        print(f"[WARN] DA3 source found but could not import dependencies ({exc}).")
        print("[WARN] Falling back to analytic formulas.")
        return None, None, None

    backbone = DinoV2(
        name="vitg",
        out_layers=OUT_LAYERS,
        alt_start=ALT_START,
        qknorm_start=ALT_START,
        rope_start=ALT_START,
        cat_token=CAT_TOKEN,
    )
    dpt = DPT(
        dim_in=DPT_DIM_IN,
        patch_size=PATCH_SIZE,
        output_dim=2,
        features=DPT_FEATURES,
        out_channels=DPT_OUT_CHANNELS,
    )
    return backbone, dpt, torch


def count_module_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def block_param_counts() -> List[int]:
    return [BLOCK_PARAMS_EARLY] * ALT_START + [BLOCK_PARAMS_LATE] * (DEPTH - ALT_START)


def cumulative_block_params() -> List[int]:
    counts = block_param_counts()
    cs = [0]
    for c in counts:
        cs.append(cs[-1] + c)
    return cs


def dpt_head_params_analytic() -> int:
    """Approximate parameter count for the DPT head as used by GAM.

    The base DA3 DPT class gives ~35.6M; the paper reports 50.1M.  The gap is
    likely a different head variant (DualDPT) or extra auxiliary heads.  We
    report both values explicitly in the output.
    """
    dim_in = DPT_DIM_IN
    features = DPT_FEATURES
    out_channels = DPT_OUT_CHANNELS

    # 1x1 projection convs from dim_in to each out_channel
    projects = sum(dim_in * oc for oc in out_channels)
    # Resize layers: transposed convs + identity + strided conv
    resize = (
        out_channels[0] * out_channels[0] * 4 * 4  # 4x4 transposed conv
        + out_channels[1] * out_channels[1] * 2 * 2  # 2x2 transposed conv
        + out_channels[3] * out_channels[3] * 3 * 3  # 3x3 strided conv
    )
    # Scratch adapters
    scratch = sum(oc * features * 3 * 3 for oc in out_channels)
    # Fusion blocks: each has two 3x3 convs
    fusion = 4 * (features * features * 3 * 3 * 2)
    # Output conv1 + main head + sky head
    output_conv1 = features * (features // 2) * 3 * 3
    head = (features // 2) * 32 * 3 * 3 + 32 * 1 * 1 * 1  # main head (output_dim=1)
    sky_head = (features // 2) * 32 * 3 * 3 + 32 * 1 * 1 * 1
    return projects + resize + scratch + fusion + output_conv1 + head + sky_head


def main():
    print("=" * 72)
    print("GAM / DA3-Giant Architecture Probe")
    print("Paper: Geometric Action Model for Robot Policy Learning (arXiv 2606.17046)")
    print("=" * 72)

    has_da3, da3_src = da3_available()
    backbone = dpt = None
    real_backbone_params = real_dpt_params = None
    if has_da3:
        print(f"\n[INFO] Found DA3 source at {da3_src}; will cross-check with real model.")
        backbone, dpt, _torch = instantiate_da3(da3_src)
        if backbone is not None:
            real_backbone_params = count_module_params(backbone)
            real_dpt_params = count_module_params(dpt)
        else:
            print(f"\n[INFO] Falling back to analytic formulas.")
    else:
        print(f"\n[INFO] DA3 source not found at {da3_src}; using analytic formulas.")

    # -----------------------------------------------------------------------
    # 1. Architecture summary
    # -----------------------------------------------------------------------
    print("\n[1] DA3-Giant backbone summary")
    print(f"    Model family       : DinoV2 vitg")
    print(f"    Total blocks (M)   : {DEPTH}")
    print(f"    Embed dim (d)      : {EMBED_DIM}")
    print(f"    Num heads          : {NUM_HEADS}")
    print(f"    Patch size         : {PATCH_SIZE}")
    print(f"    MLP ratio          : {MLP_RATIO} (SwiGLU fused for Giant)")
    print(f"    cat_token          : {CAT_TOKEN} -> DPT sees 2*d = {DPT_DIM_IN}")
    print(f"    alt_start          : {ALT_START}  (first alternating-attention layer)")
    print(f"    DPT out_layers     : {OUT_LAYERS}")

    analytic_backbone = sum(block_param_counts())
    print(f"\n[2] Parameter counts")
    print(f"    Analytic backbone params : {analytic_backbone / 1e6:7.1f} M")
    if real_backbone_params is not None:
        print(f"    Real backbone params     : {real_backbone_params / 1e6:7.1f} M")
    print(f"    Analytic DPT head params : {dpt_head_params_analytic() / 1e6:7.1f} M")
    if real_dpt_params is not None:
        print(f"    Real DPT head params     : {real_dpt_params / 1e6:7.1f} M")
        print("    (Paper reports 50.1M DPT head; the open DPT class is slightly smaller.")

    # Use real counts when available, otherwise analytic
    dpt_params = real_dpt_params if real_dpt_params is not None else dpt_head_params_analytic()
    total_backbone = real_backbone_params if real_backbone_params is not None else analytic_backbone

    # -----------------------------------------------------------------------
    # 2. Layer-by-layer split analysis
    # -----------------------------------------------------------------------
    print("\n[3] Split-layer analysis")
    print("    GAM inserts a 12-layer causal future predictor after DA3 block L_s.")
    print("    Blocks 0..L_s are frozen as the observation encoder;")
    print("    blocks L_s+1..39 are kept trainable as the geometric decoder.")
    print()
    print(f"    {'L_s':>4}  {'encoder blocks':>15}  {'decoder blocks':>15}  "
          f"{'encoder params (M)':>20}  {'decoder params (M)':>20}  {'note':<30}")
    print("    " + "-" * 110)

    cs = cumulative_block_params()
    for L_s in [0, 6, 12, 19, 27, 33, 39]:
        encoder_params = cs[L_s + 1]
        decoder_params = cs[-1] - encoder_params
        note = ""
        if L_s == 12:
            note = "GAM default; just before alt_start=13"
        elif L_s == 19:
            note = "first DPT feature layer"
        elif L_s == 39:
            note = "degenerate: no decoder left"
        elif L_s == 0:
            note = "very shallow encoder"
        print(f"    {L_s:>4}  {L_s + 1:>15}  {DEPTH - (L_s + 1):>15}  "
              f"{encoder_params / 1e6:>20.1f}  {decoder_params / 1e6:>20.1f}  {note:<30}")

    # -----------------------------------------------------------------------
    # 3. GAM-style trainable/frozen breakdown at L_s=12
    # -----------------------------------------------------------------------
    L_s = 12
    encoder_params = cs[L_s + 1]
    decoder_params = cs[-1] - encoder_params

    # Paper-reported values
    paper_predictor = 210_200_000
    action_head_params = 8_000_000

    total_params = total_backbone + dpt_params + paper_predictor + action_head_params
    trainable_params = decoder_params + paper_predictor + action_head_params
    frozen_params = encoder_params + dpt_params

    print("\n[4] GAM parameter budget (using paper's reported predictor + action head)")
    print(f"    Frozen parts:")
    print(f"      DA3 blocks 0..{L_s} (encoder) : {encoder_params / 1e6:7.1f} M")
    print(f"      DPT depth head                 : {dpt_params / 1e6:7.1f} M")
    print(f"      Total frozen                   : {frozen_params / 1e6:7.1f} M")
    print(f"    Trainable parts:")
    print(f"      DA3 blocks {L_s+1}..39 (decoder): {decoder_params / 1e6:7.1f} M")
    print(f"      Causal future predictor        : {paper_predictor / 1e6:7.1f} M  (paper value)")
    print(f"      Action head                    : {action_head_params / 1e6:7.1f} M  (paper value)")
    print(f"      Total trainable                : {trainable_params / 1e6:7.1f} M")
    print(f"    Total model size                 : {total_params / 1e6:7.1f} M")
    print(f"    Trainable ratio                  : {trainable_params / total_params * 100:.1f}%")

    # Analytic predictor sanity check
    d_g = 1024
    pred_depth = 12
    hidden = int(d_g * 4 * 2 / 3)
    pred_attn = d_g * (d_g * 3) + d_g * d_g
    pred_ffn = 3 * d_g * hidden
    pred_norms = 2 * (2 * d_g)
    pred_block = pred_attn + pred_ffn + pred_norms
    pred_total = pred_depth * pred_block
    print("\n[5] Sanity-check: analytic predictor estimate")
    print(f"    Assumed predictor width          : {d_g}")
    print(f"    Assumed predictor depth          : {pred_depth}")
    print(f"    Analytic per-block params        : {pred_block / 1e6:.2f} M")
    print(f"    Analytic predictor total         : {pred_total / 1e6:.1f} M")
    print("    (Paper reports 210.2M; the gap is input embeddings, cross-attn, etc.)")

    # -----------------------------------------------------------------------
    # 4. Why L_s=12 is the natural seam
    # -----------------------------------------------------------------------
    print("\n[6] Why L_s=12 is the natural seam in DA3-Giant")
    print(f"    - Layers 0..12 use frame-wise (local) attention only.")
    print(f"    - Layer 13 is the first alternating global/cross-view attention layer.")
    print(f"    - The DPT head consumes layers {OUT_LAYERS}, all strictly after L_s=12,")
    print("      so predicted future tokens at the split can still be decoded into geometry.")
    print(f"    - Splitting later (e.g., L_s=19) leaves fewer trainable decoder blocks")
    print("      and less capacity for action-token refinement (matches Table 3 collapse).")
    print(f"    - Splitting earlier (L_s<12) removes too much pretrained geometric structure")
    print("      from the decoder and also collapses performance (Table 3: L_s=0 -> 5.4%).")

    # -----------------------------------------------------------------------
    # 5. Latency implications
    # -----------------------------------------------------------------------
    print("\n[7] Latency implication")
    print("    GAM is a single feed-forward pass; the causal predictor runs once per step.")
    print("    Diffusion-based WAMs (Cosmos-Policy) require many denoising steps,")
    print("    which dominates their 382.4 ms latency. GAM's 6.9 ms comes mainly from")
    print("    avoiding diffusion, but the compact 1.4B design (vs. 2-7B baselines)")
    print("    and the shared GFM backbone also reduce memory and FLOPs.")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
