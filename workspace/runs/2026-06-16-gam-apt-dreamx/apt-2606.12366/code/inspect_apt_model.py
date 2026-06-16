#!/usr/bin/env python3
"""
Lightweight probe of the APT action expert.

* Instantiates the ActionExpert for Stage 0 (VA prior) and Stage 1 (VLA likelihood)
  without loading any VLM or checkpoint weights.
* Counts parameters per component and per stage.
* Runs a dummy forward pass through the HybridAttentionLayers to verify the
  two-stage layer expansion, attention-mask shapes, gated fusion, and FiLM
  timestep conditioning.

Run inside a virtualenv with the dependencies from code/APT/requirements.txt,
or with just ``torch``, ``einops`` and ``diffusers`` installed:

    python code/inspect_apt_model.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# APT lives at workspace/code/APT
ROOT = Path(__file__).resolve().parents[4]
APT_ROOT = ROOT / "code" / "APT"
if str(APT_ROOT) not in sys.path:
    sys.path.insert(0, str(APT_ROOT))

import torch
from apt.action_expert import ActionExpert, HybridAttentionLayers


def count_params(module: torch.nn.Module, only_trainable: bool = False) -> int:
    params = module.parameters()
    if only_trainable:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def component_counts(expert: ActionExpert) -> Dict[str, int]:
    actor = expert
    ctx = actor.context_encoder
    head = actor.dp_head
    attn = head.traj_context_attn
    return {
        "total": count_params(actor),
        "context_encoder": count_params(ctx),
        "diffusion_head": count_params(head),
        "  |- traj_context_attn": count_params(attn),
        "  |    |- attention_blocks": sum(
            count_params(b) for b in attn.layers
        ),
        "  |    |- layer_gates": int(attn.gate.numel()),
        "  |- input_encoders": count_params(head.hist_enc)
        + count_params(head.traj_enc)
        + count_params(head.abs_pos_enc)
        + count_params(head.denoising_time_embed),
        "  |- final_head": count_params(head.final_norm)
        + count_params(head.act_head),
    }


def print_counts(counts: Dict[str, int], title: str) -> None:
    print(f"\n=== {title} ===")
    for k, v in counts.items():
        indent = "    " if k.startswith("  |") else "  " if k.startswith("  ") else ""
        print(f"{indent}{k:<32} {v:>12,}  ({v/1e6:>6.2f}M)")
    print(f"{'':>32} {'-'*14}")
    print(f"{'total':<32} {counts['total']:>12,}  ({counts['total']/1e6:>6.2f}M)")


def trace_hybrid_layers(
    hdim: int = 192,
    num_heads: int = 3,
    num_layers: int = 8,
    batch: int = 2,
    len_vl: int = 64,
    len_a: int = 96,
) -> Dict[str, torch.Size]:
    """Symbolic forward pass through HybridAttentionLayers for Stage 1."""
    print("\n=== Dummy forward pass through HybridAttentionLayers (Stage 1) ===")
    layer = HybridAttentionLayers(hdim, num_heads, num_layers, train_stage=1)
    layer.eval()

    x = torch.randn(batch, len_vl + len_a, hdim)
    pe = (None, torch.randn(batch, len_vl + len_a, hdim // num_heads // 2, 2))
    pe_gta = (torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch, len_vl + len_a, 4, 4), None)
    mask = torch.ones(batch, len_vl + len_a, len_vl + len_a, dtype=torch.bool)
    mask_dilated = mask.clone()
    film = torch.randn(batch, hdim)
    # VLM per-layer highway features are aligned with the VL tokens only,
    # not with the concatenated VL+action sequence.
    vl_highways = [torch.randn(batch, len_vl, hdim) for _ in range(num_layers)]

    with torch.no_grad():
        out = layer(
            x=x,
            pe=pe,
            pe_gta=pe_gta,
            mask=mask,
            mask_dilated=mask_dilated,
            film_cond=film,
            vla_split_size=(len_vl, len_a),
            vl_highways=vl_highways,
        )

    print(f"  input shape               : {tuple(x.shape)}")
    print(f"  output shape              : {tuple(out.shape)}")
    print(f"  active attention layers   : {len(layer.layers)}")
    print(f"  gate parameter shape      : {tuple(layer.gate.shape)}  "
          f"({layer.gate.numel():,} scalars)")
    print(f"  pe pair (Stage-1 even idx): rope + None")
    print(f"  pe pair (Stage-1 odd idx) : None + prope")
    return {
        "input": x.shape,
        "output": out.shape,
        "num_layers": len(layer.layers),
        "gate_shape": layer.gate.shape,
    }


def main():
    # Use a small but representative config.  The real model is hdim=768,
    # num_heads=12, num_diffusion_layers=20 (from the paper / configs).
    hdim = 768
    num_heads = 12
    num_layers = 20
    idim = 1536  # Qwen3-VL-2B hidden size; VLM features are projected to hdim

    print("APT ActionExpert probe")
    print(f"  config: hdim={hdim}, num_heads={num_heads}, "
          f"num_diffusion_layers={num_layers}, idim={idim}")

    stage0 = ActionExpert(
        idim=idim,
        hdim=hdim,
        num_heads=num_heads,
        num_diffusion_layers=num_layers,
        train_stage=0,
    )
    stage1 = ActionExpert(
        idim=idim,
        hdim=hdim,
        num_heads=num_heads,
        num_diffusion_layers=num_layers,
        train_stage=1,
    )

    counts0 = component_counts(stage0)
    counts1 = component_counts(stage1)
    print_counts(counts0, "Stage 0 (VA prior)")
    print_counts(counts1, "Stage 1 (VLA likelihood)")

    diff = counts1["total"] - counts0["total"]
    print(f"\nParameter growth from Stage 0 -> Stage 1: "
          f"+{diff:,}  (+{diff/counts0['total']*100:.1f}%)")

    trace = trace_hybrid_layers(
        hdim=192, num_heads=3, num_layers=8,
        batch=2, len_vl=64, len_a=96,
    )

    # Save a JSON snapshot in the thread directory for downstream tables / figures.
    out_dir = Path(__file__).resolve().parent
    summary = {
        "config": {"hdim": hdim, "num_heads": num_heads,
                   "num_diffusion_layers": num_layers, "idim": idim},
        "stage0_counts": counts0,
        "stage1_counts": counts1,
        "growth": {"absolute": diff, "relative_pct": round(diff/counts0["total"]*100, 2)},
        "hybrid_layer_trace": {k: str(v) for k, v in trace.items()},
    }
    out_path = out_dir / "inspect_apt_model_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
