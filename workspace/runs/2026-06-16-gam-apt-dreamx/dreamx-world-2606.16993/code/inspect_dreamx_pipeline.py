#!/usr/bin/env python3
"""
Inspect the DreamX-World inference layout and emit a structured overview.

This script does not require PyTorch; it parses the bundled source snapshot
under src/ and reports file structure, configs, and the two main inference
entry points.
"""
import json
import os
from pathlib import Path
from collections import OrderedDict


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_yaml(path):
    # Tiny YAML parser sufficient for the flat configs used here.
    out = OrderedDict()
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if val == "":
                    out[key] = {}
                else:
                    # remove inline comments
                    val = val.split("#")[0].strip()
                    if val.startswith('"') and val.endswith('"'):
                        val = val[1:-1]
                    elif val in ("true", "True"):
                        val = True
                    elif val in ("false", "False"):
                        val = False
                    elif val in ("null", "None"):
                        val = None
                    else:
                        try:
                            val = int(val)
                        except ValueError:
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                    out[key] = val
    return out


def tree(root, prefix=""):
    """Return a small text tree of files under root."""
    lines = []
    entries = sorted(Path(root).iterdir(), key=lambda p: (p.is_file(), p.name))
    for i, p in enumerate(entries):
        is_last = i == len(entries) - 1
        branch = "└── " if is_last else "├── "
        lines.append(f"{prefix}{branch}{p.name}")
        if p.is_dir():
            ext = "    " if is_last else "│   "
            lines.extend(tree(p, prefix + ext))
    return lines


def main():
    script_dir = Path(__file__).resolve().parent
    src = script_dir / "src"

    print("=" * 70)
    print("DreamX-World released-code inspection (lightweight, no weights)")
    print("=" * 70)

    print("\n1. Source snapshot layout (curated)")
    print("-" * 70)
    print("\n".join(tree(src)))

    print("\n2. Key inference entry points")
    print("-" * 70)
    print("  inference_dreamx5b.py          -> 5-second bidirectional T2V/I2V")
    print("  inference_ar_forcing.py        -> long-horizon autoregressive T2V/I2V")
    print("  pipeline/pipeline_dreamxworld.py      -> diffusers-style bidirectional pipeline")
    print("  pipeline/pipeline_causal_camera.py    -> causal KV-cache pipeline")

    print("\n3. Causal AR config (configs/dreamx-ar/config.json)")
    print("-" * 70)
    causal_cfg = load_json(src / "configs" / "dreamx-ar" / "config.json")
    for k, v in causal_cfg.items():
        print(f"  {k:30s}: {v}")

    print("\n4. Bidirectional base config (configs/wan2.2/wan_ti2v_5b.yaml)")
    print("-" * 70)
    base_cfg = load_yaml(src / "configs" / "wan2.2" / "wan_ti2v_5b.yaml")
    for k, v in base_cfg.items():
        print(f"  {k:30s}: {v}")

    print("\n5. Model components present in the snapshot")
    print("-" * 70)
    components = [
        ("DiT (bidirectional)", "models/wan_transformer3d.py", "Wan2_2Transformer3DModel"),
        ("DiT (causal AR)", "wan/modules/causal_camera_model_2_2_prope_infinity.py", "CausalWanModel"),
        ("PRoPE transforms", "models/prope_utils.py", "prope_qkv"),
        ("PRoPE transforms (causal)", "wan/modules/camera_prope.py", "prope_qkv"),
        ("Attention backend", "models/attention_utils.py", "attention / flash_attention"),
        ("Camera trajectory", "utils/trajectory_processor.py", "generate_trajectory_from_json"),
        ("Camera -> PRoPE dict", "utils/inference_utils.py", "GetPoseEmbedsFromPosesPrope"),
        ("VAE wrapper", "utils/wan_wrapper.py", "WanVAEWrapper"),
        ("Text encoder wrapper", "utils/wan_wrapper.py", "WanTextEncoder"),
        ("Diffusion wrapper", "utils/wan_wrapper.py", "WanDiffusionCameraWrapper"),
        ("FP8 optimisation", "utils/fp8_optimization.py", "convert_model_weight_to_float8"),
        ("Memory/offload helpers", "utils/memory.py", "DynamicSwapInstaller"),
    ]
    for name, path, symbol in components:
        print(f"  {name:28s}  {path:45s}  {symbol}")

    print("\n6. Missing from the released repo (blockers for full reproduction)")
    print("-" * 70)
    missing = [
        "Training code for camera-aware / E-PRoPE fine-tuning",
        "Training code for memory-conditioned scene persistence",
        "Geometry-based memory retrieval implementation",
        "Event-instruction tuning data loader / event parser",
        "DMD distillation and long-rollout training code",
        "RL post-training (reward models, DiffusionNFT trainer)",
        "UE data-engine rendering / filtering scripts",
        "Checkpoints (must be downloaded separately from HF/ModelScope)",
    ]
    for item in missing:
        print(f"  • {item}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
