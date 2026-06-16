#!/usr/bin/env python3
"""
Estimate DreamX-World DiT parameter counts from the released configs.

No model weights are downloaded; this script only parses the architecture
configs and applies the standard transformer parameter formulas.
"""
import json
import math
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dit_params_from_config(cfg):
    """Compute parameter count for the CausalWanModel / WanTransformer3DModel."""
    dim = cfg["dim"]
    ffn_dim = cfg["ffn_dim"]
    text_dim = cfg.get("text_dim", 4096)
    freq_dim = cfg.get("freq_dim", 256)
    in_dim = cfg["in_dim"]
    out_dim = cfg.get("out_dim", in_dim)
    num_layers = cfg["num_layers"]
    num_heads = cfg["num_heads"]
    patch_size = cfg.get("patch_size", [1, 2, 2])
    patch_vol = math.prod(patch_size)
    add_control_adapter = cfg.get("add_control_adapter", False)
    attn_compress = cfg.get("attn_compress", 1)
    cross_attn_norm = cfg.get("cross_attn_norm", True)

    # Embeddings
    patch_embed = in_dim * dim * patch_vol + dim
    text_embed = (text_dim * dim + dim) + (dim * dim + dim)
    time_embed = (freq_dim * dim + dim) + (dim * dim + dim)
    time_proj = dim * (dim * 6) + dim * 6

    # Per transformer block
    self_attn = 4 * (dim * dim + dim)          # q, k, v, o
    cross_attn = 4 * (dim * dim + dim)         # text cross-attention
    ffn = (dim * ffn_dim + ffn_dim) + (ffn_dim * dim + dim)
    norm_params = dim if cross_attn_norm else 0

    # Optional E-PRoPE / PRoPE parallel branch (PropeSelfAttention)
    cam_attn = 0
    if add_control_adapter:
        attn_dim = dim // attn_compress if attn_compress else dim
        cam_attn += 3 * (dim * attn_dim + attn_dim)   # q_proj, k_proj, v_proj
        cam_attn += attn_dim * dim + dim               # out_proj
        cam_attn += 2 * attn_dim                       # norm_q, norm_k

    block_params = self_attn + cross_attn + ffn + norm_params + cam_attn
    blocks_total = num_layers * block_params

    # Head
    head = dim * out_dim * patch_vol + out_dim * patch_vol

    total = patch_embed + text_embed + time_embed + time_proj + blocks_total + head

    return {
        "summary": {
            "dim": dim,
            "ffn_dim": ffn_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "patch_size": patch_size,
            "add_control_adapter": add_control_adapter,
            "attn_compress": attn_compress,
        },
        "components": {
            "patch_embed_M": patch_embed / 1e6,
            "text_embed_M": text_embed / 1e6,
            "time_embed_M": time_embed / 1e6,
            "time_proj_M": time_proj / 1e6,
            "blocks_total_M": blocks_total / 1e6,
            "per_block_M": block_params / 1e6,
            "head_M": head / 1e6,
            "total_M": total / 1e6,
            "total_B": total / 1e9,
        },
    }


def main():
    script_dir = Path(__file__).resolve().parent
    ar_cfg_path = script_dir / "src" / "configs" / "dreamx-ar" / "config.json"

    if not ar_cfg_path.exists():
        print(f"Config not found at {ar_cfg_path}; run from the thread code/ directory.")
        return

    ar_cfg = load_json(ar_cfg_path)
    ar_cfg.setdefault("patch_size", [1, 2, 2])
    ar_cfg.setdefault("text_dim", 4096)
    ar_cfg.setdefault("freq_dim", 256)
    ar_cfg.setdefault("cross_attn_norm", True)

    print("=" * 70)
    print("DreamX-World DiT parameter estimate (from configs/dreamx-ar/config.json)")
    print("=" * 70)
    s = ar_cfg
    print(f"dim={s['dim']}, ffn_dim={s['ffn_dim']}, layers={s['num_layers']}, heads={s['num_heads']}")
    print(f"patch_size={s.get('patch_size')}, add_control_adapter={s.get('add_control_adapter')}")
    print()

    # Three scenarios
    scenarios = [
        ("Base DiT (no camera branch)", {**ar_cfg, "add_control_adapter": False}),
        ("With full-dim PRoPE branch (attn_compress=1)", {**ar_cfg, "add_control_adapter": True, "attn_compress": 1}),
        ("With E-PRoPE branch (attn_compress=4)", {**ar_cfg, "add_control_adapter": True, "attn_compress": 4}),
    ]

    for label, cfg in scenarios:
        r = dit_params_from_config(cfg)
        c = r["components"]
        print(f"{label}:")
        print(f"  per-block params : {c['per_block_M']:.1f} M")
        print(f"  block total      : {c['blocks_total_M']:.1f} M")
        print(f"  DiT total        : {c['total_B']:.2f} B")
    print("=" * 70)
    print("\nInterpretation:")
    print("- The base architecture is ~5.0B parameters, matching the paper's '5B' headline.")
    print("- Adding the parallel camera-attention branch increases the count;")
    print("  the released config sets add_control_adapter=True and eprope=False,")
    print("  which would default to a full-dim branch (~6.1B) unless overridden by")
    print("  checkpoint metadata or an omitted attn_compress value.")
    print("- VAE and UMT5-XXL text-encoder parameters are not included above.")


if __name__ == "__main__":
    main()
