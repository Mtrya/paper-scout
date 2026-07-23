#!/usr/bin/env python
"""Weight-level probe of the Masked Visual Actions LoRAs.

For each of the two expert LoRAs (high-noise / low-noise DiT of
Wan2.2-Fun-A14B-Control) report:
  - tensor inventory: count, dtypes, total params
  - key naming scheme and the set of targeted module types
  - per-block, per-module LoRA update geometry: ||A||_F, ||B||_F,
    ||B @ A||_F (exact Frobenius norm of the delta), and its scale
    relative to a full-rank reference
  - which blocks / module types carry the largest updates
  - high-noise vs low-noise expert comparison

Writes JSON results + a per-block norm plot.
"""
import json
import re
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

CKPT_DIR = "checkpoints"
OUT_JSON = "lora_probe_results.json"
OUT_PNG = "mva_lora_perblock_norms.png"


def parse_key(key):
    """Extract (block_idx, module_type) from a LoRA key.

    DiffSynth Wan LoRA keys look like:
      blocks.0.attn1.to_q.lora_A.default.weight   (or lora_B)
    or  blocks.12.self_attn.q.lora_A.weight  etc.
    Returns (block:int|None, module:str, kind:'A'|'B').
    """
    kind = None
    m = re.search(r"lora_(A|B)", key)
    if m:
        kind = m.group(1)
    bm = re.search(r"blocks\.(\d+)\.", key)
    block = int(bm.group(1)) if bm else None
    # module type: e.g. self_attn.q / cross_attn.k / ffn.0
    mod = None
    mm = re.search(r"(self_attn|cross_attn)\.(q|k|v|o)\.lora_", key)
    if mm:
        mod = f"{mm.group(1)}.{mm.group(2)}"
    else:
        mm = re.search(r"(ffn\.\d+)\.lora_", key)
        if mm:
            mod = mm.group(1)
    return block, mod, kind


def probe(path):
    inv = {"path": path}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        inv["num_tensors"] = len(keys)
        dtypes = defaultdict(int)
        total = 0
        shapes = {}
        for k in keys:
            sl = f.get_slice(k)
            dt = sl.get_dtype()
            dtypes[dt] += 1
            sh = sl.get_shape()
            shapes[k] = sh
            n = 1
            for s in sh:
                n *= s
            total += n
        inv["dtypes"] = dict(dtypes)
        inv["total_params"] = total
        inv["sample_keys"] = keys[:6]

        # group A/B pairs
        pairs = {}
        for k in keys:
            block, mod, kind = parse_key(k)
            if kind is None:
                continue
            base = re.sub(r"lora_(A|B)(\.default)?\.weight$", "", k)
            pairs.setdefault(base, {"block": block, "mod": mod, "key_base": base})
            pairs[base][kind] = k
        inv["num_lora_pairs"] = len(pairs)

        # rank check
        ranks = set()
        for base, d in pairs.items():
            if "A" in d:
                ranks.add(shapes[d["A"]][0])
        inv["distinct_ranks"] = sorted(ranks)

        # per-pair norms: exact ||B@A||_F computed as ||B A|| via
        # ||BA||_F^2 = sum_ij <...> -> just materialize: rank 256 small.
        per_block = defaultdict(lambda: defaultdict(float))   # block -> mod -> ||BA||_F^2
        per_block_rank = defaultdict(float)
        mods_seen = set()
        for base, d in pairs.items():
            if "A" not in d or "B" not in d:
                continue
            A = f.get_tensor(d["A"]).to(torch.float64)
            B = f.get_tensor(d["B"]).to(torch.float64)
            delta = B @ A                      # [out, in]
            n2 = float(delta.pow(2).sum())
            blk = d["block"] if d["block"] is not None else -1
            mod = d["mod"] or "other"
            per_block[blk][mod] += n2
            per_block_rank[blk] += n2
            mods_seen.add(mod)
            del A, B, delta
        inv["module_types"] = sorted(mods_seen)

        blocks = sorted(per_block_rank)
        inv["blocks"] = [int(b) for b in blocks]
        inv["per_block_norm"] = {int(b): float(per_block_rank[b] ** 0.5) for b in blocks}
        inv["per_block_mod_norm"] = {
            int(b): {m: float(v ** 0.5) for m, v in per_block[b].items()} for b in blocks
        }
        # aggregate per module type
        agg = defaultdict(float)
        for b in blocks:
            for m, v in per_block[b].items():
                agg[m] += v
        inv["per_module_total_norm"] = {m: float(v ** 0.5) for m, v in agg.items()}
        # top / bottom blocks
        sorted_blocks = sorted(blocks, key=lambda b: -per_block_rank[b])
        inv["top5_blocks"] = [int(b) for b in sorted_blocks[:5]]
        inv["bottom5_blocks"] = [int(b) for b in sorted_blocks[-5:]]
    return inv


def main():
    results = {}
    for name in ["masked_world_lora_high.safetensors", "masked_world_lora_low.safetensors"]:
        results[name] = probe(f"{CKPT_DIR}/{name}")

    # high vs low comparison
    hi = results["masked_world_lora_high.safetensors"]["per_block_norm"]
    lo = results["masked_world_lora_low.safetensors"]["per_block_norm"]
    common = sorted(set(hi) & set(lo))
    import math
    if common:
        hv = [hi[b] for b in common]
        lv = [lo[b] for b in common]
        mh, ml = sum(hv) / len(hv), sum(lv) / len(lv)
        cov = sum((a - mh) * (b - ml) for a, b in zip(hv, lv))
        r = cov / (math.sqrt(sum((a - mh) ** 2 for a in hv)) * math.sqrt(sum((b - ml) ** 2 for b in lv)) + 1e-12)
        results["high_vs_low"] = {
            "per_block_norm_pearson_r": r,
            "mean_norm_high": mh,
            "mean_norm_low": ml,
            "ratio_high_over_low": mh / ml if ml else None,
        }

    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_block_mod_norm"}
                      for k, v in results.items()}, indent=2)[:4000])

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (name, label) in zip(axes, [("masked_world_lora_high.safetensors", "high-noise expert (dit)"),
                                        ("masked_world_lora_low.safetensors", "low-noise expert (dit2)")]):
        res = results[name]
        blocks = res["blocks"]
        mods = res["module_types"]
        bottom = [0.0] * len(blocks)
        for m in mods:
            vals = [res["per_block_mod_norm"].get(b, {}).get(m, 0.0) for b in blocks]
            ax.bar([str(b) for b in blocks], vals, bottom=bottom, label=m)
            bottom = [a + b for a, b in zip(bottom, vals)]
        ax.set_title(label)
        ax.set_xlabel("DiT block index")
        ax.tick_params(axis="x", labelsize=6, rotation=90)
    axes[0].set_ylabel(r"$\|B A\|_F$ (LoRA update norm)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Masked Visual Actions LoRA: per-block update norms (rank-256, Wan2.2-Fun-A14B-Control)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print(f"wrote {OUT_JSON} and {OUT_PNG}")


if __name__ == "__main__":
    main()
