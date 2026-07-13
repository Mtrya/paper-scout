#!/usr/bin/env python3
"""
Estimate transformer parameter counts from public Hugging Face configs.
No weights are downloaded; counts are analytic.
"""
import json
from pathlib import Path

def fmt_b(n):
    return f"{n/1e9:.3f} B"

def count_dit(config):
    h = config.get("hidden_size", config.get("dim"))
    d = config.get("depth", config.get("num_layers"))
    nh = config.get("num_attention_heads", config.get("num_heads"))
    hd = h // nh
    intermediate = config.get("intermediate_size", config.get("ffn_dim"))
    num_experts = config.get("num_experts", 0)
    top_k = config.get("num_experts_per_tok", 1)
    moe_intermediate = config.get("moe_intermediate_size", intermediate)
    n_shared = config.get("n_shared_experts") or 0
    patch = config.get("patch_size", (1, 2, 2))
    in_ch = config.get("in_channels", config.get("in_dim", 36))
    out_ch = config.get("out_channels", config.get("out_dim", 16))
    text_dim = config.get("text_dim", config.get("text_len", 512) * 8)  # rough fallback

    # Patch embed + text embedder + time embedder + modulation + output
    patch_embed = (math.prod(patch) * in_ch) * h
    text_embedder = text_dim * h + h * h  # RMSNorm is param-free-ish except weight=1 (ignored)
    # time_proj is sinusoidal param-free; time_embedder + modulation
    time_embedder = 256 * h + h * h  # freq_dim hardcoded from configs
    time_modulation = h * (6 * h)
    # final norm is LayerNorm affine=False, output projection + modulation
    out_modulation = h * (2 * h)
    proj_out = h * (math.prod(patch) * out_ch)
    misc = patch_embed + text_embedder + time_embedder + time_modulation + out_modulation + proj_out

    # Per layer
    attn_per_layer = 4 * h * h  # q,k,v,o (ignoring small norms)
    if num_experts == 0:
        # dense SwiGLU MLP: gate, up, down
        ffn_per_layer = 3 * h * intermediate
        ffn_total = d * ffn_per_layer
    else:
        # routed experts total (all experts)
        routed_total = num_experts * (3 * h * moe_intermediate)
        # shared experts total
        shared_total = n_shared * (3 * h * moe_intermediate)
        # active per token FFN contribution
        ffn_active_per_layer = top_k * (3 * h * moe_intermediate) + n_shared * (3 * h * moe_intermediate)
        ffn_total = d * (routed_total + shared_total)
        ffn_active = d * ffn_active_per_layer

    total = d * attn_per_layer + ffn_total + misc
    active = d * attn_per_layer + (ffn_active if num_experts else ffn_total) + misc
    return {
        "hidden_size": h,
        "depth": d,
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_intermediate": moe_intermediate,
        "n_shared": n_shared,
        "attention": d * attn_per_layer,
        "ffn": ffn_total,
        "misc": misc,
        "total": total,
        "active": active if num_experts else None,
    }

if __name__ == "__main__":
    import math
    base = Path(__file__).parent
    for name in [
        "lingbot-video-dense-1.3b-transformer-config.json",
        "lingbot-video-moe-30b-a3b-transformer-config.json",
        "lingbot-world-v2-14b-causal-fast-config.json",
    ]:
        cfg = json.loads((base / name).read_text())
        c = count_dit(cfg)
        print(f"\n{name}")
        print(f"  h={c['hidden_size']}, layers={c['depth']}, experts={c['num_experts']}, top_k={c['top_k']}")
        print(f"  attention: {fmt_b(c['attention'])}")
        print(f"  ffn:       {fmt_b(c['ffn'])}")
        print(f"  misc:      {fmt_b(c['misc'])}")
        print(f"  total:     {fmt_b(c['total'])}")
        if c["active"]:
            print(f"  active/tok:{fmt_b(c['active'])}")
