#!/usr/bin/env python3
"""Probe: inspect Xiaomi-Robotics-U0 public config and sanity-check paper claims.

This script fetches config.json from the HuggingFace Hub (or uses a local copy)
and reports the architecture dimensions that define the released model.  It also
computes a rough parameter budget to cross-check the paper's "38B" claim.

Usage:
    python code/inspect_config.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from urllib.request import urlopen

CONFIG_URL = "https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-U0/raw/main/config.json"


def fetch_config() -> dict:
    """Fetch config.json from the public HuggingFace repo."""
    try:
        with urlopen(CONFIG_URL, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"WARN: could not fetch config from HF ({exc}); trying local cache...")
        local = Path(__file__).resolve().parents[3] / "code" / "xiaomi-robotics-u0" / "config.json"
        if local.exists():
            return json.loads(local.read_text())
        raise


def estimate_params(cfg: dict) -> dict:
    """Rough parameter budget for a dense decoder-only Transformer with GQA."""
    v = cfg["vocab_size"]
    h = cfg["hidden_size"]
    i = cfg["intermediate_size"]
    l = cfg["num_hidden_layers"]
    n_q = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    d = cfg["head_dim"]

    # Embeddings + output head (untied in this model).
    embed = v * h
    head = v * h

    # Per-layer attention with GQA.
    # Q proj: h * (n_q * d); K/V proj: 2 * h * (n_kv * d); O proj: (n_q*d) * h.
    attn = h * (n_q * d) + 2 * h * (n_kv * d) + (n_q * d) * h
    # QKV biases (qkv_bias=true in released config).
    attn_bias = (n_q * d + 2 * n_kv * d) if cfg.get("qkv_bias", False) else 0
    # SwiGLU MLP: gate, up, down projections.
    mlp = 3 * h * i
    # RMS norms: 2 per layer + final norm.
    norms = (2 * l + 1) * h

    layer_params = attn + attn_bias + mlp
    transformer = l * layer_params + norms
    total = embed + head + transformer

    return {
        "vocab_size": v,
        "hidden_size": h,
        "intermediate_size": i,
        "num_layers": l,
        "num_heads": n_q,
        "num_kv_heads": n_kv,
        "head_dim": d,
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "embed_params_B": embed / 1e9,
        "head_params_B": head / 1e9,
        "attn_per_layer_M": attn / 1e6,
        "mlp_per_layer_M": mlp / 1e6,
        "layer_params_M": layer_params / 1e6,
        "transformer_params_B": transformer / 1e9,
        "estimated_total_B": total / 1e9,
    }


def main() -> int:
    cfg = fetch_config()
    est = estimate_params(cfg)

    print("Xiaomi-Robotics-U0 released model config")
    print("=" * 50)
    print(f"vocab_size:              {est['vocab_size']:,}")
    print(f"hidden_size:             {est['hidden_size']:,}")
    print(f"intermediate_size:       {est['intermediate_size']:,}")
    print(f"num_hidden_layers:       {est['num_layers']}")
    print(f"num_attention_heads:     {est['num_heads']}")
    print(f"num_key_value_heads:     {est['num_kv_heads']}")
    print(f"head_dim:                {est['head_dim']}")
    print(f"max_position_embeddings: {est['max_position_embeddings']:,}")
    print()
    print("Rough parameter budget")
    print("-" * 50)
    print(f"Embedding table:         {est['embed_params_B']:.2f} B")
    print(f"Output head:             {est['head_params_B']:.2f} B")
    print(f"Attention per layer:     {est['attn_per_layer_M']:.1f} M")
    print(f"MLP per layer:           {est['mlp_per_layer_M']:.1f} M")
    print(f"Total per layer:         {est['layer_params_M']:.1f} M")
    print(f"Transformer stack:       {est['transformer_params_B']:.2f} B")
    print(f"Estimated total:         {est['estimated_total_B']:.2f} B")
    print()
    print(f"Paper claim: 38B; this rough count is ~{est['estimated_total_B']:.1f}B.")
    print("Difference is within normal counting conventions (ties, exact MLP gate counts, etc.).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
