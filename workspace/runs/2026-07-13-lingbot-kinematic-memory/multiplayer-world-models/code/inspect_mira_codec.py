#!/usr/bin/env python3
"""Compact probe: summarize MIRA's representation codec config and action vocabulary.

Run from the root of a cloned mira repo (https://github.com/mira-wm/mira).
This script does not download weights; it only inspects configs shipped with the repo.
"""
from pathlib import Path
import yaml


def summarize_codec(repo_root: str | Path = ".") -> None:
    repo_root = Path(repo_root)
    cfg_path = repo_root / "configs" / "model" / "raev2_codec_tdown.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    arch = cfg["architecture"]["config"]
    enc = arch["encoder"]
    dec = arch["decoder"]

    print("MIRA RAEv2 codec summary")
    print("=" * 50)
    print(f"Backbone: {enc['rae_model']} (frozen)")
    print(f"Aggregated DINO layers: {enc['aggregation_layers']}")
    print(f"Latent dim: {enc['latent_dim']}")
    print(f"Bottleneck stride (space/time): {enc['bottleneck']['stride']}x / {enc['bottleneck']['temporal_stride']}x")
    print(f"Video: {enc['video']['height']}x{enc['video']['width']} @ {enc['video']['fps']} fps, {enc['video']['timesteps']} frames")
    print(f"Decoder: {dec['vit_depth']} layers, width {dec['vit_width']}, heads {dec['vit_num_heads']}")
    print(f"Decoder patch size (space/time): {dec['patch_size']} / {dec['patch_size_t']}")

    weights = cfg["loss"].get("config", {}).get("weights", cfg["loss"].get("weights", {}))
    print("\nLoss terms:")
    for k, v in weights.items():
        if isinstance(v, (int, float)) and v:
            print(f"  {k}: {v}")


def summarize_actions(repo_root: str | Path = ".") -> None:
    repo_root = Path(repo_root)
    actions_path = repo_root / "src" / "mira" / "data" / "actions.py"
    if not actions_path.exists():
        print(f"\nAction file not found at {actions_path}; skipping action summary.")
        return
    txt = actions_path.read_text()
    marker = "default_rl"
    if marker in txt:
        idx = txt.find(marker)
        snippet = txt[idx:idx + 600]
        print("\nKeyVocab.default_rl snippet:")
        for line in snippet.splitlines()[:20]:
            print("  " + line)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    summarize_codec(root)
    summarize_actions(root)
