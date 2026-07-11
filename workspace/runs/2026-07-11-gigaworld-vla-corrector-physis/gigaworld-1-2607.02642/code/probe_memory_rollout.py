#!/usr/bin/env python3
"""
Probe: simulate GigaWorld-1 hierarchical memory and relative RoPE for long-horizon rollout.

The paper claims that reliable evaluators need persistent memory for long-horizon
rollout (Finding 10). This probe reconstructs the token budget from the Stage-1
config and shows how history_sizes [16,2,1] plus a first-frame anchor create a
hierarchical history buffer.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def relative_rope_positions(history_len: int, future_len: int):
    """Local temporal positions reinitialized at every autoregressive step."""
    return list(range(history_len + future_len))


def memory_token_budget(history_sizes, latent_window: int, patches_per_frame: int = 300):
    """
    Estimate token counts for the hierarchical memory buffer.

    history_sizes: e.g. [16, 2, 1] for short/mid/long history frames kept.
    latent_window: future latent frames generated in one denoising step.
    patches_per_frame: rough proxy for 480x640 tokens after patchify.
    """
    anchor_tokens = 1 * patches_per_frame
    short_tokens = history_sizes[0] * patches_per_frame
    mid_tokens = history_sizes[1] * patches_per_frame
    long_tokens = history_sizes[2] * patches_per_frame
    future_tokens = latent_window * patches_per_frame
    total_context = anchor_tokens + short_tokens + mid_tokens + long_tokens + future_tokens
    return {
        "anchor_frames": 1,
        "short_history_frames": history_sizes[0],
        "mid_history_frames": history_sizes[1],
        "long_history_frames": history_sizes[2],
        "future_frames": latent_window,
        "anchor_tokens": anchor_tokens,
        "short_tokens": short_tokens,
        "mid_tokens": mid_tokens,
        "long_tokens": long_tokens,
        "future_tokens": future_tokens,
        "total_context_tokens": total_context,
    }


def simulate_autoregressive_chunks(total_frames: int, latent_window: int, history_sizes):
    """
    Show which historical frames are available at each generation chunk.
    Returns a list of (chunk_idx, future_range, history_ranges).
    """
    chunks = []
    # First chunk has no generated history; it uses the first frame as anchor.
    chunks.append(
        {
            "chunk": 0,
            "future": (1, 1 + latent_window),
            "history": {"anchor": (0, 1), "short": None, "mid": None, "long": None},
        }
    )

    generated = [0]  # frame indices already generated or used as anchor
    for chunk_idx in range(1, (total_frames - 1) // latent_window + 1):
        start = 1 + chunk_idx * latent_window
        end = min(start + latent_window, total_frames)
        future_range = (start, end)

        # Hierarchical memory picks recent, mid, and long-range frames from generated.
        short_end = start
        short_start = max(0, short_end - history_sizes[0])
        mid_end = max(0, short_start)
        mid_start = max(0, mid_end - history_sizes[1])
        long_end = max(0, mid_start)
        long_start = max(0, long_end - history_sizes[2])

        chunks.append(
            {
                "chunk": chunk_idx,
                "future": future_range,
                "history": {
                    "anchor": (0, 1),
                    "short": (short_start, short_end),
                    "mid": (mid_start, mid_end),
                    "long": (long_start, long_end),
                },
            }
        )
        generated.extend(range(start, end))
    return chunks


def plot_memory_timeline(chunks, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {"anchor": "#d62728", "short": "#1f77b4", "mid": "#ff7f0e", "long": "#2ca02c", "future": "#9467bd"}
    y_positions = {"anchor": 4, "short": 3, "mid": 2, "long": 1, "future": 0}
    labels = {"anchor": "First-frame anchor", "short": "Short history", "mid": "Mid history", "long": "Long history", "future": "Future window"}

    max_frame = max(c["future"][1] for c in chunks)
    for c in chunks:
        for name, rng in [("future", c["future"])] + list(c["history"].items()):
            if rng is None:
                continue
            start, end = rng
            ax.barh(y_positions[name], end - start, left=start, height=0.6, color=colors[name], alpha=0.85)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([labels[k] for k in y_positions])
    ax.set_xlabel("Frame index")
    ax.set_title("Hierarchical memory timeline during autoregressive rollout")
    ax.set_xlim(0, max_frame)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved memory timeline figure to {out_path}")


def plot_rope_positions(out_path: Path):
    """Illustrate Relative RoPE: positions are local to each chunk."""
    history_len, future_len = 19, 9  # 16+2+1 history frames + 9 future
    positions = relative_rope_positions(history_len, future_len)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(positions, marker="o", markersize=4)
    ax.axvline(x=history_len - 0.5, color="red", linestyle="--", label="history/future boundary")
    ax.set_xlabel("Concatenated token index")
    ax.set_ylabel("Local temporal position")
    ax.set_title("Relative RoPE: positions reset each autoregressive step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved RoPE figure to {out_path}")


def main():
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path(__file__).parent / "official_snippets" / "stage_1_post_functrl_wan21.yaml"

    cfg = load_config(config_path)
    tc = cfg["training_config"]
    vc = cfg["validation_config"]

    history_sizes = tc["history_sizes"]
    latent_window = vc["validation_latent_window_size"][0]
    total_frames = vc["validation_max_num_frames"]

    budget = memory_token_budget(history_sizes, latent_window)
    chunks = simulate_autoregressive_chunks(total_frames, latent_window, history_sizes)

    out_dir = Path(__file__).parent / "probe_outputs"
    out_dir.mkdir(exist_ok=True)

    plot_memory_timeline(chunks, out_dir / "memory_timeline.png")
    plot_rope_positions(out_dir / "relative_rope.png")

    result = {
        "config_source": str(config_path),
        "history_sizes": history_sizes,
        "latent_window_size": latent_window,
        "validation_max_num_frames": total_frames,
        "token_budget": budget,
        "num_chunks_for_validation_rollout": len(chunks),
        "first_few_chunks": chunks[:5],
    }

    json_path = out_dir / "memory_rollout.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    md_path = out_dir / "memory_rollout.md"
    with open(md_path, "w") as f:
        f.write("# Hierarchical Memory & Relative RoPE Probe\n\n")
        f.write(f"Config: `{config_path}`\n\n")
        f.write("## Token budget estimate\n\n")
        f.write(f"- Total context tokens per denoising step: **{budget['total_context_tokens']:,}**\n")
        f.write(f"  - First-frame anchor: {budget['anchor_tokens']:,}\n")
        f.write(f"  - Short history ({history_sizes[0]} frames): {budget['short_tokens']:,}\n")
        f.write(f"  - Mid history ({history_sizes[1]} frames): {budget['mid_tokens']:,}\n")
        f.write(f"  - Long history ({history_sizes[2]} frames): {budget['long_tokens']:,}\n")
        f.write(f"  - Future window ({latent_window} frames): {budget['future_tokens']:,}\n")
        f.write(f"\nValidation rollout length: {total_frames} frames → {len(chunks)} chunks of size {latent_window}.\n\n")
        f.write("## Relative RoPE\n\n")
        f.write("At each autoregressive step the concatenated [history, future] sequence gets local positions ")
        f.write("`0 … history_len+future_len-1`, then RoPE is applied. This prevents the model from seeing ")
        f.write("unseen absolute positions during long rollouts.\n\n")
        f.write("![Memory timeline](memory_timeline.png)\n")
        f.write("![Relative RoPE](relative_rope.png)\n")

    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
