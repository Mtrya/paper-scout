#!/usr/bin/env python3
"""Visualize SIEVE probe results: pattern redistribution."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_probe(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pattern_counts_from_json(items):
    return Counter({tuple(k): v for k, v in items})


def plot_redistribution(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("SIEVE probe: composition-pattern redistribution (synthetic data)", fontsize=13)

    for row_idx, label in enumerate(("code", "paper")):
        data = result[label]
        counts_before = pattern_counts_from_json(data["pattern_counts_before"])
        counts_after = pattern_counts_from_json(data["pattern_counts_after"])

        # 1. Pattern-length distribution.
        len_before = Counter(len(p) for p in counts_before)
        len_after = Counter(len(p) for p in counts_after)
        max_len = max(list(len_before.keys()) + list(len_after.keys()) + [1])
        x_len = np.arange(1, max_len + 1)
        before_vals = [len_before.get(i, 0) for i in x_len]
        after_vals = [len_after.get(i, 0) for i in x_len]

        ax = axes[row_idx, 0]
        width = 0.35
        ax.bar(x_len - width / 2, before_vals, width, label="before", color="gray", alpha=0.7)
        ax.bar(x_len + width / 2, after_vals, width, label="after", alpha=0.85)
        ax.set_xlabel("Pattern length (#primitives)")
        ax.set_ylabel("Episode count")
        ax.set_title(f"{label} norm: pattern length")
        ax.set_xticks(x_len)
        ax.legend()

        # 2. Sorted pattern frequency (Lorenz-style).
        before_all = sorted(counts_before.values(), reverse=True)
        after_all = sorted(counts_after.values(), reverse=True)
        ax = axes[row_idx, 1]
        ax.plot(range(len(before_all)), before_all, "o-", label="before", color="gray", alpha=0.7, markersize=3)
        ax.plot(range(len(after_all)), after_all, "s-", label="after", alpha=0.85, markersize=3)
        ax.set_xlabel("Pattern rank")
        ax.set_ylabel("Episode count")
        ax.set_title(f"{label} norm: sorted frequency")
        ax.legend()

        # 3. Top selected patterns.
        top_after = counts_after.most_common(12)
        patterns = [p for p, _ in top_after]
        before_vals = [counts_before.get(p, 0) for p in patterns]
        after_vals = [counts_after.get(p, 0) for p in patterns]
        labels = [str(p) for p in patterns]

        ax = axes[row_idx, 2]
        x = np.arange(len(labels))
        ax.bar(x - width / 2, before_vals, width, label="before", color="gray", alpha=0.7)
        ax.bar(x + width / 2, after_vals, width, label="after", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Episode count")
        ax.set_title(f"{label} norm: top selected patterns")
        ax.legend()

    plt.tight_layout()
    out_path = output_dir / "probe_redistribution.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-result", type=Path, default=Path("../probe_result.json"))
    p.add_argument("--output-dir", type=Path, default=Path("../figures"))
    args = p.parse_args()

    result = load_probe(args.probe_result)
    plot_redistribution(result, args.output_dir)


if __name__ == "__main__":
    main()
