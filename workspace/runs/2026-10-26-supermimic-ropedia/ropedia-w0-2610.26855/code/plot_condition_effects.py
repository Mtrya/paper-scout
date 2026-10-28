#!/usr/bin/env python
"""Plot per-frame differences between full and counterfactual generations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "rice_synthcam_vs_rice_full": "synthetic camera",
    "rice_skelshift_vs_rice_full": "skeleton +40 px",
    "rice_skelshuffle_vs_rice_full": "shuffled timing",
    "rice_noskel_vs_rice_full": "skeleton off",
    "rice_none_vs_rice_full": "camera + skeleton off",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    comparisons = json.loads(args.data.read_text())["comparisons"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, label in LABELS.items():
        if name not in comparisons:
            continue
        values = np.asarray(comparisons[name]["mean_abs_rgb_by_frame"])
        ax.plot(np.arange(1, len(values) + 1), values, label=label, linewidth=1.8)
    ax.set(xlabel="Frame", ylabel="Mean absolute RGB difference",
           title="Difference from the full-condition generation")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
