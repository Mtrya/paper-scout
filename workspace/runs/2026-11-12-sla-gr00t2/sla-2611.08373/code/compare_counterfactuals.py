#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ["state_zero", "state_shuffle", "state_dropout", "image_zero", "image_shuffle"]
LABELS = ["State\n= 0", "State\nshuffled", "State tokens\nremoved", "Images\n= black", "Images\nshuffled"]


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def values(rows, condition):
    selected = sorted((r for r in rows if r["condition"] == condition), key=lambda r: int(r["frame"]))
    return [int(r["frame"]) for r in selected], np.asarray([float(r["relative_rmse"]) for r in selected])


def interval(values, seed=20261112):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = read_rows(args.base)
    post = read_rows(args.post)
    result = {
        "metric": "per-frame relative RMSE before unnormalization and absolute-action reconstruction",
        "difference": "LIBERO post-training minus SLA base",
        "conditions": {},
    }
    means = {"base": [], "post": []}
    errors = {"base": [], "post": []}
    differences = []
    difference_errors = []
    for condition in CONDITIONS:
        base_frames, base_values = values(base, condition)
        post_frames, post_values = values(post, condition)
        if base_frames != post_frames:
            raise RuntimeError(f"frame mismatch for {condition}")
        difference = post_values - base_values
        record = {}
        for name, array in [("base", base_values), ("post", post_values)]:
            mean = float(array.mean())
            lo, hi = interval(array)
            means[name].append(mean)
            errors[name].append([mean - lo, hi - mean])
            record[name] = {"relative_rmse_mean": mean, "bootstrap95": [lo, hi]}
        lo, hi = interval(difference)
        differences.append(float(difference.mean()))
        difference_errors.append([differences[-1] - lo, hi - differences[-1]])
        record["paired_difference"] = {"mean": differences[-1], "bootstrap95": [lo, hi]}
        result["conditions"][condition] = record
    result["frames"] = base_frames
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    x = np.arange(len(CONDITIONS))
    for offset, name, color in [(-0.18, "base", "#4C72B0"), (0.18, "post", "#DD8452")]:
        axes[0].bar(x + offset, means[name], width=0.36, yerr=np.asarray(errors[name]).T, capsize=3, color=color, label="SLA base" if name == "base" else "LIBERO post-training")
    axes[0].set_xticks(x, LABELS)
    axes[0].set_ylabel("Relative action RMSE")
    axes[0].set_title("Input sensitivity on the same frames")
    axes[0].legend(frameon=False)
    axes[1].bar(x, differences, yerr=np.asarray(difference_errors).T, capsize=4, color="#8172B3")
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].set_xticks(x, LABELS)
    axes[1].set_ylabel("Post-training minus base")
    axes[1].set_title("Paired change in sensitivity")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result["conditions"], indent=2))


if __name__ == "__main__":
    main()
