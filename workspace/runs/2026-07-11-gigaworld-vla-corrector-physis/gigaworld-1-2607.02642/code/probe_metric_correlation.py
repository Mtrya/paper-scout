#!/usr/bin/env python3
"""
Probe: reproduce the WMBench metric-vs-WMES correlation pattern on synthetic data.

The paper (Finding 1 & 2, Figures 4-5) reports that some automatic metrics are
strong positive predictors of human-judged evaluator quality (WMES), while
appearance-stability metrics can be negative predictors because they reward
degenerate static rollouts. This probe generates a toy dataset with that
structure and recovers the same qualitative ranking.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_submissions(n: int = 80, seed: int = 42):
    """
    Simulate N world-model submissions with varying quality.

    We assume a latent "evaluator_quality" variable that drives both WMES and
    the honest metrics. A separate "static_bias" variable creates the
    degenerate behaviour rewarded by appearance-stability metrics.
    """
    rng = np.random.default_rng(seed)
    quality = rng.normal(0, 1, size=n)
    static_bias = rng.uniform(0, 1, size=n)

    # WMES is driven primarily by quality; static videos cap out at low scores.
    wmes = 1.5 + 1.2 * quality - 1.5 * static_bias
    wmes = np.clip(wmes, 0, 3)

    metrics = {
        "Subject Consistency": 0.6 + 0.25 * quality + 0.05 * rng.normal(size=n),
        "Perspectivity": 0.55 + 0.23 * quality + 0.06 * rng.normal(size=n),
        "Instruction Following": 0.65 + 0.20 * quality + 0.05 * rng.normal(size=n),
        "JEPA Similarity": 0.50 + 0.18 * quality + 0.07 * rng.normal(size=n),
        "Aesthetic Quality": 0.45 + 0.12 * quality + 0.08 * rng.normal(size=n),
        "Image Quality": 0.50 + 0.10 * quality + 0.06 * rng.normal(size=n),
        "Semantic Alignment": 0.30 + 0.05 * quality + 0.10 * rng.normal(size=n),
        "Background Consistency": 0.40 - 0.20 * quality + 0.45 * static_bias + 0.06 * rng.normal(size=n),
        "Photometric Consistency": 0.35 - 0.18 * quality + 0.50 * static_bias + 0.06 * rng.normal(size=n),
        "Interaction Quality": 0.30 + 0.02 * quality - 0.05 * static_bias + 0.08 * rng.normal(size=n),
    }

    # Clip to [0, 1]
    for k in metrics:
        metrics[k] = np.clip(metrics[k], 0, 1)

    return metrics, wmes


def pearson(x: np.ndarray, y: np.ndarray):
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).sum() / (np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum()) + 1e-12)


def plot_correlation_bars(corrs: dict, out_path: Path):
    names = list(corrs.keys())
    values = list(corrs.values())
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, values, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with WMES")
    ax.set_title("Synthetic metric-WMES correlations (GigaWorld-1 pattern)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved correlation bar plot to {out_path}")


def plot_heatmap(metrics: dict, wmes: np.ndarray, out_path: Path):
    names = list(metrics.keys()) + ["WMES"]
    data = [metrics[m] for m in metrics] + [wmes]
    n = len(names)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = pearson(np.array(data[i]), np.array(data[j]))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Metric correlation matrix (synthetic)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved correlation heatmap to {out_path}")


def main():
    out_dir = Path(__file__).parent / "probe_outputs"
    out_dir.mkdir(exist_ok=True)

    metrics, wmes = make_submissions(n=80, seed=42)
    corrs = {m: float(pearson(metrics[m], wmes)) for m in metrics}
    corrs_sorted = dict(sorted(corrs.items(), key=lambda kv: kv[1]))

    plot_correlation_bars(corrs_sorted, out_dir / "metric_wmes_correlations.png")
    plot_heatmap(metrics, wmes, out_dir / "metric_correlation_heatmap.png")

    result = {
        "note": "Synthetic reproduction of Figures 4-5 from GigaWorld-1.",
        "metric_correlations_with_WMES": corrs_sorted,
        "interpretation": {
            "positive_predictors": [k for k, v in corrs_sorted.items() if v > 0.3],
            "negative_predictors": [k for k, v in corrs_sorted.items() if v < -0.1],
            "unreliable": [k for k, v in corrs_sorted.items() if abs(v) < 0.15],
        },
    }

    json_path = out_dir / "metric_correlation.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    md_path = out_dir / "metric_correlation.md"
    with open(md_path, "w") as f:
        f.write("# Synthetic WMBench Metric Correlation Probe\n\n")
        f.write("This probe generates synthetic world-model submissions with two latent factors:\n")
        f.write("1. **evaluator_quality** — drives WMES and honest fidelity/geometry/semantics metrics.\n")
        f.write("2. **static_bias** — drives appearance-stability metrics independently; high static_bias ")
        f.write("produces rollouts that look stable but ignore actions, yielding low WMES.\n\n")
        f.write("The resulting correlation pattern matches the paper's Finding 1 & 2:\n\n")
        f.write("| Metric | Pearson r with WMES |\n|---|---|\n")
        for metric, r in corrs_sorted.items():
            f.write(f"| {metric} | {r:.3f} |\n")
        f.write("\n![Metric-WMES correlations](metric_wmes_correlations.png)\n")
        f.write("\n![Correlation matrix](metric_correlation_heatmap.png)\n")

    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
