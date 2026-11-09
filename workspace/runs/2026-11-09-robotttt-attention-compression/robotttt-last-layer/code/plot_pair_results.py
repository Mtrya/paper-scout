"""Plot the paired RoboTTT last-layer probe from preserved summary JSON."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def moving_average(values, window=9):
    return [sum(values[max(0, i - window + 1):i + 1]) /
            len(values[max(0, i - window + 1):i + 1])
            for i in range(len(values))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    args = parser.parse_args()
    pair = json.loads((args.thread / "code/pair_T512_2500.json").read_text())
    spectral = json.loads((args.thread / "code/spectral_ttt_last_T1024.json").read_text())
    colors = {"ttt": "#bf3b45", "ttt_last": "#167e78"}
    labels = {"ttt": "all fast parameters", "ttt_last": "last readout only"}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))

    ax = axes[0, 0]
    for arm, rec in pair["arms"].items():
        x = [r["iter"] for r in rec["history"]]
        y = [r["loss"] for r in rec["history"]]
        ax.plot(x, y, color=colors[arm], alpha=0.14, linewidth=0.7)
        ax.plot(x, moving_average(y), color=colors[arm], label=labels[arm], linewidth=1.6)
    ax.set(yscale="log", xlabel="outer iteration", ylabel="sequence action MSE",
           title="A  Paired training trajectories")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    buckets = ["8-16", "16-32", "32-64", "64-128", "128-256"]
    x = list(range(len(buckets)))
    for arm, rec in pair["arms"].items():
        vals = rec["eval"]["512"]["r2_by_context_bucket"]
        ax.plot(x, [vals[b] for b in buckets], marker="o", color=colors[arm], linewidth=1.6)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xticks(x, buckets, rotation=25)
    ax.set(xlabel="context position (step)", ylabel="unsaturated action R²",
           title="B  Last-only updates pay a cold-start cost")

    ax = axes[0, 2]
    buckets = ["384-512", "512-640", "640-768", "768-896", "896-1024"]
    x = list(range(len(buckets)))
    for arm, rec in pair["arms"].items():
        vals = rec["eval"]["1024"]["r2_by_context_bucket"]
        ax.plot(x, [vals[b] for b in buckets], marker="o", color=colors[arm], linewidth=1.6)
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1,
               label="training horizon ends")
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xticks(x, buckets, rotation=25)
    ax.set(xlabel="context position (step)", ylabel="unsaturated action R²",
           title="C  The ordering reverses at long horizon")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for arm, rec in pair["arms"].items():
        layer = rec["diag"]["layers"][0]
        ax.plot(range(1, len(layer["inner_loss"]) + 1), layer["inner_loss"],
                color=colors[arm], linewidth=1.3)
    ax.set(yscale="log", xlabel="context position (step)",
           ylabel="pre-update reconstruction MSE",
           title="D  Layer-0 inner loss crosses late in the rollout")

    ax = axes[1, 1]
    for arm, rec in pair["arms"].items():
        for layer_index, layer in enumerate(rec["diag"]["layers"]):
            ax.plot(range(1, len(layer["drift_upd"]) + 1), layer["drift_upd"],
                    color=colors[arm], linestyle=["-", "--"][layer_index], linewidth=1.3,
                    label=f"{labels[arm]}, layer {layer_index}")
    ax.set(xlabel="context position (step)", ylabel="relative updated-state drift",
           title="E  Parameter drift does not predict task stability")
    ax.legend(frameon=False, fontsize=7, ncol=1)

    ax = axes[1, 2]
    for layer, color in zip(spectral["layers"], ["#386cb0", "#f0a202"]):
        rows = layer["per_step"]
        x = [r["t"] for r in rows]
        mean = [r["radial_eigenvalue_mean"] for r in rows]
        minimum = [r["radial_eigenvalue_min"] for r in rows]
        ax.plot(x, minimum, color=color, alpha=0.18, linewidth=0.7)
        ax.plot(x, moving_average(mean, 17), color=color, linewidth=1.5,
                label=f"layer {layer['layer']}: mean / faint min")
    ax.axhline(1, color="#777777", linestyle="--", linewidth=1, label="non-expansion boundary")
    ax.set(ylim=(0.75, 1.01), xlabel="context position (step)", ylabel="radial eigenvalue",
           title="F  Static-feature updates stay locally contractive")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.12)
    fig.tight_layout()
    args.assets.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.assets / "robotttt-paired-results.png", dpi=100,
                bbox_inches="tight", facecolor="white")
    print("wrote", args.assets / "robotttt-paired-results.png")


if __name__ == "__main__":
    main()
