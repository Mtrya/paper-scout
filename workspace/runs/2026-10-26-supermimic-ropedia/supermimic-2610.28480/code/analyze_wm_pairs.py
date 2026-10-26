"""Compare paired motion families, with policy return and model error from each identical rollout."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text())
    rows = probe["rows"]
    by_name = {row["clip"]: row for row in rows}
    assert len(by_name) == len(rows)
    by_family = {}
    for row in rows:
        by_family.setdefault(row["clip"].rsplit("_subject", 1)[0], []).append(row)
    train_names = (args.eval_dir / "train30.txt").read_text().split()
    heldout_names = (args.eval_dir / "heldout10.txt").read_text().split()
    train_set, heldout_set = set(train_names), set(heldout_names)

    pairs = []
    unpaired = []
    for family, group in sorted(by_family.items()):
        trained = [r for r in group if r["clip"] in train_set]
        heldout = [r for r in group if r["clip"] in heldout_set]
        if not trained or not heldout:
            unpaired.extend(r["clip"] for r in group)
            continue
        assert len(trained) == len(heldout) == 1
        a, b = trained[0], heldout[0]
        pairs.append({
            "family": family, "train_clip": a["clip"], "heldout_clip": b["clip"],
            "train_nse1": a["nse1"], "heldout_nse1": b["nse1"],
            "train_nse8": a["nse8"], "heldout_nse8": b["nse8"],
            "train_rms8": a["rms8"], "heldout_rms8": b["rms8"],
            "train_return": a["policy_return_full_clip"], "heldout_return": b["policy_return_full_clip"],
            "train_failed": a["policy_failed"], "heldout_failed": b["policy_failed"],
            "train_steps": a["policy_steps"], "heldout_steps": b["policy_steps"],
        })
    assert len(pairs) == 9, f"expected 9 pairs, got {len(pairs)}"
    summary = {"n_pairs": len(pairs), "unpaired_heldout": sorted(unpaired)}
    for key in ("nse1", "nse8"):
        a = np.array([p[f"train_{key}"] for p in pairs])
        b = np.array([p[f"heldout_{key}"] for p in pairs])
        summary[key] = {
            "train_median": float(np.median(a)), "heldout_median": float(np.median(b)),
            "mean_paired_difference_heldout_minus_train": float(np.mean(b - a)),
            "median_paired_difference_heldout_minus_train": float(np.median(b - a)),
            "pairs_with_larger_heldout_error": int((b > a).sum()),
        }
    args.out_json.write_text(json.dumps({"summary": summary, "pairs": pairs}, indent=2) + "\n")

    from matplotlib.ticker import NullFormatter
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.get_cmap("tab10").colors
    limits = [min(p[k] for p in pairs for k in ("train_nse1", "heldout_nse1", "train_nse8", "heldout_nse8")) / 1.4,
              max(p[k] for p in pairs for k in ("train_nse1", "heldout_nse1", "train_nse8", "heldout_nse8")) * 1.4]
    for ax in axes:
        ax.tick_params(labelsize=12)
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
    ax = axes[0]
    ax.plot(limits, limits, color="#999999", linewidth=1, linestyle="--")
    for i, pair in enumerate(pairs):
        ax.scatter(pair["train_nse1"], pair["heldout_nse1"], color=colors[i], marker="o", s=52)
        ax.scatter(pair["train_nse8"], pair["heldout_nse8"], color=colors[i], marker="^", s=56)
        ax.annotate(pair["family"], (pair["train_nse8"], pair["heldout_nse8"]),
                    xytext=(5, 7 if i % 2 == 0 else -13), textcoords="offset points", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Trained performer: normalized state error")
    ax.set_ylabel("Held-out performer: normalized state error")
    ax.set_title("World model on matched motion families")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="black", linestyle="", label="1 step (0.02 s)"),
                       Line2D([0], [0], marker="^", color="black", linestyle="", label="8 steps (0.16 s)")],
              loc="upper left", fontsize=11)
    ax.grid(alpha=0.18, which="both")

    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=1, linestyle="--")
    for i, pair in enumerate(pairs):
        failed = int(pair["train_failed"]) + int(pair["heldout_failed"])
        color = ("#238b45" if failed == 0 else "#fd8d3c" if failed == 1 else "#cb181d")
        ax.scatter(pair["train_return"], pair["heldout_return"], color=color, s=64)
        ax.annotate(pair["family"], (pair["train_return"], pair["heldout_return"]),
                    xytext=(5, 7 if i % 2 == 0 else -13), textcoords="offset points", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Trained performer: whole-clip return")
    ax.set_ylabel("Held-out performer: whole-clip return")
    ax.set_title("Same rollout: full-motion policy score")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="#238b45", linestyle="", label="both finish"),
                       Line2D([0], [0], marker="o", color="#fd8d3c", linestyle="", label="one ends early"),
                       Line2D([0], [0], marker="o", color="#cb181d", linestyle="", label="both end early")],
              loc="upper left", fontsize=11)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=100)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
