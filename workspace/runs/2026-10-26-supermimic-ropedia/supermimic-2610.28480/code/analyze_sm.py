"""Aggregate SuperMimic benchmark JSONs into coverage/generalization tables and plots.

Usage:
  python analyze_sm.py --eval-dir <dir-with-benchmark-jsons> --out <outdir>
Expects files named eval_<split>_<tag>.json where split in {train30, heldout10}.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_results(path):
    d = json.load(open(path))
    if isinstance(d, dict) and "clips" in d:
        return d["clips"]
    if isinstance(d, dict) and "results" in d:
        return d["results"]
    return d


def summarize(recs):
    rets = np.array([r["norm_return"] for r in recs])
    fails = np.array([bool(r["failed"]) for r in recs])
    steps = np.array([r["steps"] for r in recs])
    frames = np.array([r["frames"] for r in recs])
    full = np.array([r.get("norm_return_full_clip", r["norm_return"] * r["steps"] / max(r["frames"] - 1, 1)) for r in recs])
    return {
        "n": len(recs),
        "failed": int(fails.sum()),
        "mean_return": float(rets.mean()),
        "median_return": float(np.median(rets)),
        "step_weighted_return": float((rets * steps).sum() / steps.sum()),
        "mean_return_full_episode": float(full.mean()),
        "min_return": float(rets.min()),
        "fails": [r["name"] for r in recs if r["failed"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(Path(args.eval_dir).glob("eval_*.json")):
        split = f.stem.split("_")[1]
        tag = f.stem.split("_", 2)[2]
        recs = load_results(f)
        s = summarize(recs)
        s.update({"split": split, "tag": tag, "file": f.name})
        rows.append(s)
        print(f"{tag:24s} {split:10s} n={s['n']:3d} failed={s['failed']:3d} "
              f"mean={s['mean_return']:.4f} full={s['mean_return_full_episode']:.4f} "
              f"stepw={s['step_weighted_return']:.4f} min={s['min_return']:.3f}")
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tags = sorted({r["tag"] for r in rows}, key=lambda t: (len(t), t))
        labels = {"train30": "trained (30 clips)", "heldout10": "held-out (10 clips)"}
        ticks = [f"{int(t) // 1000}k" for t in tags]
        with plt.rc_context({"font.size": 16}):
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            width = 0.35
            for ax, metric, ylabel, title, percent in (
                (axes[0], "mean_return_full_episode", "Mean whole-clip return", "Policy score", False),
                (axes[1], "failed", "Clips ending early (%)", "Early termination", True),
            ):
                for i, split in enumerate(["train30", "heldout10"]):
                    vals = [next((r[metric] / r["n"] if percent else r[metric]
                                  for r in rows if r["tag"] == t and r["split"] == split), np.nan)
                            for t in tags]
                    if percent:
                        vals = [100 * v for v in vals]
                    ax.bar(np.arange(len(tags)) + (i - 0.5) * width, vals, width, label=labels[split])
                ax.set_xticks(np.arange(len(tags)), ticks)
                ax.set_xlabel("Training iterations")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.set_ylim(0, 100 if percent else 1)
                ax.grid(axis="y", alpha=0.2)
                ax.legend(fontsize=13, loc="lower right" if percent else "upper left")
            fig.tight_layout()
            fig.savefig(out / "policy_progress.png", dpi=100)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
