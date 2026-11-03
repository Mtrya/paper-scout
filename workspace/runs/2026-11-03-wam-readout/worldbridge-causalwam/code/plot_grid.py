"""Aggregate and plot the synthetic WAM grid."""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, "out", "grid")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(GRID, "*.json"))):
        j = json.load(open(f))
        a = j["args"]
        rows.append({
            "cond": os.path.basename(f).split("_")[0],
            "topk": a["topk"], "seed": a["seed"],
            "id_succ": j["id"]["success"], "ood_succ": j["ood"]["success"],
            "id_mse": j["id"]["action_mse"], "ood_mse": j["ood"]["action_mse"],
            "rel": j["id"]["nuisance_reliance"],
            "vid_mse": j["ood"]["video_mse"],
            "mass_g": j["id"].get("mass_goal", np.nan),
            "mass_d": j["id"].get("mass_dist", np.nan),
            "sel_g": j["id"].get("sel_goal_frac", np.nan),
            "sel_d": j["id"].get("sel_dist_frac", np.nan),
            "sel_c": j["id"].get("sel_center_dist", np.nan),
        })
    return pd.DataFrame(rows)


def agg(df):
    return df.groupby(["cond", "topk"]).agg(
        id_succ=("id_succ", "mean"), id_succ_min=("id_succ", "min"), id_succ_max=("id_succ", "max"),
        ood_succ=("ood_succ", "mean"), ood_min=("ood_succ", "min"), ood_max=("ood_succ", "max"),
        rel=("rel", "mean"), rel_min=("rel", "min"), rel_max=("rel", "max"),
        vid_mse=("vid_mse", "mean"),
        sel_g=("sel_g", "mean"), sel_d=("sel_d", "mean"), sel_c=("sel_c", "mean"),
    ).reset_index()


def fig_lines(A, df, metric, ylabel, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    order = ["base", "vid", "vid1"]
    labels = {"base": "ID only", "vid": "+ OOD video", "vid1": "+ 1 demo/task"}
    colors = {"base": "#333333", "vid": "#2f6f9f", "vid1": "#c0504d"}
    for cond in order:
        s = A[A.cond == cond].sort_values("topk")
        if len(s) == 0:
            continue
        x = np.arange(len(s))
        y = s[metric].values
        lo = s[metric + "_min"].values if metric + "_min" in s else y
        hi = s[metric + "_max"].values if metric + "_max" in s else y
        ax.plot(x, y, marker="o", color=colors[cond], label=labels[cond])
        ax.fill_between(x, lo, hi, color=colors[cond], alpha=0.18)
        raw = df[df.cond == cond]
        for i, k in enumerate(s.topk):
            vals = raw[raw.topk == k][metric.replace("succ", "succ")].values if metric == "ood_succ" else raw[raw.topk == k]["rel"].values
            ax.scatter(np.full(len(vals), x[i]) + 0.05, vals, s=14, color=colors[cond], alpha=0.65, zorder=3)
        ax.set_xticks(x, [("dense" if k >= 36 else str(k)) for k in s.topk])
    ax.set_xlabel("top-k tokens per action query")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    df = load()
    df.to_csv(os.path.join(OUT, "grid_raw.csv"), index=False)
    A = agg(df)
    A.to_csv(os.path.join(OUT, "grid_summary.csv"), index=False)
    print(A.round(3).to_string(index=False))
    fig_lines(A, df, "ood_succ", "OOD success rate", os.path.join(OUT, "fig_synth_ood.png"))
    fig_lines(A, df, "rel", "nuisance reliance (action change)", os.path.join(OUT, "fig_synth_reliance.png"))
    # pivot tables for the report
    for metric in ["id_succ", "ood_succ", "rel"]:
        p = A.pivot(index="cond", columns="topk", values=metric)
        p.to_csv(os.path.join(OUT, f"pivot_{metric}.csv"))
        print("\n", metric, "\n", p.round(3).to_string())


if __name__ == "__main__":
    main()
