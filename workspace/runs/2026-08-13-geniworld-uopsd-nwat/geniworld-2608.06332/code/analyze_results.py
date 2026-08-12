"""Analyze + plot the GeniWorld toy interface-probe results.

Input: results_v1.json produced by toy_wam.py on the remote 4090.
Output: analysis JSON + PNG figures for the report.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDS = ["numeric", "concat_static", "concat_shuffle", "concat_motion"]
LABELS = {"numeric": "numeric (EE轨迹)",
          "concat_static": "static mask (姿态冻结)",
          "concat_shuffle": "shuffle render (顺序打乱)",
          "concat_motion": "motion render (GeniWorld)"}
COLORS = {"numeric": "#d62728", "concat_static": "#7f7f7f",
          "concat_shuffle": "#ff7f0e", "concat_motion": "#1f77b4"}


def main():
    with open("results_v1.json") as f:
        data = json.load(f)
    os.makedirs("plots", exist_ok=True)
    out = {}

    # ---- convergence curves (val_mse over steps)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c in CONDS:
        curve = data[c]["curve"]
        steps = [p[0] for p in curve]
        mses = [p[1] for p in curve]
        ax.plot(steps, mses, label=LABELS[c], color=COLORS[c], lw=1.8)
    ax.set_xlabel("training steps")
    ax.set_ylabel("val MSE (50-step sampling)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/convergence.png", dpi=160)
    plt.close(fig)

    # ---- OOD generalization + few-step degradation (bars)
    mets = []
    for c in CONDS:
        d = data[c]
        mets.append({
            "cond": c,
            "val_mse": d["val_mse@50"],
            "ood1_mse": d["ood_stripes_mse@50"],
            "ood2_mse": d["ood_dots_mse@50"],
            "val_5": d["val_mse@5"],
            "val_10": d["val_mse@10"],
            "val_50": d["val_mse@50"],
            "cube_val": d["val_cube_err@50"],
            "cube_ood1": d["ood_stripes_cube_err@50"],
            "cube_ood2": d["ood_dots_cube_err@50"],
        })
    out["table"] = mets

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(CONDS))
    ax = axes[0]
    w = 0.28
    for i, key in enumerate(["val_mse", "ood1_mse", "ood2_mse"]):
        vals = [m[key] for m in mets]
        ax.bar(x + (i - 1) * w, vals, w, label=["in-domain", "OOD stripes", "OOD dots"][i],
               color=[COLORS[c] for c in CONDS], alpha=0.55 + 0.2 * i)
    ax.set_xticks(x)
    ax.set_xticklabels(["num", "static", "shuffle", "motion"], fontsize=8)
    ax.set_ylabel("pixel MSE (50-step)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    w = 0.28
    for i, key in enumerate(["val_50", "val_10", "val_5"]):
        vals = [m[key] for m in mets]
        ax.bar(x + (i - 1) * w, vals, w, label=["50 steps", "10 steps", "5 steps"][i],
               color=[COLORS[c] for c in CONDS], alpha=0.55 + 0.2 * i)
    ax.set_xticks(x)
    ax.set_xticklabels(["num", "static", "shuffle", "motion"], fontsize=8)
    ax.set_ylabel("val MSE (in-domain)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("plots/ood_fewstep.png", dpi=160)
    plt.close(fig)

    # ---- cube error (functional metric)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(CONDS))
    w = 0.28
    for i, key in enumerate(["cube_val", "cube_ood1", "cube_ood2"]):
        vals = [m[key] for m in mets]
        ax.bar(x + (i - 1) * w, vals, w, label=["in-domain", "OOD stripes", "OOD dots"][i],
               color=[COLORS[c] for c in CONDS], alpha=0.55 + 0.2 * i)
    ax.set_xticks(x)
    ax.set_xticklabels(["num", "static", "shuffle", "motion"], fontsize=8)
    ax.set_ylabel("final-frame cube error (px)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("plots/cube_err.png", dpi=160)
    plt.close(fig)

    with open("analysis.json", "w") as f:
        json.dump(out, f, indent=2)

    # console summary
    print(f"{'cond':16s} {'val':>8s} {'ood1':>8s} {'ood2':>8s} {'@10':>8s} {'@5':>8s} {'cube_v':>7s} {'cube_o1':>7s} {'cube_o2':>7s}")
    for m in mets:
        print(f"{m['cond']:16s} {m['val_mse']:8.4f} {m['ood1_mse']:8.4f} {m['ood2_mse']:8.4f} "
              f"{m['val_10']:8.4f} {m['val_5']:8.4f} {m['cube_val']:7.1f} {m['cube_ood1']:7.1f} {m['cube_ood2']:7.1f}")
    print("\nOOD degradation ratio (ood2/val):",
          {c: round(m['ood2_mse'] / max(m['val_mse'], 1e-9), 2) for c, m in
           [(c, m) for c, m in zip(CONDS, mets)]})
    print("few-step degradation ratio (5/50):",
          {c: round(m['val_5'] / max(m['val_50'], 1e-9), 2) for c, m in
           [(c, m) for c, m in zip(CONDS, mets)]})


if __name__ == "__main__":
    main()
