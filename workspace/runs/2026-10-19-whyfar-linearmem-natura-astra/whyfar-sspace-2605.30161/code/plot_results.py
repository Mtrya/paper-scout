"""Plot the WFLU / S-Space cross-probe results."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_DIR, cos  # noqa: E402

CATS = ["left", "right", "above", "below", "far", "close"]
AX = ["horizontal", "vertical", "distance"]
COL = {"horizontal": "#1f77b4", "vertical": "#2ca02c", "distance": "#9467bd"}


def style():
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 150, "font.family": "DejaVu Sans",
    })


def fig_layerwise(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for g in AX:
        axes[0].plot(df.layer, df[f"coh_{g[0].upper()}"], "-o", ms=2.5,
                     color=COL[g], label=g.capitalize())
    axes[0].axvline(28, color="gray", ls="--", lw=1)
    axes[0].set_xlabel("Transformer layer")
    axes[0].set_ylabel("Axis coherence (Coh)")
    axes[0].set_title("WFLU axis coherence")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(df.layer, df.vd_ei_eq3, "-o", ms=2.5, color="#d62728",
                 label="VD-EI (paper Eq. 3)")
    axes[1].plot(df.layer, df.vd_ei_repo, "-s", ms=2.5, color="#ff7f0e",
                 label="VD-EI (repo variant)")
    axes[1].axvline(28, color="gray", ls="--", lw=1, label="paper L*=28")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_xlabel("Transformer layer")
    axes[1].set_ylabel("VD-EI")
    axes[1].set_title("Vertical-distance entanglement")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(delta, meta, layer_idx, out):
    means = {}
    for c in CATS:
        idx = [i for i, m in enumerate(meta) if m["category"] == c]
        means[c] = delta[idx, layer_idx].astype(np.float64).mean(0)
    M = np.zeros((6, 6))
    for i, a in enumerate(CATS):
        for j, b in enumerate(CATS):
            M[i, j] = cos(means[a], means[b])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    ax.set_xticks(range(6), CATS, rotation=45)
    ax.set_yticks(range(6), CATS)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(M[i, j]) < 0.6 else "white")
    ax.set_title(f"Δ similarity, layer {layer_idx}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return M


def fig_sspace(df, out):
    d = df[df.readout_acc_overall.notna() & (df.layer < 35)]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for g in AX:
        ax.plot(d.layer, d[f"readout_acc_{g[0]}"], "-o", ms=4,
                color=COL[g], label=g.capitalize())
    ax.axhline(0.5, color="black", ls=":", lw=1, label="chance")
    ax.axvline(24, color="gray", ls="--", lw=1)
    ax.text(24.3, 0.38, "best L24", fontsize=7, color="gray")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("S-Space readout accuracy")
    ax.set_ylim(0.35, 1.0)
    ax.set_title("Per-axis linear readout (axes from 2 samples/axis)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_bridge(delta_L, hs_q_L, hs_r_L, meta, axes, L, out):
    dc_z = np.dot(hs_q_L.astype(np.float64) - hs_r_L.astype(np.float64), axes[2])
    a_z = np.dot(delta_L.astype(np.float64), axes[2])
    keep = [i for i, m in enumerate(meta) if m["group"] == "distance"]
    x = -dc_z[keep]
    y = a_z[keep]
    cats = [meta[i]["category"] for i in keep]
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for c, col in [("far", "#9467bd"), ("close", "#8c564b")]:
        idx = [i for i, cc in enumerate(cats) if cc == c]
        ax.scatter(x[idx], y[idx], s=8, alpha=0.55, color=col, label=c)
    b, a = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, a + b * xx, color="black", lw=1.2)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("S-Space distance coordinate difference (query − reference)")
    ax.set_ylabel("WFLU Δ projected on $v_z$")
    ax.set_title(f"Layer {L}: r = {r:+.2f}, $R^2$ = {r**2:.2f}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return r


def fig_geometry(metrics_row, boot, out):
    keys = ["cos_far_v", "cos_far_d", "cos_close_v", "cos_close_d"]
    labels = ["Δfar·$v_y$", "Δfar·$v_z$", "Δclose·$v_y$", "Δclose·$v_z$"]
    vals = [boot[k]["value"] for k in keys]
    lo = [boot[k]["value"] - boot[k]["ci95"][0] for k in keys]
    hi = [boot[k]["ci95"][1] - boot[k]["value"] for k in keys]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    colors = ["#2ca02c", "#9467bd", "#2ca02c", "#9467bd"]
    ax.bar(range(4), vals, color=colors, yerr=[lo, hi], capsize=3,
           error_kw={"elinewidth": 1, "ecolor": "black"})
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("cosine with S-Space axis")
    ax.set_title(f"Mean distance Δ direction, layer {int(metrics_row['layer'])}")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wflu-dir", default=os.path.join(OUT_DIR, "wflu_qwen25"))
    ap.add_argument("--sspace-dir", default=os.path.join(OUT_DIR, "sspace_qwen25"))
    ap.add_argument("--summary-dir", default=os.path.join(OUT_DIR, "summary"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "plots"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    style()

    df = pd.read_csv(os.path.join(args.summary_dir, "metrics_by_layer.csv"))
    with open(os.path.join(args.summary_dir, "summary.json")) as f:
        summary = json.load(f)
    delta = np.load(os.path.join(args.wflu_dir, "delta.npy"))
    hs_q = np.load(os.path.join(args.wflu_dir, "hs_q.npy"))
    hs_r = np.load(os.path.join(args.wflu_dir, "hs_r.npy"))
    with open(os.path.join(args.wflu_dir, "meta.json")) as f:
        meta = json.load(f)
    axes_all = np.load(os.path.join(args.sspace_dir, "axes.npy"))
    ss_layers = [int(x) for x in np.load(os.path.join(args.sspace_dir, "layer_ids.npy"))]
    best_L = int(summary["best_sspace_layer"])
    best_i = ss_layers.index(best_L)
    layer_idx = 28

    fig_layerwise(df, os.path.join(args.out, "layerwise_metrics.png"))
    fig_heatmap(delta, meta, layer_idx, os.path.join(args.out, "heatmap_L28.png"))
    fig_sspace(df, os.path.join(args.out, "sspace_readout.png"))
    r = fig_bridge(delta[:, best_L], hs_q[:, best_L], hs_r[:, best_L],
                   meta, axes_all[best_i], best_L,
                   os.path.join(args.out, "bridge_scatter.png"))
    row = df[df.layer == best_L].iloc[0].to_dict()
    fig_geometry(row, summary["bootstrap_geometry_best"],
                 os.path.join(args.out, "delta_axis_cosines.png"))
    print(f"bridge r={r:.3f}; plots in {args.out}")


if __name__ == "__main__":
    main()
