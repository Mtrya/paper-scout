#!/usr/bin/env python3
"""Figures for experiments C (LAWA anchor hypothesis) and D (mini-WorldEcho).

Reads (after scp back from inspire hydra-probe):
  runs/2026-08-27-gigabrain-lawa-worldecho/lawa-2608.24882/code/eval_cfull.json
  runs/2026-08-27-gigabrain-lawa-worldecho/lawa-2608.24882/code/eval_cfs.json
  runs/2026-08-27-gigabrain-lawa-worldecho/worldecho-2608.24885/code/eval_d1.json
Writes:
  runs/2026-08-27-gigabrain-lawa-worldecho/assets/figures/expc_sr.png
  runs/2026-08-27-gigabrain-lawa-worldecho/assets/figures/expd_ade.png

Full-data baselines from the archived 2026-08-14 toy run (eval_v3.json, n=80):
joint .2625 / currentonly .0375 / rift-fm .0 / noiseslots .075
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(__file__).resolve().parents[2]
FIG = RUN / "assets" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BASE14 = {"joint": 26.25, "currentonly": 3.75, "rift-fm": 0.0, "noiseslots": 7.5}
C_ORDER = ["joint", "currentonly", "rift-fm", "noiseslots", "lawatok"]
C_LABEL = {"joint": "joint\n(08-14)", "currentonly": "current\nonly", "rift-fm": "rift-fm",
           "noiseslots": "noise\nslots", "lawatok": "lawatok\n(ours)"}
C_COLOR = {"joint": "#b0b0b0", "currentonly": "#b0b0b0", "rift-fm": "#b0b0b0",
           "noiseslots": "#b0b0b0", "lawatok": "#d62728"}


def fig_c():
    cfull = json.load(open(RUN / "lawa-2608.24882/code/eval_cfull.json"))
    cfs = json.load(open(RUN / "lawa-2608.24882/code/eval_cfs.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    # panel A: full data — archived baselines + new lawatok
    ax = axes[0]
    vals = [BASE14[v] for v in C_ORDER[:-1]] + [cfull["lawatok"]["sr"] * 100]
    bars = ax.bar(range(len(C_ORDER)), vals,
                  color=[C_COLOR[v] for v in C_ORDER], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(C_ORDER)))
    ax.set_xticklabels([C_LABEL[v] for v in C_ORDER], fontsize=8.5)
    ax.set_title("Full data (2000 eps)", fontsize=10)
    ax.set_ylabel("Success rate (%)", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25 + 2)
    # panel B: few-shot 200 eps — all four trained fresh this run
    ax = axes[1]
    fs_order = C_ORDER[1:]
    vals = [cfs[v]["sr"] * 100 for v in fs_order]
    bars = ax.bar(range(len(fs_order)), vals,
                  color=[C_COLOR[v] for v in fs_order], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(fs_order)))
    ax.set_xticklabels([C_LABEL[v] for v in fs_order], fontsize=8.5)
    ax.set_title("Few-shot (200 eps, 10%)", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Experiment C: anchoring learned future slots with a codebook (lawatok)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "expc_sr.png", dpi=170)
    print("wrote", FIG / "expc_sr.png")


def fig_d():
    d1 = json.load(open(RUN / "worldecho-2608.24885/code/eval_d1.json"))
    cats = ["demo", "xstate", "perturb", "random"]
    cat_label = {"demo": "demonstrated", "xstate": "cross-state",
                 "perturb": "local perturb", "random": "random feasible"}
    variants = ["acwm", "acwm-cov", "acwm-ie"]
    v_label = {"acwm": "ACWM (expert-only)", "acwm-cov": "+coverage", "acwm-ie": "+IE"}
    colors = {"acwm": "#1f77b4", "acwm-cov": "#ff7f0e", "acwm-ie": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    w = 0.26
    for ax, metric, title, ylab in [
        (axes[0], "ade_med", "EE-ADE (median, px) — lower = follows queried action", "EE-ADE (px)"),
        (axes[1], "armloss", "Arm-loss rate — fraction of frames EE unreadable", "arm-loss rate"),
    ]:
        for j, v in enumerate(variants):
            vals = [d1[v][c][metric] for c in cats]
            x = np.arange(len(cats)) + (j - 1) * w
            bars = ax.bar(x, vals, width=w, color=colors[v], label=v_label[v])
            for b, val in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels([cat_label[c] for c in cats], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Experiment D: toy AC-WM under off-expert action queries", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "expd_ade.png", dpi=170)
    print("wrote", FIG / "expd_ade.png")


if __name__ == "__main__":
    fig_c()
    fig_d()
