"""实验 C 报告图:v3/v4 对照 —— 池化不确定度的制度内优势与制度外崩溃。"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})

NAMES = [("fixed-0.1", "fixed $\\beta$=0.1", "#999999", ":"),
         ("gated", "reliability-gated $\\beta_t$", "#1f77b4", "--"),
         ("kalman-iso", "Kalman isotropic (pooled)", "#d62728", "-"),
         ("kalman-diag", "Kalman diagonal (per-channel)", "#2ca02c", "-")]


def load(path):
    d = json.load(open(path))
    return d["curve"], d["drifts"]


def plot_panel(ax, curve, drifts, group, title, ylab=True):
    for key, label, color, ls in NAMES:
        xs = [c["t"] for c in curve]
        ys = [c[f"{key}_{group}"] for c in curve]
        ax.semilogy(xs, ys, ls, color=color, lw=1.8 if "kalman" in key else 1.2, label=label)
    for d in drifts:
        ax.axvline(d, color="black", alpha=0.15, lw=0.8)
    ax.set_title(title, fontsize=11)
    if ylab:
        ax.set_ylabel("recall MSE (log)")
    ax.set_xlabel("stream token")
    ax.set_ylim(5e-4, 3)


fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)

c3, d3 = load("probe_v3_curve.json")
plot_panel(axes[0], c3, d3, "d", "A. 70% noisy drifting channels: pooled wins")
axes[0].text(0.02, 0.04, "iso < diag everywhere in-regime", transform=axes[0].transAxes, fontsize=9, color="#d62728")

c4, d4 = load("probe_v4_curve.json")
plot_panel(axes[1], c4, d4, "d", "B. 100% noisy drifting channels: pooled starves", ylab=False)
axes[1].text(0.02, 0.04, "iso drift error > fixed-$\\beta$; diag stays near-best", transform=axes[1].transAxes, fontsize=9, color="#2ca02c")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Drifting-group recall MSE: when does per-direction uncertainty matter?", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("kdn_v3v4_contrast.png", bbox_inches="tight")
print("saved kdn_v3v4_contrast.png")

# 第二张:v4 stable 组 + drift 组双线,展示 diag 的全制度近最优
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
plot_panel(axes[0], c4, d4, "s", "C. stable channels (always reliable)")
plot_panel(axes[1], c4, d4, "d", "D. drifting channels (always noisy)", ylab=False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("v4 setting, both groups: diagonal is the only arm near-best in both regimes", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("kdn_v4_both_groups.png", bbox_inches="tight")
print("saved kdn_v4_both_groups.png")
