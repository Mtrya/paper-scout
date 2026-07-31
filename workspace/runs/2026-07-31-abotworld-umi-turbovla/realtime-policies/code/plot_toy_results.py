#!/usr/bin/env python3
"""Plot piR2 toy results: sync vs staircase across observation delays.

Numbers are the final 8-seed run of pir2_modality_toy.py (2026-07-31 cruise).
Output: ../../assets/pir2-toy-delay.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "assets", "pir2-toy-delay.png")

# mode, delay, err_arm, err_target, err_all (8-seed means)
ROWS = [
    ("sync", 1, 0.5507, 0.6409, 0.5443),
    ("sync", 3, 0.6295, 0.7128, 0.6200),
    ("sync", 5, 0.7224, 0.7840, 0.7337),
    ("staircase", 1, 0.6345, 0.7002, 0.6450),
    ("staircase", 3, 0.6715, 0.7539, 0.6646),
    ("staircase", 5, 0.7110, 0.8080, 0.6869),
]
STD = {  # (mode, delay) -> (std_arm, std_target)
    ("sync", 1): (0.200, 0.263), ("sync", 3): (0.190, 0.215), ("sync", 5): (0.198, 0.255),
    ("staircase", 1): (0.193, 0.244), ("staircase", 3): (0.183, 0.216), ("staircase", 5): (0.192, 0.173),
}

d = [1, 3, 5]
def col(mode, idx):
    return [r[idx] for r in ROWS if r[0] == mode]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Left: err_all vs delay — the crossover
axes[0].plot(d, col("sync", 4), "o-", color="#1f77b4", label="sync (denoise-all-at-once)")
axes[0].plot(d, col("staircase", 4), "s-", color="#d62728", label="staircase (piR2-style)")
axes[0].annotate("staircase pays a\nquality tax at low delay", xy=(1, 0.645), xytext=(1.3, 0.685),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
axes[0].annotate("crossover: last-minute\nfinalization pays off", xy=(5, 0.687), xytext=(1.15, 0.715),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
axes[0].set_xlabel("observation delay d (control ticks)")
axes[0].set_ylabel("tracking error (all perturbations)")
axes[0].set_title("The real independent variable is delay, not modality")
axes[0].legend(fontsize=9, loc="lower right")

# Right: per-modality errors overlap — modality hypothesis falsified
w = 0.18
for i, mode in enumerate(("sync", "staircase")):
    xs = np.array(d) + (i - 0.5) * 2 * w
    axes[1].errorbar(xs - w / 2, col(mode, 2), yerr=[STD[(mode, x)][0] for x in d],
                     fmt="o", color="#2ca02c" if mode == "sync" else "#9467bd",
                     label=f"{mode}: proprio-visible (arm impulse)", capsize=3, markersize=5)
    axes[1].errorbar(xs + w / 2, col(mode, 3), yerr=[STD[(mode, x)][1] for x in d],
                     fmt="^", color="#2ca02c" if mode == "sync" else "#9467bd", mfc="none",
                     label=f"{mode}: vision-only (target jump)", capsize=3, markersize=5)
axes[1].set_xlabel("observation delay d (control ticks)")
axes[1].set_ylabel("tracking error (8 seeds, mean ± std)")
axes[1].set_title("Proprio-visible vs vision-only perturbations: no asymmetry")
axes[1].legend(fontsize=7.5, loc="upper left")

fig.suptitle("piR2 staircase toy replication (1D point mass, flow-matching, diffusion forcing)", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
