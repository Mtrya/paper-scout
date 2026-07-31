#!/usr/bin/env python3
"""Plot HiFi-UMI-2K direction-reversal distribution from episode_kinematics.csv.

Output: ../../assets/umi-recovery-reversals.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "assets", "umi-recovery-reversals.png")

df = pd.read_csv(os.path.join(HERE, "episode_kinematics.csv"))
total_rev = df["right_reversals"] + df["left_reversals"]
zero_frac = (total_rev == 0).mean()
rev10_r = df["right_rev_per_10s"]
rev10_l = df["left_rev_per_10s"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Left: episodes by total direction reversals (both hands)
counts = [int((total_rev == 0).sum()), int((total_rev == 1).sum()),
          int((total_rev == 2).sum()), int((total_rev >= 3).sum())]
labels = ["0", "1", "2", ">=3"]
colors = ["#d62728", "#8c8c8c", "#8c8c8c", "#8c8c8c"]
bars = axes[0].bar(labels, counts, color=colors)
for b, c in zip(bars, counts):
    axes[0].text(b.get_x() + b.get_width() / 2, c + 8, f"{c}\n({c/len(df)*100:.1f}%)",
                 ha="center", fontsize=10)
axes[0].set_xlabel("Total direction reversals per episode (both hands, 120-deg threshold)")
axes[0].set_ylabel("Episodes")
axes[0].set_title(f"HiFi-UMI-2K, N={len(df)} episodes: {zero_frac*100:.1f}% have ZERO reversals")
axes[0].set_ylim(0, max(counts) * 1.25)

# Right: reversals per 10s, per hand (clipped tail)
bins = np.arange(0, 2.05, 0.1)
axes[1].hist(np.clip(rev10_r, 0, 2), bins=bins, alpha=0.6, label="right hand", color="#1f77b4")
axes[1].hist(np.clip(rev10_l, 0, 2), bins=bins, alpha=0.6, label="left hand", color="#ff7f0e")
axes[1].set_yscale("log")
axes[1].set_xlabel("Direction reversals per 10 s (clipped at 2)")
axes[1].set_ylabel("Episodes (log)")
axes[1].set_title("Per-hand reversal rate: mass sits at exactly 0")
axes[1].legend()

fig.suptitle("Recovery behavior is absent from UMI demonstrations (measured, 2026-07-31 cruise)", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT, "zero_frac=", zero_frac)
