#!/usr/bin/env python3
"""Experiment A figures: AllTracker-on-DROID audit (Hydra-0 training-side interface).

Reads drafts/tracker_full/tracker/*.npy + tracker_summary.json, writes
runs/2026-08-26-hydra0-ld4wam-unimem/assets/expa_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = Path("drafts/tracker_full/tracker")
OUT = Path("runs/2026-08-26-hydra0-ld4wam-unimem/assets")
OUT.mkdir(parents=True, exist_ok=True)

summary = json.loads((BASE / "tracker_summary.json").read_text())

# ---- fig 1: per-episode residual time series (log y) + summary bars ----
fig = plt.figure(figsize=(15, 5.6))
gs = fig.add_gridspec(2, 6, height_ratios=[2.4, 1.1], hspace=0.5, wspace=0.25,
                      top=0.86, bottom=0.08)
for i, r in enumerate(summary):
    ax = fig.add_subplot(gs[0, i])
    resid = np.load(BASE / f"{r['episode']}_resid.npy")
    ax.plot(np.arange(len(resid)) / 15.0, resid, lw=0.7, color="tab:blue")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 300)
    ax.axhline(8, color="gray", ls="--", lw=0.7)
    bg = r["bg_disp_median_end"]
    title = f"{r['episode']}  inlier {r['affine_inlier_frac']:.0%}"
    if bg > 5:
        title += "  [cam/scene moved]"
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.set_ylabel("|affine(EE 3D) - track|  (px, log)", fontsize=8.5)
    ax.set_xlabel("t (s)", fontsize=8)
fig.suptitle("Per-frame residual between gripper pixel track and best-fit affine map of the EE 3D "
             "trajectory (constant calibration absorbed). Dashed = 8 px RANSAC threshold.",
             fontsize=10, color="dimgray", y=0.95)

axb = fig.add_subplot(gs[1, :])
names = [r["episode"] for r in summary]
med = [r["affine_resid_median"] for r in summary]
p95 = [r["affine_resid_p95"] for r in summary]
inl = [r["affine_inlier_frac"] * 100 for r in summary]
x = np.arange(len(names))
axb.bar(x - 0.2, med, width=0.38, color="tab:blue", label="median resid (px)")
axb.bar(x + 0.2, p95, width=0.38, color="tab:red", alpha=0.75, label="p95 resid (px)")
for xi, v, p in zip(x, inl, p95):
    axb.text(xi, p * 1.18, f"inlier {v:.0f}%", ha="center", fontsize=7.5, color="dimgray")
axb.set_ylim(0.5, 400)
axb.set_yscale("log")
axb.set_xticks(x, names, fontsize=9)
axb.set_ylabel("residual (px, log)", fontsize=8.5)
axb.legend(fontsize=8, loc="upper left")
axb.set_title("Median is fine (2-7 px); the tails are the story (p95 up to 84 px). "
              "Bar label = RANSAC inlier fraction.", fontsize=9)
fig.savefig(OUT / "expa_resid.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- fig 2: overlay contact strip (3 episodes x 3 times) ----
pick = [("ep000", [0, 120, 239]), ("ep004", [0, 141, 282]), ("ep005", [0, 75, 149])]
fig, axes = plt.subplots(3, 3, figsize=(12, 5.4))
for row, (ep, ts) in enumerate(pick):
    for col, t in enumerate(ts):
        p = BASE / f"{ep}_overlay_t{t}.jpg"
        ax = axes[row, col]
        if p.exists():
            ax.imshow(Image.open(p))
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(ep, fontsize=10)
        if row == 0:
            ax.set_title(f"t={t}", fontsize=9)
fig.suptitle("AllTracker dense tracks on DROID exterior view: yellow ring = gripper-point track "
             "(follows the gripper), dots = top motion-correlated pixels (home positions)",
             fontsize=9.5)
fig.savefig(OUT / "expa_overlays.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved", OUT / "expa_resid.png", OUT / "expa_overlays.png")
