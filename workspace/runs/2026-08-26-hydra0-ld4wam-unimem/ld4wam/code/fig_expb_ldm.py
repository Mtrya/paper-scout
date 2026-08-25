#!/usr/bin/env python3
"""Experiment B figure: LDM frozen probe on OOD DROID (LD4WAM audit).

Reads drafts/ldm_probe_v4_results.json (adds magnitude control), writes
runs/2026-08-26-hydra0-ld4wam-unimem/assets/expb_probe.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("drafts/ldm_probe_v4_results.json"))
OUT = Path("runs/2026-08-26-hydra0-ld4wam-unimem/assets")
OUT.mkdir(parents=True, exist_ok=True)

DIMS = ["x", "y", "z", "roll", "pitch", "yaw"]

fig, (axL, axR, axM) = plt.subplots(1, 3, figsize=(17, 4.6), width_ratios=[1.35, 1, 1])

# ---- left: val R2 by condition, LDM vs DINOv3-diff; train R2 as markers ----
conds = [("1", "ldm_all", "dv3_all", "s1 all"),
         ("1", "ldm_mov05", "dv3_mov05", "s1 |d|>0.5"),
         ("1", "ldm_top50", "dv3_top50", "s1 top-50%"),
         ("2", "ldm_all", "dv3_all", "s2 all"),
         ("2", "ldm_top50", "dv3_top50", "s2 top-50%")]
labels, ldm_v, dv3_v, ldm_t, dv3_t = [], [], [], [], []
for s, kl, kd, lab in conds:
    r = d["strides"][s]
    if kl not in r:
        continue
    labels.append(lab)
    ldm_v.append(float(np.mean(r[kl]["r2"])))
    dv3_v.append(float(np.mean(r[kd]["r2"])))
    ldm_t.append(r[kl]["r2_train"])
    dv3_t.append(r[kd]["r2_train"])
x = np.arange(len(labels))
axL.bar(x - 0.2, ldm_v, width=0.38, color="tab:purple", label="LDM latent (val)")
axL.bar(x + 0.2, dv3_v, width=0.38, color="tab:gray", label="DINOv3 feat-diff (val)")
axL.scatter(x - 0.2, ldm_t, marker="D", s=28, color="tab:purple", edgecolor="k",
            zorder=5, label="LDM latent (train)")
axL.scatter(x + 0.2, dv3_t, marker="D", s=28, color="tab:gray", edgecolor="k",
            zorder=5, label="DINOv3 feat-diff (train)")
axL.axhline(0, color="k", lw=0.8)
axL.set_xticks(x, labels, fontsize=9)
axL.set_ylabel("R2 of delta-EE regression (6D)", fontsize=9.5)
axL.set_ylim(-1.0, 1.15)
axL.legend(fontsize=8, ncol=2, loc="upper left")
axL.set_title("OOD probe on DROID (unseen in LDM training): both stay at floor on val;\n"
              "DINOv3 memorizes train (train R2~1.0) and fails below mean on val", fontsize=9.5)

# ---- right: LDM per-dim R2, stride1 top50 & stride2 top50 ----
s1 = d["strides"]["1"]["ldm_top50"]["r2"]
s2 = d["strides"]["2"]["ldm_top50"]["r2"]
xd = np.arange(6)
axR.bar(xd - 0.2, s1, width=0.38, color="tab:blue", label="s1 (15 Hz) top-50%")
axR.bar(xd + 0.2, s2, width=0.38, color="tab:cyan", label="s2 (7.5 Hz) top-50%")
axR.axhline(0, color="k", lw=0.8)
axR.set_xticks(xd, DIMS, fontsize=9.5)
axR.set_ylabel("R2 per dimension (val)", fontsize=9.5)
axR.legend(fontsize=8.5)
axR.set_title("Per-dimension breakdown (LDM latent):\nonly vertical (z) motion decodes, rising with "
              "temporal stride", fontsize=9.5)
for xi, v in zip(xd - 0.2, s1):
    axR.text(xi, v + 0.01 if v >= 0 else v - 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
for xi, v in zip(xd + 0.2, s2):
    axR.text(xi, v + 0.01 if v >= 0 else v - 0.03, f"{v:.2f}", ha="center", fontsize=7.5)

# ---- third: magnitude control (rotation/translation-invariant targets) ----
mconds = [("1", "ldm_all_mag", "s1 all"), ("1", "ldm_mov05_mag", "s1 |d|>0.5"),
          ("1", "ldm_top50_mag", "s1 top-50%"), ("2", "ldm_all_mag", "s2 all"),
          ("2", "ldm_top50_mag", "s2 top-50%")]
mlabels, mag_xyz, mag_rot = [], [], []
for s, k, lab in mconds:
    r = d["strides"][s]
    if k not in r:
        continue
    mlabels.append(lab)
    mag_xyz.append(r[k]["r2"][0])
    mag_rot.append(r[k]["r2"][1])
xm = np.arange(len(mlabels))
axM.bar(xm - 0.2, mag_xyz, width=0.38, color="tab:green", label="||d(xyz)|| (val)")
axM.bar(xm + 0.2, mag_rot, width=0.38, color="tab:olive", label="||d(rot)|| (val)")
axM.axhline(0, color="k", lw=0.8)
axM.set_xticks(xm, mlabels, fontsize=9)
axM.set_ylabel("R2 of magnitude regression (val)", fontsize=9.5)
axM.set_ylim(-0.35, 0.5)
axM.legend(fontsize=8.5)
axM.set_title("Magnitude control (invariant to any fixed per-episode\nframe): decodes on 'all' only via the static/moving bit;\nfloor within moving frames -> rules out extrinsics story",
              fontsize=9)
for xi, v in zip(xm - 0.2, mag_xyz):
    axM.text(xi, v + 0.012 if v >= 0 else v - 0.035, f"{v:.2f}", ha="center", fontsize=7.5)

fig.tight_layout()
fig.savefig(OUT / "expb_probe.png", dpi=150, bbox_inches="tight")
print("saved", OUT / "expb_probe.png")
