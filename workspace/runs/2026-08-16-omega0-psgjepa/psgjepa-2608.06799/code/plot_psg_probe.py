#!/usr/bin/env python3
"""Plot PSG-JEPA probe comparison figures.

Inputs (produced by probe_psg.py on the notebook):
  probe_results.json  {name: {per_dim_r, group_r, rollout_mse}}
  dataset_stats.json  {var, pixel_visibility, groups}

Outputs:
  psg_probe_dims.png    per-dim ridge r, lewm vs psg, colored by group,
                        with pixel-visibility overlay
  psg_rollout_mse.png   open-loop rollout MSE at steps {5,15,30}
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "probe_results.json"))
stats = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "dataset_stats.json"))
out_prefix = sys.argv[3] if len(sys.argv) > 3 else "psg"

names = list(results.keys())  # ["lewm", "psg"]
D = len(results[names[0]]["per_dim_r"])
groups = stats["groups"]  # {name: [a, b]}
group_order = ["joint_pos", "joint_vel", "effector", "privileged"]
group_colors = {"joint_pos": "#4C8BF5", "joint_vel": "#F5A623",
                "effector": "#50B86C", "privileged": "#B45AD0"}
vis = np.array(stats["pixel_visibility"], dtype=float)

fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})
x = np.arange(D)
w = 0.4
ax = axes[0]
for gi, name in enumerate(names):
    r = np.array(results[name]["per_dim_r"], dtype=float)
    off = (gi - (len(names) - 1) / 2) * w
    for g in group_order:
        a, b = groups[g]
        m = (x >= a) & (x < b)
        ax.bar(x[m] + off, r[m], width=w,
               color=group_colors[g], alpha=0.95 if gi == len(names) - 1 else 0.45,
               edgecolor="none",
               label=None if gi else g)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("held-out Pearson r (linear probe)")
ax.set_ylim(-0.05, 1.05)
handles = [plt.Rectangle((0, 0), 1, 1, color=group_colors[g]) for g in group_order]
leg1 = ax.legend(handles, group_order, loc="upper left", ncol=4, fontsize=9,
                 title="state group", frameon=False)
ax.add_artist(leg1)
mh = [plt.Rectangle((0, 0), 1, 1, color="gray", alpha=a) for a in (0.45, 0.95)]
ax.legend(mh, names, loc="upper right", fontsize=9, title="model", frameon=False)
ax.set_title("Per-dimension identifiability of the frozen latent (PSG-JEPA vs LeWM)")

ax2 = axes[1]
for g in group_order:
    a, b = groups[g]
    m = (x >= a) & (x < b)
    ax2.bar(x[m], vis[m], color=group_colors[g], edgecolor="none")
ax2.set_ylabel("|corr(Δobs, Δpixel)|")
ax2.set_xlabel("observation dimension (0-5 joint pos, 6-11 joint vel, 12-18 effector, 19-27 privileged)")
ax2.set_ylim(0, max(0.6, np.nanmax(vis) * 1.15))
fig.savefig(f"{out_prefix}_probe_dims.png", dpi=170, bbox_inches="tight")

# rollout MSE
steps = sorted({int(s) for n in names for s in results[n].get("rollout_mse", {})})
fig, ax = plt.subplots(figsize=(5.6, 3.6))
xw = np.arange(len(steps))
for gi, name in enumerate(names):
    rm = results[name].get("rollout_mse", {})
    vals = [rm.get(str(s), np.nan) for s in steps]
    ax.bar(xw + (gi - (len(names) - 1) / 2) * 0.36, vals, width=0.36,
           label=name, alpha=0.95 if gi == len(names) - 1 else 0.45,
           color="#D05555")
ax.set_xticks(xw, [f"{s} steps" for s in steps])
ax.set_ylabel("open-loop latent MSE")
ax.legend(frameon=False)
ax.set_title("Open-loop rollout: grounding improves forward prediction")
fig.savefig(f"{out_prefix}_rollout_mse.png", dpi=170, bbox_inches="tight")
print("saved", f"{out_prefix}_probe_dims.png", f"{out_prefix}_rollout_mse.png")
