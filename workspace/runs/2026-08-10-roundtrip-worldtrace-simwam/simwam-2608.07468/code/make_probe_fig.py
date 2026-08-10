#!/usr/bin/env python3
"""SimWAM prefill 截断探针绘图:rel_l2-k 悬崖 + 轨迹叠加。
输入:results/prefill_probe.json;输出:../assets/simwam-prefill-cliff.png
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "results" / "prefill_probe.json"))
ks = d["k_list"]
scen = d["scenarios"]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

# --- Panel A: rel_l2 vs k, all scenarios + median ---
ax = axes[0]
for name, sc in scen.items():
    ys = [sc["delta"][str(k)]["rel_l2"] for k in ks]
    ax.plot(ks, ys, color="#9db8d2", lw=0.9, alpha=0.7, zorder=2)
med = [statistics.median(sc["delta"][str(k)]["rel_l2"] for sc in scen.values()) for k in ks]
ax.plot(ks, med, color="#c0392b", lw=2.4, marker="o", ms=5, zorder=3, label="median of 12 scene-command runs")
ax.set_yscale("log")
ax.set_xlabel("video-tower prefill depth $k$ (layers kept, of 30)")
ax.set_ylabel(r"action traj. rel. $L_2$ vs full prefill")
ax.axvspan(10, 15, color="#f5b041", alpha=0.25, zorder=1)
ax.annotate("the cliff:\n10→15 layers", xy=(12.5, 0.06), xytext=(16.5, 0.25),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="#7d6608"),
            color="#7d6608")
ax.axhline(med[0], color="#7f8c8d", ls=":", lw=1)
ax.text(30, med[0]*1.15, "no video tower at all ($k$=0): 0.39", ha="right", fontsize=8, color="#7f8c8d")
ax.legend(fontsize=8.5, loc="center right")
ax.set_title("The action head reads a representation that converges by layer ~15", fontsize=10)
ax.grid(alpha=0.25, which="both")

# --- Panel B: trajectory overlay, scene0-left ---
ax = axes[1]
show = {0: ("#7f8c8d", "--", "$k$=0 (no tower)"), 10: ("#e67e22", "-.", "$k$=10"),
        15: ("#2980b9", "-", "$k$=15"), 30: ("#27ae60", "-", "$k$=30 (full)")}
tr = scen["scene0-left"]["trajs_m"]
for k, (c, ls, lab) in show.items():
    pts = tr[str(k)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, ls, color=c, lw=2 if k in (15, 30) else 1.6, marker="o", ms=3.5, label=lab)
ax.set_xlabel("x (m, forward)"); ax.set_ylabel("y (m, lateral)")
ax.set_title('Predicted 8-step trajectories, scene0 + command "left"', fontsize=10)
ax.legend(fontsize=8.5)
ax.grid(alpha=0.25)
ax.set_aspect("equal", adjustable="datalim")

fig.tight_layout()
out = HERE.parent.parent / "assets" / "simwam-prefill-cliff.png"
fig.savefig(out, dpi=170)
print("saved", out)
