import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "runs/2026-11-03-wam-readout/assets/rua-harness.png"

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1.05, 1]})

ax = axes[0]
labels = ["Qwen + RoboProbe", "Qwen + RPent", "Qwen + GAP", "Qwen + uni",
          "Astra + RoboProbe", "Astra + uni"]
scores = [1.90, 7.75, 21.01, 31.39, 28.97, 51.90]
colors = ["#c9d6e3", "#c9d6e3", "#c9d6e3", "#2f6f9f", "#e3c9c9", "#a33b3b"]
y = np.arange(len(labels))
ax.barh(y, scores, color=colors, height=0.62)
for yi, s in zip(y, scores):
    ax.text(s + 0.8, yi, f"{s:.2f}", va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 60)
ax.set_xlabel("RoboDojo Score (0-100)", fontsize=9)
ax.set_title("Same model, different harness", fontsize=10.5, loc="left")
ax.grid(axis="x", color="#dddddd", linewidth=0.6)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.text(0.0, -0.22, "Qwen = Qwen3.8-Flash-Next; Astra = GPT-6 Astra", transform=ax.transAxes,
        fontsize=8, color="#555555")

ax = axes[1]
x = np.arange(3)
width = 0.36
blocking = [5.0, 10.0, 0.0]
pipelined = [4.0, 6.0, 0.0]
ax.bar(x - width/2, blocking, width, color="#c9d6e3", label="blocking (serial)")
ax.bar(x + width/2, pipelined, width, color="#2f6f9f", label="pipelined")
ax.bar([2], [1.0], width, color="#7a9e7e", label="parallel tools")
ax.text(0, 4.35, "1.0 s hidden", ha="center", fontsize=8.5, color="#2f6f9f")
ax.text(1, 6.35, "4.0 s hidden", ha="center", fontsize=8.5, color="#2f6f9f")
ax.text(1, 10.25, "2.0 s exposed", ha="center", fontsize=8.5, color="#a33b3b")
ax.set_xticks(x)
ax.set_xticklabels(["4 actions, 0.25 s\nreasoning each", "4 actions, 1.5 s\nreasoning each", "4 parallel\nvision calls"], fontsize=8.5)
ax.set_ylabel("wall time (s)", fontsize=9)
ax.set_ylim(0, 11.5)
ax.set_title("Pipelining hides at most the action time", fontsize=10.5, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.02))
ax.grid(axis="y", color="#dddddd", linewidth=0.6)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180)
print(OUT)
