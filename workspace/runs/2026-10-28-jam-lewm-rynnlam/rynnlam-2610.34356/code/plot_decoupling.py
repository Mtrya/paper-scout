#!/usr/bin/env python3
"""Plot RynnLAM decoupling probe: component-wise relative change by condition family."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

rows = pd.read_json("decoupling_results.json")


def fam(name):
    if name == "identity":
        return "identity"
    if name.startswith("content"):
        return name
    return name.split("_")[0]


df = rows.copy()
df["family"] = df.condition.map(fam)
order = ["identity", "content_later", "content_other", "translate", "rotate", "zoom"]
labels = ["identity\n(control)", "content\n+4 frames", "content\nsame-episode far frame",
          "translate\n8–24 px", "rotate\n5–10°", "zoom\n1.1–1.25×"]
g = df.groupby("family")[["d_zm", "d_za", "d_zc"]].mean().reindex(order)
fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=160)
x = np.arange(len(order))
w = 0.26
for i, (col, lab, color) in enumerate([("d_zm", "motion tokens $z^m$ (512d)", "#4878cf"),
                                       ("d_za", "action latent $z^a$ (64d)", "#ee854a"),
                                       ("d_zc", "camera latent $z^c$ (32d)", "#6acc65")]):
    bars = ax.bar(x + (i - 1) * w, g[col], w, label=lab, color=color)
    for b, v in zip(bars, g[col]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("relative change $\\|z_{cond}-z_{base}\\|/\\|z_{base}\\|$")
ax.set_ylim(0, 0.66)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("RynnLAM descriptor sensitivity: camera motion vs content change (6 LIBERO episodes, 18 frame pairs)", fontsize=9)
fig.tight_layout()
fig.savefig("decoupling_fig.png")
print("saved decoupling_fig.png")
