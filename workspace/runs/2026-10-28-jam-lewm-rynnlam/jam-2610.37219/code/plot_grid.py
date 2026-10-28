#!/usr/bin/env python3
"""Toy grid summary: success + future-state probe R^2 per arm (mean +/- std over 3 seeds)."""
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["action", "fs_oracle", "fs_random", "fs_pixels", "fs_const",
         "fs_current_random", "latent_oracle", "latent_random", "latent_pixels", "latent_const",
         "latent_oracle_long", "latent_random_long"]
LABELS = {"action": "action only", "fs_oracle": "fs oracle\n(future state)",
          "fs_random": "fs random\n(frozen CNN)", "fs_pixels": "fs pixels\n(raw patches)",
          "fs_const": "fs const\n(input-indep.)", "fs_current_random": "fs random\n(current frame)",
          "latent_oracle": "latent oracle\n(2-stage)", "latent_random": "latent random\n(2-stage)",
          "latent_pixels": "latent pixels\n(2-stage)", "latent_const": "latent const\n(input-indep.)",
          "latent_oracle_long": "latent oracle\n(long action)", "latent_random_long": "latent random\n(long action)"}

res = {}
for f in sorted(glob.glob("out/*_seed*.json")):
    d = json.load(open(f))
    res.setdefault(d["arm"], []).append(d)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=160, sharex=True)
present = [a for a in ORDER if a in res]
xs = np.arange(len(present))
for ax, key, title, color in [(axes[0], "success", "rollout success (200 eps/seed)", "#4878cf"),
                              (axes[1], "probe_r2", "frozen-feature → future state ridge $R^2$", "#ee854a")]:
    means = [np.mean([d[key] for d in res[a]]) for a in present]
    stds = [np.std([d[key] for d in res[a]]) for a in present]
    ax.bar(xs, means, yerr=stds, capsize=4, color=color, alpha=0.85)
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[a] for a in present], fontsize=7, rotation=30, ha="right")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1.0)
fig.suptitle(f"Role × target-quality grid on visual push-block ({len(present)} arms × 3 seeds × 12k steps)", fontsize=10)
fig.tight_layout()
fig.savefig("toy_grid.png")
print("saved toy_grid.png")
