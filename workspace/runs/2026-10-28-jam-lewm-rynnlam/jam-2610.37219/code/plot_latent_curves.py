#!/usr/bin/env python3
"""Stage-1 latent pretraining loss curves: oracle vs random vs pixels vs const."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["latent_oracle", "latent_random", "latent_pixels", "latent_const"]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
for arm in ARMS:
    try:
        d = json.load(open(f"out/{arm}_seed0.json"))
    except FileNotFoundError:
        continue
    steps = [l["step"] for l in d["losses"] if l["kind"] == "latent"]
    vals = [l["loss"] for l in d["losses"] if l["kind"] == "latent"]
    if steps:
        ax.plot(steps, vals, marker="o", label=arm)
ax.set_yscale("log")
ax.set_xlabel("stage-1 step (flow-matching on future representation)")
ax.set_ylabel("latent loss (MSE, log scale)")
ax.legend(fontsize=8)
ax.set_title("Generation-target role: smooth targets collapse to identity, oracle keeps signal", fontsize=9)
fig.tight_layout()
fig.savefig("latent_curves.png")
print("saved latent_curves.png")
