"""Figures for the ShadowDancer sprites probe report section.

Reads shadow_sprites_results.json (produced by sprites_probe.py) and writes:
  - sprites_probe_bars.png: grouped bars of ridge-probe R^2 (appearance leak
    vs dynamics content) + cross/self MSE ratio per variant
  - sprites_pairs_example.png: two example shadow pairs (src x_t, x_{t+1};
    shadow x~_t, x~_{t+1}) so readers can see the toy environment
"""

import json
import math
import colorsys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = 48
rng = np.random.default_rng(7)


def rand_color():
    h, s, v = rng.uniform(0, 1), rng.uniform(0.6, 1.0), rng.uniform(0.7, 1.0)
    return np.array(colorsys.hsv_to_rgb(h, s, v), np.float32)


def render(balls, bg):
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    img = np.full((IMG, IMG, 3), bg, np.float32)
    for x, y, r, c in balls:
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
        img[mask] = c
    return img


def sample_transition():
    while True:
        pa = rng.uniform(9, IMG - 9, 2)
        pb = rng.uniform(9, IMG - 9, 2)
        if np.linalg.norm(pa - pb) > 14:
            break
    theta = rng.uniform(0, 2 * math.pi)
    speed = rng.uniform(8.0, 14.0)
    vel = speed * np.array([math.cos(theta), math.sin(theta)])
    pb1 = pb + vel
    if not (6 < pb1[0] < IMG - 6 and 6 < pb1[1] < IMG - 6):
        return sample_transition()
    return pa, pb, pb1


def example_pairs(path):
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for row in range(2):
        pa, pb, pb1 = sample_transition()
        for col in range(2):  # col 0: source, col 1: shadow
            ca, cb = rand_color(), rand_color()
            ra, rb = rng.uniform(3.0, 5.5, 2)
            bg = rng.uniform(0.03, 0.15)
            xt = render([(pa[0], pa[1], ra, ca), (pb[0], pb[1], rb, cb)], bg)
            xt1 = render([(pa[0], pa[1], ra, ca), (pb1[0], pb1[1], rb, cb)], bg)
            axes[row, col * 2].imshow(np.clip(xt, 0, 1))
            axes[row, col * 2 + 1].imshow(np.clip(xt1, 0, 1))
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    for ax, t in zip(axes[0], ["src $x_t$", "src $x_{t+1}$",
                               "shadow $\\tilde{x}_t$", "shadow $\\tilde{x}_{t+1}$"]):
        ax.set_title(t, fontsize=10)
    fig.suptitle("Two shadow pairs: same dynamics, resampled appearance", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def bars(results, path):
    names = [n for n in results if not n.startswith("_")]
    labels = {"A-selfrec-b0.01": "A\nself\n$\\beta$=.01",
              "Ahb-selfrec-b1.0": "Ahb\nself\n$\\beta$=1.0",
              "B-crosshadow-b0.01": "B\ncross\n$\\beta$=.01"}
    metrics = [("r2_color_moving", "moving-ball color $R^2$ (leak)"),
               ("r2_color_static", "static-ball color $R^2$ (leak)"),
               ("r2_direction", "direction $R^2$ (dynamics)"),
               ("cross_self_ratio", "cross/self MSE ratio")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    colors = {"A-selfrec-b0.01": "#c44e52", "Ahb-selfrec-b1.0": "#dd8452",
              "B-crosshadow-b0.01": "#4c72b0"}
    for ax, (key, title) in zip(axes, metrics):
        vals = [results[n][key] for n in names]
        ax.bar([labels[n] for n in names], vals,
               color=[colors[n] for n in names], width=0.6)
        ax.set_title(title, fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        lo, hi = min(vals + [0]), max(vals)
        ax.set_ylim(min(0, lo * 1.2 - 0.05), hi * 1.25 + 0.03)
        if key == "cross_self_ratio":
            ax.axhline(1.0, color="gray", ls="--", lw=1)
        if "r2" in key:
            ax.axhline(0.0, color="gray", lw=1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    with open("shadow_sprites_results.json") as f:
        results = json.load(f)
    bars(results, "sprites_probe_bars.png")
    example_pairs("sprites_pairs_example.png")
    print("FIGS_DONE")
