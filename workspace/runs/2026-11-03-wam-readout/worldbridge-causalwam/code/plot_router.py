"""Visualize what the sparse router actually selects."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

from synth_wam import WAM, render, sample_id, sample_ood, DEVICE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def token_coords():
    c = -1 + (torch.arange(6, device=DEVICE) + 0.5) * (2 / 6)
    return torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(36, 2)


def plot_panel(ax, goal, dist, sel, title):
    img = render(goal.unsqueeze(0), dist.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
    ax.imshow(img)
    coords = token_coords().cpu().numpy()
    # pixel position of normalized coord
    px = (coords[:, 0] + 1) / 2 * 47
    py = (coords[:, 1] + 1) / 2 * 47
    for q, idx in enumerate(sel):
        ax.add_patch(Circle((px[idx], py[idx]), 4.2, fill=False, color="#4fc3f7", lw=1.2))
        ax.text(px[idx] + 5, py[idx] - 5, str(q + 1), color="#4fc3f7", fontsize=6)
    ax.scatter([(goal[0].item() + 1) / 2 * 47], [(goal[1].item() + 1) / 2 * 47],
               marker="*", s=55, c="white", edgecolors="black", linewidths=0.4)
    ax.scatter([(dist[0].item() + 1) / 2 * 47], [(dist[1].item() + 1) / 2 * 47],
               marker="x", s=30, c="#dddddd", linewidths=1.0)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    model = WAM(topk=1).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(OUT, "debug_k1.pt"), map_location=DEVICE))
    model.eval()
    torch.manual_seed(4)
    g1, d1 = sample_id(4)
    g2, d2 = sample_ood(4)
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.0))
    for i in range(4):
        with torch.no_grad():
            _, w, _ = model.head(model.encoder(render(g1[i:i+1], d1[i:i+1])))
        sel = w.argmax(-1)[0].cpu().numpy()
        plot_panel(axes[0, i], g1[i].cpu(), d1[i].cpu(), sel, f"ID {i+1}")
    for i in range(4):
        with torch.no_grad():
            _, w, _ = model.head(model.encoder(render(g2[i:i+1], d2[i:i+1])))
        sel = w.argmax(-1)[0].cpu().numpy()
        plot_panel(axes[1, i], g2[i].cpu(), d2[i].cpu(), sel, f"OOD {i+1}")
    fig.suptitle("Top-1 token selected by each of the 4 action queries (blue circles); star=goal, x=distractor",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_router.png"), dpi=220)
    print("saved", os.path.join(OUT, "fig_router.png"))


if __name__ == "__main__":
    main()
