#!/usr/bin/env python3
"""
Derivation: two data-level tricks that let GenCeption use a single RGB decoder
for heterogeneous tasks.

1. "Rothko" raymap layout: pack a 6-DoF per-pixel camera ray representation
   (3 origin + 3 direction channels) into a standard 3-channel RGB frame by
   spatially splitting the image into a central region (origins) and a
   surrounding ring (directions).  This keeps the decoder unchanged.

2. Median-normalized log depth: remove global scale ambiguity and squeeze a
   wide depth range into [0,1] RGB with a learnable/adjustable parameter alpha.

Both are illustrative only — no model weights are available.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

THREAD = Path(__file__).resolve().parent.parent
RUN = THREAD.parent
OUT = RUN / "assets" / "genception"
OUT.mkdir(parents=True, exist_ok=True)


def rothko_raymap(h=480, w=832):
    """
    Simulate the layout in Figure 5 of the paper.
    Returns an HxWx3 array where:
      - central crop holds per-pixel ray origins (R=ox, G=oy, B=oz)
      - peripheral ring holds ray directions (R=dx, G=dy, B=dz)
    """
    y, x = np.mgrid[0:h, 0:w]
    # Normalized image coordinates
    u = (x - w/2) / (w/2)
    v = (y - h/2) / (h/2)
    # Approximate pinhole ray directions in camera space
    fx, fy = 0.8 * w, 0.8 * h
    dx = (x - w/2) / fx
    dy = (y - h/2) / fy
    dz = np.ones_like(dx)
    # Normalize
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx, dy, dz = dx/norm, dy/norm, dz/norm
    # Origins are simply the camera center (0,0,0) shifted by translation
    ox = np.zeros_like(dx)
    oy = np.zeros_like(dy)
    oz = np.zeros_like(dz)

    # Build central mask (e.g. 50% area)
    cy, cx = h // 2, w // 2
    rh, rw = h // 4, w // 4
    central = ((y >= cy - rh) & (y < cy + rh) &
               (x >= cx - rw) & (x < cx + rw))

    rgb = np.zeros((h, w, 3))
    rgb[central, 0] = ox[central]
    rgb[central, 1] = oy[central]
    rgb[central, 2] = oz[central]
    rgb[~central, 0] = dx[~central]
    rgb[~central, 1] = dy[~central]
    rgb[~central, 2] = dz[~central]

    # Normalize to [0,1] for visualization
    vis = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb, vis, central


def median_log_depth(depth, alpha=0.3):
    """
    Reproduce Equation from Sec 3.5:
        d_norm = depth / median(depth)
        d' = clip(alpha * log(d_norm + 1), 0, 1)
    """
    med = np.median(depth[depth > 0])
    d_norm = depth / med
    d_prime = np.clip(alpha * np.log(d_norm + 1), 0.0, 1.0)
    return med, d_norm, d_prime


def main():
    # 1. Raymap layout figure
    rgb, vis, central = rothko_raymap()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(vis)
    axes[0].set_title("Packed 3-channel raymap")
    axes[0].axis("off")

    axes[1].imshow(central, cmap="gray")
    axes[1].set_title("Central region → ray origins")
    axes[1].axis("off")

    ring = ~central
    axes[2].imshow(ring, cmap="gray")
    axes[2].set_title("Peripheral ring → ray directions")
    axes[2].axis("off")

    fig.tight_layout()
    ray_path = OUT / "rothko_raymap_layout.png"
    fig.savefig(ray_path, dpi=150)
    print("Saved", ray_path)
    plt.close(fig)

    # 2. Depth mapping figure
    # synthetic depth with scale ambiguity
    z = np.linspace(0.1, 50.0, 1000)
    for alpha in [0.15, 0.30, 0.60]:
        _, _, d_prime = median_log_depth(z, alpha=alpha)
        plt.plot(z, d_prime, label=f"α={alpha}")
    plt.xlabel("Depth / median(depth)")
    plt.ylabel("Mapped value d' in [0,1]")
    plt.title("Median-normalized log depth mapping")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    depth_path = OUT / "median_log_depth_mapping.png"
    plt.savefig(depth_path, dpi=150)
    print("Saved", depth_path)
    plt.close()

    # Save the equations/parameters as JSON
    recipe = {
        "raymap": {
            "description": "6-DoF per-pixel rays packed into 3 RGB channels",
            "central_region": "ray origins (ox, oy, oz)",
            "peripheral_ring": "ray directions (dx, dy, dz)",
            "note": "No architectural decoder change needed; VAE sees a normal video latent."
        },
        "depth_mapping": {
            "description": "Data-level scale removal + range compression",
            "steps": [
                "d_norm = depth / median(depth)",
                "d' = clip(alpha * log(d_norm + 1), 0, 1)"
            ],
            "alpha_role": "trade-off between near-field detail and far-field structure"
        }
    }
    json_path = OUT / "data_level_tricks.json"
    json_path.write_text(json.dumps(recipe, indent=2))
    print("Saved", json_path)


if __name__ == "__main__":
    main()
