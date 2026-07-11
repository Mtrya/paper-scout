#!/usr/bin/env python3
"""
Probe: visualize the two view-specific control signals described in Sec 6.2.2.

- Head (static) camera: end-effector (EE) pose map encoding manipulation intent.
- Wrist (moving) cameras: ray map encoding camera geometry.

This script synthesizes toy robot-camera trajectories and renders the control
maps so the spatial-alignment claim can be inspected directly.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_ee_pose_map(height: int, width: int, ee_positions: np.ndarray, gripper: np.ndarray):
    """
    Render a 2-channel EE pose map: position heatmap + gripper state.

    ee_positions: (T, 2) normalized image coordinates (x, y)
    gripper: (T,) in [0, 1]
    """
    T = len(ee_positions)
    map_h, map_w = height // 8, width // 8
    pose_map = np.zeros((T, map_h, map_w, 2), dtype=np.float32)
    for t, (x, y) in enumerate(ee_positions):
        px, py = int(x * map_w), int(y * map_h)
        if 0 <= px < map_w and 0 <= py < map_h:
            pose_map[t, py, px, 0] = 1.0
            pose_map[t, py, px, 1] = gripper[t]
    return pose_map


def render_ray_map(height: int, width: int, camera_poses: np.ndarray):
    """
    Render a toy ray map: for a wrist camera, each pixel stores the ray direction
    in world coordinates. We approximate by rotating a canonical ray field by the
    camera orientation at each timestep.

    camera_poses: (T, 3) — (x, y, yaw)
    """
    T = len(camera_poses)
    map_h, map_w = height // 8, width // 8
    ray_map = np.zeros((T, map_h, map_w, 3), dtype=np.float32)

    y_coords, x_coords = np.meshgrid(np.linspace(-1, 1, map_h), np.linspace(-1, 1, map_w), indexing="ij")
    canonical = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
    canonical = canonical / (np.linalg.norm(canonical, axis=-1, keepdims=True) + 1e-8)

    for t, (cx, cy, yaw) in enumerate(camera_poses):
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        rays = canonical @ R.T
        ray_map[t] = rays
    return ray_map


def plot_control_maps(ee_map, ray_map, out_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    timesteps = np.linspace(0, ee_map.shape[0] - 1, 4, dtype=int)
    for col, t in enumerate(timesteps):
        axes[0, col].imshow(ee_map[t, :, :, 0], cmap="hot", vmin=0, vmax=1)
        axes[0, col].set_title(f"EE pose map t={t}")
        axes[0, col].axis("off")

        # Visualize ray direction as RGB
        ray_rgb = (ray_map[t] + 1) / 2
        axes[1, col].imshow(ray_rgb)
        axes[1, col].set_title(f"Ray map t={t}")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Head camera: EE pose", fontsize=12)
    axes[1, 0].set_ylabel("Wrist camera: ray dir", fontsize=12)
    plt.suptitle("Toy spatially-aligned control representations (Sec 6.2.2)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved control map visualization to {out_path}")


def main():
    out_dir = Path(__file__).parent / "probe_outputs"
    out_dir.mkdir(exist_ok=True)

    H, W = 480, 640
    T = 16

    # Toy trajectory: EE moves diagonally and grips halfway.
    ee_positions = np.stack([np.linspace(0.3, 0.7, T), np.linspace(0.3, 0.7, T)], axis=-1)
    gripper = np.where(np.arange(T) >= T // 2, 1.0, 0.0)

    # Toy wrist-camera trajectory: translates and yaws.
    camera_poses = np.stack(
        [
            np.linspace(-0.2, 0.2, T),
            np.linspace(0.1, 0.1, T),
            np.linspace(-0.3, 0.3, T),
        ],
        axis=-1,
    )

    ee_map = render_ee_pose_map(H, W, ee_positions, gripper)
    ray_map = render_ray_map(H, W, camera_poses)

    plot_control_maps(ee_map, ray_map, out_dir / "control_interface_maps.png")

    result = {
        "map_shapes": {
            "ee_pose_map": list(ee_map.shape),
            "ray_map": list(ray_map.shape),
        },
        "description": (
            "EE pose map renders end-effector trajectory as a spatial heatmap plus gripper state; "
            "ray map stores per-pixel world-coordinate ray directions for moving wrist cameras. "
            "Both are spatially aligned with the video latent and concatenated channel-wise."
        ),
    }

    json_path = out_dir / "control_interface.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    md_path = out_dir / "control_interface.md"
    with open(md_path, "w") as f:
        f.write("# Control Interface Probe\n\n")
        f.write("The paper (Sec 6.2.2) injects actions through a unified pixel-aligned representation:\n\n")
        f.write("- **Head camera**: EE pose map (manipulation intent).\n")
        f.write("- **Wrist cameras**: ray map (camera geometry).\n\n")
        f.write("This probe synthesizes toy trajectories and renders the two control modalities.\n\n")
        f.write(f"- EE pose map shape: {ee_map.shape}\n")
        f.write(f"- Ray map shape: {ray_map.shape}\n\n")
        f.write("![Control maps](control_interface_maps.png)\n")

    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
