"""
HUG mechanism probe: RGB-D point-cloud cropping around a query point.

Reimplements (in pure NumPy) the point-cloud preprocessing step that HUG uses
before its PointNeXt encoder: back-project a depth image to metric XYZ, lift a
2D click to a 3D query point via the camera intrinsics, and keep only points
inside a sphere of radius r=0.3 m around that query point.

Why it matters: HUG's full point cloud is too sparse once it is downsampled to
256 tokens. Cropping to the grasp region concentrates those tokens on the
target object; the paper's ablation (Table 2) shows that removing the crop cuts
test success rate from 73.0% to 58.0% on HUG-BENCH.

This script runs without the HUG model, MANO assets, or the 1M-HUGS dataset.
It builds a synthetic tabletop scene, crops around a user click, and writes a
visualization to the run's assets folder.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def make_synthetic_scene(image_size: int = 224):
    """Create a synthetic RGB-D tabletop scene with two objects.

    Returns:
        rgb: (H, W, 3) uint8 image.
        depth_m: (H, W) float32 depth map in meters.
        K: (3, 3) camera intrinsics.
        target_uv: (2,) click pixel on the target object.
    """
    H = W = image_size
    # Pinhole camera looking forward; 60 deg horizontal FOV at 224 px.
    fx = fy = 300.0
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    # Table at z = 1.20 m (camera frame). At this distance the 224 px FOV is
    # roughly 0.9 x 0.9 m, so a 0.30 m sphere crop removes most of the table
    # background and concentrates tokens on the clicked object.
    table_z = 1.20
    depth_m = np.full((H, W), table_z, dtype=np.float32)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Background table: gray.
    rgb[:, :] = 200

    # Pixel coordinate grids.
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    # Metric X/Y at the table plane for every pixel.
    X = (uu - cx) * table_z / fx
    Y = (vv - cy) * table_z / fy

    # Target object: a cyan box at center, size 0.07 x 0.12 m, height 0.05 m.
    target_center = np.array([0.0, 0.05])
    target_half = np.array([0.035, 0.06])
    in_target = (
        (np.abs(X - target_center[0]) < target_half[0])
        & (np.abs(Y - target_center[1]) < target_half[1])
    )
    target_z = table_z - 0.05
    depth_m[in_target] = target_z
    rgb[in_target] = [0, 180, 255]

    # Distractor object: a red sphere offset to the right, radius 0.045 m.
    dist_center = np.array([0.18, -0.08])
    dist_radius = 0.045
    dist_sq = (X - dist_center[0]) ** 2 + (Y - dist_center[1]) ** 2
    in_dist = dist_sq < dist_radius**2
    dist_z = table_z - 0.04
    depth_m[in_dist] = dist_z
    rgb[in_dist] = [255, 60, 60]

    # Click near the center of the target object.
    target_uv = np.array([W * 0.50, H * 0.46])
    return rgb, depth_m, K, target_uv


def pixel_to_xyz(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    """Back-project a single pixel to metric camera-frame XYZ."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth])


def backproject_to_pcl(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    max_depth: float = 3.0,
    center: np.ndarray | None = None,
    crop_radius: float | None = None,
):
    """Backproject depth to (xyz, rgb) and optionally sphere-crop around center.

    Mirrors src/utils/pcl_utils.py::backproject_to_pcl.
    """
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_m.astype(np.float32)
    valid = (z > 0) & (z < max_depth)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    mask = valid.flatten()
    if center is not None and crop_radius is not None:
        center = np.asarray(center, dtype=np.float32).reshape(3)
        dist_sq = ((xyz - center) ** 2).sum(axis=-1)
        mask = mask & (dist_sq < crop_radius * crop_radius)
    return xyz[mask], rgb_flat[mask]


def sample_fixed_n(
    xyz: np.ndarray, rgb: np.ndarray, n_points: int, rng: np.random.Generator
):
    """Randomly subsample/repeat to exactly n_points."""
    M = xyz.shape[0]
    if M >= n_points:
        idx = rng.choice(M, size=n_points, replace=False)
    else:
        idx = rng.choice(M, size=n_points, replace=True)
    return xyz[idx], rgb[idx]


def crop_fraction(xyz: np.ndarray, center: np.ndarray, radius: float) -> float:
    """Fraction of points retained by a sphere crop."""
    if xyz.shape[0] == 0:
        return 0.0
    inside = ((xyz - center) ** 2).sum(axis=-1) < radius * radius
    return float(inside.mean())


def main():
    rng = np.random.default_rng(42)
    rgb, depth_m, K, target_uv = make_synthetic_scene()

    # Query point: back-project the click.
    u, v = target_uv
    d = depth_m[int(round(v)), int(round(u))]
    query_xyz = pixel_to_xyz(u, v, float(d), K)

    # Full point cloud.
    xyz_full, rgb_full = backproject_to_pcl(depth_m, rgb, K)
    # Cropped point cloud at HUG's radius.
    xyz_crop, rgb_crop = backproject_to_pcl(
        depth_m, rgb, K, center=query_xyz, crop_radius=0.30
    )
    # Resampled to 4096 (matching HUG's PointNeXt input).
    xyz_sample, rgb_sample = sample_fixed_n(xyz_crop, rgb_crop, 4096, rng)

    print("=" * 60)
    print("HUG point-cloud crop probe")
    print("=" * 60)
    print(f"Image resolution: {rgb.shape[:2]}")
    print(f"Click (u, v, d): ({u:.1f}, {v:.1f}, {d:.3f} m)")
    print(f"Query point (x, y, z): ({query_xyz[0]:.3f}, {query_xyz[1]:.3f}, {query_xyz[2]:.3f})")
    print(f"Full point cloud: {xyz_full.shape[0]} points")
    print(f"After 0.30 m crop: {xyz_crop.shape[0]} points "
          f"({crop_fraction(xyz_full, query_xyz, 0.30) * 100:.1f}% retained)")
    print(f"Subsampled to PointNeXt input: {xyz_sample.shape[0]} points")

    # Sweep crop radii to show the sparsity/density trade-off.
    radii = np.linspace(0.05, 0.60, 50)
    fractions = [crop_fraction(xyz_full, query_xyz, r) for r in radii]

    # Visualization.
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # RGB + click.
    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.scatter([u], [v], c="lime", s=80, marker="x", linewidths=2)
    ax.set_title("Synthetic RGB + click")
    ax.axis("off")

    # Depth.
    ax = axes[0, 1]
    im = ax.imshow(depth_m, cmap="viridis")
    ax.scatter([u], [v], c="red", s=60, marker="x")
    ax.set_title("Depth (m)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.axis("off")

    # Crop-fraction curve.
    ax = axes[0, 2]
    ax.plot(radii * 100, np.asarray(fractions) * 100, lw=2)
    ax.axvline(30, color="r", linestyle="--", label="HUG radius = 30 cm")
    ax.set_xlabel("Crop radius (cm)")
    ax.set_ylabel("Points retained (%)")
    ax.set_title("Crop radius vs. point retention")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3D top-down: full cloud.
    ax = fig.add_subplot(2, 3, 4, projection="3d")
    step = max(1, xyz_full.shape[0] // 4000)
    ax.scatter(
        xyz_full[::step, 0] * 100,
        xyz_full[::step, 1] * 100,
        xyz_full[::step, 2] * 100,
        c=rgb_full[::step] / 255.0,
        s=1,
        alpha=0.5,
    )
    ax.scatter(
        [query_xyz[0] * 100],
        [query_xyz[1] * 100],
        [query_xyz[2] * 100],
        c="lime",
        s=60,
        marker="x",
    )
    ax.set_title("Full point cloud")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.view_init(elev=20, azim=-70)

    # 3D top-down: cropped cloud.
    ax = fig.add_subplot(2, 3, 5, projection="3d")
    step = max(1, xyz_crop.shape[0] // 4000)
    ax.scatter(
        xyz_crop[::step, 0] * 100,
        xyz_crop[::step, 1] * 100,
        xyz_crop[::step, 2] * 100,
        c=rgb_crop[::step] / 255.0,
        s=2,
    )
    # Draw crop sphere wireframe.
    u_sph = np.linspace(0, 2 * np.pi, 30)
    v_sph = np.linspace(0, np.pi, 15)
    r = 30.0  # cm
    x_sph = query_xyz[0] * 100 + r * np.outer(np.cos(u_sph), np.sin(v_sph))
    y_sph = query_xyz[1] * 100 + r * np.outer(np.sin(u_sph), np.sin(v_sph))
    z_sph = query_xyz[2] * 100 + r * np.outer(np.ones(np.size(u_sph)), np.cos(v_sph))
    ax.plot_wireframe(x_sph, y_sph, z_sph, color="lime", alpha=0.2, rstride=5, cstride=5)
    ax.scatter(
        [query_xyz[0] * 100],
        [query_xyz[1] * 100],
        [query_xyz[2] * 100],
        c="lime",
        s=80,
        marker="x",
    )
    ax.set_title("After 0.30 m crop")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.view_init(elev=20, azim=-70)

    # 2D bird's eye view with crop circle.
    ax = axes[1, 2]
    ax.scatter(
        xyz_full[::step, 0] * 100,
        xyz_full[::step, 1] * 100,
        c=rgb_full[::step] / 255.0,
        s=1,
        alpha=0.3,
        label="full cloud",
    )
    ax.scatter(
        xyz_crop[::step, 0] * 100,
        xyz_crop[::step, 1] * 100,
        c=rgb_crop[::step] / 255.0,
        s=3,
        label="cropped cloud",
    )
    ax.scatter(
        [query_xyz[0] * 100],
        [query_xyz[1] * 100],
        c="lime",
        s=80,
        marker="x",
        label="query point",
    )
    ax.add_patch(
        Circle(
            (query_xyz[0] * 100, query_xyz[1] * 100),
            30.0,
            fill=False,
            edgecolor="lime",
            linewidth=2,
            linestyle="--",
        )
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("Bird's-eye view")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_dir = Path("runs/2026-06-17-aceego-actworld-motionvla/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hug_pcl_crop_probe.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization: {out_path.resolve()}")


if __name__ == "__main__":
    main()
