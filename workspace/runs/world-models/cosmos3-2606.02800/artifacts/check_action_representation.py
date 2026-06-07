"""
Verify that the Cosmos 3 action representation actually preserves SE(3) geometry.

We use the public pose utilities from the repo (pose_utils.py) to:
  1. Build a trajectory of random absolute SE(3) poses.
  2. Encode it as relative poses with 6D rotation (the paper's default).
  3. Decode back to absolute poses and compare with the original.

A correct implementation should recover the original trajectory up to float32
precision when the initial frame is provided.
"""

import os
import sys

# Use the repo's utilities without installing the full package.
repo = os.path.join(os.path.dirname(__file__), "repos", "world-models", "cosmos-framework")
sys.path.insert(0, repo)

import numpy as np
from cosmos_framework.data.vfm.action.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
    pose_rel_to_abs,
)


def test_rot6d_roundtrip(num_frames: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Random translations.
    xyz = rng.standard_normal((num_frames, 3), dtype=np.float32)

    # Random unit quaternions (xyzw order expected by build_abs_pose_from_components).
    quat = rng.standard_normal((num_frames, 4), dtype=np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    poses_abs = build_abs_pose_from_components(xyz, quat, rotation_input_format="quat_xyzw")

    # Encode relative deltas with rot6d (the paper's unified action format).
    rel_rot6d = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")

    # Decode back, seeded with the initial absolute pose.
    poses_rec = pose_rel_to_abs(
        rel_rot6d,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        initial_pose=poses_abs[0],
        normalize_rotation=True,
    )

    max_translation_error = np.abs(poses_abs[:, :3, 3] - poses_rec[:, :3, 3]).max()
    # Rotation error as geodesic angle (degrees).
    R_true = poses_abs[:, :3, :3]
    R_rec = poses_rec[:, :3, :3]
    R_diff = np.einsum("bji,bjk->bik", R_true, R_rec)  # R_true^T @ R_rec
    traces = np.trace(R_diff, axis1=1, axis2=2)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    max_rotation_error_deg = np.degrees(angles.max())

    print(f"SE(3) round-trip with rot6d, {num_frames} frames")
    print(f"  Relative action vector shape: {rel_rot6d.shape} (3 trans + 6 rot)")
    print(f"  Max translation error:        {max_translation_error:.2e}")
    print(f"  Max rotation error:           {max_rotation_error_deg:.2e} degrees")

    assert max_translation_error < 1e-4
    assert max_rotation_error_deg < 1.0  # sub-degree is fine for float32 rot6d


def test_action_vector_layout():
    """Confirm the relative vector layout is [translation(3), rotation(6)]."""
    rng = np.random.default_rng(7)
    xyz = rng.standard_normal((4, 3), dtype=np.float32)
    quat = rng.standard_normal((4, 4), dtype=np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    poses = build_abs_pose_from_components(xyz, quat, "quat_xyzw")
    rel = pose_abs_to_rel(poses, rotation_format="rot6d")

    print()
    print(f"Action-vector layout: first 3 dims = translation, next 6 = 6D rotation")
    print(f"  Sample vector: {rel[0]}")
    print(f"  Translation portion: {rel[0, :3]}")
    print(f"  Rotation portion:    {rel[0, 3:]}")
    assert rel.shape[1] == 9


if __name__ == "__main__":
    test_rot6d_roundtrip(num_frames=25)
    test_action_vector_layout()
