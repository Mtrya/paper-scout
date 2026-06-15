# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pose-format helpers used by the HDF5 dataset and offline norm-stats scripts.

Conventions:
* Quaternions are always (x, y, z, w).
* Dual-arm 16-d EE state layout: ``[left_xyz(3), left_quat(4), left_gripper(1),
  right_xyz(3), right_quat(4), right_gripper(1)]``.
* Dual-arm 20-d state-with-rotmat layout: ``[left_xyz(3), left_rot6d(6),
  left_gripper(1), right_xyz(3), right_rot6d(6), right_gripper(1)]``, where
  ``rot6d`` is the first two rows of the rotation matrix flattened
  ``[r00, r01, r02, r10, r11, r12]``.
* Dual-arm relative-action 20-d layout used by Hy-VLA's RT-relative target:
  ``[left_dxyz(3), left_relRot6d(6), left_gripper(1), right_dxyz(3),
  right_relRot6d(6), right_gripper(1)]``, with deltas computed against
  the chunk's t=0 frame in each arm's wrist frame.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def convert_PosQuat2PosRotationMatrix_batch(pos_quat_gripper, quat_order="xyzw"):
    """Batched 16-d (PosQuat) -> 20-d (PosRotMat6d) dual-arm conversion.

    Input shape: (N, 16) = ``[left_xyz(3) + left_quat(4) + left_gripper(1)
    + right_xyz(3) + right_quat(4) + right_gripper(1)]``.
    Output shape: (N, 20) = ``[left_xyz(3) + left_rot6d(6) + left_gripper(1)
    + right_xyz(3) + right_rot6d(6) + right_gripper(1)]``.
    """
    assert quat_order == "xyzw"
    N = pos_quat_gripper.shape[0]
    output = np.zeros((N, 20), dtype=pos_quat_gripper.dtype)

    # Left arm
    left_pos = pos_quat_gripper[:, :3]
    left_quat = pos_quat_gripper[:, 3:7]
    left_gripper = pos_quat_gripper[:, 7:8]
    left_matrix = R.from_quat(left_quat).as_matrix()
    output[:, 0:3] = left_pos
    output[:, 3:6] = left_matrix[:, 0, :]
    output[:, 6:9] = left_matrix[:, 1, :]
    output[:, 9:10] = left_gripper

    # Right arm
    right_pos = pos_quat_gripper[:, 8:11]
    right_quat = pos_quat_gripper[:, 11:15]
    right_gripper = pos_quat_gripper[:, 15:16]
    right_matrix = R.from_quat(right_quat).as_matrix()
    output[:, 10:13] = right_pos
    output[:, 13:16] = right_matrix[:, 0, :]
    output[:, 16:19] = right_matrix[:, 1, :]
    output[:, 19:20] = right_gripper

    return output


def poses_to_relative_matrices(pose_sequence):
    """Single-arm chunk -> per-frame RT-relative (xyz, rot6d).

    Input  ``pose_sequence``: (N, 7) = ``[xyz(3) + quat_xyzw(4)]``.
    Output: (N, 9) = ``[delta_xyz(3) + relRot6d(6)]``, with the delta
    expressed in the t=0 wrist frame: ``T_rel = T0^{-1} @ Ti``.
    """
    positions = pose_sequence[:, :3]
    quats = pose_sequence[:, 3:]
    R_all = R.from_quat(quats).as_matrix()

    # Inverse of T0 (t=0 wrist frame).
    R0_T = R_all[0].T
    pos0 = positions[0]
    T0_inv = np.eye(4)
    T0_inv[:3, :3] = R0_T
    T0_inv[:3, 3] = -R0_T @ pos0

    # All Ti as a (N, 4, 4) tensor.
    N = pose_sequence.shape[0]
    Ti_all = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    Ti_all[:, :3, :3] = R_all
    Ti_all[:, :3, 3] = positions

    # delta_T = T0_inv @ Ti, broadcast along batch.
    delta_T_all = T0_inv @ Ti_all

    rotation_6d = delta_T_all[:, :2, :3].reshape(N, -1)
    translation_3d = delta_T_all[:, :3, 3]
    return np.concatenate([translation_3d, rotation_6d], axis=1)


def dual_arm_poses_to_relative(dual_pose_sequence):
    """Dual-arm 16-d PosQuat chunk -> 20-d RT-relative chunk.

    Input  ``dual_pose_sequence``: (N, 16), see module docstring.
    Output: (N, 20) RT-relative chunk, see module docstring.
    """
    num_poses = dual_pose_sequence.shape[0]

    pose_seq_arm1 = dual_pose_sequence[:, 0:7]
    gripper_seq_arm1 = dual_pose_sequence[:, 7]
    pose_seq_arm2 = dual_pose_sequence[:, 8:15]
    gripper_seq_arm2 = dual_pose_sequence[:, 15]

    delta_arm1 = poses_to_relative_matrices(pose_seq_arm1).reshape(num_poses, 9)
    delta_arm2 = poses_to_relative_matrices(pose_seq_arm2).reshape(num_poses, 9)
    gripper_arm1 = gripper_seq_arm1.reshape(-1, 1)
    gripper_arm2 = gripper_seq_arm2.reshape(-1, 1)

    return np.concatenate(
        [delta_arm1, gripper_arm1, delta_arm2, gripper_arm2], axis=1
    )
