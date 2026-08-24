"""SE(3) 与 CamVLA 确定性几何合成（论文式 3–4）。

相机系相对动作在手眼旋转 R_t 下线性变换到机器人基座系：
  Δp_b = R_t Δp_c
  Δr_b = R_t Δr_c
夹爪 g 与手眼平移 τ 不参与相对动作合成。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def axis_angle_to_matrix(omega: np.ndarray) -> np.ndarray:
    """轴角向量 (..., 3) → 旋转矩阵 (..., 3, 3)。"""
    omega = np.asarray(omega, dtype=np.float64)
    flat = omega.reshape(-1, 3)
    mats = Rotation.from_rotvec(flat).as_matrix()
    return mats.reshape(*omega.shape[:-1], 3, 3).astype(np.float64)


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 (..., 3, 3) → 轴角 (..., 3)。"""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    aa = Rotation.from_matrix(flat).as_rotvec()
    return aa.reshape(*R.shape[:-2], 3).astype(np.float64)


def hand_eye_rotation(omega: np.ndarray) -> np.ndarray:
    """由 Geometric Head 的轴角输出得到 R_t ∈ SO(3)。"""
    return axis_angle_to_matrix(omega)


def compose_action(
    delta_p_c: np.ndarray,
    delta_r_c: np.ndarray,
    gripper: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """相机系相对动作 → 基座系相对动作（论文式 3–4）。

    Parameters
    ----------
    delta_p_c, delta_r_c : (..., 3)
        相机系平移 / 轴角旋转增量。
    gripper : (...,) 或 (..., 1)
        夹爪开合，原样传递。
    R : (..., 3, 3)
        手眼旋转 R_t。

    Returns
    -------
    delta_p_b, delta_r_b, gripper
    """
    delta_p_c = np.asarray(delta_p_c, dtype=np.float64)
    delta_r_c = np.asarray(delta_r_c, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    delta_p_b = np.einsum("...ij,...j->...i", R, delta_p_c)
    delta_r_b = np.einsum("...ij,...j->...i", R, delta_r_c)
    return delta_p_b, delta_r_b, gripper


def decompose_action(
    delta_p_b: np.ndarray,
    delta_r_b: np.ndarray,
    gripper: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基座系相对动作 → 相机系（R^T 逆变换）。"""
    R = np.asarray(R, dtype=np.float64)
    R_t = np.swapaxes(R, -1, -2)
    return compose_action(delta_p_b, delta_r_b, gripper, R_t)


def pack_action(delta_p: np.ndarray, delta_r: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """打包为 7 维动作 [Δp(3), Δr(3), g(1)]。"""
    delta_p = np.asarray(delta_p, dtype=np.float64)
    delta_r = np.asarray(delta_r, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64)
    if gripper.ndim == delta_p.ndim - 1:
        gripper = gripper[..., None]
    return np.concatenate([delta_p, delta_r, gripper], axis=-1)


def unpack_action(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """拆分 7 维动作。"""
    action = np.asarray(action, dtype=np.float64)
    return action[..., :3], action[..., 3:6], action[..., 6]


def rotation_angle_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> np.ndarray:
    """测地距离角度误差（度）。"""
    R_pred = np.asarray(R_pred, dtype=np.float64)
    R_gt = np.asarray(R_gt, dtype=np.float64)
    R_err = np.einsum("...ij,...kj->...ik", R_pred, R_gt)
    # trace(R_err) = 1 + 2 cos θ
    tr = np.einsum("...ii->...", R_err)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def random_rotation(rng: np.random.Generator, shape: Tuple[int, ...] = ()) -> np.ndarray:
    """均匀随机 SO(3)。"""
    n = int(np.prod(shape)) if shape else 1
    quats = rng.normal(size=(max(n, 1), 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    mats = Rotation.from_quat(quats).as_matrix()
    if not shape:
        return mats[0]
    return mats.reshape(*shape, 3, 3)
