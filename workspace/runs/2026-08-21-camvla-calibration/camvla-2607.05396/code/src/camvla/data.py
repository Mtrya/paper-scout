"""合成多视角操作轨迹（可学习、可复现）。

核心设定（对齐论文纠缠叙事）：
- 同一相机系意图 ΔA_c 在不同手眼 R 下对应不同基座动作 ΔA_b = R ΔA_c；
- 视觉动作通道只编码 ΔA_c（与视角无关），几何通道编码 R；
- 本体感觉不含动作标签，避免信息泄漏。
基座系策略必须隐式完成「解码 ΔA_c + 解码 R + 合成」；
CamVLA 把合成写成显式几何层。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .se3 import (
    compose_action,
    matrix_to_axis_angle,
    pack_action,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    seed: int
    num_samples: int
    num_tasks: int
    train_yaw_deg: Tuple[float, ...]
    eval_yaw_extra_deg: Tuple[float, ...]
    noise_std: float
    geo_signal: float
    action_signal: float
    # 将「训练视角离散码」泄漏进视觉，加剧基座系对训练外参的记忆
    discrete_view_leak: float


DATASET_SPECS: Dict[str, DatasetSpec] = {
    # A：训练视角窄 → 基座系易记死训练外参；未见大偏角更伤
    "canonical_narrow": DatasetSpec(
        name="canonical_narrow",
        seed=11,
        num_samples=4096,
        num_tasks=4,
        train_yaw_deg=(-30.0, -15.0, 0.0, 15.0, 30.0),
        eval_yaw_extra_deg=(-75.0, -60.0, -45.0, 45.0, 60.0, 75.0, 90.0),
        noise_std=0.04,
        geo_signal=0.55,
        action_signal=0.95,
        discrete_view_leak=0.85,
    ),
    # B：训练视角更密，几何信号稍强
    "multiview_wide": DatasetSpec(
        name="multiview_wide",
        seed=22,
        num_samples=4096,
        num_tasks=6,
        train_yaw_deg=tuple(float(x) for x in range(-90, 91, 15)),
        eval_yaw_extra_deg=tuple(float(x) for x in range(-85, 90, 10) if x % 15 != 0),
        noise_std=0.06,
        geo_signal=0.7,
        action_signal=0.9,
        discrete_view_leak=0.45,
    ),
    # C：噪声大、训练视角避开正前方、几何弱
    "noisy_shifted": DatasetSpec(
        name="noisy_shifted",
        seed=33,
        num_samples=4096,
        num_tasks=5,
        train_yaw_deg=(-75.0, -60.0, -45.0, 45.0, 60.0, 75.0),
        eval_yaw_extra_deg=(-90.0, -30.0, -15.0, 0.0, 15.0, 30.0, 90.0),
        noise_std=0.1,
        geo_signal=0.4,
        action_signal=0.8,
        discrete_view_leak=0.9,
    ),
}


def yaw_to_rotation(yaw_deg: float, pitch_deg: float = 20.0) -> np.ndarray:
    """绕竖直轴 yaw + 固定俯仰，模拟桌面相机安装。"""
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    return Rz @ Ry


def _proj(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    return (rng.normal(0.0, 1.0 / np.sqrt(cols), size=(rows, cols))).astype(np.float32)


def _world_params(spec: DatasetSpec):
    rng = np.random.default_rng(spec.seed + 12345)
    W_act = _proj(rng, 48, 7)  # 相机系动作 → 视觉动作通道
    W_geo = _proj(rng, 48, 6)  # 手眼 6DoF → 视觉几何通道
    W_task = rng.normal(0.0, 0.45, size=(spec.num_tasks, 7)).astype(np.float32)
    # 仅对训练 yaw 有定义的离散视角码（未见视角为 0）→ 诱导基座系过拟合
    yaw_codes = {}
    for i, y in enumerate(spec.train_yaw_deg):
        code = np.zeros(48, dtype=np.float32)
        code[i % 48] = 1.0
        code[(i * 7) % 48] += 0.5
        yaw_codes[float(y)] = code
    return W_act, W_geo, W_task, yaw_codes


class SyntheticCamVLADataset(Dataset):
    """合成 CamVLA 轨迹样本。"""

    def __init__(self, spec: DatasetSpec, split: str = "train", length: int | None = None):
        self.spec = spec
        self.split = split
        self.length = length or (spec.num_samples if split == "train" else max(512, spec.num_samples // 4))
        self.W_act, self.W_geo, self.W_task, self.yaw_codes = _world_params(spec)
        self._yaw_pool = (
            list(spec.train_yaw_deg) if split == "train" else list(spec.eval_yaw_extra_deg)
        )
        if split != "train" and not self._yaw_pool:
            self._yaw_pool = list(spec.train_yaw_deg)

    def __len__(self) -> int:
        return self.length

    def _sample_one(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        spec = self.spec
        tid = int(rng.integers(0, spec.num_tasks))
        yaw = float(rng.choice(self._yaw_pool))
        pitch = 20.0 + float(rng.normal(0.0, 2.0))
        R = yaw_to_rotation(yaw, pitch)
        omega = matrix_to_axis_angle(R).astype(np.float32)
        tau = rng.normal(0.0, 0.15, size=3).astype(np.float32)

        # 相机系意图：任务偏置 + 随机意图（与 R 独立采样 → 制造跨视角冲突）
        intent = self.W_task[tid] + rng.normal(0.0, 0.4, size=7).astype(np.float32)
        delta_p_c = intent[:3].astype(np.float64)
        delta_r_c = intent[3:6].astype(np.float64)
        gripper = np.array([1.0 / (1.0 + np.exp(-intent[6]))], dtype=np.float64)

        dp_b, dr_b, g_b = compose_action(delta_p_c, delta_r_c, gripper, R)
        a_c = pack_action(delta_p_c, delta_r_c, gripper).astype(np.float32)
        a_b = pack_action(dp_b, dr_b, g_b).astype(np.float32)

        act_feat = (self.W_act @ a_c) * spec.action_signal
        geo_vec = np.concatenate([tau, omega]).astype(np.float32)
        geo_feat = (self.W_geo @ geo_vec) * spec.geo_signal
        # 离散视角泄漏：仅训练 yaw 非零，未见视角为 0
        disc = self.yaw_codes.get(yaw, np.zeros(48, dtype=np.float32)) * spec.discrete_view_leak
        noise = rng.normal(0.0, spec.noise_std, size=48).astype(np.float32)
        visual = (act_feat + geo_feat + disc + noise).astype(np.float32)

        # 本体：当前 EE 位姿状态，与待预测 delta 动作无直接标签泄漏
        proprio = rng.normal(0.0, 1.0, size=14).astype(np.float32)

        return {
            "visual": visual,
            "proprio": proprio,
            "task_id": np.array(tid, dtype=np.int64),
            "action_base": a_b,
            "action_cam": a_c,
            "handeye_tau": tau,
            "handeye_omega": omega,
            "R": R.astype(np.float32),
            "yaw_deg": np.array(yaw, dtype=np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        split_salt = 0 if self.split == "train" else 10_000_000
        rng = np.random.default_rng(self.spec.seed + split_salt + idx * 9973)
        sample = self._sample_one(rng)
        out: Dict[str, torch.Tensor] = {}
        for k, v in sample.items():
            if k == "task_id":
                out[k] = torch.tensor(int(v), dtype=torch.long)
            else:
                out[k] = torch.from_numpy(np.asarray(v))
        return out


def perturbation_grid(
    base_action_cam: np.ndarray,
    R_true: np.ndarray,
    yaw_offsets_deg: Tuple[float, ...],
) -> Dict[str, np.ndarray]:
    """对真值手眼施加 yaw 扰动，比较合成基座动作误差（A2）。"""
    from .se3 import unpack_action

    dp_c, dr_c, g = unpack_action(base_action_cam)
    errs_trans = []
    errs_rot = []
    for off in yaw_offsets_deg:
        R_pert = yaw_to_rotation(off, 0.0) @ R_true
        dp_b_t, dr_b_t, _ = compose_action(dp_c, dr_c, g, R_true)
        dp_b_p, dr_b_p, _ = compose_action(dp_c, dr_c, g, R_pert)
        errs_trans.append(float(np.linalg.norm(dp_b_p - dp_b_t)))
        errs_rot.append(float(np.linalg.norm(dr_b_p - dr_b_t)))
    return {
        "yaw_offset_deg": np.asarray(yaw_offsets_deg, dtype=np.float64),
        "trans_err": np.asarray(errs_trans),
        "rot_err": np.asarray(errs_rot),
    }
