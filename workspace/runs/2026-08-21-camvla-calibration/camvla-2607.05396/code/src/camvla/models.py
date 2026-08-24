"""轻量策略头：基座系直接回归 vs CamVLA（Action Head + Geometric Head）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def axis_angle_to_matrix_torch(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues：轴角 (B,3) → 旋转矩阵 (B,3,3)。"""
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp(min=1e-8)
    k = omega / theta
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zeros = torch.zeros_like(kx)
    K = torch.stack(
        [
            torch.stack([zeros, -kz, ky], dim=-1),
            torch.stack([kz, zeros, -kx], dim=-1),
            torch.stack([-ky, kx, zeros], dim=-1),
        ],
        dim=-2,
    )
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0).expand(omega.shape[0], -1, -1)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    return I + sin_t * K + (1.0 - cos_t) * (K @ K)


def compose_action_torch(
    action_cam: torch.Tensor, R: torch.Tensor
) -> torch.Tensor:
    """Δp_b = R Δp_c, Δr_b = R Δr_c, g 不变。action: (B,7)。"""
    dp_c = action_cam[:, :3]
    dr_c = action_cam[:, 3:6]
    g = action_cam[:, 6:7]
    dp_b = torch.bmm(R, dp_c.unsqueeze(-1)).squeeze(-1)
    dr_b = torch.bmm(R, dr_c.unsqueeze(-1)).squeeze(-1)
    return torch.cat([dp_b, dr_b, g], dim=-1)


@dataclass
class ModelConfig:
    visual_dim: int = 48
    proprio_dim: int = 14
    num_tasks: int = 4
    hidden: int = 96
    action_dim: int = 7
    variant: str = "camvla"  # "base" | "camvla"


class TinyEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.task_emb = nn.Embedding(cfg.num_tasks, 32)
        in_dim = cfg.visual_dim + cfg.proprio_dim + 32
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
        )

    def forward(self, visual, proprio, task_id) -> torch.Tensor:
        t = self.task_emb(task_id)
        x = torch.cat([visual, proprio, t], dim=-1)
        return self.net(x)


class BaseFramePolicy(nn.Module):
    """直接预测基座系相对动作。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = TinyEncoder(cfg)
        self.head = nn.Linear(cfg.hidden, cfg.action_dim)

    def forward(self, visual, proprio, task_id) -> Tuple[torch.Tensor, Dict]:
        h = self.enc(visual, proprio, task_id)
        return self.head(h), {}


class CamVLAPolicy(nn.Module):
    """Action Head（相机系）+ Geometric Head（手眼）+ 确定性合成。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = TinyEncoder(cfg)
        self.action_head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.action_dim),
        )
        self.geo_head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, 6),  # tau(3) + omega(3)
        )

    def forward(self, visual, proprio, task_id) -> Tuple[torch.Tensor, Dict]:
        h = self.enc(visual, proprio, task_id)
        a_c = self.action_head(h)
        geo = self.geo_head(h)
        tau = geo[:, :3]
        omega = geo[:, 3:6]
        R = axis_angle_to_matrix_torch(omega)
        a_b = compose_action_torch(a_c, R)
        aux = {
            "action_cam": a_c,
            "tau": tau,
            "omega": omega,
            "R": R,
        }
        return a_b, aux


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.variant == "base":
        return BaseFramePolicy(cfg)
    if cfg.variant == "camvla":
        return CamVLAPolicy(cfg)
    raise ValueError(f"未知 variant: {cfg.variant}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def camvla_losses(
    pred_base: torch.Tensor,
    aux: Dict,
    batch: Dict[str, torch.Tensor],
    lambda_cam: float = 1.0,
    lambda_geo: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """基座动作监督 + 相机系动作 + 手眼几何辅助损失。"""
    gt_base = batch["action_base"]
    loss_base = F.mse_loss(pred_base, gt_base)
    stats = {"loss_base": float(loss_base.detach().cpu())}

    if not aux:
        return loss_base, stats

    loss_cam = F.mse_loss(aux["action_cam"], batch["action_cam"])
    loss_tau = F.mse_loss(aux["tau"], batch["handeye_tau"])
    loss_omega = F.mse_loss(aux["omega"], batch["handeye_omega"])
    loss_geo = loss_tau + loss_omega
    total = loss_base + lambda_cam * loss_cam + lambda_geo * loss_geo
    stats.update(
        {
            "loss_cam": float(loss_cam.detach().cpu()),
            "loss_geo": float(loss_geo.detach().cpu()),
            "loss_total": float(total.detach().cpu()),
        }
    )
    return total, stats
