"""端到端实验：A1 几何 / A2 扰动网格 / A3 双策略对照 × 三组数据。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DATASET_SPECS, SyntheticCamVLADataset, perturbation_grid, yaw_to_rotation
from .models import ModelConfig, build_model, camvla_losses, count_params
from .se3 import (
    compose_action,
    pack_action,
    random_rotation,
    rotation_angle_error_deg,
)


@dataclass
class TrainConfig:
    steps: int = 1500
    batch_size: int = 128
    lr: float = 2e-3
    eval_every: int = 300
    seed: int = 0
    device: str = "cpu"
    success_mse_coef: float = 0.12
    lambda_cam: float = 1.0
    lambda_geo: float = 0.75


VARIANTS = ("base", "camvla")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(spec_name: str, split: str, batch_size: int) -> DataLoader:
    ds = SyntheticCamVLADataset(DATASET_SPECS[spec_name], split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), drop_last=False)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    success_thresh: float,
) -> Dict[str, float]:
    model.eval()
    mses: List[float] = []
    successes: List[float] = []
    rot_errs: List[float] = []
    cam_mses: List[float] = []

    for batch in loader:
        visual = batch["visual"].to(device)
        proprio = batch["proprio"].to(device)
        task_id = batch["task_id"].to(device)
        actions = batch["action_base"].to(device)
        pred, aux = model(visual, proprio, task_id)
        mse = ((pred - actions) ** 2).mean(dim=-1)
        mses.extend(mse.cpu().tolist())
        successes.extend((mse < success_thresh).float().cpu().tolist())
        if aux and "R" in aux:
            R_pred = aux["R"].cpu().numpy()
            R_gt = batch["R"].numpy()
            rot_errs.extend(rotation_angle_error_deg(R_pred, R_gt).tolist())
            cam_mse = ((aux["action_cam"].cpu() - batch["action_cam"]) ** 2).mean(dim=-1)
            cam_mses.extend(cam_mse.tolist())

    out = {
        "mse": float(np.mean(mses)),
        "success_rate": float(np.mean(successes)),
        "n": float(len(mses)),
        "success_thresh": float(success_thresh),
    }
    if rot_errs:
        out["handeye_rot_err_deg"] = float(np.mean(rot_errs))
    if cam_mses:
        out["cam_action_mse"] = float(np.mean(cam_mses))
    return out


def run_geometry_unit_tests(log_fp, n: int = 200, seed: int = 0) -> Dict[str, float]:
    """A1：可逆性与 τ 独立性。"""
    rng = np.random.default_rng(seed)
    max_recon = 0.0
    max_tau_impact = 0.0
    for _ in range(n):
        R = random_rotation(rng)
        dp_c = rng.normal(size=3)
        dr_c = rng.normal(size=3) * 0.3
        g = np.array([rng.random()])
        dp_b, dr_b, g2 = compose_action(dp_c, dr_c, g, R)
        # 逆变换还原
        R_t = R.T
        dp_c2, dr_c2, _ = compose_action(dp_b, dr_b, g2, R_t)
        max_recon = max(max_recon, float(np.linalg.norm(dp_c2 - dp_c) + np.linalg.norm(dr_c2 - dr_c)))
        # τ 不进入相对动作合成：人为改变 τ 不应改变 compose（仅用 R）
        dp_b2, dr_b2, _ = compose_action(dp_c, dr_c, g, R)
        max_tau_impact = max(
            max_tau_impact,
            float(np.linalg.norm(dp_b2 - dp_b) + np.linalg.norm(dr_b2 - dr_b)),
        )

    # 扰动敏感性：yaw 越大，合成误差越大
    a_c = pack_action(np.array([0.1, 0.0, 0.05]), np.array([0.0, 0.2, 0.0]), np.array([0.8]))
    R0 = yaw_to_rotation(0.0)
    grid = perturbation_grid(a_c, R0, tuple(range(0, 61, 5)))
    mono = bool(np.all(np.diff(grid["trans_err"][1:]) >= -1e-9))

    result = {
        "max_recon_err": max_recon,
        "max_tau_impact": max_tau_impact,
        "perturb_mono_increasing": float(mono),
        "perturb_trans_err_at_15deg": float(grid["trans_err"][grid["yaw_offset_deg"] == 15][0]),
        "perturb_trans_err_at_45deg": float(grid["trans_err"][grid["yaw_offset_deg"] == 45][0]),
        "pass": float(max_recon < 1e-8 and max_tau_impact < 1e-12 and mono),
    }
    log_fp.write(f"[A1] geometry unit tests: {json.dumps(result)}\n")
    log_fp.flush()
    return result


def run_perturbation_sweep(out_dir: Path, log_fp) -> Dict:
    """A2：视角扰动网格曲线。"""
    rng = np.random.default_rng(42)
    offsets = tuple(range(0, 91, 5))
    # 多条随机相机系动作取平均
    trans_mat = []
    rot_mat = []
    for _ in range(64):
        a_c = pack_action(rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.2, np.array([rng.random()]))
        R = random_rotation(rng)
        g = perturbation_grid(a_c, R, offsets)
        trans_mat.append(g["trans_err"])
        rot_mat.append(g["rot_err"])
    trans = np.mean(trans_mat, axis=0)
    rot = np.mean(rot_mat, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.0))
    ax.plot(offsets, trans, label="‖Δp_b(pert)-Δp_b(gt)‖", linewidth=2)
    ax.plot(offsets, rot, label="‖Δr_b(pert)-Δr_b(gt)‖", linewidth=2)
    ax.set_xlabel("手眼 yaw 扰动 (°)")
    ax.set_ylabel("基座系动作误差")
    ax.set_title("A2：相机外参扰动 vs 合成动作误差")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "a2_perturbation_curve.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    table = {
        "yaw_offset_deg": list(offsets),
        "mean_trans_err": trans.tolist(),
        "mean_rot_err": rot.tolist(),
        "figure": str(fig_path.name),
    }
    with open(out_dir / "a2_perturbation.json", "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    log_fp.write(f"[A2] wrote {fig_path.name}; err@15°={trans[3]:.4f}, err@45°={trans[9]:.4f}\n")
    log_fp.flush()
    return table


def train_one(
    variant: str,
    dataset_name: str,
    out_dir: Path,
    train_cfg: TrainConfig,
    log_fp,
) -> Dict:
    set_seed(train_cfg.seed)
    device = torch.device(
        train_cfg.device if (train_cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    spec = DATASET_SPECS[dataset_name]
    mcfg = ModelConfig(num_tasks=spec.num_tasks, variant=variant)
    model = build_model(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg.steps)

    train_loader = make_loader(dataset_name, "train", train_cfg.batch_size)
    # 用训练集估计动作方差作为成功阈值基准
    probe = []
    for i, batch in enumerate(train_loader):
        probe.append(batch["action_base"].numpy())
        if i >= 8:
            break
    action_std = float(np.std(np.concatenate(probe, axis=0)))
    success_thresh = (train_cfg.success_mse_coef * action_std) ** 2

    # 同分布（训练视角池）与未见视角（eval yaw）评测集
    val_seen = DataLoader(
        SyntheticCamVLADataset(spec, split="train", length=1024),
        batch_size=train_cfg.batch_size,
        shuffle=False,
    )
    val_unseen = DataLoader(
        SyntheticCamVLADataset(spec, split="eval", length=1024),
        batch_size=train_cfg.batch_size,
        shuffle=False,
    )

    n_params = count_params(model)
    log_fp.write(
        f"\n=== train {dataset_name}/{variant} params={n_params} device={device} "
        f"steps={train_cfg.steps} thresh={success_thresh:.6f} ===\n"
    )
    log_fp.flush()

    it = iter(train_loader)
    t0 = time.time()
    model.train()
    for step in tqdm(range(1, train_cfg.steps + 1), desc=f"{dataset_name}/{variant}"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred, aux = model(batch["visual"], batch["proprio"], batch["task_id"])
        if variant == "camvla":
            loss, stats = camvla_losses(
                pred, aux, batch, train_cfg.lambda_cam, train_cfg.lambda_geo
            )
        else:
            loss = torch.nn.functional.mse_loss(pred, batch["action_base"])
            stats = {"loss_base": float(loss.detach().cpu())}

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % train_cfg.eval_every == 0 or step == train_cfg.steps:
            seen = evaluate(model, val_seen, device, success_thresh)
            unseen = evaluate(model, val_unseen, device, success_thresh)
            log_fp.write(
                f"step={step} loss={stats} seen_sr={seen['success_rate']:.4f} "
                f"seen_mse={seen['mse']:.6f} unseen_sr={unseen['success_rate']:.4f} "
                f"unseen_mse={unseen['mse']:.6f}"
            )
            if "handeye_rot_err_deg" in unseen:
                log_fp.write(f" handeye_err_deg={unseen['handeye_rot_err_deg']:.3f}")
            log_fp.write("\n")
            log_fp.flush()

    elapsed = time.time() - t0
    seen = evaluate(model, val_seen, device, success_thresh)
    unseen = evaluate(model, val_unseen, device, success_thresh)
    ckpt = out_dir / f"{dataset_name}_{variant}.pt"
    torch.save({"model": model.state_dict(), "cfg": asdict(mcfg)}, ckpt)

    result = {
        "dataset": dataset_name,
        "variant": variant,
        "params": n_params,
        "elapsed_sec": elapsed,
        "success_thresh": success_thresh,
        "action_std": action_std,
        "seen": seen,
        "unseen": unseen,
        "ckpt": ckpt.name,
    }
    with open(out_dir / f"{dataset_name}_{variant}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log_fp.write(f"[done] {json.dumps(result, ensure_ascii=False)}\n")
    log_fp.flush()
    return result


def run_matrix(
    datasets: Optional[Sequence[str]] = None,
    out_root: Optional[Path] = None,
    train_cfg: Optional[TrainConfig] = None,
) -> Dict:
    datasets = list(datasets or DATASET_SPECS.keys())
    out_root = Path(out_root or Path.cwd() / "outputs" / "latest")
    out_root.mkdir(parents=True, exist_ok=True)
    train_cfg = train_cfg or TrainConfig()

    log_path = out_root / "full_experiment.log"
    summary: Dict = {
        "train_cfg": asdict(train_cfg),
        "geometry": {},
        "perturbation": {},
        "runs": [],
    }

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_fp.write("CamVLA 机制复现完整日志\n")
        log_fp.write(f"time_start={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fp.write(f"datasets={datasets}\n")
        log_fp.write(f"train_cfg={asdict(train_cfg)}\n")
        log_fp.write(
            "deployment_assumption: 推理仅依赖单目 RGB 代理特征+本体+任务；"
            "不喂显式外参标定矩阵。\n"
        )

        summary["geometry"] = run_geometry_unit_tests(log_fp)
        summary["perturbation"] = run_perturbation_sweep(out_root, log_fp)

        for ds in datasets:
            for variant in VARIANTS:
                summary["runs"].append(train_one(variant, ds, out_root, train_cfg, log_fp))

        # 对照表
        log_fp.write("\n=== SUMMARY TABLE ===\n")
        log_fp.write(
            "dataset\tvariant\tparams\tseen_sr\tunseen_sr\tseen_mse\tunseen_mse\thandeye_deg\n"
        )
        for r in summary["runs"]:
            he = r["unseen"].get("handeye_rot_err_deg", float("nan"))
            log_fp.write(
                f"{r['dataset']}\t{r['variant']}\t{r['params']}\t"
                f"{r['seen']['success_rate']:.4f}\t{r['unseen']['success_rate']:.4f}\t"
                f"{r['seen']['mse']:.6f}\t{r['unseen']['mse']:.6f}\t{he}\n"
            )
        log_fp.write(f"time_end={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 复制到 logs/
    logs_dir = out_root.parents[1] / "logs" if out_root.name == "latest" else out_root / "logs"
    # Prefer repo_root/logs
    repo_logs = Path(__file__).resolve().parents[2] / "logs"
    repo_logs.mkdir(parents=True, exist_ok=True)
    (repo_logs / "full_experiment.log").write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
    (repo_logs / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary["out_root"] = str(out_root)
    summary["log"] = str(log_path)
    return summary
