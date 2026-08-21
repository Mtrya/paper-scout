#!/usr/bin/env python3
"""运行 CamVLA 机制复现矩阵（三组数据 × base/camvla）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from camvla.data import DATASET_SPECS  # noqa: E402
from camvla.experiment import TrainConfig, run_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CamVLA 相机中心动作机制复现")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_SPECS.keys()),
        choices=list(DATASET_SPECS.keys()),
        help="要跑的数据集（默认三组不同数据）",
    )
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out) if args.out else ROOT / "outputs" / "latest"
    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    result = run_matrix(datasets=args.datasets, out_root=out, train_cfg=cfg)
    print(json.dumps({k: result[k] for k in ("geometry",) if k in result}, ensure_ascii=False, indent=2))
    print(f"LOG={result['log']}")
    print(f"SUMMARY={Path(result['out_root']) / 'summary.json'}")


if __name__ == "__main__":
    main()
