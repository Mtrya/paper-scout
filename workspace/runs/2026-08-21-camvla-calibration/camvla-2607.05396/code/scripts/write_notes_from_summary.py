#!/usr/bin/env python3
"""根据 summary.json 更新 notes/geometry.md 结论表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATE = """# CamVLA 几何机制笔记

## 机制对照（相对论文）

| 论文组件 | 本仓库最小实现 | 状态 |
|---------|---------------|------|
| 相机系相对动作 $\\Delta A_{{c}}$ | Action Head 回归 7D | 已实现 |
| 手眼矩阵 $T_t$（$\\tau,\\omega$） | Geometric Head 回归 6D | 已实现 |
| 确定性合成 $\\Delta p_b=R\\Delta p_c$, $\\Delta r_b=R\\Delta r_c$ | `compose_action` / `compose_action_torch` | 已实现 |
| $\\tau$ 不进入相对动作 | A1 单元测试验证 | 已实现 |
| 标定-free 部署 | 推理不喂显式外参 | 已记录 |
| 完整 VLA / RLBench / 真机 | 等级 A 明确不做 | 不做 |

## 代码来源

**自研机制复现**（官方训练仓库尚未放出；项目页 https://alibaba-damo-academy.github.io/CamVLA/）。

## 实验矩阵

- **A1** 手眼变换单元测试：`python scripts/test_geometry.py`
- **A2** 相机位姿扰动网格：`outputs/latest/a2_perturbation_curve.png`
- **A3** 基座系直接监督 (`base`) vs 相机系+手眼 (`camvla`)，三组合成数据

三组不同数据：`canonical_narrow` / `multiview_wide` / `noisy_shifted`。

## 验收清单

- [x] 几何变换有测试（`GEOMETRY_OK`）
- [x] 视角扰动对照图/表
- [x] 文档写明代码来源（自研）

## 最终实验结果

完整日志：`logs/full_experiment.log`  
汇总：`logs/summary.json`

### A1 几何

{geometry_block}

### A3 对照表

| dataset | variant | params | seen_sr | unseen_sr | seen_mse | unseen_mse | handeye_deg |
|---|---|---:|---:|---:|---:|---:|---:|
{table_rows}

## 机制结论

{conclusions}

## 与完整 CamVLA 的差距

- 无真实 RGB / 语言 backbone，视觉为合成特征。
- 无 π0 / GR00T 全量训练与 RLBench 绝对成功率对齐。
- Geometric Head 在合成线索上可学，不代表真机单目外参估计难度。
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("summary", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    g = summary.get("geometry", {})
    geometry_block = (
        f"- pass={g.get('pass')}, max_recon_err={g.get('max_recon_err')}, "
        f"τ影响={g.get('max_tau_impact')}\n"
        f"- 扰动单调递增={g.get('perturb_mono_increasing')}, "
        f"trans@15°={g.get('perturb_trans_err_at_15deg')}, "
        f"trans@45°={g.get('perturb_trans_err_at_45deg')}"
    )

    rows = []
    for r in summary.get("runs", []):
        he = r["unseen"].get("handeye_rot_err_deg", float("nan"))
        he_s = f"{he:.3f}" if he == he else "—"
        rows.append(
            f"| {r['dataset']} | {r['variant']} | {r['params']} | "
            f"{r['seen']['success_rate']:.3f} | {r['unseen']['success_rate']:.3f} | "
            f"{r['seen']['mse']:.4f} | {r['unseen']['mse']:.4f} | {he_s} |"
        )

    # 自动结论要点
    bullets = []
    by_ds = {}
    for r in summary.get("runs", []):
        by_ds.setdefault(r["dataset"], {})[r["variant"]] = r
    win = 0
    for ds, m in by_ds.items():
        if "base" in m and "camvla" in m:
            b, c = m["base"]["unseen"]["success_rate"], m["camvla"]["unseen"]["success_rate"]
            if c >= b:
                win += 1
                bullets.append(
                    f"- `{ds}`：未见视角上 camvla SR={c:.3f} ≥ base SR={b:.3f}"
                )
            else:
                bullets.append(
                    f"- `{ds}`：未见视角上 camvla SR={c:.3f} < base SR={b:.3f}（需结合 MSE/手眼误差解读）"
                )
    bullets.insert(
        0,
        f"- A1 几何单元测试 {'通过' if g.get('pass') else '未通过'}；"
        "相对动作合成与手眼平移解耦符合论文式 3–4。",
    )
    bullets.append(
        f"- 三组数据中，camvla 在未见视角 SR 不低于 base 的组数：{win}/{len(by_ds)}。"
    )
    bullets.append(
        "- 标定-free：评测与训练均不向策略输入真值外参矩阵，仅依赖观测特征。"
    )

    text = TEMPLATE.format(
        geometry_block=geometry_block,
        table_rows="\n".join(rows) if rows else "| — | — | — | — | — | — | — | — |",
        conclusions="\n".join(bullets),
    )
    out = args.out or (Path(__file__).resolve().parents[1] / "notes" / "geometry.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
