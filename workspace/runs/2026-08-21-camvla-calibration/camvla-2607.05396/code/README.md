# camvla-camera-robust

CamVLA：相机中心动作 + 手眼矩阵，降低标定依赖 — **等级 A（机制最小）复现**。

- **复现方案：** [复现方案.md](./复现方案.md)
- **几何机制笔记：** [notes/geometry.md](./notes/geometry.md)
- **完整实验日志：** [logs/full_experiment.log](./logs/full_experiment.log)
- **结果汇总：** [logs/summary.json](./logs/summary.json)
- **扰动曲线：** [notes/a2_perturbation_curve.png](./notes/a2_perturbation_curve.png)
- **论文：** https://arxiv.org/abs/2607.05396v1
- **项目页：** https://alibaba-damo-academy.github.io/CamVLA/
- **官方代码：** 未放出（本仓库为自研机制复现）

## 目标口径

验证「相机中心动作 + 手眼旋转合成 → 机器人系动作」在视角扰动下优于直接预测机器人系动作。

| ID | 设置 | 实现 |
|----|------|------|
| A1 | 手眼变换单元测试 | `scripts/test_geometry.py` → `GEOMETRY_OK` |
| A2 | 相机位姿扰动网格 | `notes/a2_perturbation_curve.png` |
| A3 | 基座系直接监督 vs CamVLA | `base` / `camvla` × 三组数据 |

三组不同合成数据：`canonical_narrow` / `multiview_wide` / `noisy_shifted`。

## 快速复现

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\test_geometry.py
.\.venv\Scripts\python.exe scripts\run_experiment.py --out outputs\latest --steps 1500 --device cpu
.\.venv\Scripts\python.exe scripts\write_notes_from_summary.py logs\summary.json
```

## 主要结论（本机一次完整跑）

- A1：SE(3) 合成可逆，手眼平移 τ 不进入相对动作（符合论文式 3–4）。
- A2：yaw 扰动增大时，合成基座动作误差单调上升。
- A3：`multiview_wide` 未见视角 SR：camvla **0.946** > base **0.889**；`noisy_shifted`：**0.117** > **0.013**。
- 标定-free：推理不喂显式外参矩阵。详见 `notes/geometry.md`。

## 明确不做

完整 CamVLA 大规模训练、RLBench/真机无标定部署、官方 checkpoint 对齐。
