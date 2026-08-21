# CamVLA 几何机制笔记

## 机制对照（相对论文）

| 论文组件 | 本仓库最小实现 | 状态 |
|---------|---------------|------|
| 相机系相对动作 $\Delta A_{c}$ | Action Head 回归 7D | 已实现 |
| 手眼矩阵 $T_t$（$\tau,\omega$） | Geometric Head 回归 6D | 已实现 |
| 确定性合成 $\Delta p_b=R\Delta p_c$, $\Delta r_b=R\Delta r_c$ | `compose_action` / `compose_action_torch` | 已实现 |
| $\tau$ 不进入相对动作 | A1 单元测试验证 | 已实现 |
| 标定-free 部署 | 推理不喂显式外参 | 已记录 |
| 完整 VLA / RLBench / 真机 | 等级 A 明确不做 | 不做 |

## 代码来源

**自研机制复现**（官方训练仓库尚未放出；项目页 https://alibaba-damo-academy.github.io/CamVLA/）。

## 实验矩阵

- **A1** 手眼变换单元测试：`python scripts/test_geometry.py`
- **A2** 相机位姿扰动网格：`notes/a2_perturbation_curve.png`
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

- pass=1.0, max_recon_err=1.9845870143629155e-15, τ影响=0.0
- 扰动单调递增=1.0, trans@15°=0.02899515862677869, trans@45°=0.08500950373668632

### A3 对照表

| dataset | variant | params | seen_sr | unseen_sr | seen_mse | unseen_mse | handeye_deg |
|---|---|---:|---:|---:|---:|---:|---:|
| canonical_narrow | base | 19239 | 0.998 | 0.006 | 0.0008 | 0.1181 | — |
| canonical_narrow | camvla | 38445 | 1.000 | 0.007 | 0.0007 | 0.0999 | 35.431 |
| multiview_wide | base | 19303 | 0.982 | 0.889 | 0.0017 | 0.0029 | — |
| multiview_wide | camvla | 38509 | 0.991 | 0.946 | 0.0015 | 0.0022 | 2.817 |
| noisy_shifted | base | 19271 | 0.694 | 0.013 | 0.0039 | 0.0294 | — |
| noisy_shifted | camvla | 38477 | 0.716 | 0.117 | 0.0035 | 0.0288 | 17.786 |

## 机制结论

- A1 几何单元测试 通过；相对动作合成与手眼平移解耦符合论文式 3–4。
- `canonical_narrow`：未见视角上 camvla SR=0.007 ≥ base SR=0.006
- `multiview_wide`：未见视角上 camvla SR=0.946 ≥ base SR=0.889
- `noisy_shifted`：未见视角上 camvla SR=0.117 ≥ base SR=0.013
- 三组数据中，camvla 在未见视角 SR 不低于 base 的组数：3/3。
- 标定-free：评测与训练均不向策略输入真值外参矩阵，仅依赖观测特征。

## 与完整 CamVLA 的差距

- 无真实 RGB / 语言 backbone，视觉为合成特征。
- 无 π0 / GR00T 全量训练与 RLBench 绝对成功率对齐。
- Geometric Head 在合成线索上可学，不代表真机单目外参估计难度。
