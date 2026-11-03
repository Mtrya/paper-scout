# Probing Spatial State:发布物重分析

对论文公开的验证结果表做独立重算,不重跑模型。数据来自 `Probing_Spatial_State_Release/results/`:

- `canonical_v2_val_seed42.csv`:十个模型、六种探针、四个前缀的主表。
- `c3_from_c1_val_seed42.csv`:训练好的直接预测头在同轨迹三个端点上的动作敏感性评测。

## 重算内容

- 主表在指定层、四个前缀上取均值;缺 CogVideoX 第 20 层整行,所以是九模型。
- 跨模型 Spearman 相关与留一法:持久化—相机位姿 ρ=+0.80(n=9,p=0.010,LOO +0.71..+0.98);预测—可见场景六对 0.48–0.65,显著性不足;预测—持久化在九模型上为负,论文的 |ρ|<0.1 依赖被排除的 CogVideoX。
- 动作可归因份额 = (真实 R@1 − 静态基线)/真实 R@1;静态基线取去掉动作与打乱动作中较高者。
- 动作敏感性:DINOv2 边距 +0.029、敏感度 0.100;Qwen2.5-VL 0.034/−0.0026;其余模型敏感度 ≤0.014,边距接近零。

## 复现

```bash
python probe_analysis.py
```

输出 `out/probe_main_table.csv`、`probe_action_share.csv`、`probe_c3_summary.csv`、`probe_loo.csv` 与三张图。L4(可行动性)不在发布物里,未复核;论文主文的 L4 结果只在报告中按论文口径转述。
