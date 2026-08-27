# WorldEcho/WorldSync 线程:AC-WM 动作跟随诊断 + 玩具 IE 消融(实验 D)

论文:WorldEcho (arXiv 2608.24885,2026-08,无代码)。问题:动作条件世界模型
(AC-WM)生成的视频"看起来对"但真的跟随了动作吗?——把"动作跟随"从视觉
质量里拆出来测量。

## 论文要点(全文通读)

- 诊断协议 WorldEcho:五类动作查询(demonstrated / cross-state /
  local perturb / policy rollout / feasible random)+ 视觉完整性门
  (MUSIQ 无参考画质 + 运动平滑 + SAM 跟踪 EE 可见性 + VLM 判手臂完整性)
  + SE(3) NDTW 轨迹对齐(Φ = Tan et al. 2026 从视频提 EE 位姿)。
- 诊断结论:六个专家训练的 AC-WM,off-expert 查询下 gated error 升
  0.029–0.099m;两种失败形态 = 视觉崩溃 / 看似合理但不跟动作(乐观偏差:
  视觉门通过 ≠ 动作跟随)。
- 修复方案 WorldSync:扩动作覆盖 + AFE(从视频特征解码未来 EE 轨迹的
  对齐头,推理时摘除)+ IE 监督(同一观测、同一噪声、不同动作的两个样本,
  对齐 Δv 与 Δx0——动作差异必须反映为生成差异)。
- **消融关键:IE 是轨迹对齐的主驱动;AFE 单独几乎不涨动作指标,只涨视觉
  通过率**——与 08-17 PSG-JEPA"对齐头是外科手术,不修潜变量本身"互证。

## 实验 D 设计(玩具 AC-WM,启智 hydra-probe 4090)

动机:论文的诊断是观察性的(六个现成 AC-WM)。玩具环境可以做**因果**
消融:同一基座模型,只改训练信号,看 off-expert 恶化是否出现、IE 是否
真的比"扩覆盖"更能修。

- 三变体同脚本 `wam_if_toy.py`:
  - acwm:专家动作训练的基线 AC-WM;
  - acwm-cov:动作分布扩覆盖(更宽动作先验)对照;
  - acwm-ie:加 IE 式一致性损失(同观测同噪声、成对动作)。
- 查询四类:demonstrated / local-perturb / random-feasible / cross-state,
  指标 EE-ADE(像素),EE 轨迹用骨架拟合读出(本地冒烟:mean 0.8px,
  max 1.45px,0.1s/帧;重放相位约定已验证)。
- 判定:1)acwm 在 off-expert 查询上 EE-ADE 显著恶化(复现诊断形态);
  2)acwm-ie 比 acwm-cov 更能降 off-expert EE-ADE(IE 主驱动)。

## 结果(EE-ADE px,均值/中位,n=60/类)

| 变体 | demo | xstate | perturb | random |
|---|---|---|---|---|
| acwm(仅专家) | 1.17 / 1.14 | 2.89 / 2.27 | 3.21 / 3.13 | 8.79 / 8.64 |
| acwm-cov(扩覆盖) | 1.10 / 1.06 | 2.76 / 2.57 | 2.49 / 2.23 | 9.10 / 8.93 |
| acwm-ie(+IE) | 1.19 / 1.17 | 2.69 / 2.34 | 2.44 / 2.26 | **8.28 / 7.70** |

- 诊断形态复现:off-expert 单调退化,demo 1.17 → random 8.79;arm-loss 全 0
  (玩具环境无纹理,视觉崩溃形态按预期不出现,只复现"不跟动作"半边)。
- 扩覆盖基本不解决问题:perturb 3.21→2.49 略降,random 反而 8.79→9.10——
  与论文消融(Base/Expanded raw NDTW 0.0306→0.0258,温和)同形态。
- IE 是唯一在最远分布查询上改善的变体(random 中位 8.64→7.70,-11%);
  perturb 上与扩覆盖打平(2.26 vs 2.23);demo 上无损。方向与论文
  "IE 主驱动"一致,但幅度弱得多(论文 -34% raw NDTW,玩具 -11%)。
  玩具无纹理/外观多样性,IE 红利可能要在视觉更丰富的环境里才充分显形。
- 局限:逐样本 ADE 未落盘(只存汇总),无法做配对显著性检验;n=60/类。
  下次实验脚本应把 per-sample 数组写进 JSON。

## 置信度边界

玩具数据无纹理/光照复杂性,视觉崩溃形态无法复现,只能复现"不跟动作"
一半;EE 读出是骨架拟合而非学习解码器,系统性优于论文的 Φ,差错误差
被压低,相对趋势可信、绝对值不可比。

## 文件

- `code/wam_if_toy.py`:与 LAWA 线程共用脚本(实验 D 入口 --d-variants)。
- `code/eval_d1.json`:结果(待回传)。
