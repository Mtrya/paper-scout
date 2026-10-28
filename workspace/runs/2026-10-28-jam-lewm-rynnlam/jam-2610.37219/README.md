# JAM(2610.37219)深挖线程:代码核验 + 玩具"角色×目标质量"网格

## 论文核心(精读自 papers/vla/jam-2610.37219.md)

JAM 把 VLA/WAM 之争拆成三个独立选择:骨干初始化、未来观测上的目标、推理时保留的模块。
方法:动作损失 + foresight 损失(当前帧特征经小 transformer 头预测冻结未来锚点的表示,
余弦距离,sg 停梯度;训练后丢掉锚分支与预测头,不生成任何图像)。
关键实验(LIBERO / LIBERO-Plus,固定架构对照):
- VAE-direct 目标最优(Wan):64.54 vs 生成损失 55.31 vs 无未来损失 45.55(LIBERO-Plus)。
- 随机冻结 Wan 锚点 58.63 > 预训练 Wan 锚点 55.58;随机骨干时预训练锚点反而最差(71.15 vs 随机锚 84.40)。
- 未来位置目标([1,2])远好于当前位置([0]:44.62 低于无未来损失基线 45.55)。
- Qwen3.5-2B + foresight:81.14 vs 动作单训 77.03;DINOv2-L:70.39 vs 68.26。
- 成本:缓存目标后在线 FLOPs 比 FastWAM 少 73%。
- 统计:3 个独立训练 run(Wan 64.61±0.37、Ovis 79.92±0.43)、McNemar p<1e-12、任务 bootstrap CI。
- 代码核验(github.com/y0urOy/JAM):发布版只实现 VAE-direct 目标(best config);
  余弦损失 target.detach()、共享 trunk + 每 (layer,offset) 一个线性 readout、
  bf16+zstd 目标缓存;预训练/随机锚点的消融不在发布代码内(属论文实验,非可复现接口)。

## 我们的实验:角色×目标质量网格(玩具视觉推块)

动机:JAM 的随机锚点结果与 RynnLAM 的"目标质量决定论"表面冲突。把未来表示的
**角色**(辅助 foresight 损失 vs 生成式预训练目标)与**目标质量**(oracle 真值状态 /
随机冻结 CNN / 原始像素)在同一任务、同一架构、同一数据与预算下交叉。
另加两个对照:输入无关常数目标(排除"任何辅助损失都有用")与当前帧随机锚点(offset 对照)。

设置:2D 视觉推块(48×48 渲染,白手推蓝块进绿圈;连续脚本专家 2500 集);
CNN→36 个空间 token→2 层 transformer 骨干,动作专家为 2 层联合 transformer
(视觉 token 不能反向影响动作 token),flow matching x-prediction,chunk H=4、执行 2。
9 臂 × 3 seeds × 12000 步(batch 128,AdamW 1e-3→1e-4 cosine,λ_F=0.5);
latent_ 臂为两段式(前 6000 步 flow 匹配未来全局表示,后 6000 步换头训动作)。
测量:每臂 200 episode 成功率;冻结特征→未来真值状态的 ridge 探针 R²(独立 500 集)。

预登记预测见 `code/PREDICTIONS.md`(P1 随机锚点有效、P2 常数无效、P3 未来>当前、
P4 目标质量敏感性 latent>fs、P5 信息侧探针、P6 像素探索项)。

## 结果

全量网格(3 seeds 均值;成功率为 200 episode/seed;probe 为冻结特征 ridge R²):

| 臂(角色 / 目标) | 成功率 | sT 探针 R² | delta 探针 R² |
|---|---|---|---|
| action only(基线) | 0.860 | 0.757 | 0.674 |
| fs oracle(辅助 / 真值未来状态) | 0.863 | 0.808 | 0.671 |
| fs random(辅助 / 随机冻结 CNN 未来帧) | 0.885 | 0.750 | 0.674 |
| fs pixels(辅助 / 原始像素未来帧) | 0.875 | 0.741 | 0.668 |
| fs const(辅助 / 输入无关常数) | 0.862 | 0.621 | 0.563 |
| fs current random(辅助 / 随机 CNN 当前帧) | 0.873 | 0.743 | 0.670 |
| latent oracle(生成 / 真值状态) | 0.810 | 0.804 | 0.684 |
| latent random(生成 / 随机 CNN) | 0.618 | 0.721 | 0.636 |
| latent pixels(生成 / 原始像素) | 0.668 | 0.720 | 0.628 |
| latent const(生成 / 常数) | 0.590 | 0.709 | 0.622 |
| latent oracle_long(生成 / 真值,动作 12000 步) | 0.868 | 0.795 | — |
| latent random_long(生成 / 随机,动作 12000 步) | 0.820 | 0.739 | 0.656 |

预训练曲线:latent_random / latent_pixels / latent_const 的 stage-1 损失在 2000–4000 步内
塌到 ~1e-4(未来帧≈当前帧,输出当前帧的随机/像素特征即成恒等任务);
latent_oracle 保持 0.05–0.09(未来真值状态需要真动力学)。

判定(对照 PREDICTIONS):
- P1 成立:fs_random 0.885 > action 0.860(随机锚点在辅助角色下有效)。
- P2 成立(均值):fs_const 0.862 ≈ action,输入无关目标无平均增益;但 note 它把 delta 探针压到 0.563。
- P3 弱成立:fs_random 0.885 > fs_current_random 0.873(未来略优于当前)。
- P4 强成立:latent_oracle 0.810 ≫ latent_random 0.618,生成角色下目标质量是生死线;
  且 fs_oracle 与 fs_random 的差距(−0.022)远小于 latent 两侧差距(+0.192)——目标质量敏感性:辅助≪生成。
- P5 部分:fs_oracle 的 sT 探针升高(0.808),其余不升;delta 探针全臂不优于基线。
  即成功率增益与动力学信息增益脱钩。
- P7 待 latent_const 全量:生成角色下常数目标预训练同样塌缩,预期成功率进一步走低。
- 预算对照:latent_oracle(0.810)低于 action(0.860),但两段式动作步数减半;
  latent_oracle_long(动作 12000 步)0.868 > action 0.860——两段式 oracle 预训练在充足动作预算下是净收益,
  前一段差距主要是预算效应。latent_random_long 三 seed 均值 0.820:随机预训练在长预算下仍低于基线 0.860,
  但远高于短预算的 0.618——崩塌的大头也是预算,角色效应(oracle_long − random_long = +4.8pp)保留。
  关键对照:同预算下,辅助随机(0.885)优于生成随机(0.820),角色差异成立。

机制读法(写报告的骨干):
- 辅助角色 ≈ 对当前观测做"随机教师蒸馏":未来帧≈当前帧,foresight 实际在让骨干
  同时拟合一个冻结随机特征器,是输入依赖的表示正则,不是 anticipation;
  oracle 虽提高绝对状态可解码性,但不提高 delta 可解码性,也不提高成功率。
- 生成角色 ≈ 预训练一个"未来标签生成器":标签若是帧的光滑确定函数,预训练退化为恒等,
  标签若含真动力学,才有非平凡监督;标签质量直接决定下游成功率。
- 独立复算的两个量(本机 scout-gpu 实测):随机锚点未来/当前目标余弦相似度 0.99987,
  与无关帧也 0.99977;目标平方范数的 99.97% 由数据集平均方向解释(cos² 口径),
  像素目标为 65.0%。即随机锚点的 foresight 损失在数值上退化为"对齐到数据统计给出的
  固定方向",而像素目标仍有 35% 的范数来自逐输入变化——但两臂成功率差只有 ~1pp。

## 复跑

```bash
# 依赖:CUDA torch、numpy(本机 3060 即可;CPU 也行但慢)
cd code/toy-foresight
python train.py --gen-demos 2500 --data data/demos.pt --probe-data data/demos_probe.pt --stats data/stats.pt
python train.py --steps 12000 --batch 128 --eval-episodes 200 --device cuda --out out
```

env.py / data.py 复用自 runs/2026-10-07-astra-ascent-vlarl/lowrank-flow(连续脚本专家与准静态推块动力学)。
