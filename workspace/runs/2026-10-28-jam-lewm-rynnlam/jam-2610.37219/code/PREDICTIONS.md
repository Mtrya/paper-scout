# 预登记预测 — 玩具实验：未来表示的"角色×目标质量"网格

日期：2026-10-28。在任何正式训练运行之前写定。

动机：JAM(2610.37219) 发现随机冻结未来锚点也能提升策略(辅助 foresight 损失的角色)，
RynnVLA-Latent(2610.34356) 则显示目标质量(3D 几何接地 vs LAPA)影响 latent-action 预训练
(表示作为"伪动作标签"的角色)。本实验在同一视觉推块任务、同一策略架构、同一数据与预算下，
把未来表示的两种角色(辅助损失 vs 生成目标)与三档目标质量(oracle 真值状态 / 随机冻结 CNN /
原始像素)交叉，另加两个对照(输入无关常数目标、当前帧随机锚点)。

测量：每臂 3 seeds；200 episode 成功率；冻结特征→未来状态的 ridge 探针 R²(独立 episode)。

P1(JAM 复现)：fs_random 成功率 > action(随机锚点在辅助角色下有帮助)。
P2(输入依赖性)：fs_const 成功率 ≤ action + 1pp(输入无关目标无增益，排除"任何辅助损失都有用")。
P3(未来 vs 当前)：fs_random > fs_current_random(JAM 的 offset 结论)。
P4(角色调和)：latent_oracle > latent_random(目标质量在生成目标角色下重要)，
   且 fs_oracle 与 fs_random 的差距 < latent_oracle 与 latent_random 的差距
   (目标质量敏感性：辅助角色 < 生成角色)。
P5(信息侧)：fs_random / fs_oracle 的 probe R² > action；fs_const 的 R² 不高于 action。
P6(目标质量探索项，不押方向)：fs_pixels vs fs_random 的排序记录为探索；
   JAM 的 VAE-direct 最优提示 fs_pixels 可能不弱于 fs_random。

地板检查：action 基线成功率若 < 0.5，先把训练预算加倍到基线可用再比较
(2026-09-14 教训：机制对照要求被试能力高于评测地板)。

边界：单任务、48×48 合成视觉、~0.5M 参数、12000 步预算(初版 3000 步基线 0.24,按地板检查上调);
结论只声称机制方向,不声称幅度向大模型外推。

追加(网格运行中、latent_const 启动前登记):
P7(角色对输入依赖性的要求):latent_const(生成目标角色 + 输入无关常数目标)
   成功率 < latent_oracle,且预期 ≤ action——生成目标角色下,表示就是动作标签,
   输入无关的标签无法提供可迁移的预训练;这与 fs_const(辅助角色下同型对照)
  形成同任务直接对照。

中途观察记录(非预测):fs_const 两个 seed 成功率均不低于 action,probe R² 明显更低,
P2 已被证伪——辅助角色下连输入无关目标也能带来优化侧收益。
注:第三个 seed 后 fs_const 均值与 action 持平,P2 方向部分恢复;最终以全量 3 seeds 为准。

预算公平性补充(latent_oracle_long / latent_random_long 启动前登记):
两段式 latent 臂的动作训练步数只有 fs 臂的一半(6000 vs 12000),
成功率差距里混有预算混淆。追加两臂 stage1=6000、stage2=12000 的长臂,
总步数 18000;用于把"角色效应"与"动作预算减半效应"分开。
若 latent_oracle 仍低于 action 而 latent_oracle_long 追平,则前半差距主要是预算。
结果(已出):latent_oracle_long 0.868 > action 0.860(预算效应坐实);
latent_random_long 三 seed 均值 0.820 < action 0.860 但 ≫ latent_random 0.618——
随机预训练的崩塌大头也是预算,角色效应(oracle_long − random_long = +4.8pp)保留。
另:latent_const 三 seed 均值 0.590,远低于 oracle 与基线(P7 成立)。
