# Round-Trip Consistency (2608.00675) — 耗散边界探针

论文:Alexander Scheinker (LANL),双向潜扩散的 round-trip 误差自检。无代码。

## 我们的研究动作

搭建最小代理(条件 MLP 双向去噪器 + DDIM-50,架构精神对齐论文:时序对条件、方向旗标、前向 x_{t+1} / 反向 x_{t-2} 目标),在两类动力系统上扫耗散结构:

- **Lorenz-63**,r ∈ {0.5, 3, 10, 20, 28}:r=0.5 原点全局稳定(世界死亡),r=3/10/20 稳定不动点(阻尼递减),r=28 奇怪吸引子。散度解析常数 −(σ+1+b) = −13.67(强耗散)。
- **无阻尼单摆**(正对照):Hamiltonian,散度恒零,相体积守恒——论文 Assumption 1(co-Lipschitz 逆)天然成立的形态。

每个机制训练 8000 步(单步误差 ~1e-4,双向同精度),128 个留出 IC 上分解三个量:E_i(前向 rollout 真实误差)、C_i(round-trip 差异)、δ_i(真值终点起逆向 rollout 的误差 = 逆向腿噪声地板),外加固定深度 Spearman(C,E) 与逆向映射 Jacobian 奇异值。

## 核心发现

- 猜想"吸引子处 C 失明"**方向被证伪**:实际失效模式是虚警——E 平在 1e-3,C 与 δ 重合爆炸(r=0.5 深度 40:C/E ≈ 10⁴;r=28 深度 80:C ≈ 1.5×10⁶)。
- 机制定位:浅层 ρ(C,δ)=0.92-0.97 vs ρ(C,E)≤0.38——C 测量逆向腿自身的不稳定性(耗散系统时间反演是扩张映射),不是前向误差。
- 单摆对照:C/E 稳定 1.3-2.5,ρ(C,δ)≈0——体积守恒时 round-trip 按设计工作。判别变量是相体积收缩率,正是 Assumption 1 的 μ。
- 未复现论文的 0.91-0.98 排序力(代理实现差距:MLP vs 潜扩散、每转移 50 步采样噪声复合);失效的定性结构不依赖此差距。

## 内容

- `code/`:systems.py(Lorenz 五机制 + 单摆)、model.py(双向去噪器 + DDIM)、train.py、probe.py、analyze_roundtrip.py、make_figs.py
- `code/results/`:各机制 metrics.json + roundtrip_summary.json(汇总);完整 arrays.npz 与模型权重在本机 `code/roundtrip-probe/`(体量大未晋升)
