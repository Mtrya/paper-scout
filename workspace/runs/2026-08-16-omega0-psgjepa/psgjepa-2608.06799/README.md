# PSG-JEPA 真实复训 + 逐维可辨识性探针(2608.06799)

## 尝试了什么

论文(official code 完整开源)称 LeWM 系动作条件 JEPA 的潜变量存在"本体感知可辨识性缺口"
(EE-yaw 线性探针 r=0.08),并提出训练时接地头(λ_g=0.1)修复。论文只报分组均值探针,
没有追问缺口的机制来源。我们把官方代码与 OGBench cube 数据(10000 集,102GB)完整复训:
baseline(λ_g=0)与 PSG(λ_g=0.1)同种子同配置各训 10 个 epoch(H200,约 9.5h),
然后在 400 集上(按集 9:1 划分)做三个探针:

1. **逐维岭探针**:冻结潜变量 → 28 维本体状态逐维回归,held-out Pearson r。
   28 维布局:joint_pos [0:6] | joint_vel [6:12] | effector+gripper [12:19] | privileged [19:28]。
2. **像素可见性**:各维度 |Δobs_d| 与像素帧差的相关(数据集侧,不需要模型)。
3. **开环 rollout MSE**:3 帧上下文 + 记录动作序列递归预测,模型步 {5,15,30}。

## 发现了什么

- baseline 逐维 r 与像素可见性相关 0.42:前向预测只编码像素里看得见的东西——缺口的机制证据。
- 缺口高度集中:腕部偏航(dim 5,r 0.03→0.76)、末端姿态(dim 15/16,0.21/-0.23→0.79/0.75)、
  关节速度(dim 7-9,r ≤0.20→0.73-0.91,单帧像素原理上不可见,由动态转移头 Δq 监督直接注入)。
- 未被接地头监督的 privileged 维(19-27)PSG 后原地不动:接地是外科手术式的,不是泛泛表示变好。
- 长程 rollout 改善复现(15 步 1.48 vs 1.72;30 步 1.65 vs 1.90),5 步持平。
- 退化维警示:dim 4 在数据中基本不动(var 7e-8),两个模型都读不出——分组均值会掩盖这种维度。

## 保留了什么

- `code/probe_psg.py`:探针脚本(在 Inspire notebook 上运行,依赖 repo + 密封 venv)。
- `code/plot_psg_probe.py`:报告图生成(逐维对照 + rollout MSE)。
- `code/probe_results.json` / `code/dataset_stats.json`:两个模型的完整逐维结果与数据集统计。
- `code/train_psg_job.sh`:H200 job 训练入口(set -ex + preflight + h5 拷贝到 /tmp + 训练)。
- `../assets/psg_probe_dims.png`、`../assets/psg_rollout_mse.png`:报告图。
- 训练 ckpt 与密封 venv 在 GPFS:`/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/psgjepa/`(out_baseline/、out_psg/、.venv/)。

## 环境坑(详见 INSPIRE.md 与 memories/2026-08-17-memory.md)

- cube h5 是 flat 结构,pixels 列需 `import hdf5plugin` 才能解码。
- 动作编码器输入是 frameskip×action_dim=25 维展平块(train.py `effective_act_dim`),不是原始 5 维。
- job 日志流会中途停更,但训练仍在推进——以 ckpt 落盘时间为准,勿凭日志判死。
