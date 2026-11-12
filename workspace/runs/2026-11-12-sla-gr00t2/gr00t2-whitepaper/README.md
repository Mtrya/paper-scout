# GR00T 2：任务与身体怎样在联合生成中重新组合

## 研究问题

技术报告《GR00T 2: Scaling Robot Foundation Models for Generalization, Dexterity, and Adaptation》把人类或 UMI 演示与目标机器人的 task-agnostic play 共同训练，让模型从目标机器人的初始画面生成它自己的身体执行任务。本线程追踪三个机制问题：联合视频—动作生成为什么能分离任务知识与本体知识；共享动作头与相机对齐分别承担什么工作；跨轮上下文记忆怎样改变 GRPO 的信用分配。

论文缓存：`papers/vla/gr00t2-whitepaper.md`。官方项目页 https://research.nvidia.com/labs/gear/gr00t-2/ 当前标注 **Code Available in December**，尚无官方 GR00T 2 权重。以下两项实验分别使用公开 Human-1M 数据与白皮书 GRPO 公式，证据对象不是 GR00T 2 模型推理。

## 论文与代码路径得到的机制判断

GR00T 2 用单个 2B Cosmos-Predict2.5 DiT 联合去噪未来视频和动作，动作 token 能读取正在生成的未来视频。未来视频提供跨身体共享的场景变化接口，目标机器人 play 提供外观、运动学与动力学。X-Adapt 因此把“任务该发生什么”和“这个身体怎样做到”放在同一个生成问题里。

UMI→YAM 的消融把这条链拆得更具体：共享 YAM 头在 Toy / NIST / Conveyor 上为 60% / 50% / 86%，独立 UMI 头为 10% / 60% / 64%。共享头提供目标身体动作先验；独立头在 NIST 的特殊卸螺母策略上保留得更好。这个臂同时改变动作表示与头，不能归结为纯单因子比较。

腕视角对齐为 60% / 50% / 86%，随机两视角为 0% / 0% / 60%，无对齐为 0% / 0% / 46%。对齐在 112 对 Isaac Sim 渲染上拟合，场景位置误差由 74–94px 降至 8–11px，抓取区尺度由 6.9 提至 12.5px/cm。视频模型能吸收外观变化，相机几何仍需显式约束。

人类视频缩放实验在 0 / 0.87k / 8.1k / 84.4k 小时上得到真机成功率 18% / 27% / 28% / 41%，视频 rollout 成功率 30% / 33% / 38% / 51%。每档只有一次训练运行，epoch 对齐也使大数据档有更多更新，误差棒来自 rollout。这些数字支持视频规模作为跨本体能力的来源，不足以给出纯数据量幂律。

## 实验一：Human-1M 的多视角与三维手部监督

### 问题与协议

需要确认“视频监督场景、三维轨迹监督动作”是否有真实数据基础，而不是只存在于架构图。使用官方 `nvidia/PhysicalAI-Human-1M-100K` 的 `stereo_flower_wrist` 完整样本，读取 parquet 元数据与四路 MP4，检查时间对齐、T2 子任务分段、三维手部点、置信度与投影一致性。视频使用 FFmpeg 的 select 滤镜按帧号解码，`-fps_mode passthrough` 保留原始帧序。

### 观察

- 836 帧，30fps，27.835 秒；四路视频为主视角、右眼视角与左右腕视角。
- 六段连续 T2 子任务覆盖采花、拢枝、取皮筋、缠绕、固定与放回花束。帧间隔最小 0.032887 秒，最大 0.033686 秒。
- 双手各 21 个三维关键点，每点三维；另有逐关节四元数、头相机位姿、内参、掩码与置信度。
- 四个代表帧的双手三维点均按样本内参投回主相机，每只手 21/21 点。投回结果见 `code/results/human1m/human1m_aligned_episode.png`。
- 左手 10 帧、右手 1 帧的三维位置全零，这些是缺失哨兵；左手置信度 5% 分位 0.80196，右手 0.77157，图中可见对应的低质量区间。
- 样本没有直接的 grasp/release 标签。T2 提供子任务级语言与时间边界，不自动等同于接触事件标注。

### 结论

这段真实数据把多视角视频、三维手部运动、相机几何与子任务语言放在同一时间轴上，能够分别监督场景变化与动作几何。投影可恢复只是几何一致性检查；跨本体执行还需要目标身体的动力学与动作统计，这正是 play 数据在 GR00T 2 中的作用。

证据：`code/probe_human1m.py`、`code/results/human1m/summary.json`、两张结果图。

## 实验二：历史 GRPO 基线的独立数值实验

### 问题与协议

普通 leave-one-out 基线在全成功或全失败组中给出零优势。白皮书引入每上下文历史均值 `m_g`，把 `h_g` 个历史伪回报加入基线。需要测量它究竟只修复全同组，还是改变所有组的优势结构，以及旧表现在策略突变后怎样退出。

`code/grpo_context_memory.py` 按白皮书公式执行：每组八次尝试；首次访问 `h=0`，随后 `h=2`；先计算优势，再以 `α=0.25` 更新历史 EMA。固定种子 20261112；七个成功概率各跑 128 条独立链，每条 160 次访问，前 20 次作为 warm-up 排除。另构造由差变好与由好变差的确定性轨迹，测量历史优势的衰减。

### 结果与推导

- 普通 RLOO 在全同奖励组中的非零优势比例始终为 0。
- 历史基线在全同奖励组恢复非零优势的比例为 **99.7709%–100%**。极少数未达到 1 的情况来自历史均值恰好等于当前奖励。
- 对轨迹 `i`，令 `R̄₋ᵢ = Σⱼ≠ᵢ Rⱼ/(N−1)`，逐轨迹恒等式为 `A_memory,i = A_RLOO,i + h(R̄₋ᵢ−m)/(h+N−1)`。对全组取平均后，`A_RLOO` 的均值为零，`R̄₋ᵢ` 的均值等于组均奖励 `μ`，所以组均优势为 `h(μ−m)/(h+N−1)`。`μ−m` 描述组平均修正，不能直接代替逐轨迹式中的 `R̄₋ᵢ−m`；混合组也会受到历史项影响，它不是全同组专属补丁。
- 将全同组与混合组一并统计，每组平均绝对优势的 RMS 在各概率档均小幅收缩。历史项改变所有组的优势构造，新增更新集中在原本零优势的全同组；这与公式推导一致。
- EMA 每轮将历史偏差乘以 0.75，半衰期为 `ln(0.5)/ln(0.75)≈2.41` 次到访。突变后第四轮到访保留首轮优势的 31.6%。这是访问次数的衰减，不随未访问场景的墙钟时间变化。

实验测的是优势构造，不涉及真实策略训练或成功率提升。其作用是把跨轮记忆明确定义为“当前组表现相对历史表现的偏移”，并给出伪计数与 EMA 两个可调的尺度。

证据：`code/results/summary.json`、`code/results/stationary_summary.csv`、`code/results/abrupt_trajectory.csv`、`code/results/grpo_context_memory.png`、`code/results/run.log`。

## 可继续的问题

1. 训练期三维教师能否替代一部分推理期视频生成，同时保留 X-Adapt 所需的目标身体想象？判别实验应分别测场景泛化、相机错位与动作精度，而不是只看视频生成指标。
2. 共享动作头的身体先验与稀有子技能的覆盖范围能否按通道或技能阶段分配？NIST 的反向结果提供了具体检验点。
3. 上下文历史表能否与按事件索引的机器人记忆结合，使 GRPO 的信用衰减由有意义的事件而非固定访问次数驱动？

## 重跑

```bash
cd runs/2026-11-12-sla-gr00t2/gr00t2-whitepaper

# GRPO 数值实验
uv run --with numpy --with matplotlib python \
  code/grpo_context_memory.py --output code/results

# Human-1M 样本，下载在远程实例执行
export HF_ENDPOINT=https://hf-mirror.com
hf download nvidia/PhysicalAI-Human-1M-100K \
  samples/downloads/stereo_flower_wrist.tar.gz \
  --repo-type dataset --local-dir "$DATA/human1m"
tar -xzf "$DATA/human1m/samples/downloads/stereo_flower_wrist.tar.gz" -C "$DATA/human1m"
FFMPEG_BINARY=/path/to/ffmpeg uv run --with numpy --with pyarrow --with matplotlib --with pillow python \
  code/probe_human1m.py \
  --dataset "$DATA/human1m/stereo_flower_wrist" \
  --output code/results/human1m
```
