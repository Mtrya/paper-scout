# 线程一：VLA 的视觉记忆不是“把更多帧塞进上下文”

## 研究问题

本线程只讨论 VLA 的视觉历史税：像素历史高维、高频、多视角，保留、编码和消费都昂贵。文本型 LLM agent 也有上下文与检索问题，但文本轨迹已经是廉价的符号序列，不是同一个问题。

真正需要记忆的判据是观察混淆：存在两段历史 `H`、`H'`，使当前观察与本体状态相同，但正确动作不同。

```text
o_t(H) = o_t(H')    but    a*(H) != a*(H')
```

因此“长任务成功率提升”最多是间接证据。更强的评测要固定当前观察，交换历史，看动作分布是否沿语义正确的方向改变。

## 论文与外部信号

- [NativeMEM（arXiv:2607.06678）](https://arxiv.org/abs/2607.06678)，2026-07-07；[项目页](https://opendrivelab.com/NativeMEM/)。主论文。每视角每帧压成一个 token，以两阶段训练把 token 对齐到预训练 VLA。
- [SOMA（arXiv:2605.22283）](https://arxiv.org/abs/2605.22283)，ICML 2026。关键相邻工作：把目标移出视野、主动扫视并建立对象级空间记忆，比一般“长时程”任务更直接地制造记忆必要性。
- [MemoryVLA 官方实现](https://github.com/shihao1895/MemoryVLA)，审计 commit `d732ea9072bc063399ccc817aed74ab172eb50be`。它不是本轮第三个主题，而是用于检查固定容量视觉记忆的代码级外部信号。

## 从论文读到的边界

NativeMEM 的成绩很强：仿真平均成功率从最强列示基线的 32.4% 提升到 84.0%，真实机器人从 34.7% 到 98.7%。它也把历史视觉重编码从动作主路径拆出，并把每帧每视角压成一个 token。问题是：队列长度、存储量与被 VLA 消费的 token 数仍随 `T` 线性增长；它是压缩后的视觉证据轨迹，不是固定大小、动作充分的 belief state。

SOMA 使“为什么必须记忆”更清楚：当前视锥里没有目标，正确动作必须依赖扫视历史。但它用 YOLO、DINOv3、VGGT 先构建对象级空间记忆，扫描/几何预处理昂贵；附录失败中 44% 来自噪声空间 token、32% 来自无关记忆激活、24% 来自缺乏任务阶段意识。对象位置记忆不等于控制状态记忆。

## MemoryVLA 代码审计与机制探针

发布实现维护认知与感知两套 16 槽 bank。超长时，寻找余弦最相似的相邻槽，直接做 `0.5 * (left + right)`，不保存每个槽已经代表多少帧；合并后还保留左槽（更早）的时间戳。检索时，这个时间戳会进入时间位置编码。

`code/audit_memoryvla_consolidation.py` 对这一规则做了单 token 的代数镜像，并显式追踪代码丢弃的叶节点权重。400 条、每条 128 步的平滑 64 维随机游走，容量固定为 16：

- 一个最坏槽覆盖 32 帧，但有效样本数 `1 / Σw_i²` 只有 3；
- 最新叶节点权重为 0.5，最早两个各约 `4.66e-10`，相对均匀平均相差 16 倍与约 `6.7e7` 倍；
- 槽内容的加权时间中心在 62，而保留时间戳是 32，误差 30 步。

不均匀权重和陈旧时间戳是发布算法的确定性代数结果；上述发生频率来自合成随机游走，不能当成策略性能估计。它说明的是固定槽数需要带质量的合并（保存 count/span、任务阶段与不确定度），而不只是一个容量上限。

证据：`code/evidence/memoryvla-consolidation-audit.json`。

## 发布权重的结构核验

本轮完整取得官方 LIBERO-100 checkpoint（33,507,487,606 bytes），并用 `map_location=meta` 做了不加载 tensor storage 的只读检查。文件含 1,406 个 tensor、约 83.77 亿参数；认知 memory bank、感知 memory bank 与 perceptual compressor 共 74 个参数 tensor，两套 bank 都含两层 retrieval block、gate fusion 与 timestep encoder。checkpoint 没有序列化 episode slot、历史 token 或 timestamp；它发布的是记忆模块及基座的训练权重，不是某条轨迹的运行态记忆。证据：`code/inspect_memoryvla_checkpoint.py` 与 `code/evidence/memoryvla-checkpoint-structure.json`。

## 可直接建造的下一步：History-Swap VLA benchmark

对每个当前观察构造有效历史对 `(H, H')`，动作标签分别为 `(a_H, a_H')`。用对称 swap margin 测模型是否真的消费历史：

```text
S_swap = 1/2 E[
  log π(a_H | o,H)  - log π(a_H | o,H')
+ log π(a_H'| o,H') - log π(a_H'| o,H)
]
```

再加入动作不应改变的无关历史，单独测鲁棒性。训练时可在普通 action loss 外加入 paired-history margin；扩散动作头必须固定初始噪声或比较完整条件分布，否则随机采样会伪造 swap effect。

这会把研究目标从“能否保存过去”改成“能否把过去压成对当前动作充分、对无关视觉变化不敏感的信念状态”。NativeMEM 适合作为高保真证据轨迹基线；固定槽 memory 应与它在相同 history-alias 对上比较，而不是只比平均长任务成功率。

## 复现边界

NativeMEM 与 SOMA 在本轮检查时都未发布可运行代码/权重。MemoryVLA 代码、配置和 33.5GB LIBERO-100 权重已发布并完成结构核验。完整动作级 history swap 还需要基座 CogACT、LIBERO 资产、环境轨迹与确定性扩散采样；本轮不把 checkpoint 元数据检查或合成 consolidation 探针冒充该实验。
