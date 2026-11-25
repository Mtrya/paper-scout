# 线程：Motion Mamba 2.0 的慢状态快读出（2611.29897）

论文：Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control（Yue Huang 等，北大/高丽/莫纳什/南洋理工；arXiv 2611.29897v1，ECCV 2024 Motion Mamba 的期刊扩展版）。代码 https://github.com/AIGeeksGroup/MotionMamba2.0 ，权重与数据在 HuggingFace（AIGeeksGroup/MotionMamba2.0）。

## 论文核心主张

1. 任务：长时程（最长 39.52 s）、实时、可控的 G1 人形运动生成——文本指令与空间目标在生成中途到达，输出 52 帧（25 Hz）窗口流。
2. 架构：root–body 两段式自回归扩散（ARDY 表示），每段 DiT 的序列混合器 = 局部注意力（W=4 窗有界缓存 + 4 个静态 sink）+ 可复位 Mamba-3 递推 + PCM（Persistent Content Memory）。
3. PCM（核心新意）：把 Mamba 递推态的池化摘要 s̄_t、相邻窗口摘要差 d_t、文本嵌入 e_t 拼成候选内容，经 FiLM 文本调制与 LN 后写进每层 2 个内容槽；写入门受 η=0.25 上限约束；读取时把槽投影成注意力键值插在 sink 之后，读出残差用 tanh(ρ) 控制（ρ 零初始化 → 精确退化为无 PCM 的 Native 混合器）。递推态本身不是注意力 token，PCM 是"压缩递推态 → 注意力可读内容"的学习接口。
4. accepted-only 历史提交：去噪中间候选只读不可变历史快照；干净窗接受后，缓存追加、递推态替换、PCM 写入三者在一次规范提交中原子完成，失败则整体回滚——把扩散采样迭代与物理时间流分开。
5. 推理优化：执行流简化（不物化未消费的状态提议）+ Mamba 预处理融合核 + 三 CFG 分支联合 CUDA Graph 回放；RTX 4090 上 52 帧完整窗 p50 从 1685.35 → 276.14 ms（6.138×，CI [6.094,6.290]），且 4×100 窗 rollout 输出与提交态逐位一致。
6. 结果：retargeted HiPHI 39.52 s 处 Root CE 7.55 cm / End CE 13.64 cm（全场最低）；消融 Minimal（无 Mamba）60.00/64.86 → Native（仅 Mamba）9.36/15.54 → 全量 7.55/13.64——长时程收益主要来自递推支路，PCM 再贡献 1–2 cm；短窗上 Mamba 支路略有代价（52 帧 Root CE 3.86→4.49，PCM 拉回 2.78）。真机 Unitree G1 100 试验 94 次全任务成功、0 摔倒，键盘/鼠标交互物理响应 p50 351/360 ms。

## 本轮做了什么与看到了什么

### 1. 全文精读
全文（3717 行，含附录 A–D）通读，缓存于 `papers/robotics/motion-mamba2-2611.29897.md`。Eq.6（Mamba-3 梯形+旋转递推）、Eqs.8–13（PCM 写读）、Eq.14（accepted-only）逐式核对。

### 2. 官方仓库源码审计（code/MotionMamba2.0，浅克隆，模型代码 1 万余行）
- **PCM 与论文逐条吻合**（motion_mamba2/model/content_memory.py，类 PersistentContentMemory）：候选内容 [s̄; (1+tanh γ)⊙LN(d)+β; e]、write gate = η·σ(f_w([j; vec(P_{t-1})]))、槽更新 (1-w)⊙P_{t-1}+w⊙Z、读出残差 tanh(read_gate)——与 Eqs.8–13 一字不差。
- **递推态四元组确认**（model/mamba.py 的 allocate_inference_cache / step）：(angle_dt_state 累积相位 φ, ssm_state 递推矩阵 H, k_state 上一步状态输入 B̃, v_state 上一步值 u)——正是论文"carried state includes the recurrent matrix, accumulated phase, and previous state-input and value vectors"。step 走 CUDA kernel（cutlass/quack），CPU 不可直接跑。
- **accepted-only 提交**（model/history_commit.py）：`prepare_accepted_commit`/`cancel_accepted_commit`，并断言"canonical root/body history prime must commit exactly once"——论文的原子提交+失败回滚在代码里是显式契约。
- **发布配置核对**（configs/model.yaml）：content_memory_slots=2、max_write_rate=0.25（η）、window_size_chunks=4（W）、num_sink_tokens=4、state_summary_dim=128、denoising_steps=10、CFG (2,2)、cuda_graph 加速、token_state_interaction=naive_gated_add。代码里还保留着 state_query / state_kv_modulation / h0_parallel / h1_innovation / h2_delta_rule 等未采用的交互设计——论文选择"内容写入而非状态依赖查询"是消融后的取舍。
- 文本编码用 LLM2Vec-Llama-3-8B（冻结），评测用缓存文本特征，不依赖文本编码器。

### 3. 机制探针（code/mm2_pcm_resumability_probe.py，CPU torch，输出 probe_output.txt）
- **A. PCM（直接 import 官方类）**：写入门恒 ≤ η=0.25（随机输入下 max 0.128）；槽更新有界不等式逐槽成立；ρ 零初始化时 tanh(ρ)=0，PCM 读出对输出贡献精确为零（论文"零初始化恢复 Native"成立）；同一状态转移换文本 → proposal 差 26.19（内容强调制）而 gate 差仅 0.005（速率弱调制——文本主要通过内容进入记忆，对写入速率的调制在随机初始化下很小）；20 步会话写入槽范数演化有界。
- **B. Mamba-3 可复位递推（Eq.6 纯 torch 复推）**：携带 (H, φ, u_prev, B̃_prev) 四元组做 32+32 分块续扫，与 64 步全程扫描相比 max|Δy|=0、max|ΔH|=0、max|Δφ|=0——**逐位一致**；只携带 H 的朴素续扫（φ/u/B 清零）max|Δy|=0.762，显著失配。这从机制上解释了论文为什么必须携带那三个"额外"状态量：梯形项引用上一步的 u·B̃^T，旋转相位丢失则历史全部错位。
- 结论：论文的两个关键设计（PCM 接口、可复位流式递推）在代码与数学两个层面都得到独立核验。

### 4. 图片素材
arXiv HTML 已嵌入 Fig.2（交互生成 teaser，introduction.png）、Fig.3（数据集）、Fig.4（总览，overview.png）、Fig.8（真机 G1，exp5-g1.png）、Fig.9（SONIC 跟踪）、Fig.10（远距目标对比），均下载至 drafts/images/；Fig.5/6/7 为 SVG（Fig.7 已抽取）。报告选用 overview.png（机制）+ exp5-g1.png（真机）。

## 如何重跑
```
python3 runs/2026-11-26-loom-motionmamba/motion-mamba2-2611.29897/code/mm2_pcm_resumability_probe.py
```
（依赖 torch；CPU 即可，约 5 秒。Part A 需要官方仓库在任一上级 code/ 目录：`git clone --depth 1 https://github.com/AIGeeksGroup/MotionMamba2.0`；Part B 为纯公式复推，不依赖仓库。完整模型冒烟需下载 MotionMamba2.pt 与前端资产，本轮未做——探针已覆盖两个核心机制的核验。）

## 对报告的含义
- 可信的核心判断：MM2 把"快慢系统耦合"落成了一层可见的接口工程——慢通道（固定尺寸递推态）携带任意长历史，快通道（局部注意力）管毫秒级反应，PCM 是两者之间的学习式翻译层；accepted-only 提交保证扩散迭代不污染物理时间。
- 与上轮快慢耦合战线（2026-11-20 判官门/停止权）的连接：上轮回答"停止权属于谁"，本轮展示"历史归属谁、如何被对侧读出"——慢通道信息必须经学习接口才能进入快通道的决策视野，且接口的归纳偏置（内容写入而非状态查询）是消融选出来的。
- 边界与不确定：长时程指标仍随时间退化（7.55 cm @39.5 s vs 2.78 cm @2 s），PCM 槽极少（每层 2 个）、不主张精确恢复任意远的历史；文本对写入门的调制在初始化下很弱，训练后是否增强未知（未训练验证）；6.138× 加速的分解（Flow 1.525×/Graph 5.943×）来自论文自报，其"逐位一致"声称我们未在 GPU 上复测。
