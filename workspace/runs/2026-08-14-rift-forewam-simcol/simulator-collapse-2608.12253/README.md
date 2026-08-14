# simulator-collapse-2608.12253 — 模拟器坍缩最小复现

## 线程问题

One Frozen Simulator Is Not Enough(2608.12253,Levine/Manning/Shi)证明:多轮 RL 用单个冻结 LLM 当用户模拟器时,模拟器的 mode collapse 会把策略梯度偏向 mode-user 目标(Thm 3.2),杀掉奖励方差的模拟器侧对比(Lemma 3.3,式 6 方差分解),让策略熵几何收敛到 mode-exploit 集合(Cor 3.5),held-out 泛化先涨后崩。两个修复:口头化采样(VS,推理期恢复 reference 分布)与共同训练(CoT,训练期移动靶点)。SCOPE 框架论文声明开源,但截至 2026-08-14 仓库不可寻(GitHub/网页搜索均无)。

本线程:用 Qwen3-4B-Instruct 在 P4G-mini 任务上最小复现核心现象——single 的"训练奖励涨、策略熵坍缩、held-out 面板先涨后跌",vs/cot 的修复,以及三个变体的熵坍缩签名对比。

## 设计(`simcol_probe.py`)

- 任务:P4G-mini,募捐者(策略)说服捐赠者(模拟器)≤5 轮,奖励 = 解析出的捐款额 ∈{0,1,2}(末条消息须以 "Donation: $X" 结尾,正则解析)。
- 训练:REINFORCE,G=8,group-relative z-score advantage;LoRA r=32 挂在策略上;每轮按真实上下文(chat template)算策略自身话语的逐 token NLL,乘 advantage 更新。
- 三变体(各 40 步):
  - `single`:冻结模拟器直接采样(标准做法,预期坍缩)
  - `vs`:每轮先让模拟器输出 4 候选带概率的分布,再按分布采样回复
  - `cot`:模拟器同样 LoRA 更新,其 token advantage = |R_τ − R̄|/σ_R(SPICE 式方差课程;论文:对抗/合作奖励都会让模拟器重新坍缩,方差课程保持在信息区间)
- 指标:每步训练奖励、策略多样性(distinct-2、self-BLEU-2,熵坍缩签名)、held-out 三人设面板(emotional/busy/hostile,训练期未见)每 8 步评估。
- 后端:优先 vLLM(`.venv3`/`.venv3b`,双实例 util 0.25 同卡);失败退回 transformers 批生成(`--backend hf`,venv2 系统 torch 2.7)。

## 执行环境

- 启智 notebook `wam-rift-probe`(1×4090,ngc-pytorch:25.02)。Qwen3-4B 权重:`<workroot>/cache/models/Qwen3-4B`。
- 重跑:`bash run_simcol.sh`(single→vs→cot 串行);产物 `simcol_results/{single,vs,cot}.json`。

## 结果

见 `simcol_results/*.json` 与报告"模拟器坍缩/我们的最小复现"节。

## 局限(相对论文的降级)

- 4B 模型、40 步、8 轨迹/步,远小于论文的 4B/8B × 数百步 × 3 基准;held-out 面板是同一基座的三个人设 prompt,不是异族模型面板——面板差异只能读相对趋势。
- REINFORCE 的 NLL 按轮次算、未含系统提示之外的采样细节;CoT 的 SPICE 奖励是 |R−R̄| 的简化实现。
- 论文缓存:papers/llm-agents/simulator-collapse-2608.12253.md 因本机内存 OOM 未生成(全文经 arXiv HTML 版完整读取)。
