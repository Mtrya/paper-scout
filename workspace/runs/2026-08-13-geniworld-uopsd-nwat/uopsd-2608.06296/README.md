# U-OPSD 线程 — 一致性作为自生成监督:错误共识下的训练动态

- 论文: On-Policy Self-Distillation without Any Supervision (arXiv:2608.06296, UCSD + Gatech + UMD + ByteDance)
- 代码: github.com/williamium3000/u-opsd — "Code will be available soon"(2026-08-13 核验,未放出)
- 论文全文: papers/llm-agents/onpolicy-selfdistill-2608.06296.md

## 线程问题

08-08 巡航确立的战线:**权威状态的可信度——一致性可以被构造,正确性不能,二者必须分开
测量**。U-OPSD 把这条战线推到了极端:它用模型自身的多数投票共识当"伪金标",沿模型自己
的不一致 rollout 做前向 KL 蒸馏(Qwen3 4B/8B,非思考模式 +8.5/+10.7%,宣称超过用真值监督
的 OPSD 与 GRPO)。论文 Limitations 自认:"we did not measure training dynamics under
deliberately corrupted votes"——错误共识下的训练动态正是未检验的缺口。

## 方法核验(从论文重建,官方代码未放出)

每个 prompt:采样 G=8 rollout(冻结教师 = 初始策略)→ 提取 \boxed{} 答案取多数 →
一致性 c(x) ≥ τ(0.5) 且存在不一致 rollout 时,选最长一致 rollout y+ 作教师参照,
对每个不一致 rollout y- 最小化前向 KL:KL(π̄(·|x, y+, y-<t) || π_θ(·|x, y-<t))。
LoRA r=64, lr 5e-6, 150 步,全词表 logit 蒸馏。教师冻结 → 同一 prompt 的 rollout
每步相同,训练实质是对静态自采样数据集的重复蒸馏。

伪标金质量(论文自报):96.3% rollout 可解析,94% prompt 过阈值,86.7% 伪标金匹配
真值 → 13.3% 错误伪标金。逐类命运由机制决定:
- maj_correct(多数对):向正确参照收敛 → 预期提升;
- split_wrong(多数错但不一致):**沿不一致 rollout 蒸馏 = 朝错误参照收敛 → 预期放大**
  错误的一致性;
- unanimous_wrong(全体一致错):V- 为空 → 被跳过,无训练信号;
- low_signal(<2 可解析):被阈值跳过。

## 研究动作:错误共识下的训练动态探针(远端 4090)

`code/uopsd_probe.py`(prep/train/metrics)+ `vllm_rollout.py`(采样)+
`compute_metrics.py` / `plot_perclass.py`(分析)。Qwen3-4B 非思考模式,
MATH-500 前 200 题训练 + 后 100 题 held-out;G=8 vLLM 采样(MATH prompt 无
\boxed 指令,解析率显著低于论文的 96.3%)。类分布(训练集):maj_correct 131 /
split_wrong 12 / unanimous_wrong 6 / low_signal 51。

## 结果(150 步,38 条蒸馏样本:35 来自多数正确、3 来自 split_wrong)

整体:train pass@1 0.485→0.515、maj@8 0.670→0.650;held pass@1 0.400→0.510、
maj@8 0.630→0.650——巩固红利真实且泛化,但 200 题规模下远不到论文的 +8.5%。

逐类命运(机制三侧,证据在 `code/results/final_metrics.json`):
1. **巩固侧成立**:maj_correct train pass@1 0.740→0.797(+5.7),唯一答案数
   1.40→1.16;held 同向(unique 1.55→1.30)。
2. **失明侧成立**:unanimous_wrong 训练前后一字不变(maj8 0.000、一致度
   1.000、unique 1.00);low_signal 的可解析率不升反微降。
3. **放大通道窄但存在**:split_wrong 在训练集几乎无信号(τ=0.5 把共识 <0.5
   的 10/12 题全部跳过,蒸馏集仅 3 条来自 split_wrong,wrong_agree 0.575→0.577);
   但 held-out 上出现放大签名:split_wrong 错误一致度 0.589→0.701(+0.11,
   n=9)——蒸馏出的"跟随自身共识"行为泛化到未见题,连失败模式一起。

## 结论(已写入报告)

U-OPSD 是"把已有能力磨利"的机器,不是"纠错"的机器:巩固(可泛化)+ 失明
(系统性错误原样保留)+ 一条被 τ 阈值挡住大半、但通过行为泛化泄漏的放大通道。
与 GRPO 的差别不在平均成绩,在错误的命运。08-08 判断("一致性可构造、正确性
不可")在本线程获得逐类实证。

## 局限与不确定

- 探针规模:200+100 题、4B 档、单一数据集;论文的 30k 提示/8B/思考模式未复现。
- 蒸馏目标的重建与 OPSD 官方配方可能存在细节偏差(per-token pointwise clipping
  未实现);对机制性结论(逐类命运)影响有限,对绝对数字影响更大。
- 反向 KL 发散(论文表 5)未复现。
