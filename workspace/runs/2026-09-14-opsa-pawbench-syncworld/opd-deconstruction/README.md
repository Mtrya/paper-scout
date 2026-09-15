# OPD 解构线程(2608.31046 + 2609.04172)

两篇论文从不同端解构 on-policy distillation(OPD),与本实例的"监督信号三轴"框架直接咬合。

## 论文核心主张

**2608.31046 "Does OPD Really Distill?"(Purdue)**:
- 教师对 student 采样的前缀打分本质是 off-policy 打分;噪声率(符号与可验证正确性相悖)随教师规模增长:4B 30.6% → 30B-A3B 34.7% → 235B-A22B 50.6%;最大教师 97.8% 的正确答案 token 拿到负优势。
- 学生对该噪声不敏感:只用噪声轨迹/剔除噪声轨迹/标准 OPD 收敛到同一水平。
- 收益来源:低 logp token(20% 最低)上做负优势;固定负优势(-0.5)≈ OPD;固定正优势(+0.2)40 步内坍缩。
- OPSA:无监督,最低 20% logp token 上施熵自适应负优势 A=-1/2-(H-H_min)/(2(H_max-H_min))。Qwen3-1.7B AIME24 avg@32 13.44→48.85(OPD 32.08);9B +11.46;OOD 小(MBPP+ +1.2, GPQA +4.48)。
- 机制:高熵 fork 位抑制尾 token、在头 token 间再分配质量;mask 掉 reflective fork(wait/but/however…)增益几乎消失。

**2609.04172 "One Training Example"(Thinking-Space, Rethinking OPD 系列 Part II)**:
- 单 query OPD 恢复 full-data OPD 收益的 ~87%(300 步);状态覆盖率 71.5%(1 query)/98.9%(16 query);16 个语义多样 query ≈ full-data。
- 吸收率(absorption rate)随时间下降的方式与数据量无关——"data-overfed but algorithm-starved"。
- 内容无关性:空模板 <think>、WildChat 闲聊 prompt 也接近真 query 基线;query 只是状态生成器。
- 与 one-shot RLVR 对比:同一 query 下 OPD 增益 >2× RLVR(RLVR 一旦解出该题就耗尽信号)。

## 我们的研究动作

- **A1 噪声率独立复测**(远程 4090,真实权重):Qwen3-1.7B 学生 × {Qwen3-4B-Instruct, Qwen3-14B-Instruct} 教师,120 题 DAPO-17k,每题循环采样直到同时拿到 ≥1 条正确与 ≥1 条错误(每轮 4 条新样本、最多 5 轮、命中即停),按 \boxed{} 答案 token 区间算"教师 − 学生"平均优势的符号,共 505 条响应、367 条可用(93 正确 / 274 错误)。**结果**:4B 教师合计噪声 31.3%(正确答案被否定 41.9%、错误答案被肯定 27.7%)——论文同口径 4B 报 30.6%,几乎逐点复现;14B 教师合计 27.2%,但结构翻转:正确答案被否定率升到 66.7%、错误答案被肯定率降到 13.9%(两类拉平 34.8% → 40.3%),复现"教师越大、否定越不分对错"的实质。证据:`code/evidence/a1_noise.json`、`code/evidence/a1_score.log`;图 `assets/plot_a1_noise.png`。
- **A2 机制训练**(0.6B 学生 × 1.7B 教师,自研轻量 PG 环,无 Megatron/SGLang):五条件 opd / opd-oneshot / fixed-neg / opsa / fixed-pos,各 40 步(固定正 25),batch 12,max_new 2048;每 20 步在 MATH-500 子集前 20 题上评测(每题至多 4 次采样、命中即停——"4 次内解出率"口径)。检验:固定负 ≈ OPD、OPSA 最优、固定正坍缩、单 query OPD 恢复大部分收益。token 选择与优势口径按官方 slime 实现:跨 batch 全局取最低 20% logp token,OPSA 的 A = −1/2 − (H−H_min)/(2(H_max−H_min))。**动态结果**(前 3 步 → 后 5 步平均,全部 token 上平均 |A|):opd 长度 1575→1616、熵 1.97→1.53、|A| 3.54→2.18;opd-oneshot 1811→1906、1.32→0.92、2.00→0.96;fixed-neg 1610→1485、2.08→1.54、|A| 恒 0.10;fixed-pos 1557→1682、2.05→2.37、恒 0.04(未复现论文的 40 步内坍缩);opsa 1507→1631、1.92→1.62、0.17→0.15。教师幅度是固定负优势的约 20 倍而动态几乎重合。**终态精度**(MATH-500 前 50 题 × 每题 4 条,六臂 + 未训练 base 同协议):**六臂 pass@4 全部 = 6.0%(3/50,且是同一批三道题),avg@4 3.5%–5.5%(base 7/200 条成功、opd 11、单 query 8、固定负 9、固定正 7、OPSA 10)**;按 MATH 难度 1–2 级(17 题)/3–5 级(33 题)切开也无分离。地板是模型真实能力而非协议:同一批 15 题在评测配置 / 训练采样配置 / 4096 token 三种配置下未训基座都是 1/15、格式合规率 100%。证据:`results/a2_*_final.json`、`a2_breakdown.json`、`a2_eval_probe.log`、图 `assets/plot_a2_curves.png`。
- **A3 发布权重核验**:Tuwhy/Qwen3-1.7B-OPSA vs Qwen3-1.7B base,AIME24(官方 zhuzilin/aime-2024 评测文件),非思考模式(与 checkpoint README 一致),论文解码(temp 0.7 / top-k 20 / top-p 0.8);每题 4 次采样**批量生成、不提前停止**(同一 prompt 的 4 条一次生成,单条逐次生成只有 ~33 tok/s,撞上限时每题要 30+ 分钟),同时报 pass@4(4 次内至少一次正确)与 avg@4(4 次平均正确率——与论文 avg@32 同族);校验用从官方 slime 仓库 vendored 的 grade_answer_verl(boxed 抽取 + mathd 归一 / sympy 等价)。偏差:样本数 4 vs 论文 32、max_new 16384 vs 32768——绝对数值不可与论文直接比,只比两个模型之间的差距。**结果(30/30)**:base pass@4 = 20.0% / avg@4 = 10.0%(论文 avg@32 13.44 / pass@32 40.00);发布 OPSA 权重 pass@4 = 40.0% / avg@4 = 24.2%;pass@4 翻倍、avg@4 +14.2pp。**诊断**:OPSA 有 104/120 = 86.7% 的样本撞 16384 上限(base 16.7%),自然结束的 16 条样本全部正确(16/16),截断样本正确率 12.5%;论文表 6 自己给出同一事实的另一面(base 平均 4457 token / OPSA 23205 token)。证据:`results/a3_{base,opsa}.json`、`a3_report_stats.py`,图 `assets/plot_a3_bar.png`。
- **A3-ext 双倍预算重测**:对截断最重且未解的 6 题(0,4,6,7,8,10)用 32768 预算、每题 4 条重跑,并存文本做定性检查(`a3_eval.py --save-text`、`a3_ext_compare.py`、`a3_ext_positions.py`)。**结果(交付时完成 5 题)**:0,4,6,7,8 → 原本 0/20 条正确,双倍预算下 3/20(第 0 题 0/4→2/4、第 6 题 0/4→1/4;第 4、7、8 题仍 0/4)。三种失败形态:12/20 撞满 32768(其中 6 条连一个 \boxed{} 都没写出)、第 4 题四条自然结束在 23189 token 但答案全错(240/240/100/240,正确是 80)、第 8 题四条自然结束在 31168 token 全错;所有答对的样本,答案都落在生成文本的最末端(≈100% 位置)。证据:`results/a3_opsa_ext.json`、`results/a3_opsa_ext.log`。
- **A2b(报告内推理)**:自推导"教师优势 ≈ 学生自身不确定度的噪声重标度"机制解释。

## 实现事故与修复(影响结果可信度的都记在这里)

1. **左填充 + 未填充长度切片**(影响 A1/A2 全部响应采集):`pad_left` 把较短的行左侧补齐,但生成后的响应切片用 `g[len(pids):]`(未填充长度)取,导致批次里非最长 prompt 的响应前带一段 prompt 尾部(实测 376/505 条、平均 185 字符);A1 打分阶段按"最长 prompt 尾匹配"剥离(≥8 字符,均 8~1500 上限),剥离不改变 \boxed{} 答案 token 的位置。A2 的 rollout 与 batch_forward 有同源问题(rollout 响应带前缀;batch_forward 用 `Lp-1` 定位而非 pad-aware 起点),**已修复并从头重跑五条件**(旧的五条件结果移到 `results/a2_prerun_bug/` 留档,不用)。
2. **打分阶段两个内存事故**(A1):(a) 三个模型同时驻留 GPU(1.7B+4B+14B ≈ 39GB 权重)在 48GB 卡上 CUDA OOM;(b) 每条响应保留整行词表 logits(`(Lr, V)` float32 ≈ 1.2GB × 505 条)把容器 100GB 内存配额打爆,被 cgroup OOM killer 杀掉。修复:学生 → 释放 → 逐教师串行打标,只留采样 token 上的 logp,并把 token 级分数缓存到 `results/a1_scores.pt`(聚合出错无需重打分)。
3. **官方 grader 替换**:A3 初版用自写 verify;发现官方 slime 评测用 `grade_answer_verl`(boxed 抽取 + mathd 归一 / sympy 等价,需要 `pylatexenc`)后,改为直接 vendor 官方 `math_utils.py`,与论文评测管线一致。

## 代码与发布状态(审计)

- DripNowhy/On-Policy-Self-Adaptation(48★):slime(0.2.4)+Megatron+SGLang 全量训练栈,`--preset opsa/fixed-negative/fixed-positive` 条件齐备;W&B 公开;数据 zhuzilin/dapo-math-17k(17,398 行)+ aime-2024(30 行)。checkpoints:Tuwhy/Qwen3-{1.7B,4B,3.5-9B}-OPSA、Olmo-3-7B-{Instruct,Think}-OPSA(09-14 当日还发布了新模型)。
- Thinking-Space/One-Shot-OPD(65★):veRL 0.9.0+vllm 0.28/torch 2.13/transformers 5.10,需 8-GPU;我方 4090(驱动 cu128)跑不了 vLLM,故训练复现改自研环。放出 Qwen3-1.7B-Base-OPD、Qwen3-4B-Base-GRPO、Qwen3-1.7B-SFT。
- Part I 是 arXiv 2604.13016。

## 关键文件(远端)

- workroot `opd-pawb/`:a1_noise.py、a2_train.py、a3_eval.py、verify.py;models/{Qwen3-0.6B,Qwen3-1.7B,Qwen3-4B-Instruct,Qwen3-14B-Instruct,Qwen3-1.7B-OPSA};data/{dapo-math-17k,aime-2024,math500-subset}.jsonl;results/ 各条件 log.json 与 ckpt_*.pt。
- 本地:code/opd-exp/*(脚本源)、code/opsa、code/oneshot-opd(审计 clone)。
