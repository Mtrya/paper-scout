# Not Worth Another Token 线程 — 位置 > 打分器:模糊效用信号下的价值模型

- 论文: Not Worth Another Token: Marginal Value Estimation for Efficient Deep Research Agents (arXiv:2608.08389, UMass Amherst + Adobe Research)
- 代码: 未找到(2026-08-13 核验;论文未附链接)
- 论文全文: papers/llm-agents/marginal-token-value-2608.08389.md

## 线程问题

深研 agent 的上下文随检索迭代膨胀,而证据的边际价值递减。论文在 GPT-Researcher +
DeepResearch-Gym(100 条查询)上做系统性的阶段感知剪枝对比:Pre-Retrieval /
Post-Retrieval / Pre-Synthesis 三个干预点 × 启发式(MMR/GRN/CD/SC/DPP)+ LLM 打分器 +
学习型控制器。

## 核心发现(论文数字)

- **位置 > 打分器**:Post-Retrieval MMR 把 token 从 375.4k 砍到 114.6k(-69.5%)、
  节点 29.0→8.84、时延 -59.7%,质量保持基线的 97.9%;Pre-Synthesis 剪枝只能精修
  最终上下文,救不回上游搜索成本。
- 两阶段组合(Post-Retrieval + Pre-Synthesis)质量-效率最优(CD+SC:+1.64 质量、
  -63.4% token);三阶段 MMR 压到 -73.3% token 但质量掉到 55.90(基线 57.83)。
- **学习型控制器只是"探索性概念验证"**:在少量设置下可行,"does not provide strong
  evidence of consistent superiority over the strongest heuristic baselines";
  论文自评:大部分增益来自"well-matched pruning objective and stage placement"。
- 质量与证据留存分叉:所有剪枝方法的 KPR+KPC 都没超过基线——压缩保住/改善了报告
  质量,但丢弃了证据覆盖(引用召回与质量不是一回事)。

## 与主线的连接(轻线程的产出)

这一期三条线在同一个变量上汇合:**自生成监督信号的可信度由它的清晰度决定**。

- RynnValue(08-12):干净、内容接地的监督(演示数据里的真实时间戳)让学习型价值函数
  成立且防捷径有效。
- NWAT:效用信号是 LLM-judge 的 rubric 分 + 引用召回——慢、噪、判据相关(论文自报
  judge 敏感度分析,绝对分随 judge 漂移)→ 学习型价值模型打不过手写启发式。
- U-OPSD(本期重线程):自我一致性作为伪监督,在多数正确时有效、在系统性错误时放大
  或失明——一致性是"可构造的清晰",不是"正确"。

结论:价值模型的成败首先由监督信号的干净程度决定,其次才是模型容量。NWAT 的价值
在它诚实地报告了这条边界——学习型组件在模糊效用信号下的平庸,不是工程失败,是
信息论意义上的上限。

## 研究动作说明

论文无代码;重建 DeepResearch-Gym + GPT-Researcher 全管线(检索+LLM 判官)的成本远超
轻线程的产出价值。本线程选择:完整精读 + 与 RynnValue/U-OPSD 的对照分析,把判别性
实验的设计写清楚(用什么替换 LLM-judge 才能让学习型价值模型有机会赢——例如用
citation-grounded 的自动核验信号),留给后续巡航。精确障碍:官方实现未放出,重建
管线需要 OpenAI/GPT-Researcher 多步检索基础设施与 100 查询的完整运行,资源与产出
不成比例。

## 报告材料

- NWAT Fig.1(管线图,MinerU 图片 6c690dad...)
- 表 1-3 的浓缩版:基线 375.4k token / Post-Retrieval MMR 114.6k / 学习型控制器
  的平庸数字。
- 监督信号清晰度对照表:RynnValue(时间戳)/ U-OPSD(一致性)/ NWAT(LLM-judge)。
