# 线程二：hindsight 什么时候是学习，什么时候只是合理化

## 先拆开三个经常混用的对象

1. **事后解释过去动作**：解释在动作之后产生，不可能是过去动作的原因；只能检验它是否忠实报告了原因。
2. **hindsight 训练或重标注**：用结果修正训练信号，是优化技术，不自动承担解释义务。
3. **反思/凝练经验指导未来动作**：事后文本成为下一轮决策输入；此时可用反事实干预检验它是否真的被消费。

所以问题不是“hindsight 好不好”，而是系统是否把事后的文本伪装成过去的原因，或在没有干预证据时声称它驱动了未来。

## 论文与代码

- [Post-Hoc Reasoning in Chain of Thought（arXiv:2603.01437v2）](https://arxiv.org/abs/2603.01437)，2026-07-23 修订。主论文：在 CoT 前 residual 上训练 answer probe，再沿 probe direction 做 activation steering。
- [Large Language Model Agents Are Not Always Faithful Self-Evolvers（arXiv:2601.22436v3）](https://arxiv.org/abs/2601.22436)，ICML 2026；[官方代码](https://github.com/Dreamcatcher0622/Faithfulness)，审计 commit `cc8c37da02896bb935233dec6990177a5124a923`。它把问题扩展到 agent 的事后经验是否因果影响未来行为。
- [Turpin et al. 2023](https://arxiv.org/abs/2305.04388) 是较早的输入偏置→答案→CoT 合理化证据，本轮用于概念定位，不是深挖对象。

## 主论文留下的因果缺口

论文的 difference-of-means probe 在多数组合上得到大于 0.9 AUC；沿该方向持续 steering 也能翻转答案并诱发 confabulation / non-entailment。v2 限制段明确承认：activation addition 施加于 prompt 后的每个 decoding position，包括最终答案 token；因此一部分效应可能是直接推 final-answer token，而不是修改 CoT 前信念，作者把分离实验留给未来。

这意味着论文稳健地证明了“CoT 前存在可解码的 answer-related direction”，但标题更强的“pre-committed answer”仍需要阶段定位的因果实验。

## 真实权重分阶段干预

脚本 `code/posthoc_phase_probe.py` 在 RTX 4090 上运行真实 `Qwen3-VL-8B-Instruct` 权重（纯文本路径）与官方 BIG-Bench Sports Understanding 数据。240 条随机样本中 238 条答案可解析；前 159 条拟合 difference-of-means direction，后 79 条测 layer AUC。A/B 与 yes/no 映射逐样本随机化。干预集是测试集中 baseline 正确且尽量 yes/no 平衡的 24 条。

### 预测不等于稳健机制定位

最佳层 22 的 held-out AUC 为 0.9737。但在每次打乱训练标签、再从 36 层取最大 AUC 的 200 次零假设中，95 分位是 0.9706，8 次超过实测，经验 `p=(8+1)/(200+1)=0.0448`。方向确实强，但在小测试集和择层后只是刚过门槛，不能只报 0.974。

### 把 steering 按生成阶段切开

固定 `|α|=4`，每条都朝 baseline 相反语义答案 steering；括号内为目标翻转数/24：

- prefill 最后一个 token（1）；整个 prefill（2）；推理最初四个 token state（2）；
- 整段推理、遇到 `FINAL` 即停（14）；只在 `FINAL` 后答案段（3）；
- 论文式全 decoding（17）；等范数逐样本随机正交方向（1）。

整段推理相对正交控制的成对 exact sign test 为 `p=0.000244`；相对 early-4 为 `p=0.000488`。全 decoding 的 17 次中，14 次也由 reasoning-only 翻转；全 decoding 比 reasoning-only 多出的 3 次在这个样本量下不显著（`p=0.25`）。因此全程效应不主要是最后答案 token 的直接偏置，而是在 CoT 生成期间累积形成。

重要限制：整段推理接受的 addition 次数远多于 brief scope，阶段与累计剂量仍有混杂。因此本实验不能反向声称“答案在推理阶段某个瞬间才决定”；它能否定的是更强主张——当前 steering 证据没有单独隔离出 CoT 前因果承诺。brief pre-CoT 干预几乎不动，持续 in-CoT 干预才稳定翻转。

### 合理化在输出里长什么样

错误目标被推入后，模型常保留正确事实，却把结论反过来。例如它先说 Cruyff turn 是足球动作、Samir Nasri 是足球运动员，最后却据此断言句子“不合理”；另一些样本会即时捏造“direct kick 是棒球术语”等假前提。方向不只推答案字母，也在塑造一条为目标结论服务的叙事。这支持“合理化会在生成中形成”，不等于证明目标在生成前已经固定。

证据：`code/evidence/probe_summary.json`、`code/evidence/phase_analysis.json`、`code/evidence/interventions.jsonl`；图为 `../assets/posthoc-phase-localization.png`。

## Self-Evolvers 的一维 faithfulness 会奖励“容易受骗”

该论文把 experience faithfulness 定义为：扰动经验后行为显著变化，说明 agent 忠实使用经验。这个方向对“删除一个确实相关的有效经验”合理，但不能无条件推广到无关或损坏输入。

官方 ExpeL 代码的 condensed `corrupt` 会显式插入 `[ERROR_INFO]`、`[CORRUPTED_n]` 并删词；`irrelevant` 以 80% 概率替换成 “The weather is sunny today.” 等通用句；`filler` 使用 `$$$`、`###` 等占位符。ReasoningBank 复用了同类生成器。这些是明显 OOD 或无意义输入，不是两个都可能成立的、格式匹配的有效反事实经验。模型忽略它们可以是校准良好，而不是“不忠实”。论文还主要用任务成功率承载“behavior change”：行为可能变而成功率不变，或经验被消费但 pretrained prior 已足以成功。

代码位置：

- [ExpeL `expel.py` 693–1085](https://github.com/Dreamcatcher0622/Faithfulness/blob/cc8c37da02896bb935233dec6990177a5124a923/ExpeL/agent/expel.py#L693-L1085)
- [ReasoningBank `load.py` 39–178](https://github.com/Dreamcatcher0622/Faithfulness/blob/cc8c37da02896bb935233dec6990177a5124a923/ReasoningBank/load.py#L39-L178)

## 更好的二维定义

经验忠实性至少要拆成两轴：

- **有效反事实 uptake**：把相关、内部一致、在另一个世界里同样有效的经验 `M` 换成 `M'`，未来动作是否按新经验正确改变；
- **无关输入 robustness**：加入长度与文体匹配但对当前决策无关的经验，动作是否保持稳定。

| | 无关输入稳健 | 无关输入敏感 |
|---|---|---|
| 有效反事实会正确改变动作 | 忠实整合 | 轻信/易受污染 |
| 有效反事实不改变动作 | prior-only 或 memory inert | 不稳定 |

下一套 benchmark 应构造 world-valid memory pairs：同一个当前状态，两个经验分别描述不同但自洽的环境规则，正确动作也随之翻转；另配语义连贯、同长度、但对本状态无关的经验。评分直接用成对动作变化和轨迹差异，而不是只看汇总成功率。这样才区分“听懂过去”与“见什么都跟着变”。

## 初稿阶段保留下来的相邻负结果

同一巡航的初稿曾在真实 Qwen3-VL-8B 与 6 个 AD-MCQ 场景上交换自由文本 `HIGH_LEVEL_DECISION`：删除它会使 3/6 选择改变，但把其速度/方向反向覆写只改变 1/6，且 0/6 被新拉向注入目标。这个小样本不能估准确率，却与本线程的广义判断一致：一段事后或中间文本出现在上下文里，不等于未来行为按其语义受约束。证据与脚本保存在 `code/deft-hld-pilot/`；它作为 revision history 并入本线程，不再作为第二份报告或独立覆盖记录。
