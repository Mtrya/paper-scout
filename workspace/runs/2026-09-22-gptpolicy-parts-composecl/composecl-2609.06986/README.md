# composecl-2609.06986 线程:ComposeCL 精读 + 玩具尺度机制解剖

锚点论文 ComposeCL(2609.06986,百任务长程记忆的 2⁴ 因子实验)与本巡航实验三的证据包。

- 论文缓存:`papers/llm-agents/compose-cl-2609.06986.md`
- 官方参考实现:`code/compose-cl/`(工作区,本玩具不依赖不修改)
- 玩具实现:`code/toy-compose-cl/`(data/model/mechanisms/run/plots;CPU,`.venv` 未入库;运行方式与全部偏差依据见其 README)
- 结果:`code/toy-compose-cl/results/`(12 个 run JSON + aggregates.json + summary.md + 5 张图 + results_full.log 完整训练日志)

核心结果(2 层 transformer d=128、20 任务 × 100 条 key→value、LoRA r=8、6 臂 × 2 种子、CPU ~90 分钟):

- **replay×merge 超可加复现**:replay 单独 +1.10pp、merge 单独 +0.15pp,之和 +1.25pp,组合 +4.40pp(交互项 +3.15,3.5 倍;论文 Symbol-QA 为 4.0 倍);最终保持率 D 9.2±0.1% vs A 4.9±0.2%
- **SI×merge 负交互复现**:E−B = −1.7pp,F−D = −3.5pp;SI 的即时习得 90.6%,唯一明显低于满分的臂
- **机制(i) merge 保护什么**:同一 replay 锚下换成 merged LoRA,每记忆年龄存活率都上升(age1 0.23→0.49,age2 0.09→0.23,age3 0.05→0.14)——旧记忆搬出可训练坐标,新任务干扰只剩新适配器一条窄通道;单独 merge 几乎无用,是超可加性的来源
- **机制(ii) replay 是拉力不是噪声**:cos(∇L_replay, ∇L_SFT) = −0.057(92% 步为负),replay 梯度自一致性 +0.171(当前任务的 5 倍),范数比 0.28
- **机制(iii) SI 坐标错配直证**:任务初始惩罚 0.05→3.1(等价于把上一任务更新再施加一遍);旧 Ω 与新坐标重要性相关系数仍 0.78–0.88(排序有效、语义已变);‖θ_init−θ*_old‖² 持续增长
- **诊断插曲(重要边界)**:replay 的上游是教师的生成分布——base 预训练于 key 无上限随机文本时回放样本命中真实 key 为 0,replay 完全失效;换固定 key 池后 60–100% 命中才起效
- 边界:半衰期延伸远比论文温和(玩具 ≈1 vs 0.5 任务,论文 19 vs 1);SI 臂种子间方差大(1.9% vs 4.7%)
