# WorldExam（arXiv:2608.02603）— 线程记录

## 这条线做了什么

论文把 controllable video world model 的诊断分成 Visual Quality、Control
Adherence、Spatial Consistency 与 World Reactivity，重点检查输入未明说的场景后果。
本线程全文审读结果表、judge-human 对齐、接口适配与限制，并核验项目页/GitHub 的
发布范围。

## 关键发现

- 1,474 cases、8 tasks、20 models；camera/action/language 三种接口分轨评价。
- language-driven 一般视觉均分只跨 79.64–81.04，任务均分跨 39.85–65.02；
  appearance 指标无法替代后果任务。
- WorldPlay / LingBot-World 的 Subject Control 为 49.75 / 55.47，Physical
  Reaction 仅 26.91 / 33.43：能控制主体，不代表世界会响应。
- checklist judge 在 800 实例、5,793 条目上总体 Spearman 0.8614；Social
  Interaction 仅 0.7019，是当前 judge 最弱任务。
- 三条接口输入带宽不等价：language track 有详细场景描述且闭源系统可能有 prompt
  enhancement；排名测端到端接口，不能据此定位内部因果表征。
- 截至 2026-08-05，case、评测代码与模型输出仍未公开；结果表无法外部复跑。

## 线程产物

- 报告图：`../assets/worldexam-overview.jpg`（论文 Figure 1，已人工检查）。
- 无自研代码；精确障碍是 benchmark 资产尚未发布。
