# Paper Scout 巡航清单

## 运行
- 运行 id: 2026-08-14-rift-forewam-simcol
- 覆盖时段: 2026-08-12 至 2026-08-14
- 报告: runs/2026-08-14-rift-forewam-simcol/report.docxxml
- 飞书文档: https://fudan-nlp.feishu.cn/docx/P9EvdAocBodeppxMXwocrPbVnrd

## 修订记录(2026-08-15,用户反馈后)
- 自足性:mode/simulator 身份/distinct-2/self-BLEU-2/CUSUM/点估计与分布保持(操作性定义:损失的 argmin 是不是分布本身)全部随文定义。
- SimCollapse 定理链完整呈现(角色/假设/证明思路),两个 load-bearing assumptions 显式写出。
- WAM 线:Drive-WM 纳入四落点图;补归因缺口(接口×训练信号交叉格空缺);LIBERO 参照质量段(rule-based 100% 事实,筛选纪律)。
- 跨主题观察:ForeWAM 噪声槽从"分布保持"撤下(证据不足);CoT/SPICE 降级为效果层面"防锁定"。
- 飞书文档已同步(block_replace/insert 19 步,验证:新文本 20/20、旧文本 0、图 7 张无损)。

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。
- [x] 每个深度线程都有建设性的研究动作,或一个精确的障碍说明。
- [x] 关键论断有代码、探针、补丁、相关工作、数据样本、推导、产出物支撑,或明确陈述的障碍。
- [x] 报告讲清了这次巡航学到的、仅靠重读论文文本无法看出的东西。

## 报告契约
- [x] 报告可扫读:清晰的开篇综述、有力的分节、有用的图/表/公式/代码片段、流畅的逻辑。
- [x] `report.docxxml` 中至少有两个图锚点。
- [x] 深度线程读起来像研究叙事,而不是填模板的摘要。
- [x] 轻量留意的论文与深度线程被干净地区分开。

## 保存契约
- [x] 持久证据在线程目录中。
- [x] 面向报告的资产在 `assets/` 中。
- [x] `code/` 和 `drafts/` 不持有持久工作的唯一副本。
- [x] 工作区校验器 prepublish 模式通过。

## 发布契约
- [x] 报告已发布。
- [x] 用户通知已确认。
- [x] `runs/INDEX.md` 已更新。
- [x] 工作区校验器 final 模式通过。
