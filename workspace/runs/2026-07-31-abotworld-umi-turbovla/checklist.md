# Paper Scout 巡航清单

## 运行
- 运行 id: 2026-07-31-abotworld-umi-turbovla
- 覆盖时段: 2026-07-18 to 2026-07-30
- 报告: runs/2026-07-31-abotworld-umi-turbovla/report.docxxml
- 飞书文档: https://qcn0umnxrmj2.feishu.cn/docx/B10IdsJp0okcqXxwei2cPQlGnkg (bot 身份创建,已授权用户 full_access;4 图已上传替换锚点)

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。(三个真实验的发现开篇即给出:episodic collapse、recovery 盲区、staircase 雷区)
- [x] 每个深度线程都有建设性的研究动作,或一个精确的障碍说明。(ABot:4090 实跑 640-block 可控性实验;UMI:全量数据 recovery 运动学测量;πR²:可运行玩具复现 + TurboVLA 代码/权重核验)
- [x] 关键论断有代码、探针、补丁、相关工作、数据样本、推导、产出物支撑,或明确陈述的障碍。(三线程 code/ 目录均有可重跑脚本与原始数据)
- [x] 报告讲清了这次巡航学到的、仅靠重读论文文本无法看出的东西。(可控性崩溃模式、90.2% 零反转、staircase 实现三雷区、口径三问)

## 报告契约
- [x] 报告可扫读:清晰的开篇综述、有力的分节、有用的图/表/公式/代码片段、流畅的逻辑。
- [x] `report.docxxml` 中至少有两个图锚点。(4 个)
- [x] 深度线程读起来像研究叙事,而不是填模板的摘要。
- [x] 轻量留意的论文与深度线程被干净地区分开。(独立 h1 表格,arXiv API 核实标题)

## 保存契约
- [x] 持久证据在线程目录中。
- [x] 面向报告的资产在 `assets/` 中。
- [x] `code/` 和 `drafts/` 不持有持久工作的唯一副本。
- [x] 工作区校验器 prepublish 模式通过。

## 发布契约
- [x] 报告已发布(lark-cli 创建飞书文档,4 图上传替换锚点)。
- [x] 用户通知已确认(飞书 URL + 本地路径均已告知)。
- [x] `runs/INDEX.md` 已更新。
- [x] 工作区校验器 final 模式通过。
