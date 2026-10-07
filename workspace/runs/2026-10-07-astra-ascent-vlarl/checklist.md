# Paper Scout 巡航清单(交付前逐项核对,未完成项标 [ ])

## 运行
- 运行 id: 2026-10-07-astra-ascent-vlarl
- 覆盖时段: 2026-09-23 至 2026-10-06(Hugging Face Papers,616 篇去重)
- 报告: runs/2026-10-07-astra-ascent-vlarl/report.docxxml
- 飞书文档: https://fudan-nlp.feishu.cn/docx/Aq96db6SmoxASlxDTDrcuhxSnWe

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。(实验 A 测了论文没有扫过的先验质量网格并证伪自有猜想"危险中间态";实验 B 同场检验两篇 TTT 论文的结合断言 B3;实验 C 在玩具规模复测低秩机制链 C1-C5)
- [x] 每个深度线程都有建设性的研究动作,或一个精确的障碍说明。(三线程各有实验;ASCENT 官方仓库为空壳已在报告中注明)
- [x] 关键论断有代码、探针、实验、推导支撑。(36 格真实 astra 决策 + 逐格 cells.jsonl;TTT 臂对照 JSONL;玩具复测谱分析;预测预登记于 assets/PREDICTIONS.md)
- [x] 报告讲清了仅靠重读论文文本无法看出的东西。(先验质量的接受结构、审查粒度在内容层而非信任层、门控与 hindsight 目标的可分离性、玩具规模低秩复测的边界)

## 报告契约
- [x] 报告可扫读:开篇综述、三条主题线、图/表/公式齐全。
- [x] `report.docxxml` 中至少有两个图锚点。(12 个,推送后须全部消除)
- [x] 深度线程按"是什么→为什么→怎么样→我们的实验"展开,术语先定义再使用。
- [x] 轻量留意论文与深度线程干净区分。(7 篇短名单独立成节)

## 保存契约
- [x] 持久证据在线程目录中。(astra-hybrid/ 含 README+code(附录含代表 trace);lowrank-flow/ 含 README+code(含 analysis/);ttt-arms/ 含 README+code(含 analysis/),六臂全跑完)
- [x] 面向报告的资产在 `assets/` 中。(10 张论文图 + 2 张实验 A 图 + 3 张实验 C 图 + 4 张实验 B 图 + figures.json + PREDICTIONS.md)
- [x] `code/` 和 `drafts/` 不持有持久工作的唯一副本。(三个实验线程的代码与结果均已复制进线程目录)

## 发布契约
- [x] 工作区校验器 prepublish 通过。
- [x] 报告已发布(分段追加、插图、锚点消除、回读验证)。(8/8 段,19/19 图,锚点无残留,img 数回读一致)
- [x] 用户通知已确认(IM 已发送)。(第一次用旧 open_id 失败,换缓存的当前值 ou_4fb6… 后送达,om_x100b6357f341b4a0b4c641eb6729c5a)
- [x] `runs/INDEX.md` 已更新。
- [x] 记忆已写。(memories/2026-10-07-memory.md)
