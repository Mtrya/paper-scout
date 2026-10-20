# Paper Scout 巡航清单(交付前逐项核对,未完成项标 [ ])

## 运行
- 运行 id: 2026-10-19-whyfar-linearmem-natura-astra
- 覆盖对象: 用户点名制,无论文池筛选——WFLU(arXiv 2605.30161)、S-Space(2610.13527)、HLAR(2609.33093)、Natura-1 博客(2026-10-13)、SenseNova Astra 具身评测(2610.22471)
- 报告: runs/2026-10-19-whyfar-linearmem-natura-astra/report.docxxml
- 飞书文档: https://fudan-nlp.feishu.cn/docx/Jxhkddm2CoQPlYx81bcc7Ty9nRg (v2 修订版,2026-10-20 交付,当前有效;v1 原版 https://fudan-nlp.feishu.cn/docx/TqavdJunyo7nEOxbqkUcRaNVnzd 保持不动)

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。(线程A:同一权重同批题上两探针对撞,L24 的 0.665 与 0.032 共存、VD-EI 官方实现≠论文式;线程B:独立复现发现 write→retain 成立、read 断裂,新增 τ 交叉证伪"hub=最慢头";线程C:协议光谱对质 + 12 维总分不可复算 + 视频逐帧审计)
- [x] 每个深度线程都有建设性的研究动作,或一个精确的障碍说明。(A:Qwen2.5-VL-3B 上复现 WFLU+移植 S-Space 轴识别;B:纯 torch GDN 栈全链复现+τ 交叉;C:分数复算、视频 scene 审计、两点 scaling 换算、RoboRSI 仓库盘点)
- [x] 关键论断有代码、探针、实验、推导支撑。(三线程的脚本与结果均在线程 code/ 下,results 在 code/results/)
- [x] 报告讲清了仅靠重读论文文本无法看出的东西。(VD-EI 口径差 0.463/0.439;HLAR 的 46.9% 与 L20H2 在公开 checkpoint 上不可复现;Natura 16.7 按简单均值为 30.3、权重未公开;柠檬段"长杆"文字与画面不符)

## 报告契约
- [x] 报告可扫读:开篇综述、主题线、图/表/公式齐全。
- [x] `report.docxxml` 中至少有两个图锚点。(19 个,与 figures.json 逐键核对一致)
- [x] 深度线程按逻辑链展开,术语先定义再使用。
- [x] 轻量留意论文与深度线程干净区分。(本轮点名制无短名单,开篇已说明)

## 保存契约
- [x] 持久证据在线程目录中。(三线程 README + code/;结果在 code/results/)
- [x] 面向报告的资产在 `assets/` 中。(19+ 张图,含自绘 12 维对比、视频拼图、论文抠图,均已亲眼过目)
- [x] `code/` 和 `drafts/` 不持有持久工作的唯一副本。(脚本快照已入线程包)

## 发布契约
- [x] 工作区校验器 prepublish 通过。(v2 重修后重跑仍 PASS)
- [x] 报告已发布(分段追加、插图、锚点消除、回读验证)。(v1: 15 段;v2: 17 段全 ok,19 图插入,回读验证 19 图、无锚点残留)
- [x] 用户通知已确认(IM 已发送)。(v2: 默认通知 + 修订说明 IM,messages-search 回读确认)
- [x] `runs/INDEX.md` 已更新。(v2: 2026-10-20 重修条目,沿用 10-12 重修先例)
- [x] 记忆已写。(memories/2026-10-19-memory.md)
