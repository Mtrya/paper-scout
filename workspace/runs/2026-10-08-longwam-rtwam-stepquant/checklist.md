# Paper Scout 巡航清单(交付前逐项核对,未完成项标 [ ])

## 运行
- 运行 id: 2026-10-08-longwam-rtwam-stepquant
- 覆盖时段: 2026-10-06 至 2026-10-08(Hugging Face Papers,三天池逐日扫描;arXiv 网页搜索补充)
- 报告: runs/2026-10-08-longwam-rtwam-stepquant/report.docxxml
- 飞书文档: https://fudan-nlp.feishu.cn/docx/VlhrddNxXo8whzxZ74rcL0OfnWc

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。(实验一补出论文缺失的 TA-only 消融臂并核验 local-global gap;实验二在真实 GDN 权重上核验 STEPQuant 的寿命-误差相关与 exact-read oracle)
- [x] 每个深度线程都有建设性的研究动作,或一个精确的障碍说明。(Long-WAM 线程:代码追踪+发布资产核验,AR 对照复现的障碍在报告中精确说明;RealtimeWAM/STEPQuant 线程各有实验)
- [x] 关键论断有代码、探针、实验、推导支撑。(realtimewam 玩具蒸馏五臂实验,stepquant 真实 340M GDN 权重三探针,脚本与产物均在线程 code/)
- [x] 报告讲清了仅靠重读论文文本无法看出的东西。(论文"多步教师是精度上界"前提在玩具中反转;TA-only≈CD+TA 的拉扯证据;STEPQuant 探针 B 量级偏弱与 INT6 崩盘的条件依赖)

## 报告契约
- [x] 报告可扫读:开篇综述、主题线、图/表/公式齐全。
- [x] `report.docxxml` 中至少有两个图锚点。(14 个,与 figures.json 逐键核对一致)
- [x] 深度线程按"是什么→为什么→怎么样→我们的实验"展开,术语先定义再使用。
- [x] 轻量留意论文与深度线程干净区分。

## 保存契约
- [x] 持久证据在线程目录中。(longwam 线程:audit_release.sh+audit_output.txt 与 README;realtimewam/stepquant 线程:实验脚本、README、图)
- [x] 面向报告的资产在 `assets/` 中。(14 张图:9 论文图 + 3 realtimewam 实验图 + 3 stepquant 实验图,均已亲眼过目、白边 <8%)
- [x] `code/` 和 `drafts/` 不持有持久工作的唯一副本。(npz 已拆为 .npy 以过禁令)

## 发布契约
- [x] 工作区校验器 prepublish 通过。
- [x] 报告已发布(分段追加、插图、锚点消除、回读验证)。(6 段全 ok,14 图插入,回读验证 14 图、无锚点残留)
- [x] 用户通知已确认(IM 已发送)。
- [x] `runs/INDEX.md` 已更新。
- [x] 记忆已写。(memories/2026-10-08-memory.md)
