# Paper Scout 巡航清单(交付前逐项核对,未完成项标 [ ])

## 运行
- 运行 id: 2026-10-12-cyfa-mmflow-scorecenter-dreamtrue-art
- 覆盖时段: 2026-10-09 至 2026-10-12(Hugging Face Papers 逐日池 + 周榜;用户点名三篇 9 月论文 CyFA/MMF/SC 并入)
- 报告: runs/2026-10-12-cyfa-mmflow-scorecenter-dreamtrue-art/report.docxxml
- 飞书文档: https://fudan-nlp.feishu.cn/docx/IlJNdkts7oIWyfxfjtxcwT9DnFh

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。(CyFA 回绕安全性部分证伪+时钟内容自适应+状态低秩;MMF 文本潜 round-trip 通道几乎无损、瓶颈不在 decoder;SC 漂移=向采样器蒸馏的 toy 精确验证与诊断陷阱;DiffusionART 聚合残差 6.8% 的温和修正+β 空操作发现;DreamTrue 标定发布物独立复算)
- [x] 每个深度线程都有建设性的研究动作,或一个精确的障碍说明。(五线程各有真实研究动作:真实权重探针/官方组件独立实验×2/玩具精确验证×2/发布资产复算;DreamTrue 的 IoU 声明与全量标定包缺环已精确说明)
- [x] 关键论断有代码、探针、实验、推导支撑。(五个线程目录各有 README+脚本+结果 JSON+图;复算与探针数字均可重跑)
- [x] 报告讲清了仅靠重读论文文本无法看出的东西。(时钟内容自适应、回绕残迹量化、round-trip 容忍预算、漂移-蒸馏夹角 +1.000000、β 空操作、恢复力 4.9× 未复现、AgiBot 非方像素解释、DROID euler 约定勘误)

## 报告契约
- [x] 报告可扫读:开篇综述、三条主题线、图/表/公式齐全。
- [x] `report.docxxml` 中至少有两个图锚点。(24 个,与 figures.json 逐键核对一致)
- [x] 深度线程按"是什么→为什么→怎么样→我们的实验"展开,术语先定义再使用。
- [x] 轻量留意论文与深度线程干净区分。(短名单表 9 篇 + 行文顺带 3 篇)

## 保存契约
- [x] 持久证据在线程目录中。(cyfa/score-centering/diffusionart/multimodal-flow/dreamtrue 五个线程目录,各含 README.md + code/)
- [x] 面向报告的资产在 `assets/` 中。(29 张图:15 论文图 + 14 实验/复算图,均已亲眼过目)
- [x] `code/` 和 `drafts/` 不持有持久工作的唯一副本。(线程脚本与结果均已入 run 包;cyfa-probe venv 等临时物留在 code/)

## 发布契约
- [x] 工作区校验器 prepublish 通过。
- [x] 报告已发布(11 段全 ok,24 锚点 29 图插入,锚点段落 14 个已删,回读验证 29 图)。
- [x] 用户通知已确认(IM 已发送,om_x100b63fe760a90b4b044d173a343374;注意 bot 私信 open_id 已轮换为 ou_4fb6…,旧 ou_9a0c… 失效)。
- [x] `runs/INDEX.md` 已更新。
- [x] 记忆已写。(memories/2026-10-12-memory.md + MEMORY.md 标定战线一行)
