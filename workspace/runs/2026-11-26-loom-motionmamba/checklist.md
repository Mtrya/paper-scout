# Paper Scout 巡航清单

## 运行
- 运行 id: 2026-11-26-loom-motionmamba
- 覆盖时段: 2026-11-20 至 2026-11-26（HF Papers 池 11-20/11-23/11-24/11-25，共 187 篇去重后；11-21/22 为周末空档，11-26 榜单未出；博客 RSS 十三路，17 条入池）
- 报告: runs/2026-11-26-loom-motionmamba/report.docxxml
- 飞书文档: https://fudan-nlp.feishu.cn/docx/TBXvdE1tkoUGpZxhP5Oc0NeYnCc

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见，而不是论文内容的重组。
- [x] 每个深度线程都有建设性的研究动作，或一个精确的障碍说明。（Loom：6 层玩具栈机制探针 + JAX/PyTorch 双实现审计；MM2：官方 PCM 类 CPU 核验 + Mamba-3 递推纯 torch 复推 + 仓库审计）
- [x] 关键论断有代码、探针、补丁、相关工作、数据样本、推导、产出物支撑，或明确陈述的障碍。
- [x] 报告讲清了这次巡航学到的、仅靠重读论文文本无法看出的东西。（写门三实现恒等性的裁决；重连容量/秩界的玩具核验；PCM 文本-门弱调制的实测；四元组状态对精确续扫的必要性）

## 报告契约
- [x] 报告可扫读：清晰的开篇综述、有力的分节、有用的图/表/公式/代码片段、流畅的逻辑；读者读完后能复述每条深度线程的核心判断。（冷读者裁决：能）
- [x] `report.docxxml` 中至少有两个图锚点。（4 个：loom-attention:architecture、loom-attention:probe-rank、motion-mamba2:overview、motion-mamba2:realworld）
- [x] 全文提纲与各线程提纲范围匹配；提纲标明线程开场锚点，每份简报含白板底稿（矛盾、切口、判定边界）、叙事提纲、衔接契约、图示计划与硬约束，写手只获得其职责所需的提纲与材料。（drafts/report-outline.md、drafts/briefs/*.md）
- [x] 每个深度线程先给具体问题/失败案例再引入符号与前作，机制从剥离后的简化情形讲到真实系统。
- [x] 写作与冷读已显式选用文字能力与结构判断俱佳的模型；无模型选择能力时的回退已记录。（写手：gemini-4-hydrogen；审校与冷读原定 gemini-4-hydrogen，因 zenmux 订阅配额耗尽，回退 fnlp/nex-n2.6 并已记录于此）
- [x] 深度线程读起来像同事在白板前讲透前因后果，而不是填模板或堆砌事实。
- [x] 轻量留意的论文与深度线程被干净地区分开。（短名单 6 条，均不与深挖线程重复）
- [x] 事实与深度锐评关、冷读者顺序首读关全部通过，冷读仅基于正文与图注、能分辨论文结论与自得判断、结构性修订后已重读。（第一关：修改后发布，17 处修订已全部应用；第二关：能——局部 clarifications 非结构性修订，未触发重读条款）

## 保存契约
- [x] 持久证据在线程目录中。
- [x] 面向报告的资产在 `assets/` 中。
- [x] `code/` 和 `drafts/` 不持有持久工作的唯一副本。（探针脚本与输出已晋升进线程包；图片已入 assets；提纲/简报/片段为过程稿，留在 drafts 可覆盖）
- [x] 工作区校验器 prepublish 模式通过。

## 发布契约
- [x] 报告已发布。（https://fudan-nlp.feishu.cn/docx/TBXvdE1tkoUGpZxhP5Oc0NeYnCc）
- [x] 用户通知已送达，或无 `lark-cli` 时已走仅本地兜底。（push_report.py IM 已发送）
- [x] `runs/INDEX.md` 已更新。
- [x] 工作区校验器 final 模式通过。

## 附注
- 短名单超纲细节复核（第一关要求）：187 篇=本轮去重池实测数；INTACT 2.41×/DynaGoal/SAFI 机制/WorldLeague 三任务名均出自当日抓取的 arXiv 摘要原文，核验一致，保留。
- 判官门等历史运行包与文档保持原样，本轮未触碰。
