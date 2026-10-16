# Paper Scout 巡航清单(交付前逐项核对,未完成项标 [ ])

## 运行
- 运行 id: 2026-10-16-greenwam-onewm-xtrace-cortex
- 覆盖时段: 2026-10-13 至 2026-10-16(Hugging Face Papers 逐日池 + 周榜;XTrace 为用户预约论文)
- 报告: runs/2026-10-16-greenwam-onewm-xtrace-cortex/report.docxxml
- 飞书文档: https://fudan-nlp.feishu.cn/docx/KlBmdNghjonODmxJl6WcQ6T3n6e

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。(GreenWAM:79 槽注册表逐条核验+safetensors 头部实测专家塔+双榜单全量重算+真机 Fisher 检验;OneWM:掩码逐 token 穷举+监督量 0.725× 量化+初始化逐 key 量化+玩具归因;XTrace:sm_86 十六次真实拼接+字节级保真修正论文解释+逐地址可行性;Cortex:FFN-only LoRA 玩具探针单独立起"预期查询"设计轴)
- [x] 每个深度线程都有建设性的研究动作,或一个精确的障碍说明。(四线程均有真实研究动作;XTrace 运行期的三层障碍已逐层定位并留下 H100 对照假设;OneWM 的 WAM_mask 臂用自改代码重建)
- [x] 关键论断有代码、探针、实验、推导支撑。(四个线程目录各有 README+脚本+结果;RoboTwin 榜单重排、掩码可见性图、xtrace_sm86 图、cortex 探针图均可重跑)
- [x] 报告讲清了仅靠重读论文文本无法看出的东西。(控制位重排占字节级失真 86%;站点可行性是地址级属性;监督量更少反而赢;动作专家是整塔孪生的实测;预期查询未显著而 rehearsal 内容决定恢复范围)

## 报告契约
- [x] 报告可扫读:开篇综述、三个主题线、图/表/公式/代码块齐全。
- [x] `report.docxxml` 中至少有两个图锚点。(15 个,与 figures.json 逐键核对一致)
- [x] 深度线程按逻辑链组织,术语先定义再使用;三道人工闸全过(完整性闸主代理逐项核对;编辑闸 agent-66 一轮全面检查,6 个必须改+12 个建议改全部落地;首尾闸主代理核对综述与收尾如实)。
- [x] 轻量留意论文与深度线程干净区分。(短名单表 6 篇,行文顺带 TFP/NoteBank/MaLoW 3 篇)

## 保存契约
- [x] 持久证据在线程目录中。(greenwam/onewm/xtrace/cortex 四个线程目录,各含 README.md + code/)
- [x] 面向报告的资产在 `assets/` 中。(16 张图:12 论文图 + 4 自产图,均已亲眼过目,白边全检 <8%)
- [x] `code/` 和 `drafts/` 不持有持久工作的唯一副本。(已清理到只剩 README.md)

## 发布契约
- [x] 工作区校验器 prepublish 通过。
- [x] 报告已发布(11 段全 ok,15 锚点 16 图,锚点段落已删,回读验证 16 图)。
- [x] 用户通知已确认(push_report 以 bot 私信发出,收件 open_id ou_4fb6…,auth status 核实)。
- [x] `runs/INDEX.md` 已更新。
- [x] 记忆已写。(memories/2026-10-16-memory.md)
