# Paper Scout 巡航清单

## 运行
- 运行 id: 2026-11-03-wam-readout
- 覆盖时段: 2026-10-29 至 2026-11-03
- 报告: runs/2026-11-03-wam-readout/report.docxxml
- 飞书文档: https://fudan-nlp.feishu.cn/docx/UMDPdhA0vovI4wxc3ZYcJFIQnud
- 额外线程: Democratizing Robot-Use Agents (2611.01560),调度器微基准

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。
- [x] 每个深度线程都有建设性的研究动作,或一个精确的障碍说明。
- [x] 关键论断有代码、探针、补丁、相关工作、数据样本、推导、产出物支撑,或明确陈述的障碍。
- [x] 报告讲清了这次巡航学到的、仅靠重读论文文本无法看出的东西。

## 报告契约
- [x] 报告可扫读:清晰的开篇综述、有力的分节、有用的图/表/公式/代码片段、流畅的逻辑。
- [x] `report.docxxml` 中至少有两个图锚点(实际 6 个)。
- [x] 深度线程读起来像研究叙事,而不是填模板的摘要。
- [x] 轻量留意的论文与深度线程被干净地区分开。
- [x] 审校闸门(完整性/编辑/首尾/冷读者)全部通过,整改简报闭环;逐句受益者测试与正面改写完成。
- [x] 正文、表格与图注的叙述否定式语句比例 ≤ 10%(verify_run prepublish 通过)。

## 保存契约
- [x] 持久证据在线程目录中(`worldbridge-causalwam/`、`probing-spatial-state/`、`rua-2611.01560/`)。
- [x] 面向报告的资产在 `assets/` 中(6 张图 + figures.json)。

## 边界与障碍
- 启智 notebook IDE 仍报 JupyterTerminal 不可达;CPU/SSH 路径可用,GPU 交互走不通,OSWAM 探针改用 GPU job。
- OSWAM 的 5B 底座与约 12GB checkpoint 超出本机 12GB 显存;checkpoint 已在本机完整下载并通过 SHA-256,正在准备远端探针;若未跑完,报告只按静态审读记录。
- CausalWAM 的 Cosmos 3 与我们自己的闭环真机实验未复现;报告中的闭环数字按论文口径转述。
- Probing 的 L4(可行动性)不在发布物里,未独立复核;发布物缺 CogVideoX 第 20 层整行,主表重算是九模型。
- 合成实验是机制测试,不是机器人 benchmark;报告明确限制在"供给是否到达动作头"。
- RUA 的调度器测试用合成工具,不是 RoboDojo/真机 rollout;论文真机 agent 额外用 D405 深度、基线 RGB-only,报告明确标出这个混淆。

## 机器门禁备注
- verify_run 的 non-universal all-caps 警告:WAM、CWAM、AUC 首现均有全称与定义;OSWAM、REA、LIBERO、GAP、CAD、ERQA 是方法/基准/标准专名。
