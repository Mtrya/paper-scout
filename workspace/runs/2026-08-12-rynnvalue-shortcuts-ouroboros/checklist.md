# Paper Scout 巡航清单

## 运行
- 运行 id: `2026-08-12-rynnvalue-shortcuts-ouroboros`
- 覆盖时段: 2026-08-11 至 2026-08-12(HF papers 近日刊 + arXiv 补扫)
- 报告: `report.docxxml`
- 飞书文档: https://fudan-nlp.feishu.cn/docx/AwcedMOnpo0EyAxs5mFclI3inAe
- 远程实例: 启智 notebook `rynnvalue-probe`(可上网GPU资源),RynnValue-4B 八条件捷径压力测试 + 三编码器不可见捷径探针均在 4090 实例真实权重上完成;Ouroboros 考古为本地 git 历史与代码审计

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。
- [x] 每个深度线程都有建设性的研究动作:RynnValue 八条件扰动压力测试(含追加的 frozen80/loop/loopdense 加密条件);Invisible Shortcuts 三真实编码器 MP/SPD/位移实测(随机指派设计,SigLIP 为论文未测对象);Ouroboros 947 commit 考古 + BIBLE 修订史 + 保护机制代码审计 + 官方榜 PR 状态核验。
- [x] 关键论断有代码、指标 JSON、git 历史与代码行号支撑。
- [x] 报告讲清了仅靠重读论文文本无法看出的东西(循环重播的价值上跳、SigLIP/CLIP/DINOv2 的痕迹梯度、Success 假阴性模式、86.74% 的 PR 未合入状态)。

## 报告契约
- [x] 报告可扫读:开篇结论、分节叙事、论文主图 + 实验图、关键数字具体。
- [x] `report.docxxml` 中有 6 个图锚点(论文图 3 张 + 实验图 3 张)。
- [x] 深度线程逐篇精读,未硬凑共同主题(三篇三个方向,开篇仅作背景关联)。
- [x] 短名单织入结尾"同期池子里的其他信号"一节,行文顺带提及,不堆砌列表。

## 保存契约
- [x] 持久证据在线程目录(`rynnvalue-2608.09853/`、`shortcuts-2608.05424/`、`ouroboros-2608.08311/`)。
- [x] 面向报告的资产在 `assets/`(含 figures.json)。
- [x] `code/` 和 `drafts/` 不持有持久工作的唯一副本(脚本与指标 JSON 已晋升)。
- [x] 工作区校验器 prepublish 模式通过。

## 发布契约
- [x] 报告已发布。
- [x] 用户通知已确认。
