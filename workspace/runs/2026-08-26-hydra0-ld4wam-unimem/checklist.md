# Paper Scout 巡航清单

## 运行
- 运行 id: 2026-08-26-hydra0-ld4wam-unimem
- 覆盖时段: 2026-08-25(HF 日榜 32 篇 + W35 周视图 50 篇 + arXiv cs.RO 补网 40 篇)
- 报告: runs/2026-08-26-hydra0-ld4wam-unimem/report.docxxml
- 飞书文档: https://fudan-nlp.feishu.cn/docx/JZA7djXUEoVGTwxXQVIcpxXXnch

## 线程
- `hydra0/`: Hydra-0 精读(2608.18077)+ 实验 A:AllTracker×DROID 6 集地基噪声实测
  (阵发性/结构性噪声,p95 达 84px,一集相机中途被动)。
- `ld4wam/`: LD4WAM 精读 + 官方 LDM 代码核验(2608.22403)+ 实验 B:冻结 LDM 的
  OOD(DROID)运动探针,三重对照(过拟合对照/逐维分解/幅度对照),roll 环绕假警报修复。
- `unimem/`: UniMem 精读 + openpi fork 代码审计(2608.22869,实现与论文一致,
  作者自文档化 seeding 期 PE 位置 bug;无权重)。

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。
- [x] 每个深度线程都有建设性的研究动作,或一个精确的障碍说明。
- [x] 关键论断有代码、探针、补丁、相关工作、数据样本、推导、产出物支撑,或明确陈述的障碍。
- [x] 报告讲清了这次巡航学到的、仅靠重读论文文本无法看出的东西。

## 报告契约
- [x] 报告可扫读:清晰的开篇综述、有力的分节、有用的图/表/公式/代码片段、流畅的逻辑。
- [x] `report.docxxml` 中至少有两个图锚点(9 个)。
- [x] 深度线程读起来像研究叙事,而不是填模板的摘要。
- [x] 轻量留意的论文与深度线程被干净地区分开。

## 保存契约
- [x] 持久证据在线程目录中。
- [x] 面向报告的资产在 `assets/` 中。
- [ ] `code/` 和 `drafts/` 不持有持久工作的唯一副本(例外:远端 LDM 特征缓存 cache_s*.npz 约 41MB 留在启智 workroot,可由 probe_ldm_droid.py 重算;DROID 21 集子集同)。
- [x] 工作区校验器 prepublish 模式通过。
