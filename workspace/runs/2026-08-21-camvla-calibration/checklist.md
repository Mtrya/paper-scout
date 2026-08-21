# Paper Scout 巡航清单

## 运行
- 运行 id: 2026-08-21-camvla-calibration
- 覆盖时段: 2026-08-21
- 报告: runs/2026-08-21-camvla-calibration/report.docxxml
- 飞书文档: (交付后填)

## 线程
- `camvla-2607.05396/`: CamVLA 精读(相机系 delta + 几何头自估 hand-eye,平移相消),含第三方玩具复现核验。
- `ocvla-2508.13103/`: OC-VLA 精读 + 代码核验(官方仓空壳 = Dita + flag;训练目标只依赖 R,部署端 R^⊤δt 泄漏)。
- `calibration-line/`: 外参估计四代谱系,ARC-Calib / FastCal 深挖(可观性、TSVD 秩揭示、时间衰减),轻扫 Kalib/FEEPE/LRBO/Continual Hand-Eye/Fiducial Exoskeletons。
- `action-interface/`: 动作接口横切(BARX / ContactFlow / AxisGuide),误差路径谱系(全T→只R→只提示→无外参→整链)。
- `exp-handeye-ekf/`: 实验 A——误差态 EKF 在线 hand-eye 外参估计(可观性/漂移跟踪/误差时间结构/「自信地错」),code/ 含全部脚本,figures/ 为图副本。
- `exp-action-repr/`: 实验 B——blob-world 相机系 vs 基座系动作 + 误差时间结构,iid/static/AR(1) × K∈{1,5},code/ 含全部脚本,figures/ 为图副本。
- `exp-openvla-view/`: 实验 C——OpenVLA-7B × LIBERO 视角扰动分解(raw vs rescue),code/ 含远端脚本;启智平台跑批中,结果待回。

## 研究契约
- [x] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。
- [x] 每个深度线程都有建设性的研究动作,或一个精确的障碍说明。
- [x] 关键论断有代码、探针、补丁、相关工作、数据样本、推导、产出物支撑,或明确陈述的障碍。
- [x] 报告讲清了这次巡航学到的、仅靠重读论文文本无法看出的东西。

## 报告契约
- [x] 报告可扫读:清晰的开篇综述、有力的分节、有用的图/表/公式/代码片段、流畅的逻辑。
- [x] `report.docxxml` 中至少有两个图锚点。
- [x] 深度线程读起来像研究叙事,而不是填模板的摘要。
- [x] 轻量留意的论文与深度线程被干净地区分开。

## 保存契约
- [x] 持久证据在线程目录中。
- [x] 面向报告的资产在 `assets/` 中。
- [ ] `code/` 和 `drafts/` 不持有持久工作的唯一副本。
- [x] 工作区校验器 prepublish 模式通过。

## 发布契约
- [ ] 报告已发布。
- [ ] 用户通知已确认。
- [ ] `runs/INDEX.md` 已更新。
- [ ] 工作区校验器 final 模式通过。
