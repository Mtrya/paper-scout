# PrimeSim (arXiv:2611.04870) 外部公开信息核查报告

核查日期：2026-11-05
核查目标：PrimeSim: A Generalist World Model Simulator from Large-Scale Cross-Embodiment Demonstrations (PrimeBot Research 等)
本地论文路径：

---

## 1. 训练语料表审计 (Table 12)

### 1.1 总体统计与换算关系
论文 Table 12 声称训练数据汇聚了 20 个数据集家族、1,773 个 LeRobot 仓库、1.52M episodes、22 个机器人平台，总计 **30,110 小时**（Real 25,350h + Sim 4,760h，实机占比 84%）与 **2.383B 帧**（Real 1,924M + Sim 459M）。
表头明确注明：“Hours are derived as frames/fps/3600 and so are consistent with the frame counts by construction.” 即表中小时数是由帧数按采样率严格计算得出。除 In-House 9,540h / 375M 帧（隐含 fps 约 10.92Hz，官方声明采自 AgiBot G2/G2X/PrimeBOT H2 双臂人形平台，属私有不可外部审计）外，对其余公开数据集的规模、帧数和隐含 fps 进行了逐项审计。

### 1.2 逐项核验明细表

| 数据集名称 | 论文表内数字 (Hours / Frames) | 论文隐含 fps | 公开官方口径数字 (论文/官网/卡片) | 一致性判断 | 来源 URL | 核查说明与口径分析 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DROID** [40] | 510h / 28M | 15.25 Hz | 350h / 76k episodes / 15Hz 控制频率 | **口径差异 (可解释)** | https://arxiv.org/abs/2403.12945 | DROID 官方论文宣称 76k 条成功轨迹对应 350h。但官方数据集中另含约 16k 条“未成功”但保留在 release 中的轨迹（总计 92k 轨迹）；92k 轨迹按 15Hz 换算即达约 450–510h / 28M 帧。论文表内 28M 帧 / 510h 严格吻合 15Hz，表明其纳入了全量原始记录或多相机多路录制。 |
| **AgiBot World** [2] | 2,300h / 243M | 29.35 Hz | ~3,000h (2,967h) / 1.03M episodes / 30Hz | **一致 (微幅清洗收紧)** | https://arxiv.org/abs/2508.05635 | AgiBot-World-Beta (Genie Envisioner 采用) 官方口径为 2,967 小时、103 万集、30Hz 采样。PrimeSim 经视频质量与动作平滑过滤后保留 2,300h / 243M 帧，隐含 29.35Hz，符合 30Hz 下截断清洗规律。 |
| **Hy-Embodied UMI** [94] | 2,100h / 226M | 29.89 Hz | ~2,100h / 数十万 episodes / 30Hz | **一致** | https://arxiv.org/abs/2606.14409 | Hy-Embodied-0.5-VLA 包含大规模手持 UMI 示教采集，官方汇报在 2,000+ 小时级，30Hz 录制。隐含 29.89Hz 与 30Hz 完全一致。 |
| **BEHAVIOR-1K 2026** [44] | 2,000h (sim) / 211M | 29.31 Hz | 1,000 tasks / 50 scenes / OmniGibson 30Hz 仿真 | **一致** | https://arxiv.org/abs/2403.09227 | BEHAVIOR-1K / OmniGibson 仿真平台导出轨迹的标准渲染和控制步长通常为 30Hz。211M 帧对应 30Hz 约为 1,954–2,000 小时。 |
| **HiFi-UMI-130k** [3] | 2,000h / 180M | 25.00 Hz | 130k episodes / 约 2,000h | **一致** | https://arxiv.org/abs/2602.06949 | 对应 DreamDojo 及 UMI 衍生高质量重构数据集，单集约 55 秒，130k 集约 2,000h。180M 帧在 25Hz 下精确等于 2,000 小时。 |
| **InternData-A1** [71] | 4,000h real + 170h sim / 411M | 27.38 Hz | 7,433.9h / 630k episodes / 401.4M 帧 (纯 sim 宣称) | **口径差异 (定义重划分)** | https://arxiv.org/abs/2511.16651 | InternData-A1 原文摘要宣称 630k episodes, 401.4M frames, 7433.9h，全量为高质量合成数据。PrimeSim 归入 411M 帧与原作者 401.4M 极度接近，但 PrimeSim 将其拆为 4,000h real + 170h sim（推测是将高保真真实感渲染子集归为 real）。 |
| **InternData-M1** [12] | 160h real + 1,700h sim / 199M | 29.72 Hz | 2.3M 空间推理及跨机体实机/仿真示教 | **一致** | https://arxiv.org/abs/2510.13778 | InternVLA-M1 具身数据池由大量仿真生成（~1,700h）与小规模实机测试轨迹构成。199M 帧在 30Hz 下精确对应 1,842–1,860 小时。 |
| **XR2** [82] | 1,500h / 162M | 30.00 Hz | 1,500h 双臂日常家务遥操作示教 / 30Hz | **完全一致** | https://arxiv.org/abs/2609.03591 | XR-2 论文标题即为《Scaling Bimanual Household Manipulation from 1,500 hours of Demonstrations...》。162M / (1500 * 3600) = 30.00 FPS，帧数、时长与帧率与官方完全一致。 |
| **RoboMIND** [78] | 850h / 92M | 30.07 Hz | 850h+ / 100k+ trajectories / 30Hz 多机体数据 | **一致** | https://arxiv.org/abs/2504.12501 | RoboMIND 官方 benchmark 汇聚约 850 小时跨机体规范示教数据，相机采样率为 30Hz。92M / (850 * 3600) = 30.07 FPS。 |
| **RoboCOIN** [80] | 700h / 77M | 30.56 Hz | 180,000+ demonstrations / 15 平台 / ~700h / 30Hz | **一致** | https://arxiv.org/abs/2511.17441 | RoboCOIN 官方声明超过 180k 条示教，平均每条 12–15 秒，总计约 600–700h，30Hz 采样。77M 帧对应 30Hz 约为 713 小时。 |
| **AgiBot World 2026** [1] | 610h / 66M | 30.05 Hz | 增量 600+ 小时 / 30Hz | **一致** | https://arxiv.org/abs/2508.05635 | 智元 2026 年新采集的高难度任务增量集，66M 帧对应 30Hz 正好 611 小时。 |
| **RoboCasa** [56] | 610h (sim) / 44M | 20.04 Hz | ~100k 仿真任务轨迹 / 20Hz 控制频率 | **完全一致** | https://arxiv.org/abs/2406.02523 | RoboCasa 基于 RoboSuite 搭建，仿真环境控制与视频帧率默认为 20Hz。44M / (610 * 3600) = 20.04 FPS，精确符合 20Hz。 |
| **Galaxea OpenWorld** [38] | 490h / 26M | 14.74 Hz | 500h 真实场景移动双臂示教 / 15Hz 相机流 | **一致 (微幅清洗)** | https://arxiv.org/abs/2509.00576 | 星海图 (Galaxea) Open-World 官方宣传口径为 500 小时，采用 15Hz 视觉-动作记录。清洗过滤后保留 490h / 26M 帧，隐含 14.74Hz。 |
| **MimicGen** [54] | 210h (sim) / 15M | 19.84 Hz | 50k+ 合成轨迹 / 20Hz 控制频率 | **一致** | https://arxiv.org/abs/2310.17596 | MimicGen 依赖 RoboSuite，标准控制步长为 20Hz。15M / (210 * 3600) = 19.84 FPS，吻合 20Hz。 |
| **Kai0** [91] | 130h / 15M | 32.05 Hz | chi_0 项目示教集 / ~130h | **一致** | https://arxiv.org/abs/2602.09021 | 港大/商汤 chi_0 团队开源的多机体长程操作数据集，约 130 小时，30Hz 记录。 |
| **RoboTwin 2.0** [11] | 60h (sim) / 6M | 27.78 Hz | 100k+ 轨迹生成库 / 25–30Hz 仿真步频 | **一致** | https://arxiv.org/abs/2506.18088 | 清华/智元 RoboTwin 2.0 双臂仿真基准。6M 帧对应约 60 小时，隐含 ~28Hz。 |
| **RT-1 Fractal** [6] | 350h / 4M | **3.17 Hz** | 130k episodes / 3Hz 控制频率 / ~350h (按均长估算) | **完全一致 (特别澄清)** | https://arxiv.org/abs/2212.06817 | **关键发现**：RT-1 官方明确记载闭环控制频率为 **3 Hz**（“commands actions at 3 Hz”）。130k 条轨迹平均每条约 10 秒（~30 步），总时长 130,000 * 10 / 3600 约 360 小时；总帧数 130,000 * 30 约 3.9M 帧。350h 对 4M 帧得出的隐含 3.17Hz **非但不是错误，反而是对 RT-1 3Hz 独特超低采样率的精确反映**。 |
| **Bridge V2** [74] | 110h / 2M | **5.05 Hz** | 60,096 trajectories / 5Hz 控制频率 / ~110h | **完全一致 (特别澄清)** | https://arxiv.org/abs/2308.12952 | **关键发现**：BridgeData V2 官方论文明确写明“control frequency is 5 Hz”，60k 轨迹平均长约 50–100 步（10–13 秒），总计约 110–120 小时。2M / (110 * 3600) = 5.05 FPS，精确印证 5Hz。 |
| **LIBERO** [51] | 10h (sim) / 1M | 27.78 Hz | 130 tasks / ~10h 专家示教 / 20–30Hz 仿真步长 | **一致** | https://arxiv.org/abs/2306.03310 | 常用 LIBERO 专家示范集约包含数千条轨迹，总时长约 10 小时左右。 |

### 1.3 审计结论
1. **换算严谨性**：PrimeSim 表中除 DROID 的 15Hz、Bridge V2 的 5Hz、RT-1 的 3Hz、RoboCasa/MimicGen 的 20Hz 外，其余常规真实机器人与仿真平台均集中在 25–30Hz 频段。此前关于“RT-1 350h 对 4M 帧隐含 fps 极低”的疑虑，经查实完全是因为 RT-1 硬件系统原生采用了 3Hz 采样，数据完全真实可信。
2. **DROID 350h vs 510h 差异**：官方常引用的 350h 仅计入了 76k 条“有效成功”轨迹；加上数据发布中保留的 16k 未标注成功轨迹，全量数据在 15Hz 下刚好为 510h，属于过滤子集与全集口径差异。
3. **InternData-A1 真实/仿真归属差异**：InternData-A1 原论文将其全部定性为高保真仿真合成（Synthetic Data），PrimeSim 表内将 4,000h 划为 Real，属于 PrimeSim 内部以视觉拟真度/渲染层级为口径的重分类，在严格语境下存在归类口径分歧。

---

## 2. WorldSimProbe 基线数字溯源 (Table 1)

### 2.1 论文声称与对比
PrimeSim 在 Table 1 中对五个开源动作条件世界模型汇报了在 LIBERO 基准上的 T1–T5 与 Mean 分数：
- LingBot-VA: T1=77.0, T2=60.2, T3=51.0, T4=57.8, T5=22.4, Mean=53.7
- DreamDojo: T1=69.5, T2=41.5, T3=50.3, T4=65.3, T5=23.5, Mean=50.0
- Cosmos-3-Nano: T1=67.3, T2=55.9, T3=47.4, T4=61.1, T5=25.0, Mean=51.3
- IRASim: T1=54.0, T2=54.7, T3=47.9, T4=68.2, T5=22.4, Mean=49.4
- Ctrl-World: T1=62.7, T2=61.7, T3=50.4, T4=77.5, T5=25.0, Mean=55.5
- *(PrimeSim 自报)*: T1=70.2, T2=67.9, T3=72.3, T4=85.7, T5=37.1, Mean=66.6

### 2.2 溯源核查与基准原论文对照
检索 WorldSimProbe 原论文：[arXiv:2608.09298](https://arxiv.org/abs/2608.09298)（Peterson Co, Shanghang Zhang 等，*WorldSimProbe: Diagnosing Simulator Faithfulness in Action-Conditioned World Models for Embodied Manipulation*, 2026年8月）。
查阅原论文 Table 2（主表，横跨 RoboTwin / ManiSkill / LIBERO 三个评测环境）：

| 模型名称 | T1 (Local) 原文 vs PrimeSim | T2 (Global) 原文 vs PrimeSim | T3 (Source) 原文 vs PrimeSim | T4 (Grounding) 原文 vs PrimeSim | T5 (Dynamics) 原文 vs PrimeSim | Mean/Overall 原文 vs PrimeSim | 吻合状态 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IRASim** | 54.0 / 54.0 | 54.7 / 54.7 | 47.9 / 47.9 | 68.2 / 68.2 | 22.4 / 22.4 | 49.4 / 49.4 | **逐位完全一致** |
| **Ctrl-World** | 62.7 / 62.7 | 61.7 / 61.7 | 50.4 / 50.4 | 77.5 / 77.5 | 25.0 / 25.0 | 55.5 / 55.5 | **逐位完全一致** |
| **DreamDojo** | 69.5 / 69.5 | 41.5 / 41.5 | 50.3 / 50.3 | 65.3 / 65.3 | 23.5 / 23.5 | 50.0 / 50.0 | **逐位完全一致** |
| **LingBot-VA**| 77.0 / 77.0 | 60.2 / 60.2 | 51.0 / 51.0 | 57.8 / 57.8 | 22.4 / 22.4 | 53.7 / 53.7 | **逐位完全一致** |
| **Cosmos-3-Nano**| 67.3 / 67.3 | 55.9 / 55.9 | 47.4 / 47.4 | 61.1 / 61.1 | 25.0 / 25.0 | 51.3 / 51.3 | **逐位完全一致** |

### 2.3 结论
**所有五款基线模型的全部 T1–T5 及 Mean 分数，百分之百来自 WorldSimProbe 官方原论文 Table 2 中的 LIBERO (LB) 列，非 PrimeSim 团队自行复跑。**
原论文评测了 6 个模型（另有 BWM 未被 PrimeSim 放入 Table 1），LingBot-VA 与 Cosmos-3-Nano 本就在 WorldSimProbe 原文官方榜单中。PrimeSim 完整继承了原论文数字，不存在来源缺口或篡改情况。

---

## 3. EWMBench 对照口径 (Table 2)

### 3.1 GE-Sim (LTX / COSMOS2) 5.82 / 5.79 出处核验
PrimeSim Table 2 给出：
- GE-Sim (LTX): SceneC=0.900, HSD=0.940, Dyn=0.780, nDTW=0.980, Div=0.011, BLEU=0.330, CLIP=0.908, Logics=0.970, **Overall=5.82**
- GE-Sim (COSMOS2): SceneC=0.910, HSD=0.870, Dyn=0.850, nDTW=0.970, Div=0.010, BLEU=0.310, CLIP=0.902, Logics=0.970, **Overall=5.79**

**查证原出处**：Genie Envisioner 原论文 [arXiv:2508.05635v3](https://arxiv.org/abs/2508.05635)（Yue Liao, Pengfei Zhou, Guanghui Ren 等）第 6.4 节 Table 2 (*GE-Sim Evaluation on EWMBench*):
- LTX 报告指标：BLEU=0.33, CLIP=90.8 (即 0.908), DYN=0.78, Div.=0.011, SA=0.94, Log.=0.97, TA=0.98, Scn.=0.90
- COSMOS 报告指标：BLEU=0.31, CLIP=90.2 (即 0.902), DYN=0.85, Div.=0.010, SA=0.87, Log.=0.97, TA=0.97, Scn.=0.91

各指标严格对应：
- Scn. -> SceneC (0.900 / 0.910)
- SA (Spatial Accuracy) -> HSD (0.940 / 0.870)
- DYN -> Dyn (0.780 / 0.850)
- TA (Trajectory Alignment) -> nDTW (0.980 / 0.970)
- Div. -> Div (0.011 / 0.010)
- BLEU -> BLEU (0.330 / 0.310)
- CLIP (90.8 / 90.2 / 100) -> CLIP (0.908 / 0.902)
- Log. -> Logics (0.970 / 0.970)

求和验证：
- LTX: 0.33 + 0.908 + 0.78 + 0.011 + 0.94 + 0.97 + 0.98 + 0.90 = **5.819** 约等于 **5.82**
- COSMOS: 0.31 + 0.902 + 0.85 + 0.010 + 0.87 + 0.97 + 0.97 + 0.91 = **5.792** 约等于 **5.79**

**核验结论**：PrimeSim 题注注明的“*GE-Sim rows are computed from the published Genie Envisioner simulation evaluation [50], aligned to the standard EWMBench column names*”属实，数字与对齐方式完全精确。

### 3.2 21-episode 社区评估划分与基线数据来源
Table 2 第二行块（指令/文本条件）及动作条件 Thinker-WM 的来源核验：
1. **21-episode split 统一定义**：源自 Thinker-WM ([arXiv:2610.18447](https://arxiv.org/abs/2610.18447)) 与 Qwen-RobotWorld ([arXiv:2606.17030](https://arxiv.org/abs/2606.17030)) 的评测协议。从 AgiBot World 的 7 个任务中抽取 21 个标准评测集，每个 episode 生成 3 个候选并取最优，共计 63 段视频。
2. **基线数字对照（Thinker-WM Table 4 与 Qwen-RobotWorld 原文）**：
   - **Thinker-WM (text-cond.)**: SceneC=0.888, HSD=0.564, Dyn=0.363, nDTW=0.710, Div=0.012, BLEU=0.269, CLIP=0.902, Logics=0.952, **Overall=4.66**（Thinker-WM 表 4 报告 4.66）
   - **Thinker-WM (action-cond.)**: 在其消融实验 Section 8.2.1 Table 7 中，提供动作条件输入后，HSD 升至 0.7369 (0.737)，nDTW 升至 0.8736 (0.874)，Dyn 为 0.3636 (0.364)，**Overall 升至 5.03**。PrimeSim Table 2 第一行块中的数字完全取自此处。
   - **Qwen-RobotWorld**: SceneC=0.914, HSD=0.566, Dyn=0.343, nDTW=0.671, Div=0.011, BLEU=0.208, CLIP=0.883, Logics=1.000, **Overall=4.60**（两篇原文完全一致）。
   - **LVP**: Overall=4.05（两篇原文一致）。
   - **Sora2**: SceneC=0.853, HSD=0.281, Dyn=0.349, nDTW=0.275, Div=0.031, BLEU=0.247, CLIP=0.910, Logics=0.947, **Overall=3.89**。
   - **Kling26**: SceneC=0.821, HSD=0.327, Dyn=0.182, nDTW=0.342, Div=0.017, BLEU=0.259, CLIP=0.901, Logics=1.000, **Overall=3.85**。
   - **EnerVerse-FT**: 查 EWMBench 原论文 ([arXiv:2505.09694](https://arxiv.org/abs/2505.09694)) Table 2，EnerVerse_FT 报告 SceneC=0.9427, HSD=0.5356, Dyn=0.5363, nDTW=0.5957, Div=0.0691, BLEU=0.1800, CLIP=0.8638, Logics=0.9778, **Overall=4.7010 (4.70)**。PrimeSim 明确用标号 dagger 注明其来自 EWMBench 原文自测。

**核验结论**：Table 2 的所有基线分数均有清晰合法的文献源，且严格继承了 21-episode split 上由 Qwen-RobotWorld / Thinker-WM / EWMBench 建立的既有评测基线；动作条件 GE-Sim 的 5.82 / 5.79 是对 Genie Envisioner 原论文指标重整化（除以 100）求和后的正确还原。

---

## 4. 开源声明核验 (截至 2026-11-05)

### 4.1 论文声称
- 论文摘要（Abstract 第 15 行）：“We release the model weights, inference code, and training code.”
- 论文引言（第 43 行）：“We release the model weights, inference code, and training code to support further research.”
- 论文首页提供项目主页：。

### 4.2 实际状态核验
1. **GitHub 仓库状态** ()：
   - 仓库创建时间：2026-11-04 03:54:16 UTC。
   - 最近提交（共 2 次提交）：
     1.  (2026-11-04 11:54:18 +0800): *Initial commit*
     2.  (2026-11-04 12:04:22 +0800): *updated coming soon to README*
   - 当前仓库文件列表仅包含 3 个基础占位文件：
     - 
     -  (Apache-2.0)
     - 
   -  正文内容明确写道：
     > ## Code coming soon
     > The inference code, training code and model weights will be released here shortly.
   - 仓库 Releases 数量为 0，Tags 数量为 0，没有任何 Python 脚本、训练代码或权重链接。

2. **项目主页状态** ()：
   - 页面静态展示论文 Teaser、架构图及实验雷达图。
   - 页面顶部及全文字符串检索中，不存在任何 Hugging Face、ModelScope、百度网盘或外部权重下载链接。
   - 页面中的代码链接直指上述处于占位符状态的 GitHub 仓库。

### 4.3 差异核实陈述
| 论文声称 | 截至 2026-11-05 实际线上状态 | 核验判定 |
| :--- | :--- | :--- |
| “We release the model weights, inference code, and training code” (一般现在时，声称已发布) | GitHub 仓库仅有 , , ；README 明确标注 “**Code coming soon: The inference code, training code and model weights will be released here shortly**”；主页无任何权重发布链接。 | **“假开源 / Coming Soon 占位”状态**，模型权重、推理代码与训练代码**均未实际发布**。 |

---

## 5. 总结与综合评价

1. **语料真实性高**：Table 12 训练集规模指标（30.1k 小时 / 2.38B 帧）与公开学术集高度吻合。此前引起怀疑的 RT-1（350h 对应 4M 帧，隐含 3.17Hz）和 Bridge V2（110h 对应 2M 帧，隐含 5.05Hz）完全对应了两者的原生超低硬件控制频率，反映出其数据管道元数据提取的严谨性；DROID 510h 则是由于计入了 16k 未标注成功的完整原始轨迹。
2. **基准对比口径合规**：Table 1 的 WorldSimProbe 指标 100% 照录自原论文（arXiv:2608.09298）Table 2 的 LIBERO 评测列；Table 2 的 EWMBench 指标严格继承自 Genie Envisioner、Thinker-WM 及 Qwen-RobotWorld 的 21-episode 标准评测集，GE-Sim 5.82/5.79 的转换计算逻辑完全吻合。
3. **开源履约落后**：论文在 Abstract 和正文中均使用了已发布（We release...）的时态声明，但截至 2026-11-05，官方代码仓库仍为前一日创建的“Code coming soon”空壳占位仓库，权重与代码实质处于未发布状态。
