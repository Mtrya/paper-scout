# Run Index

Compact coverage log and dedup source of truth, newest first. Before serious scouting, read this to avoid repeating papers or research threads already covered. After each delivered run, append one block. Run reports and preserved evidence live in `runs/<run-id>/`, not here.

Entry format:

```
## YYYY-MM-DD — <period covered>
- Doc: <url>
- Run: runs/<run-id>/
- Deep threads: <thread name> (<paper ids if any>), <thread name>
- Covered papers: <id>, <id>, ...
- Shortlisted papers: <id>, <id>, ...
```

## 2026-08-24 — 2026-08-18 to 2026-08-24(推理时算力专题)
- Doc: https://fudan-nlp.feishu.cn/docx/SZx8d42kDoom9Hxn5FvcPSSBnXe
- Run: runs/2026-08-24-tau0vla-dalewm/
- Deep threads: τ₀-VLA 精读 + 代码解剖 + 启智实跑(混合线性注意力骨干 18/24 层 GatedDeltaNet;零样本夹爪常数偏移失败 MSE 0.01126;冻骨干微调 500 步 202 秒 MSE→0.00264;FM 步数 K=1≈K=10 平台期;from_pretrained 静默解冻 bug 及补丁)(2608.16885), DA-LeWM 诊断复刻到自家 LeWM/PSG 检查点(Plan-Real 0.665→0.670 接地头不修决策对齐预言证实;CEM elite 崩塌复现;Cube 缺口以扰动演示候选补上;Claim-1 0.583→0.722)(2608.18746), ForgeWM 精读(四阶段因果化+少步化;1 步 72.1 FPS 质量反超;test-time scaling 走平与 τ₀ FM 平台期互证)(2608.14022)
- Covered papers: 2608.16885, 2608.18746, 2608.14022
- Shortlisted papers: 2608.19891, 2608.16590, 2608.14036, 2608.19197, 2608.18701, 2608.14441, 2608.15008, 2608.17393
- 背景: 三篇共同指向"推理算力往哪花"(高层束搜索/低层砍步数/排序保真是地基);低维控制分布上多步去噪边际收益薄的跨域互证;决策对齐与信息充分性独立轴在自家模型上证真,下一步动作=加 inverse/goal-action 头

## 2026-08-24 — 外参标定 × 相机中心动作专题第二日(组会终稿)
- Doc: https://fudan-nlp.feishu.cn/wiki/PA11wO4UXiwLV1kyJuyc8Xcenkb(组会终稿,写入用户 wiki;巡航版见 08-21 条目)
- Run: runs/2026-08-21-camvla-calibration/(新增线程 exp-intrinsic-ekf / exp-frame-rep / exp-imagespace-precision / exp-recovery-channels / action-interface 测绘;终稿 assets/report_final.docxxml)
- Deep threads: 内参三实验(管线审计:有效焦距离散 2–3.4 倍、预处理沉默重写几何;OpenVLA 焦距 ±10% 仍保 60–70%;EKF 联合估计:可观性靠深度多样性不靠旋转、联合 CRB 恶化 53×、串行标定烙错、焦距失配被外参无声吸收无告警、热维护分层 Q 滞后<0.6%), 实验 B2 动作表示公平对照(差别=信息可得性:Base+稠密位姿可达 ~1.0 但需 10× 数据;B2b 跨本体 Cam 无标签未见本体 0.983 vs 基座系全条件 ~0.4——相机系=构造性解耦), 图像空间动作表示 23 篇测绘(四族=时间密度轴切片;三元悖论在轨迹层消解;免标定仅 ATM/TraceVLA/PIVOT;RVT-2 唯一真机毫米级;ReKep 自动接地 44.3% vs 人工 68.6%)(2010.14406, 2209.05451, 2306.14896, 2406.08545, 2402.07872, 2403.03174, 2401.00025, 2412.10345 等), 实验 G/H 精度预算(横向 Z·δu/f 免费、深度 δZ=Z²δd/(bf) 买基线 b·f≳50 抓取/130–270 插入、双目间旋转 1°→38mm vs delta 0.87mm 44×、平面内旋转两点白送 0.57°、出平面 1° 级必须外观模板)
- Covered papers(新增): 2010.14406, 2209.05451, 2306.14896, 2406.08545, 2306.17817, 2402.10885, 2402.07872, 2403.03174, 2406.10721, 2310.11441, 2401.00025, 2405.01527, 2407.15208, 2401.11439, 2410.11758, 2311.01977, 2412.10345, 2302.00111, 2310.10639, 2307.05973, 2409.01652, 2501.03841, 1903.06684
- 背景: 组会终稿四章按行推进(内参→外参→仅变换表示→图像空间表示),沿安装形态收句;混合架构扩为四段式(学习式冷启动+批处理对准+滤波热维护+协方差进策略);执行经验:push_report --existing-doc/--as user、lark-cli ok:true 静默失败守卫、GitHub 100MB 硬限(实验数据集不入库)

## 2026-08-21 — 外参标定 × 相机中心动作(老师交办专题,不限近期)
- Doc: https://fudan-nlp.feishu.cn/docx/XyOpdswlmokrp9xMAzPcBgPSnJl
- Run: runs/2026-08-21-camvla-calibration/
- Deep threads: CamVLA 精读 + 第三方复现代码核验(相机系 delta + 几何头自估 hand-eye,平移相消,特征源消融)(2607.05396), OC-VLA 精读 + Dita 代码核验(官方仓空壳=Dita+flag;训练只依赖 R、部署端平移误差以 R^⊤δt 泄漏)(2508.13103), 标定线(ARC-Calib 探索运动 3 次 0.0225rad;FastCal 松耦合+FIM 秩揭示 TSVD+时间衰减;结构可观≠数据集可观)(2503.14701, 1902.10585), 动作接口线误差路径谱系(AxisGuide 渲染坐标轴 6°/3cm 内不掉、BARX EE traces 收益在训练期、ContactFlow 3D 接触点条件整链敏感)(2606.06761, 2607.27549, 2607.26579), 实验 A:EKF 在线 hand-eye(直线运动留 1 维不可观 FIM 1.3e-9 vs 体积 1.9e3;体积运动近 CRB 0.21°/0.11cm;Q 匹配阶跃半恢复 ~9s;误差 ACF(1)=0.90-0.96 慢变偏差;Q=0 自信地错 vs 批处理 GN 达 CRB), 实验 B:blob-world 孪生头(Base 单视角 8/19 点脆断 vs Cam 全 1.0;K=1 三种误差形态全补偿,K=5 static 唯一破功 0.90/0.98——边界=重规划频率), 实验 C:OpenVLA-7B×LIBERO-Spatial 真实权重 222 集(baseline 85% 复现;raw ±15°=0 崩溃复现;rescue 只在 task0@+10° 恢复 0.40→1.00,±15° 双符号均 0——崩溃以视觉编码漂移为主、动作系失配为辅)
- Covered papers: 2607.05396, 2508.13103, 2503.14701, 1902.10585, 2606.06761, 2607.27549, 2607.26579
- Shortlisted papers: 2604.15814, 2601.08034, 2408.10562, 2503.14051, 2311.01335, 2603.05868, 2506.22242, 2510.02268, 2601.08414, 2601.01188
- 背景: 外参来源四代谱系 × 动作表示误差路径谱系;三条判据:delta 动作只需旋转(平移相消)、误差时间结构×重规划频率是成败边界、动作系修复是感知修复的补集;架构空白=学习式冷启动+滤波热维护+协方差进策略没人做全

## 2026-08-17 — 2026-08-15 to 2026-08-17
- Doc: https://fudan-nlp.feishu.cn/docx/RE8YdxIiaomTHWxQusBc9RGxnMb
- Run: runs/2026-08-16-omega0-psgjepa/
- Deep threads: PSG-JEPA 真实复训 + 逐维可辨识性探针(H200 双模型同种子 10 epoch;baseline 逐维 r 与像素可见性相关 0.42=缺口机制证据;接地头外科手术式修补:腕部偏航 r 0.03→0.76、关节速度 ≤0.20→0.73-0.91 由动态头 Δq 监督注入、未监督 privileged 维原地不动;长程 rollout 改善复现 30 步 1.65 vs 1.90)(2608.06799), ω-0/ω-HOME 数据集探针(并发度 mop 0.866 vs apple 0.048;遥操作时延 70-230ms;target 只覆盖腕部 6 关节)(2608.06375), Marionette 精读(显式状态外置 + 零参数图形桥;软惩罚被优化器交易掉、硬约束成立的阴性结果)(2608.14530), Cosmos 3 精读(五模态 MoT、动作=token、条件方案即噪声掩码、15000 位置间隔)(2606.02800), Beyond Final Scores 精读(经验复用弱模型收益最大、harness 演化不跨任务族、钻空子五倍于真创新)(2608.13417)
- Covered papers: 2608.06799, 2608.06375, 2608.14530, 2606.02800, 2608.13417
- 背景: 主线四篇构成"动作/状态如何住进世界模型"的谱系:PSG-JEPA 钉潜变量 → ω-0 蒸馏控制器潜变量 → Marionette 状态外置 → Cosmos 3 动作 token 化;横切判据=软损失塑表征、硬约束管记账

## 2026-08-15 — 2026-08-13 to 2026-08-15
- Doc: https://fudan-nlp.feishu.cn/docx/M51udmbAiomUdZx0EhDcM6rynhc
- Run: runs/2026-08-15-cmd-evoke/
- Deep threads: CMD 精读 + 线性高斯玩具三对照(acausal 梯度比解析单调增 0.08→0.38;关键发现=错配代价是训练瞬态而非稳态:W=A 良性不动点,同预算下双向学生被拖慢,ar=0.9 因果 16 轮到真值/双向 32 轮,途中部署误差差 0.42=4 倍噪声底;Prefix Scoring 同量级,ar=0.9 误差 0.441→0.169)(2608.13391), EVOKE 代码审计 + H200 真机实跑(57GB 权重;diffusion 2.01s/chunk 复现论文 2.11s 口径,完整管线 10.2s;segment prompt 切换实测生效;warp-off 对照证实"warp=唯一相机通道"但零样本 i2v 下 warp on 反而色彩漂移)(2608.13546), DreamX-Phi 作交叉邻居(离线完整轨迹下双向 DMD 合法,补全规则另一面)(2608.13489)
- Covered papers: 2608.13391, 2608.13546, 2608.13489
- Shortlisted papers: 2608.13552, 2608.13049, 2608.11739, 2608.12990, 2608.10538, 2608.13560, 2608.11350
- 背景: 监督信号第三轴(部署信息集对齐)确立——与清晰度轴、分布形状轴并列;CMD(时间边界)/EVOKE(时域+条件调度+空间状态)/DreamX-Phi(离线合法性)三篇同日收敛

## 2026-08-13 — 2026-08-11 to 2026-08-13
- Doc: https://fudan-nlp.feishu.cn/docx/UlnrdNhbKoG2x8xeuD0c5A1cnnd
- Run: runs/2026-08-13-geniworld-uopsd-nwat/
- Deep threads: GeniWorld 精读 + Ctrl-World 代码三角验证 + 四条件接口消融玩具重建(渲染接口域内/接地优势属实,但论文盲区答案偏"接地":static≈shuffle≈motion;OOD 泛化与 few-step 鲁棒性在玩具尺度不成立→归因预训练先验而非接口)(2608.06332), U-OPSD 精读 + 方法重建 + 错误共识训练动态探针(巩固侧 pass@1 +5.7/+11.0 泛化成立、失明侧一字不动、放大侧在 held-out 露出尾巴 wrong_agree 0.589→0.701;τ=0.5 把"自信地错∩不一致"交集切掉)(2608.06296), NWAT 轻线程(位置>打分器;模糊效用信号下学习型价值模型打不过启发式,与 RynnValue/U-OPSD 拼出监督信号清晰度光谱)(2608.08389)
- Covered papers: 2608.06332, 2608.06296, 2608.08389
- Shortlisted papers: 2608.10915, 2608.10744, 2608.10299, 2608.09096, 2608.02508, 2608.10875
- 背景: Dyna-2 新闻(2026-08-10,1M 小时人类视频 WAM,宣称首个纯人类数据 scaling law;技术报告未放出,数字待核验)

## 2026-08-12 — 2026-08-11 to 2026-08-12
- Doc: https://fudan-nlp.feishu.cn/docx/AwcedMOnpo0EyAxs5mFclI3inAe
- Run: runs/2026-08-12-rynnvalue-shortcuts-ouroboros/
- Deep threads: RynnValue 精读 + 官方 4B 权重八条件捷径压力测试(内容接地证实 shuffle ρ=0.76;多尺度回退检测;frozen 时序泄漏与 Success 假阴性两个警告)(2608.09853), Invisible Shortcuts 精读 + 三真实编码器 MP/SPD 实测(SigLIP 57%/CLIP 53%/DINOv2 18% 痕迹梯度)(2608.05424), Ouroboros 精读 + 947 commit git 考古(74% agent 署名、BIBLE 自修订 12 次、86.74% 官方 PR 未合入)(2608.08311)
- Covered papers: 2608.09853, 2608.05424, 2608.08311
- Shortlisted papers: 2608.09888, 2608.09819, 2608.07169, 2608.08285, 2608.08097, 2608.08722

## 2026-08-10 — 2026-08-08 to 2026-08-10
- Doc: https://fudan-nlp.feishu.cn/docx/ViE2dEgqmonnOqx3CnZcHAWOnDd
- Run: runs/2026-08-10-roundtrip-worldtrace-simwam/
- Deep threads: Round-Trip Consistency 精读 + 六机制 Lorenz/单摆对照探针("吸引子失明"猜想证伪为虚警,C_i 测的是逆向腿)(2608.00675), WorldTrace 精读 + Qwen3 真实权重 RoPE 相消三组测量(逐频率存活、logit 平凡性、softmax 读权重)(2608.07408), SimWAM 精读 + 推理路径审计(视频塔未退场)+ 双 checkpoint census/IL-RL delta + 4090 prefill 截断探针(表征第 ~15 层收敛、prefill 仅占延迟 4.5%)(2608.07468)
- Covered papers: 2608.00675, 2608.07408, 2608.07468
- Shortlisted papers: 2608.05424, 2608.06729, 2608.06375, 2608.06994, 2608.06799, 2608.01851, 2608.05219, 2608.05703, 2608.06756

## 2026-08-08 — 2026-08-06 to 2026-08-08
- Doc: https://fudan-nlp.feishu.cn/docx/TKxSdihK6oEffBxiCUhcnSlxnHh
- Run: runs/2026-08-08-mass-memorytrust/
- Deep threads: MASS 精读 + typed-carrier 核心消融独立复现（5.61M/2.85M 双模型）+ 三探针（合法平行世界线 70.6%、确定性流 tick 1 全分歧、世界死亡吸引子 24–50 唯一状态）(2608.06257), When Memory Lies 精读 + SpatialSTALE 测试床重建与校准 + 模态鸿沟开源复跑（Qwen3-VL-8B 文本 1.000 vs 视觉 0.158)+ 图像消融（"对图像敏感但读不懂"第三种失败模式）+ 四因素仲裁曲线（文本天花板 null result)(2608.04574)
- Covered papers: 2608.06257, 2608.04574
- Shortlisted papers: 2608.05369, 2608.01964, 2608.05013, 2607.23783, 2608.06197, 2608.05042, 2608.03392, 2608.06374

## 2026-08-07 — 2026-08-04 to 2026-08-06
- Doc: https://fudan-nlp.feishu.cn/docx/N50edTJGso5vkzxtW9FceyWJnzf
- Run: runs/2026-08-07-worldcycle-wamspace/
- Deep threads: WorldCycle 精读 + ABot-World-0 六协议可逆循环探针（sink 锚定骗过闭环指标；avatar 转身代逆平移；回归后功能衰减 3×）(2608.04964), WAM 未来监督空间之争：ST-WAM/SG-WAM 精读与代码审计 + 三元组诊断独立复现 99 组（核心主张成立、VAE 判别率数字翻转、DINO 不变性过剩新发现）(2607.28993, 2608.01397)
- Covered papers: 2608.04964, 2607.28993, 2608.01397
- Shortlisted papers: 2608.03207, 2608.00486, 2608.02603, 2608.02580, 2608.02713, 2608.01127, 2607.29613, 2608.03994

## 2026-08-05 — 2026-07-20 to 2026-08-05
- Doc: https://fudan-nlp.feishu.cn/docx/Mc34d9LkeounK4xcyrLcpqIDnmf
- Run: runs/2026-08-05-visual-memory-posthoc-faithfulness/
- Deep threads: VLA 视觉记忆：NativeMEM/SOMA 精读 + MemoryVLA 固定槽 consolidation 代码审计与完整 checkpoint 结构核验 + History-Swap benchmark 设计（2607.06678, 2605.22283）, hindsight 合理化：Post-Hoc Reasoning 分阶段 Qwen3-VL-8B activation steering + Faithful Self-Evolvers 扰动代码审计与 uptake×robustness 二维重定义（2603.01437, 2601.22436）
- Covered papers: 2607.06678, 2605.22283, 2603.01437, 2601.22436
- Shortlisted papers: —（四篇均进入两条深挖线；本条为同日初稿经用户反馈后的唯一 canonical 大修版）

## 2026-08-03 — 2026-08-02 to 2026-08-03
- Doc: https://fudan-nlp.feishu.cn/docx/IOIwdEw2hoSguvxr1qNcs3YunWg
- Run: runs/2026-08-03-vtla-vipe-spatialcli/
- Deep threads: N₀-VTLA 精读 + 代码解剖(放出权重≠论文架构)+ 真实权重合成批次 z 敏感性探针 (2607.23782), VIPE 精读 + Wan2.2-5B 条件帧编辑反向检验(8 场景 × 3 编辑真实生成) (2607.25537), SpatialCLI 精读 + 放出范围核验(Internalize 数据未放出) (2607.27703)
- Covered papers: 2607.23782, 2607.25537, 2607.27703
- Shortlisted papers: 2607.28415, 2607.22561, 2607.21848, 2607.22393, 2607.23806, 2607.25308

## 2026-08-02 — 2026-07-31 to 2026-08-02
- Doc: https://fudan-nlp.feishu.cn/docx/Pfa4dqGlVoC9q4xmUd6cudzDnzh
- Run: runs/2026-08-02-ace-phizero-shadowdancer/
- Deep threads: ACE-Data-0 测量引擎规格核验 + 发布状态三角验证 (2607.28625), PhiZero 精读 + Wan2.2-5B 8 场景物理失败模式解剖实验 (2607.28624), ShadowDancer 精读 + sprites cross-shadow 正则/配对机制探针 (2607.28362)
- Covered papers: 2607.28625, 2607.28624, 2607.28362
- Shortlisted papers: 2607.27180, 2607.26056, 2607.26037, 2607.26754, 2607.27380, 2607.26760, 2607.28227, 2607.28568, 2607.22798, 2607.23402

## 2026-07-31 — 2026-07-18 to 2026-07-30
- Doc: https://qcn0umnxrmj2.feishu.cn/docx/B10IdsJp0okcqXxwei2cPQlGnkg (本地副本: runs/2026-07-31-abotworld-umi-turbovla/report.docxxml)
- Run: runs/2026-07-31-abotworld-umi-turbovla/
- Deep threads: ABot-World-0 4090 实跑 + 640-block 可控性阵发崩溃实验 (2607.19191), UMI 数据 fidelity×scale + HiFi-UMI-2K recovery 含量测量 (2607.15330, 2607.25895), 实时策略口径核验 + πR² staircase 玩具复现 (2607.27205, 2607.26055)
- Covered papers: 2607.19191, 2607.15330, 2607.25895, 2607.27205, 2607.26055
- Shortlisted papers: 2607.26754, 2607.26037, 2607.18367, 2607.23909, 2607.19343, 2607.18703, 2607.17977, 2607.27180, 2607.13429, 2607.24744, 2607.11498, 2607.25337, 2607.16401, 2607.14183, 2607.21655, 2607.24653

## 2026-07-18 — 2026-07-17
- Doc: https://fudan-nlp.feishu.cn/docx/P3S7dtwhyomKNZxULvocMplknX6
- Run: runs/2026-07-18-badwam-robotttt-gamestate/
- Deep threads: BadWAM world-action drift toy-WAM reconstruction + attack probe (2607.15207), RoboTTT fast-weight context probe (2607.15275), Pixels-to-States game-engine taxonomy triangulation (2607.14076)
- Covered papers: 2607.15207, 2607.15275, 2607.14076
- Shortlisted papers: 2607.13399, 2607.14777, 2607.15038, 2607.14187, 2607.14935, 2607.14952

## 2026-07-17 — 2026-07-16
- Doc: https://fudan-nlp.feishu.cn/docx/DVafdVjjvowpWPxo7x8cIJzmnQe
- Run: runs/2026-07-17-gigaworld05-harness-spear/
- Deep threads: GigaWorld-Policy-0.5 action-centered WAM code+weights verification (2607.13960), Harness Handbook behavior localization + mini-handbook probe on real Terminus-2 (2607.13285), SPEAR UE reflection-driven simulator code trace (2607.06701)
- Covered papers: 2607.13960, 2607.13285, 2607.06701
- Shortlisted papers: 2607.13104, 2607.12747, 2607.07702, 2607.12625, 2607.13921, 2607.12477, 2607.12395, 2607.13125, 2607.13639, 2607.09786

## 2026-07-16 — 2026-07-14 to 2026-07-15
- Doc: https://fudan-nlp.feishu.cn/docx/UIt3d2fE0oc8F7xZ0aAcPaFMnyf
- Run: runs/2026-07-16-densereward-terrazero-flowwam/
- Deep threads: DenseReward failure synthesis for dense rewards (2607.13033), TerraZero procedural driving sim and zero-demo self-play (2607.13028), FlowWAM optical flow as unified action representation for WAMs (2607.13017)
- Covered papers: 2607.13033, 2607.13028, 2607.13017
- Shortlisted papers: 2607.12992, 2607.12931, 2607.12892, 2607.12659, 2607.12571, 2607.12356

## 2026-07-14 — 2026-07-08 to 2026-07-14
- Doc: https://fudan-nlp.feishu.cn/docx/CQBSdrQELoOXcExGR83cZzasnvc
- Run: runs/2026-07-14-genception-lhtb-robodojo/
- Deep threads: GenCeption generative video as universal vision prior (2607.09024), Long-Horizon-Terminal-Bench dense grading for long-horizon agents (2607.08964), RoboDojo sim-and-real manipulation diagnosis (2607.04434)
- Covered papers: 2607.09024, 2607.08964, 2607.04434
- Shortlisted papers: 2607.09657, 2607.09661, 2607.06403, 2607.06291, 2607.05373, 2607.02403, 2607.04988, 2607.04425, 2607.06838, 2607.08716

## 2026-07-13 — 2026-07-08 to 2026-07-13
- Doc: https://fudan-nlp.feishu.cn/docx/UC4pdUL3bo8OS1xXokIch228nKb
- Run: runs/2026-07-13-lingbot-kinematic-memory/
- Deep threads: LingBot world models (2607.07675, 2607.07534), Imagined Rollouts kinematic diagnosis (2607.05966), MIRA multiplayer world models (2607.05352)
- Covered papers: 2607.07675, 2607.07534, 2607.05966, 2607.05352
- Shortlisted papers: 2607.07608, 2607.06442, 2607.06018, 2607.04434, 2607.03723, 2607.05765, 2607.08716, 2607.08768, 2606.30111, 2607.02501, 2607.07508, 2607.06987

## 2026-07-12 — 2026-07-08 to 2026-07-11
- Doc: https://fudan-nlp.feishu.cn/docx/XjeTdcYjroyDJUxikhPcEAE3nld
- Run: runs/2026-07-12-lamem-rynnworld-sieve/
- Deep threads: LaMem-VLA latent-memory probe (2607.07608), RynnWorld-4D code trace and 4D-policy analysis (2607.06559), SIEVE structure-aware selection probe (2607.06442)
- Covered papers: 2607.07608, 2607.06559, 2607.06442
- Shortlisted papers: 2607.08716, 2607.04988, 2607.03751, 2607.06558, 2607.06291, 2607.05352, 2607.06018, 2607.05765, 2607.05390, 2607.07534, 2607.07675, 2607.02466, 2607.04434, 2607.06403, 2607.02646, 2607.07508, 2607.08763, 2607.03748, 2607.03723, 2607.08768, 2607.06838

## 2026-07-11 — 2026-06-28 to 2026-07-11
- Doc: https://fudan-nlp.feishu.cn/docx/JaRGdeEawoAvAKxPQCgc3ZHFnUg
- Run: runs/2026-07-11-gigaworld-vla-corrector-physis/
- Deep threads: GigaWorld-1 world-model policy evaluation (2607.02642), PhysisForcing physics-reinforced world simulator (2606.28128), VLA-Corrector detect-and-correct inference (2607.01804)
- Covered papers: 2607.02642, 2606.28128, 2607.01804
- Shortlisted papers: 2607.06559, 2607.05390, 2607.02403, 2607.02517, 2607.05966, 2607.04434, 2607.07608, 2607.03751, 2607.00678, 2607.00272, 2607.02501

## 2026-06-27 — 2026-06-26 to 2026-06-27
- Doc: https://fudan-nlp.feishu.cn/docx/RXoKdXYYtowcXjxSELKckfghnRb
- Run: runs/2026-06-27-icwm-fastlew-hallucination/
- Deep threads: In-Context World Modeling for Robotic Control (2606.26025), Fast LeWorldModel (2606.26217), Hallucination in World Models is Predictable and Preventable (2606.27326)
- Covered papers: 2606.26025, 2606.26217, 2606.27326
- Shortlisted papers: 2606.27364, 2606.26790, 2606.26907

## 2026-06-25 — 2026-06-23 to 2026-06-25
- Doc: https://fudan-nlp.feishu.cn/docx/NXqedB7DAo2mx3xLcyhct759ndd
- Run: runs/2026-06-25-robotwin-foresight-beyond-gradients/
- Deep threads: RoboTwin 2.0 synthetic bimanual data engine (2506.18088), Foresight failure detection with action-conditioned world-model latents (2606.23085), Learning Beyond Gradients heuristic-learning paradigm (blog)
- Covered papers: 2506.18088, 2606.23085
- Shortlisted papers: 2606.20092, 2606.24742, 2606.22540

## 2026-06-22 — 2026-06-15 to 2026-06-22
- Doc: https://fudan-nlp.feishu.cn/docx/WA6hdqAhWox75Zx0DkRcAxULnTG
- Run: runs/2026-06-22-humanscale-playful-imagewam-wrbench/
- Deep threads: HumanScale egocentric vs. real-robot pretraining (2606.20521), ImageWAM image-editing world-action model (2606.19531), WRBench persistent-state diagnosis (2606.20545), Playful RATS code-as-policy play learning (2606.19419)
- Covered papers: 2606.20521, 2606.19531, 2606.20545, 2606.19419
- Shortlisted papers: 2606.15133, 2606.20515, 2606.19980, 2606.16122, 2606.20083, 2606.19495, 2606.18847, 2606.17480, 2606.18558, 2606.17030, 2606.14667, 2606.19338, 2606.00793, 2606.18180

## 2026-06-18 — 2026-06-16 to 2026-06-18
- Doc: https://fudan-nlp.feishu.cn/docx/Yc1HdNNXEoTGk2xuG3rcVtpEnSb
- Run: runs/2026-06-18-guava-kairos-omniagent/
- Deep threads: Guava harness for embodied manipulation (2606.18363), Kairos native world model stack (2606.16533), OmniAgent active perception (2606.19341)
- Covered papers: 2606.18363, 2606.16533, 2606.19341
- Shortlisted papers: 2606.18375, 2606.18208, 2606.18180, 2606.18216, 2606.17628, 2606.17861

## 2026-06-17 — 2026-06-16 to 2026-06-17
- Doc: https://fudan-nlp.feishu.cn/docx/B5IxdNyLAoqHcNxyoZHc7CGenMJ
- Run: runs/2026-06-17-aceego-actworld-motionvla/
- Deep threads: ACE-Ego-0 egocentric-robot VLA pretraining (2606.17200), ActWorld action-aware interactive world model (2606.17730), MotionVLA dual-stream humanoid motion tokenizer (2606.15142)
- Covered papers: 2606.17200, 2606.17730, 2606.15142
- Shortlisted papers: 2606.17054, 2606.17030, 2606.15768, 2606.17043, 2606.16519

## 2026-06-16 — 2026-06-15 to 2026-06-16
- Doc: https://fudan-nlp.feishu.cn/docx/NtxjdBv3AoNSQPxWD4XcqFpqnce
- Run: runs/2026-06-16-gam-apt-dreamx/
- Deep threads: Geometric Action Model (2606.17046), APT action-expert pretraining (2606.12366), DreamX-World 1.0 (2606.16993)
- Covered papers: 2606.17046, 2606.12366, 2606.16993
- Shortlisted papers: 2606.15631, 2606.09813, 2606.06194, 2606.14777, 2606.16295, 2606.17030, 2606.16519

## 2026-06-15 — 2026-06-15
- Doc: https://fudan-nlp.feishu.cn/docx/SVnEd0IsyobD8fx71yPcHPCynsd
- Run: runs/2026-06-15-mu0-hyvla/
- Deep threads: μ₀ 3D interaction-trace world model (2606.13769), Hy-Embodied-0.5-VLA full robot learning stack (2606.14409)
- Covered papers: 2606.13769, 2606.14409
- Shortlisted papers: 2606.13679, 2606.14249, 2606.14579, 2606.12384

## 2026-06-07 — 2026-05-28 to 2026-06-07
- Doc: https://fudan-nlp.feishu.cn/docx/G8dFd8ry2oC3yZxdIb3cGvwonnc
- Run: runs/2026-06-07-cosmos3-grail-qwenvla/
- Deep threads: Cosmos 3 omnimodal world models (2606.02800), Qwen-VLA text-to-action pretraining (2605.30280), GRAIL synthetic humanoid loco-manipulation (2606.05160)
- Covered papers: 2606.02800, 2605.30280, 2606.05160
- Shortlisted papers: 2606.03603, 2606.01247, 2606.03985

## 2026-06-10 — 2026-06-08 to 2026-06-10
- Doc: https://fudan-nlp.feishu.cn/docx/I2QadXXR6o0i9Kx8U5UcKbb5npf
- Run: runs/2026-06-10-oasis-ahawam-tbdvla/
- Deep threads: OASIS sim-to-real humanoid loco-manipulation (2606.08548), AHA-WAM async world-action modeling (2606.09811), TBD-VLA temporal block diffusion VLA (2606.07895), QGF test-time gradient guidance for flow policies (2606.11087)
- Covered papers: 2606.08548, 2606.09811, 2606.07895, 2606.11087
- Shortlisted papers: 2605.25077, 2606.06556, 2606.09669, 2606.11129, 2606.09828, 2606.07723, 2606.06476

## 2026-06-14 — 2026-06-08 to 2026-06-14
- Doc: https://fudan-nlp.feishu.cn/docx/VpjwdvelZo2tW6xUjP3ccv8lnp5
- Run: runs/2026-06-14-weaver-eurekagent-repwam/
- Deep threads: WEAVER latent world model for manipulation (2606.13672), RepWAM representation visual-action tokenizer (2606.13674), EurekAgent environment engineering for autonomous research (2606.13662)
- Covered papers: 2606.13672, 2606.13674, 2606.13662
- Shortlisted papers: 2606.12072, 2606.01027, 2606.11482, 2606.09426, 2606.08039, 2606.13681, 2606.11926, 2606.11119

## 2026-06-12 — 2026-06-11 to 2026-06-12
- Doc: https://fudan-nlp.feishu.cn/docx/YT9Rd6YeLoHZxQx1TczcBWpCn1e
- Run: runs/2026-06-12-spatialclaw-moverse-labvla/
- Deep threads: LabVLA scientific-lab VLA (2606.13578), MoVerse panoramic Gaussian world model (2606.13376), SpatialClaw code-as-action spatial reasoning (2606.13673)
- Covered papers: 2606.13578, 2606.13376, 2606.13673
- Shortlisted papers: 2606.12195, 2606.13681, 2606.13662, 2606.11926, 2606.12373

## 2026-06-11 — 2026-06-08 to 2026-06-11
- Doc: https://fudan-nlp.feishu.cn/docx/Li5YdEumooOkIrxTQgfcBivRnKe
- Run: runs/2026-06-11-alebench-worldpilot-embodiedr1-nextforcing/
- Deep threads: ALE-Bench long-horizon algorithm engineering (2506.09050), World Pilot steering VLAs with WAM priors (2606.12403), Next Forcing multi-chunk prediction for causal world models (2606.11187), Embodied-R1.5 unified embodied foundation model (2606.11324)
- Covered papers: 2506.09050, 2606.12403, 2606.11187, 2606.11324
- Shortlisted papers: 2606.09828, 2606.12072, 2606.07100

---
## 2026-08-14 — 2026-08-12 to 2026-08-14
- Doc: https://fudan-nlp.feishu.cn/docx/P9EvdAocBodeppxMXwocrPbVnrd
- Run: runs/2026-08-14-rift-forewam-simcol/
- Deep threads: 免 rollout WAM 谱系精读 + 玩具重建(RIFT 干预协议全复现:四项破坏干预 0% 成功率、终态重放部分恢复;五变体同骨干:单趟生产者 0-7.5% vs rollout 26.2%,噪声槽 7.5%>learned token 0%;计划注入探针:动作跟随指令不跟随注入缓存,读动力学值不读计划身份)(2608.11521, 2608.11605), 模拟器坍缩最小复现(Qwen3-4B×P4G-mini REINFORCE G=4×20 步:single/cot 坍缩签名可见,口头化采样保多样性方向复现;cot 修复在玩具尺度未复现)(2608.12253)
- Covered papers: 2608.11521, 2608.11605, 2608.12253, 2608.04404, 2603.16666, 2604.25859
- Shortlisted papers: 2608.11739, 2608.11671, 2608.12063, 2608.11350, 2608.12078, 2608.12314
- 背景: 免 rollout WAM 三方案同日收敛(RIFT anticipation token / ForeWAM 噪声槽 / Faster-WAM),监督信号"分布形状轴"确立(点估计退化梯度、分布保持留信息区间)
