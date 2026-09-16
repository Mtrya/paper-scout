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

## 2026-09-17 — 单锚点纯实验巡航:有技巧地找 astra+code 的能力边界(Polanyi 三机制探针族)
- Doc: https://fudan-nlp.feishu.cn/docx/J9uDdAakXo12g7x0zkEccX6SnUd
- Run: runs/2026-09-17-boundary-astra/
- Deep threads: Show-Harness 精读 + D/E 条件复测(论文唯一保留锚点;5 条件 × 5 回合 25/25 全破解——论文 1/20 的 D 条件(任意符号无约定)被 astra 开局 6 次调用探测全映射、零冗余;论文未做的 E 条件(任意符号+仅图像)5/5,三处泄漏堵净后成绩不变;论文负结果是被试的而非范式的)(2609.10522), 拦截族(机制③:每次桥调用世界前进 0.5s 不冻结;速度档 5/5/5/3/0,全知可行性上界逐回合审计——19 可捕捕 18、6 不可捕全部 ≤3 调用带数字论证放弃,唯一漏判 v4/t3 是"观测吃掉窗口"的刀锋局且 astra 自算与独立复算毫米级一致;抖动档 σ∈{5,15,30,60}° 打出本轮唯一真实破口:j5 全可捕只捕 2、j30 4 可捕捕 2——拦截落空后错选数学上无望的等速追击 43–82 步不止损,而 j15/j60 达全知最优;破口在策略元认知不在物理), 投掷族(机制②:2 维隐藏参数+5 档随机游走非平稳,25/25 与 oracle 同效率;d4 改用压平弹道缩短飞行时间的灵敏度削减策略,风漂移按 ½aT² 自动衰减), 绳索族(机制①:MuJoCo 单臂四关梯子 L0 刚杆/L1 链拉直/L2 摆形/L3 绕柱 ≥0.85 圈;L0 3/5(180° 反转未核验,余 56 预算)、L1 v1 2/5→v2 5/5(prompt 判据 A/B)、L2 5/5、L3 3/5(两败绕数达标 0.8875/0.883 但死从未告知的平坦性判据 on_table)、L3X 屏蔽泄漏环绕数 5/5(不算积分,过绕 ~20% 余量对冲,Δw 均值 0.99 vs L3 0.92);全部失败归因判据可观测性,无一例归因塑形能力;感知探针 10 例双通道:判据操作化后环绕判断 8/10(失误全是 |w|=0.500 刀锋)、自交叉 9/10 与 8/10、端点距离 10/10;环境三 bug 史含 astra 自行诊断持久化 bug)
- Covered papers: 2609.10522
- Shortlisted papers: (本轮按用户指令不看池内其他论文)
- 背景: 框架=Polanyi 默会通道三失效机制(①状态不可符号化/②结果空间病态/③关键快区间);边界图 v1:符号接地[实证攻破]、机制③[攻破一半]、机制①[实证守住但破口不在预想处——失败类=判据可观测性]、机制②[实证守住+仅论证];三路径含义:benchmark 要让错误策略有诱惑力且审计决策质量(全知上界法);蒸馏的最佳对象=可行性审计元行为;小脑领土修正为"状态说不清 × 判据只能体感核验"的交集,且余量策略(过绕/压平)能自发对冲不可观测性。

## 2026-09-16 — 2026-09-14 至 2026-09-16(astra 时代的接口三线:下行=程序、上行=逐方向不确定度、第三通道协议名存实亡)
- Doc: https://fudan-nlp.feishu.cn/docx/Yg12deKX7ogmbRx4nfVcawkQnXg
- Run: runs/2026-09-16-agp-syncworld-kdn/
- Deep threads: AGP 精读 + 实验 A(astra 本人在 2D 沙盒复刻 AGP 7 命令桥:6 任务 26/26;balance 全部直接合成 PD 控制器(θ:ω≈4:1 与 LQR 参考同构);windx 单样本系统辨识——第一投当探针反推风加速度,第二投偏差 0.6mm;balpush 禁控制器后 agent 写 shell 循环包住桥命令原地重建小脑,98/250 预算;副产物:同一份力序列开环回放发散、反馈重算收敛——反馈是控制律本体)(2609.12541), KDN 精读 + 玩具探针(循环联想记忆=线性高斯状态空间滤波,delta 规则=不追踪协方差特例;v1-v4 弧:稠密键机制无意义→同质各向同性最优→异质 70% 噪声池化仍优→100% 噪声各向同性崩溃 0.41 比固定 β 还差、对角唯一全制度近最优;机制=池化置信度被可靠多数锚定后饿死持续噪声方向;陈述:逐方向不确定度的价值在跨制度鲁棒性,不在制度内)(2609.07816), SyncWorld 复现 + 可观性探针跟进(发布权重首跑:272 帧 PSNR 27.78/SSIM 0.954,逐段无累积漂移,抓放接触段语义失败而指标钝感;证伪:标定收益真实——全零标定 4/4 组一致变差 -0.35~-0.93dB,但不按逐自由度协议分解——抹旋转三槽在转动主导 ep(4.24×)上零效应,段级相关 4 组仅 1 组 -0.56;协议结构在发布权重上名存实亡)(2609.09155)
- Covered papers: 2609.12541, 2609.07816, 2609.09155
- Shortlisted papers: 2609.12641, 2609.07398, 2609.11561, 2609.10522, 2609.05324, 2609.04280, 2609.10895, 2609.05588, 2609.07064, 2609.10540, 2609.07470, 2609.02886, 2608.16590, 2609.12419
- 背景: astra+π0.5 刷爆 RoboDojo(62.6 vs 第二 38.26;Astra Direct 仅 37.81),分层架构工业级验证,问题从"要不要分层"移到"层间怎么说话";接口三通道各投一票(下行=程序/上行=逐方向不确定度/上下文=协议名存实亡);RL 讨论留存(大脑=agentic RL 成熟;小脑=goal-conditioned RL 难在奖励;封建制解耦、版本闸口;危险区=双侧同速共适应)。

## 2026-09-14 — 2026-08-28 至 2026-09-14(OPD 解构:教师噪声、单 query、自监督熵整形 × 世界模型的结局分布)
- Doc: https://fudan-nlp.feishu.cn/docx/RdUmdp9eZodfQmxxeZ2c7NdEn3e
- Run: runs/2026-09-14-opsa-pawbench-syncworld/
- Deep threads: A1 教师噪声独立复测(真实权重:4B 教师符号噪声 31.3% vs 论文同口径 30.6%;14B 教师合计 27.2% 但结构翻转——正确答案被否定率 41.9%→66.7%、错误答案被肯定率 27.7%→13.9%;505 条响应/367 可用)(2608.31046), A2 五条件机制训练(自研轻量 PG 环,0.6B 学生 × 1.7B 教师,opd/单 query/固定负/固定正/OPSA + base 六臂同协议:**pass@4 全部 6.0% 且解出同一批三题**,avg@4 3.5–5.5%;动态按预测分离:教师 |A| 是固定负优势的 20 倍量级而长度/熵轨迹几乎重合、单 query 信号衰减最快(|A| 2.00→0.96)、固定正优势 25 步未复现论文坍缩反而熵升;三配置探针证明地板是模型真实能力而非协议)(2608.31046 + 2609.04172), A3 发布权重核验 + 截断诊断(AIME24 官方数据与 grader:base pass@4 20.0%/avg@4 10.0% → OPSA 40.0%/24.2%;30 题里 26 题的批次至少一条回答撞 16384 上限,整批自然结束的 16 条回答全对;双倍预算(32768)重测最截断的 6 题:24 条从 0 正确变 3 正确,其余失败分"没写完"(9 条连 \boxed{} 都没写出)与"写错了"(第 4 题自然结束于 23189 token 而答案 240 vs 80))(2608.31046), 实验 B 随机结局玩具世界(解析参考分布,五动作 500 次 rollout 平均 TVD:det 0.721 / VAE 0.208 / z-diffusion 0.102 / oracle 0.039;支持覆盖率 1/9、6/9、6/9、8/9)(2608.27345), SyncWorld 代码审计(标定 episode 进上下文;发布权重与论文四点差异:主干 Wan2.2-5B→Cosmos3-Nano、动作单位米→厘米、跨注意→MoT 联合自注意、标定 12→6 段——论文主表数字无法用发布权重复现)(2609.09155)
- Covered papers: 2608.31046, 2609.04172, 2608.27345, 2609.07398, 2609.09155
- Shortlisted papers: 2608.12564, 2608.24646, 2608.26872, 2609.00188, 2609.02886, 2609.03241, 2609.05588, 2609.08798, 2609.10522, 2609.10540, 2609.11561
- 背景: 两条主线在"分布"上会合——OPSD 侧证明教师内容近乎可剥离(收益 = 压制学生自己采样的低概率 token 的分布整形),世界模型侧证明单条可信轨迹 ≠ 结局分布对齐。可复用判据两条:其一,长 CoT 评测必须报告 token 预算与自然结束率(小预算下 avg@k 只是下界);其二,机制类对照实验要求学生能力高于评测地板(0.6B × MATH-500 上六臂共享同一地板)。

## 2026-08-27 — 2026-08-26(WAM 接口的抽象层级)
- Doc: https://fudan-nlp.feishu.cn/docx/C3UVd4N4cosmCxxkVFIcf43anxf
- Run: runs/2026-08-27-gigabrain-lawa-worldecho/
- Deep threads: GigaBrain-0.7 精读(三系统:S2 PaliGemma2-3B 规划 / S1 双流 VLA 3.5B+0.5B / S3 世界价值模型 5B,子目标图+二元优势注入 prompt 推理恒取 1;骨干更大更差 Gemma4 8.5B 叠衣服 0%;双流 vs 交叉注意 0.221s vs 0.073s;System3 消融礼物包装 0→80%;RoboTwin Hard 67.9% vs π0.5 46.0% 但 Easy 反输 66.8 vs 70.7;EBench 33.30 非第一)(2608.15875), LAWA 精读 + 实验 C(潜动作意图接口=冻结 ViPRA 式分词器+SAM2 掩码辅助+第一人称预训练;RoboCasa 65.6/80.8 比 Fast-WAM +9.6/+4.5,承重墙=无预训练时比 Joint-WAM 低 3.4/2.0;实验 C 玩具 WAM-IF:全量 lawatok 3.0% vs joint 26.25% 锚点假说全量证伪,少样本 lawatok 16.0% vs 其他 3/7/7% 锚点收益只在数据稀缺处显现,与论文 few-shot 放大模式同向;全量<少样本倒挂悬案)(2608.24882), WorldEcho/WorldSync 精读 + 实验 D(off-expert 动作跟随诊断:五类查询+视觉门+SE(3) NDTW 门控误差;WorldSync=扩覆盖+AFE+IE;实验 D 玩具 AC-WM EE-ADE 中位 px n=60/类:demo 1.14/xstate 2.27/perturb 3.13/random 8.64 单调退化复现;扩覆盖 random 反略差 8.93;IE 唯一改善最远分布查询 -11%,方向同论文、幅度弱得多,论文 -34%;arm-loss≈0 玩具无纹理视觉崩溃半边复现不了)(2608.24885)
- Covered papers: 2608.15875, 2608.24882, 2608.24885
- Shortlisted papers: 2608.24115(PonderPounce), 2608.24101(TrAct), 2608.23041(AutoSaddler), 2608.24714(GaussianWAM), 2608.23831(ARLI), 2608.24845(LAION-BVD)
- 背景: 三篇按接口抽象层级排开(LAWA 潜动作 token 最贴动作 / GigaBrain 标量价值+子目标图最贴语义 / WorldEcho 不立接口而给视频式未来立度量+修复配方);锚点假说分裂结果(全量证伪、few-shot 成立且放大),IE 因果消融方向成立幅度弱;逐样本 ADE 未落盘做不了显著性检验,教训记入记忆

## 2026-08-26 — 2026-08-25(WAM 接口形态专题)
- Doc: https://fudan-nlp.feishu.cn/docx/JZA7djXUEoVGTwxXQVIcpxXXnch
- Run: runs/2026-08-26-hydra0-ld4wam-unimem/
- Deep threads: Hydra-0 精读 + 实验 A(AllTracker×DROID 6 集地基噪声:中位 2-7px 干净但阵发性爆发,半数集 p95 16-84px、inlier 低至 50%,ep004 相机中途被动漂移 48px;仿射拟合吸收未知外参的残差法)(2608.18077), LD4WAM 精读 + LDM 代码核验 + 实验 B(冻结 LDM 在 OOD DROID 上的运动探针:val R²≈0.02-0.07 全地板、仅 z 维可解 0.20→0.39;DINOv3 过拟合对照 train 0.99/val 负;幅度对照排除逐集外参错位;检索 same_ep 0.376 vs cross_ep 0.162=外观泄漏;roll ±π 环绕假警报修复)(2608.22403), UniMem 精读 + openpi fork 代码审计(事件驱动记忆、关键帧缓存 90ms 恒定;null 权重 0.02 的 append-only 不对称;作者自文档化 seeding PE bug;MemER 失败模式实证分层接口脆性;无权重)(2608.22869)
- Covered papers: 2608.18077, 2608.22403, 2608.22869
- Shortlisted papers: 2608.22591(WorldToken), 2608.22364(WAM-OPD), 2608.23486(GeoWAM), 2608.20430(RISE), 2608.23224(TOWN-VLA), 2608.20169(Task-CoEvolve);2608.23189(EchoWM)、2608.23565(ReWorld)按窄定义减分
- 背景: WAM 动作接口三形态拼齐(Dyna-2 分离边际场 / Hydra-0 像素流 / LD4WAM 潜动力学桥),两个实验各咬后两者一口:像素流地基噪声阵发、潜桥 OOD 掉崖;记忆线 UniMem 投"单骨干+自我监测上行"一票。悬念:z 维幸存或可作跨域鲁棒性探针;阵发坏段是否=接触密集段

## 2026-08-24 — 2026-08-18 to 2026-08-24(推理时算力专题)
- Doc: https://fudan-nlp.feishu.cn/docx/SZx8d42kDoom9Hxn5FvcPSSBnXe
- Run: runs/2026-08-24-tau0vla-dalewm/
- Deep threads: τ₀-VLA 精读 + 代码解剖 + 启智实跑(混合线性注意力骨干 18/24 层 GatedDeltaNet;零样本夹爪常数偏移失败 MSE 0.01126;冻骨干微调 500 步 202 秒 MSE→0.00264;FM 步数 K=1≈K=10 平台期;from_pretrained 静默解冻 bug 及补丁)(2608.16885), DA-LeWM 诊断复刻到自家 LeWM/PSG 检查点(Plan-Real 0.665→0.670 接地头不修决策对齐预言证实;CEM elite 崩塌复现;Cube 缺口以扰动演示候选补上;Claim-1 0.583→0.722)(2608.18746), Dyna-2 官方博客按技术报告精读(100 万小时纯人类视频 WAM;嵌套等比例 ladder;首个人→机零样本转移 scaling law;视频=独立 scaling axis 归因实验;单步蒸馏追逐博弈 10203ms→110ms)
- Covered papers: 2608.16885, 2608.18746, dyna-2-blog(https://www.dyna.co/dyna-2);2608.14022(ForgeWM)曾精读但因不符合窄定义世界模型从报告撤下
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
