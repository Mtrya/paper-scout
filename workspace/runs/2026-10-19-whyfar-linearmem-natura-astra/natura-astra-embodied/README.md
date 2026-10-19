# Natura-1 × GPT-6-Astra:同日两份具身基础模型报告的交叉审计

线程 C,2026-10-19。审计对象是 2026-10-13 的 Natura-1 博客(即技术报告)与 2026-10-15 的 arXiv 报告 *Has GPT-6-Astra Achieved Embodied Intelligence?*(arXiv:2610.22471v1)。两份报告相隔两天,一份是 50B MoE、100 万小时人类第一视角视频预训练、20 自由度灵巧手的具身基础模型;另一份是对 GPT-6-Astra 做 12 个基准、8 种真机配置的大规模评测。它们看起来在互相打脸:Natura-1 说 GPT-6-Astra 在灵巧手平台上"零得分"、在夹爪平台上只有 16.7;SenseNova 说 GPT-6-Astra 在 RoboDojo 的 10× 预算下 81.9%、在 DexJoCo 的 10× 下 87.3%、跨 8 种本体都能干活。这份线程包的任务就是把这两个数字放在同一张协议表上,判断哪些是真矛盾,哪些只是协议差,哪些是两边都没说清。

一句话结论:**两份报告并不矛盾,它们测的是协议光谱的两端。** 在两者的交集——真机、零样本、约 60 分钟、没有人类写死的任务原语——双方都同意 GPT-6-Astra 在刚性抓放和双臂持物上可用,在高自由度灵巧手、精密接触和长程移动上失败。Natura-1 的 16.7 和"灵巧手零分"是另一套更严/不同协议下的测量,不是对 SenseNova 测量结果的反证;SenseNova 的 81.9/87.3 是 ICL + 最高 10× 步数预算下的上界,也不是"标准条件下已经解决"。真正无法对齐的是两个标签:Natura-1 的"bimanual coordination = 0.0"与 SenseNova 的"dual Piper 14/14",以及双方对"recovery"的定义。它们暴露的是评测构念(operationalization)差异,不是模型能力差异。

---

## 1. 两份产物各自是什么

### 1.1 Natura-1(Nature Will,2026-10-13)

Natura-1 是 50B 参数的 MoE transformer,在一个序列里自回归建模语言、图像、视频和动作 token;视频与动作用 flow matching 联合去噪,作者的说法是"video and action as one state"(`papers/vla/natura-1-blog.md:109-113`)。它在超过 100 万小时的第一视角人类视频上预训练,再用"有限量的机器人数据"做后训练,驱动 20 自由度灵巧手,声称维持 100 ms 的 observation-to-action 闭环(`papers/vla/natura-1-blog.md:14,26`)。数据规模是 61,702,471 个 episode、317,842 个任务、14,286 个场景;数据管线每周处理 35 万小时第一视角视频,湖仓管理数十 PB;运动重建的 MPJPE 比开源方案低 41%;训练 MFU > 50%、GPU 利用率 98%;自研 attention kernel 平均约 1,200 TFLOP/s,是调优过的 FlexAttention 的 6 倍以上,训练端到端加速 >1.7×、推理加速 2×(`papers/vla/natura-1-blog.md:81-103`)。

关键数字表:

| 项目 | Natura-1 数字 | 备注 |
|---|---:|---|
| 参数量 | 50B MoE | 单一序列模型,语言/图像/视频/动作联合 |
| 预训练人类视频 | 1,000,000 小时 | 61,702,471 episodes / 317,842 tasks / 14,286 scenes |
| 灵巧手 | 20 DoF | ROKAE AR5 双臂 + Wuji 手 |
| 控制环 | 100 ms | 博客称 end-to-end inference latency within 100 ms |
| 数据吞吐 | 350,000 小时/周 | 数十 PB lakehouse |
| 运动重建 | MPJPE −41% | 相对开源方案,内部 mocap 真值 |
| 训练效率 | MFU > 50%,GPU 98% | 数百 GPU 的长任务 |
| Attention kernel | ~1,200 TFLOP/s | >6× tuned FlexAttention |
| 12 维基准总分 | 53.4(Base)/ 59.1(Agentic) | 灵巧手平台 |
| 夹爪平台基线 | π0.5 13.5 / GPT-6-Astra 16.7 | ARX AC One 双臂夹爪 |
| 平均执行时间 | 63.16 s(Base)/ 139.31 s(Agentic) | π0.5 289.11 s;GPT-6-Astra 2,734.56 s |

12 维基准的分数(来自博客内嵌的 `score-embed.html`,不是论文表格;Natura-1 只公布了分数,没有公布每个维度的任务数、试验次数和聚合权重):

| 维度 | Natura Base | Natura Agentic | GPT-6-Astra(夹爪) | π0.5(夹爪) |
|---|---:|---:|---:|---:|
| D01 Pose control & contact | 56.9 | 63.2 | 14.4 | 8.6 |
| D02 Bimanual coordination | 50.0 | 61.5 | 0.0 | 0.0 |
| D03 In-hand manipulation | 16.7 | 16.7 | not scored | not scored |
| D04 Tool use | 57.3 | 66.7 | 10.8 | 0.0 |
| D05 Goal-directed reasoning | 32.8 | 35.4 | 15.5 | 10.4 |
| D06 Spatial & scene generalization | 90.9 | 100.0 | 39.3 | 20.6 |
| D07 Tool & object generalization | 100.0 | 100.0 | 40.8 | 10.6 |
| D08 New-action generalization | 29.4 | 64.3 | 13.8 | 7.8 |
| D09 Compositional generalization | 73.7 | 77.8 | 65.1 | 36.2 |
| D10 Planning | 38.5 | 55.6 | 20.6 | 25.0 |
| D11 Memory | 57.7 | 70.0 | 37.5 | 10.9 |
| D12 Recovery | 68.6 | 83.3 | 75.0 | 44.6 |

博客用四个示例任务说明这套 benchmark 的设计:tray loading & transport、extract/wipe/dispose、adding sugar & stirring、preparing a pot;四个任务都不在 Natura-1 的后训练数据里,场景、物体和动作组合也都是新的。它们不是 12 个维度的全部题目,只是让读者判断"泛化距离"的样例。

三点必须先说清。第一,**53.4/59.1 不是"无机器人数据"意义上的零样本**:博客说所有评估使用"a single fixed training set of real-robot trajectories, with no task-specific tuning",也就是说 Natura-1 已经在机器人数据上后训练过,基准测的是从固定训练集到未见任务/场景的泛化(`papers/vla/natura-1-blog.md:39-41`)。它对基准任务是零任务微调的,但不是无机器人数据的零样本;把它与 Astra 的零样本 coding policy 并排时,必须带上这个区别。第二,**16.7/13.5 是夹爪平台,53.4/59.1 是灵巧手平台**,这是一次跨本体的并排,不是同本体对比;博客自己承认"没有公开的灵巧手基础模型,严格 like-for-like 的基线不存在"(`papers/vla/natura-1-blog.md:39`)。第三,**16.7 的总体分不能用已公布的 12 个维度分按简单平均复算**:12 维的未加权平均是 30.3(Astra,跳过 not scored),而公布总体是 16.7;Base 的未加权平均是 56.0,公布 53.4;Agentic 的未加权平均是 66.2,公布 59.1。总体分显然按题目/维度加权,但权重、题目数、每维试验数都没有公布。这意味着 16.7 只能当作"博客口径下的一个总分",不能当作可复算的均值。另外,**"Agentic" 没有定义**:博客只给了 59.1 分和 139.31 s,比 Base 高 5.7 分、慢 2.2×,但没有说它加了什么(测试时搜索?工具?harness?),所以不能把它当作"同一模型加 agent 壳"的干净对照。

### 1.2 SenseNova 报告(arXiv:2610.22471,2026-10-15)

报告把 GPT-6-Astra 当作被测系统,在 12 个基准、8 种真机配置上做了评测,并额外做了 S2(System 2,即 GPT-6-Astra)+ S1(System 1,即 π0.5)的 harness 消融,以及一个"下一代基准"原型。核心协议是两条:允许从训练集演示做 in-context learning(ICL),以及把执行步数预算放宽到官方的 10×(`papers/robotics/astra-embodied-2610.22471.md:17,33-37`)。12 个基准不是 12 个能力维度,而是 12 个 benchmark suite,分属桌面操作、灵巧手、人形、长程家务和视觉语言导航五类(`papers/robotics/astra-embodied-2610.22471.md:61`)。

12 个基准的关键结果(Table 2,`papers/robotics/astra-embodied-2610.22471.md:1150`;SOTA 为论文报告的最佳已发表结果,Agentic SOTA 为最佳已发表 agentic 方法):

| 类别 | 基准 | SOTA | Agentic SOTA | Astra 1× | Astra 10× |
|---|---|---|---:|---:|---:|
| 桌面 | LIBERO-Long++ | 82.0(LiLo-VLA) | — | 97.5 | 100.0 |
| 桌面 | LIBERO-PRO | 50.0(πRLinf) | 92.6(RPent/GPT-6 Astra) | 98.3 | 99.3 |
| 桌面 | LIBERO-Ultra-Long | 44.0(LiLo-VLA) | — | 82.2 | 98.9 |
| 桌面 | RoboDojo | 29.6(Awomo-0.5) | 31.4(PhysicalRSI) | 27.6 ICL+直接 / 54.2 ICL+VLA | 81.9 |
| 桌面 | RoboLab | 42.9(FLUX 3 Action) | — | 57.1 | 69.2 |
| 人形 | SIMPLE | 92.5(Ψ0) | — | 96.3 | 98.8 |
| 移动/家务 | BEHAVIOR | 14.8(Rondo) | 5.3(Agent) | 37.0 | 未扩展 |
| 灵巧手 | DexJoCo | 40.3(π0.5) | — | 78.2 | 87.3 |
| 灵巧手 | Bench2Dex | 25.5(π0.5) | — | 25.6 | 61.5 |
| 导航 | VLN-CE RxR | 76.5(Qwen-RobotNav) | — | 82.0 | 90.0 |
| 导航 | VLN-CE R2R | 77.4(Robostral Navigate) | 81.3(GPT-6-Astra ultra) | 80.0 | 88.0 |
| 导航 | LHPR-VLN | 2.4(FantasyVLN) | 5.3(Claude Code Opus-5) | 21.1 | 28.9 |

这张表最容易被误读的地方是 RoboDojo 和 Bench2Dex。RoboDojo 的 27.6 是 ICL + 直接控制、标准预算下的成功率,低于 SOTA 29.6 和 Agentic SOTA 31.4;54.2 是 ICL + closed-loop VLA 的成功率;81.9 是 10× 预算下的成功率。没有 ICL 的直接控制基线是 15.56(Table 13)。Bench2Dex 的 25.6 与 π0.5 的 25.5 基本打平,61.5 是 10× 预算下的结果。也就是说,**在标准预算、直接控制的条件下,Astra 并没有全面超过已发表方法;它的"unprecedented performance"主要来自 ICL、VLA 协作和 10× 步数**。这一点是后面与 Natura-1 对齐的关键。

ICL 与预算的具体口径(按 benchmark 分):

| 基准 | 任务/试验数 | ICL 演示来源与数量 | 1× 预算 | 10× |
|---|---|---|---|---|
| LIBERO-PRO | 8 suites × 10 tasks × 10 init states = 800 episodes | 从公开 exploration 蒸馏的 package;不含测试轨迹/初始状态 | 10,000 env steps 或 100 planner turns | 累计 10×,同一 episode 续跑 |
| LIBERO-Long++ / Ultra-Long | Long++ 12 configurations,Ultra 9;各 10 次 bare random reset | 4,050 条 atomic-skill 轨迹(150/skill;351,999 transitions) | Long++ 600/800 steps;Ultra 1,800/2,000/3,200 steps | 累计 10× |
| RoboDojo | 42 tasks × 5 layouts = 210 episodes | 34 个任务各 10 条 demo;Open 8 任务共享 340 条 demo(10/task) | 每任务 200–1,900 steps | 累计 10× |
| RoboLab | 120 tasks;基线各 1,200 episodes | 无训练集,无 ICL | 官方预算 | 10× |
| SIMPLE | 6 tasks × 3 levels × 10 instances = 180;Matched-8 80,Common-4 120 | task-level ICL,数值训练轨迹;数量未披露 | 400–1,200 steps | 10× |
| BEHAVIOR | 100 rollouts | 训练任务中的物体 affordance/恢复知识 | 官方长预算 | 不扩展 |
| DexJoCo | 11 tasks × 5 episodes = 55 | phase-indexed motion references;数量未披露 | 官方预算 | 10× |
| Bench2Dex | 26 tasks × 3 episodes = 78 | 每任务 100 条 seen-split replay | 官方预算 | 10× |
| VLN-CE R2R/RxR | 各 100 episodes(val-unseen) | 训练场景的 exploration notes/code | 官方预算 | 10× |
| LHPR-VLN | 114 episodes(从 205 中筛出) | 训练场景的 exploration notes/code | 500 actions | 5,000 actions |

这些数字来自 `papers/robotics/astra-embodied-2610.22471.md` 的 Appendix C(尤其 C.1–C.3、C.5、C.9、C.10)与 Table 2/5/7/8/9/10/11/12。

### 1.3 Harness 消融:S2+S1 vs 单 S1 / 单 S2

这是报告里最有信息量的一张对照表,也是理解"81.9 从哪来"的关键。42 个 RoboDojo 任务,3 个 seed;S2 = GPT-6-Astra,S1 = 在 RoboDojo 上微调的 π0.5(`papers/robotics/astra-embodied-2610.22471.md:237-251`,Table 13/Fig. 7):

| 配置 | 成功率 SR(%) | Token(M) | 相对基线 |
|---|---:|---:|---|
| S2 直接控制基线 | 15.56 | 2.56 | — |
| S1 单独(π0.5,无 S2) | 6.91 | — | S1 单独远弱于 S2 |
| S2→S1 开环 | 5.83 | 0.387 | 开环协作反而低于 S2 单独 |
| S2↔S1 闭环,L=1 | 23.61 | 7.664 | 比开环 +20.00 pp |
| S2↔S1 闭环,L=4 | 25.83 | 2.796 | 比 L=1 再 +2.22 pp,token 少 63.5% |
| S2 + ICL | 29.67 | 9.567 | 比基线 +14.11 pp |
| S2↔S1 闭环 + ICL | 54.17 | 19.335 | 最高,但 token 是 L=4 的 6.9× |

读法:S1 单独只有 6.91,说明"把执行全交给 VLA"在当前 π0.5 上不够;S2 单独 15.56,说明"只靠 agent 直接控制"也不够;闭环把两者接起来能到 23.61–25.83;ICL 再把它推到 29.67;ICL + 闭环到 54.17,但成本 19.3M token。**这条链的结论不是"S2+S1 最好",而是"能力提升与 token 成本强耦合";报告自己用"costly but effective"来概括,并在结论里把"selective verification/recovery"列为下一步。**另外,L=4 的 2.796M token 比 L=1 的 7.664M 少 63.5%,成功率还略高,说明"审查更长的动作序列"是性价比更高的接口设计。

### 1.4 真机 8 配置结果

报告的真机部分是零样本、人类上下文、墙钟预算,和仿真的 ICL + 10× 是两套协议(`papers/robotics/astra-embodied-2610.22471.md:286-296,1473-1477`,Table 14,App. E):

| # | 平台 | 本体 | 任务与结果(成功/试验) | 关键上下文 | 时间 |
|---|---|---|---|---|---|
| 1 | Franka 3 + Robotiq 2F-85 | 单臂夹爪 | block stacking 6/6;cloth folding 2/4;plug insertion 0/3 | robot settings;无演示 | 7–16 min |
| 2 | Franka Panda + Franka Hand | 单臂夹爪 | tape-roll stacking 3/3;towel folding 2/2;dark-blue cloth 1/2;plug insertion 0/5 | tool schemas、joint limits、safety | 7–37 min;一次 196 min |
| 3 | AgileX PiPER + Pika | 单臂夹爪 | container nesting 3/3;cloth folding 3/3;plug insertion 0/3 | step prompt、可抓点 | 60 min/trial |
| 4 | Dual AgileX Piper | 双臂夹爪 | 12 tasks,14/14 | robot specs/manuals;无时间上限 | 6–21 min |
| 5 | Tianji M6S + Wuji Gen-1 | 双臂灵巧手 | toy:模型写 grasp code 0/3;toy:人类写 script 3/3;bottle:人类写 script 1/3 | tool schemas、hand presets、人类 grasp scripts | 60 min/attempt |
| 6 | Galaxea R1 Pro | 轮式移动操作 | find washer/press button 2/2;flowers 0/2;6 个桌面任务 9/9 | 一次性机器人探索;无演示 | 6–60 min |
| 7 | AgileX Tracer + xArm7 | 轮式移动操作 | navigation/search 9/23(5 个子任务) | 无 robot model/SLAM/collision;无演示 | 0.2–8.3 min |
| 8 | Unitree G1 + 两个 Wuji Gen-1 | 人形 | toy pick-and-drop 2/3;turn/approach/stop 3/3 | tool schemas、joint limits、binary hands;无演示 | pick >1 h;walk 8–13 min |

报告的概括是:刚性抓放 17/18;可变形物体(如 Franka 3 上的 cloth folding)2/4;精密接触(plug insertion)跨 3 个平台 0/11;长程移动(navigation/search)9/23(`papers/robotics/astra-embodied-2610.22471.md:374`)。这几个数字比 8 行表更有用,因为它们把"哪些能力已经可靠、哪些还不行"分开了。

### 1.5 失败归因:论文没有给统一分布

报告没有一张"失败原因占比"的图或表。它给的是 Fig. 6 的三类行为模式(验证物理成功、从后果性动作恢复、长程保持进展),加上 App. E 每个平台的逐条记录。下面是我从这些记录归并的计数/证据,不是论文的正式统计,不能当作百分比:

| 归因 | 报告中的证据 | 说明 |
|---|---|---|
| 物理成功验证失败 | RoboDojo Hang mugs、Fill egg holder;Bench2Dex Jigsaw Puzzle Assembly(Fig. 6a) | 视觉重叠/自报完成 ≠ 实际接触/支撑;修正过程不收敛 |
| 不可逆后果 | RoboDojo Pour liquid into cup(杯子翻倒、液体洒出);DexJoCo Assembly(peg 掉地)(Fig. 6b) | 后续重试无法撤销;预算被浪费 |
| 长程进展丢失 | RoboDojo Store laptop and headphones;BEHAVIOR can_meat(罐子掉落)(Fig. 6c) | 早期子目标被后续动作破坏;需要重新建立 |
| 硬件/安全约束 | Franka 3 保护性停止(薄布/插头);PiPER 插头 2/3 次 joint collision fault;Tianji 手指 62 °C 堵转;Galaxea 躯干过热终止 2/2 次采花;G1 一次 run 26 次手臂运动 18 次 stalled_short/8 次 timeout | 仿真里不存在的失败源;agent 无法从硬件级 fault 恢复 |
| 人类先验/演示缺失 | Tianji 无 script 0/3 vs 有 script 3/3;plug 0/11;Tracer 9/23 | 不是"不会控制",而是"不知道该用哪个抓法/先验" |
| 视角/遮挡 | 单臂 Franka 3 插花失败(腕相机看不到瓶口);双臂 Piper 移动自由臂相机后成功 | 主动视觉是能力的一部分;遥操作演示很少包含"移相机" |
| 成本/延迟 | G1 中位 24.84 s/次 API,88.8–91.7% 墙钟;约 90% 时间在 deliberation;Natura 基准上 Astra 平均 45.6 min/任务 | 物理世界不暂停;时间和 token 是硬成本 |

如果要在主报告里用一个数字概括失败结构,最稳的是报告自己的三分类(验证/不可逆/长程进展),而不是我归并的计数。计数只适合说明"硬件/人类先验/视角"这些仿真里看不到的失败源在真机上占比不小。

---

## 2. 交叉审计 A:Natura-1 的基线数字 vs SenseNova 自己测的 Astra

### A.1 先把协议摆在同一张表上

两份报告的关键差异不是"谁测得更准",而是"测的是哪一种信息/时间/本体条件"。把两者压成一个 2×2 矩阵,矛盾立刻变成可解释的:

| 条件 | SenseNova 的测量 | Natura-1 的测量 |
|---|---|---|
| 仿真 + ICL + 1× | RoboDojo 27.6 ICL+直接 / 54.2 ICL+VLA;Bench2Dex 25.6;DexJoCo 78.2 | 不适用 |
| 仿真 + ICL + 10× | RoboDojo 81.9;Bench2Dex 61.5;DexJoCo 87.3 | 不适用 |
| 真机 + 零样本 + 人类上下文 + ~60 min | 刚性抓放 17/18;dual Piper 14/14;plug 0/11;Tracer 9/23;Tianji 手 0/3(无人类脚本) | 灵巧手平台:GPT-6-Astra 60 min 内无得分;夹爪平台:16.7 |
| 真机 + 固定机器人训练集 + 60 min + 无 ICL | 不适用 | Natura-1 Base 53.4 / Agentic 59.1(灵巧手);π0.5 13.5 / Astra 16.7(夹爪) |

这张表说明:Natura-1 的 16.7 和"灵巧手零分"落在最严格的右下格;SenseNova 的 81.9/87.3 落在最宽松的左上格。把它们并排说成"同一件事的两个答案",是范畴错误。真正可比较的是右下格与 SenseNova 的真机零样本格(第三行),以及左上格与 Natura-1 的固定训练集泛化格(第四行)。

### A.2 Natura-1 的数字与 SenseNova 的 Astra 能力是否相容?

**结论:在"高自由度灵巧手不能自主控制"这一点上,两份报告高度相容;在"夹爪/双臂操作"上,数字差异主要由协议解释,不能直接互证或互否。**

先看灵巧手。Natura-1 说,在 ROKAE AR5 + Wuji 手上,π0.5 即使在本体数据上微调也失败,GPT-6-Astra 在每任务 60 分钟内"fails to score"(`papers/vla/natura-1-blog.md:39`)。SenseNova 在 Tianji M6S + Wuji Gen-1 手上测得:GPT-6-Astra 自己写 grasp code 时 0/3;人类写 grasp script 后 toy grasp 3/3、bottle grasp 1/3(其中一次是超时后经人类反馈成功)(`papers/robotics/astra-embodied-2610.22471.md:2100-2136`,Table 14)。两边的手都是 Wuji,任务都是抓取,结论完全一致:**没有人类先验时,Astra 不能自主完成高自由度手的抓取;有了人类写的 grasp primitive,简单抓取可以成功。**Natura-1 没有给 Astra 人类写的 grasp script,所以"no score"是"无人先验"条件下的下界;SenseNova 的 3/3 是"有人类先验"条件下的上界。两者不是矛盾,而是同一个变量的两端。

这也解释了一个容易被忽略的细节:Natura-1 的灵巧手基线只有 π0.5 和 Astra 两个,而 π0.5 的失败是"即使在本体数据上微调也失败"。SenseNova 没有测 π0.5 在 Tianji 手上的表现,所以无法判断 π0.5 的失败是本体/数据问题还是方法问题。但两边的 Astra 结论可以互相印证。

再看夹爪。Natura-1 的 Astra 16.7 与 SenseNova 的 RoboDojo 27.6(ICL+直接、1×)、RoboLab 57.1、dual Piper 14/14 看起来差距很大。拆开看,至少有四个协议差:

1. **训练制度不同。**Natura-1 的 53.4/59.1 是机器人数据后训练后的模型,基准测的是从固定训练集到未见任务的泛化;Astra 的 16.7 是零样本 coding policy,没有机器人后训练(`papers/vla/natura-1-blog.md:39-41`)。这不是"同一个模型有没有后训练"的对比,而是"有机器人后训练 vs 没有"的对比。
2. **本体不同。**Natura-1 用 20 DoF 灵巧手,基线用 ARX AC One 夹爪;灵巧手与夹爪的任务难度、动作空间、接触条件都不同。博客自己说严格 like-for-like 基线不存在。16.7 是"夹爪上的 Astra",不是"灵巧手上的 Astra"。
3. **时间预算不同且是硬约束。**Natura-1 每任务 60 分钟;Astra 的平均执行时间是 2,734.56 s = 45.6 分钟,已经用掉 60 分钟上限的 76%,均值被上限截断。SenseNova 的 RoboDojo 10× 结果 81.9 说明 Astra 的很多成功需要更多交互/修正;在 60 分钟硬上限下,这些交互被截掉(步数与墙钟不是同一量,但方向上一致)。反过来,SenseNova 的真机实验多数是 60 分钟/次,但它的仿真结果不是。
4. **任务选择不同。**Natura-1 的 12 维基准包含 in-hand manipulation、tool use、new-action generalization 等对灵巧手/泛化更敏感的维度;RoboDojo 是 42 个桌面任务,RoboLab 是 120 个语言条件桌面任务,dual Piper 是 12 个双臂演示任务。它们不是同一张考卷。

因此,16.7 与 27.6/57.1/14/14 的差异不能用来证明"某一方测错了"。它们能证明的是:**Astra 的能力强烈依赖本体、训练制度、时间预算和任务分布;在 Natura-1 选的那组条件下,它只有 16.7。**

补一个本仓库的第一方校验:2026-09-16 我们用 Astra + π0.5 在 RoboDojo 上测到 62.6(第二名 38.26),Astra Direct 37.81(`runs/INDEX.md:78`)。这三个数与 SenseNova 的 54.17(VLA-assisted)、27.6(ICL+直接)、81.9(10×)同一量级但排序不同,差异同样来自 harness/预算/ICL;它至少说明 SenseNova 的 RoboDojo 数字不是孤例。2026-09-17 的 boundary 线程则把 Astra 的失败定位在"策略元认知/判据可观测性",而不是物理能力——这与 SenseNova 的"验证/不可逆/长程进展"三分类可以互相翻译。

### A.3 "12 维能力基准"与"12 个基准"不是同类

Natura-1 的"12"是一个单平台的能力 rubric:3 个一级组(Dexterous control、Instruction following、Long-horizon execution),12 个二级维度;Natura-1 自己的 12 个分数全部来自同一个双臂灵巧手平台,而 GPT-6-Astra 和 π0.5 的分数来自另一个双臂夹爪平台,所以这张表本身已经是"两个平台 + 两套训练制度"的拼表。SenseNova 的"12"是 12 个独立的 benchmark suite,跨桌面操作、灵巧手、人形、长程家务、导航五类,使用多个仿真器,另有 8 种真机配置;每个 suite 有自己的任务、观测、动作接口和评分。把它们都叫"12 维/12 基准"会制造一种虚假的可比性。下面这张映射表只在对齐"构念"的层面成立,不能把两边的分数直接相减:

| Natura-1 维度 | SenseNova 最接近的证据 | 判断 |
|---|---|---|
| Pose control & contact | DexJoCo 78.2(1×)/87.3(10×);Bench2Dex 25.6/61.5;真机 plug 0/11 | Astra 在仿真灵巧接触上强,在真机精密插入上弱;方向一致 |
| Bimanual coordination | RoboDojo 双臂整体 27.6 ICL+直接/54.2 ICL+VLA;dual Piper 14/14 | **表面矛盾**:Natura 给 Astra 0.0,SenseNova 给 14/14;见 A.4 |
| In-hand manipulation | 无直接对应;Bench2Dex/DexJoCo 最接近 | 不能比较;Natura 的 16.7 是唯一直接测量 |
| Tool use | RoboDojo 工具任务;真机 dual Piper 插花成功 | 方向一致:Astra 能用工具,但不稳 |
| Goal-directed reasoning | LIBERO-PRO 98.3;RoboLab 57.1 | 不可比:LIBERO-PRO 接近饱和,RoboLab 中等 |
| Spatial & scene generalization | LIBERO-PRO 98.3;SIMPLE L2 31/40 | 混合:仿真强,真机空间泛化弱 |
| Tool & object generalization | RoboDojo Generalization 28.3(1×)/75.0(10×) | 方向一致:ICL/预算能大幅提升 |
| New-action generalization | 无直接对应;RoboDojo Open 组最低 | 方向一致:跨任务迁移弱 |
| Compositional generalization | LIBERO-Long++ 97.5;Ultra-Long 82.2 | SenseNova 更高,但任务不同 |
| Planning | BEHAVIOR 37.0;LHPR-VLN 21.1 | 方向一致:长程规划仍是短板 |
| Memory | RoboDojo Memory SR 70.0(1×)/96.7(10×) | SenseNova 更高,任务不同 |
| Recovery | 10× 预算带来大幅恢复;但不可逆失败无法恢复 | **标签冲突**:Natura 75.0 是"可重试的恢复",SenseNova 说的是"不可逆后果";见 A.4 |

### A.4 两个真正对不上的标签

**Bimanual coordination = 0.0 vs dual Piper 14/14。**Natura-1 给 GPT-6-Astra 在夹爪平台的 bimanual coordination 是 0.0,π0.5 也是 0.0;而 SenseNova 的 dual AgileX Piper 做了 12 个双臂任务、14/14 全成功,包括双臂举篮、双手持牛奶盒、双手折毛巾等(`papers/robotics/astra-embodied-2610.22471.md:1949-1992`,Table 14)。这不是小差异,是 0 与 100% 的差异。可能的解释有三类:(a) Natura 的 bimanual 维度要求的是灵巧手之间的协同(例如双手交接、双手同时操作一个物体),而夹爪基线的 0.0 是因为夹爪根本不该被期待完成这些任务;(b) 两边的任务难度/评分阈值不同,Natura 的 bimanual 任务可能包含更细的接触/时序要求;(c) Natura 的 0.0 是某个小子集的全部失败,样本量很小,不能外推。Natura 没有公布维度任务数和试验数,所以无法判定是哪一种。**在拿到 Natura 的原始 rubric 之前,这个 0.0 不应被用作"Astra 不会双臂协调"的证据;SenseNova 的 14/14 也不应被用作"Astra 的双臂协调已经解决"的证据,因为它的任务是演示性 rollout,不是随机化 benchmark。**

**Recovery = 75.0 vs "cannot recover from consequential/irreversible failures"。**Natura-1 给 Astra 的 recovery 是 75.0,是它在夹爪平台上的最高维度,甚至高于 Natura-1 Base 的 68.6;但 SenseNova 的失败分析明确说,Astra 不能从不可逆后果中恢复:倒洒的液体、掉到地上的 peg、被撞倒的耳机,都会让后续子任务失败(`papers/robotics/astra-embodied-2610.22471.md:223-225`,Fig. 6)。这两个"recovery"不是同一个构念:Natura 的 recovery 更可能是"抓取失败后重试/换策略",SenseNova 的 recovery 是"对已经改变且不可撤销的世界状态做恢复"。两者可以同时为真:Astra 会重试,但不会撤销。Natura 的 75.0 不能用来反驳 SenseNova 的失败分析,反之亦然。

### A.5 一个缺失的关键交叉实验

两份报告都缺一个能一锤定音的实验:**把 Natura-1 放到 SenseNova 的 12 个基准上,把 Astra + SenseNova 的 ICL package 放到 Natura-1 的 12 维基准上。**现在两边各说各话,是因为:

- Natura-1 的 Astra 基线是零样本 coding policy,没有 ICL,没有人类 grasp script,没有 10× 预算;SenseNova 已经证明这三项都能显著改变 Astra 的成绩。
- SenseNova 的 Astra 评测没有灵巧手平台的 12 维 rubric,也没有 Natura-1 的固定训练集泛化协议;它的 DexJoCo/Bench2Dex 是仿真,不是 ROKAE/Wuji 真机。
- Natura-1 的 53.4/59.1 是后训练模型;SenseNova 的 Astra 是冻结权重的通用模型。两者不是同一训练制度的对比。

在上述交叉实验完成之前,最诚实的表述是:**Natura-1 证明了"人类视频预训练 + 灵巧手后训练"可以在它自己的 12 维 rubric 上拿到 53.4/59.1;SenseNova 证明了"GPT-6-Astra + ICL + 10× 预算"可以在 12 个 benchmark 上拿到很高的分数;两者都没有证明对方条件下的能力。**把 53.4 与 16.7 直接相减,或者把 81.9 当作 Astra 的"真实水平",都是把协议当成了模型。

---

## 3. 交叉审计 B:Natura-1 的两点 scaling 与 Dyna-2 的人→机 scaling

### B.1 换算

Natura-1 的报告只给了两个数据点:约 44,000 小时和 1,000,000 小时;指标是 held-out 人类数据的 Action MSE/MAE,以及"same weights"在 held-out 真机验证集上的 MSE/MAE;两者都是 teacher forcing,不是闭环成功率(`papers/vla/natura-1-blog.md:117`)。把相对变化代入幂律 `q(D) = q0 (D/D0)^α`,得到:

| 指标 | 44k→1M 的相对变化 | 拟合 α | 每翻倍变化 |
|---|---:|---:|---:|
| 人类 Action MSE | −47.7% | −0.2075 | −13.4% |
| 人类 Action MAE | −37.3% | −0.1494 | −9.8% |
| 真机 Action MSE | −67.7% | −0.3618 | −22.2% |
| 真机 Action MAE | −49.3% | −0.2175 | −14.0% |

计算脚本是 `code/scaling_fit.py`,输出在 `code/scaling_fit.txt`。数据比是 22.7×,所以 α 的绝对值看起来比常见的 LLM scaling 指数大;但这不是"每个数量级"的指数,而是"22.7 倍数据"的等效指数。

### B.2 与 Dyna-2 的对比

Dyna-2 的博客给了明确的幂律形式(以小时为 D):

- 人类 acc@0.5 = 0.357 · D^+0.0203,R² = 0.865
- 人类 acc@0.1 = 0.0116 · D^+0.0606,R² = 0.926
- 零样本机器人 acc@0.5 = 0.0241 · D^+0.139,R² = 0.918

Dyna-2 的 ladder 是 1k/10k/100k/1M 四个 rung,39 个机器人任务,固定动作数据、只变人类视频小时;它报告的是"人类域内"和"人→机零样本"两条曲线。Natura-1 只有两个 rung,没有中间点、没有 R²、没有误差棒,而且它的"as training advances"不是纯数据 ladder——训练算力、checkpoint schedule、可能还有后训练配方都随"小时数"一起变。把两者放在同一张表上:

| 对比项 | Natura-1 | Dyna-2 |
|---|---|---|
| 数据 ladder | 44k → 1M(2 点) | 1k → 10k → 100k → 1M(4 点) |
| 指标 | 人类/真机 Action MSE、MAE | 人类/机器人 acc@0.5、acc@0.1、MSE |
| 拟合 | 两点精确拟合,无残差自由度 | 幂律拟合,R² 0.865–0.926 |
| 控制变量 | "as training advances",算力/配方共变 | 固定动作数据,只变视频小时 |
| 评测 | teacher forcing 的离线动作误差 | 离线动作预测 + 39 任务机器人集 |
| 人→机指数 | 真机 MSE α=−0.362;人类 MSE α=−0.208;比值 1.74× | 机器人 acc@0.5 α=+0.139;人类 acc@0.5 α=+0.0203;比值 6.85× |
| 证据强度 | 弱到中:方向性、假设生成 | 中到强:多点、多指标、有 R²;仍为自报、无代码/权重 |

方向上的确有一致性:两份报告都看到"机器人侧/人→机侧的斜率比人类域内更陡"。Dyna-2 的机器人指数是人类指数的 6.85 倍;Natura-1 的真机 MSE 指数是人类 MSE 指数的 1.74 倍。但**这个一致性只能当作方向性互证,不能当作定量复现**,因为:(a) 指标不同(MSE vs accuracy);(b) 数据规模区间不同(44k 起 vs 1k 起);(c) Natura-1 的 44k→1M 不是纯数据 ladder;(d) Natura-1 的"真机验证"是否对每个 checkpoint 都做了相同的机器人后训练,博客没有说清——"same weights ... without further training for this evaluation"只能推出"没有为这次评测额外训练",不能推出"这些权重没有做过机器人后训练"。

### B.3 两点式 scaling 的证据强度边界

两点式拟合的本质是:任何单调曲线都能穿过两个点。它不能区分幂律、对数、指数饱和、分段线性;它没有残差,所以没有 R²、没有置信区间、没有 model comparison。Natura-1 甚至没有公布两个点的绝对值,只公布相对变化,所以连"曲线长什么样"都无法画出来。再加上 MSE 和 MAE 给出不同的 α(人类 −0.208 vs −0.149;真机 −0.362 vs −0.218),说明这两点之间误差分布的形状也在变,不是简单的"整体乘一个因子"。

因此,对 Natura-1 的 scaling 最合适的表述是:**它给出了一个很强的存在性信号——100 万小时人类视频的预训练确实让真机动作误差大幅下降,而且下降幅度大于人类域内——但它没有建立一条可检验的 scaling law。**把它称为"scaling law"会过度声称;把它称为"two-point scaling signal"更准确。Dyna-2 的曲线更强,但也是公司自报、无代码无权重;两者都应打折。

一个能真正推进的实验:用 Natura-1 的中间 checkpoint(如果作者愿意放)在 Dyna-2 的 39 任务机器人集上评测,或者反过来把 Dyna-2 的 1k/10k/100k/1M 四个 checkpoint 放到 Natura-1 的 12 维基准上。只有交叉评测才能判断"人→机指数"是模型/数据的性质,还是评测集的性质。

---

## 4. 交叉审计 C:100 ms 控制环 vs "物理约束下不能用更长执行换成功率"

### C.1 两个数字测的不是同一层

Natura-1 的 100 ms 是**策略推理层**的 observation-to-action 延迟:博客说 Rust orchestrator 重叠 observation streaming、computation 和 action transfer,fused CUDA kernels 压缩 critical path,端到端推理在 100 ms 内,并能在多台物理机器人上并发(`papers/vla/natura-1-blog.md:26,97-105`)。SenseNova 的 24.84 s 是**agent deliberation 层**的单次 API 请求中位延迟:G1 上 108 次归档调用,中位 24.84 s,API 请求墙钟占整个 run 的 88.8–91.7%,图像采集占 4.5–9.0%;Astra 在每个 run 里约 90% 的时间在"思考",一次操作 trial 超过一小时(`papers/robotics/astra-embodied-2610.22471.md:394`,App. E.9)。一个是 10 Hz 的 policy loop,一个是 0.04 Hz 的 agent loop,相差两个数量级以上。

所以,这两者并不直接矛盾:**Natura-1 用快速策略避免了 deliberation 时间这一项成本;SenseNova 说的是,当你的策略是慢速 agent 时,物理世界不会等你,额外的执行步数/时间不能像仿真里那样免费换成功率。**两者其实在说同一件事的两面:物理约束把"时间"变成硬成本,Natura-1 的答案是"把策略做快",SenseNova 的答案是"不要用 agent 的长 deliberation 去换成功率"。

### C.2 但 100 ms 的声明本身需要打折

博客没有给 100 ms 的测量协议:硬件型号、batch size、active parameter 数、action chunk 长度、是否包含图像传输/预处理、是单次 forward 还是 chunk 级 amortized latency、并发几台机器人、p99 还是均值。50B MoE 在 100 ms 内完成一次完整 forward 是极不寻常的;更可能的实现是 action chunking + 异步流水线:一次 forward 生成一段动作,控制环以 100 ms 的节奏消费,而不是每 100 ms 重新跑一次 50B 模型。如果是这样,"100 ms control loop"与"100 ms end-to-end inference"就不是同一个量。博客同时写了 "strict 100-millisecond control-loop budget" 和 "End-to-end inference latency within 100 ms"(`papers/vla/natura-1-blog.md:97,105`),但没有解释这两者是否等价。**在拿到延迟分布和 action horizon 之前,100 ms 应当被当作一个系统级声明,而不是可复现的测量。**

### C.3 成本对比:数字本身很有说服力

即使对 100 ms 打折,Natura-1 与 Astra 的成本差距是量级性的:

| 模型 | 平台 | 12 维基准总分 | 平均执行时间 | 相对 Natura Base |
|---|---|---:|---:|---:|
| Natura-1 Base | 20-DoF 灵巧手 | 53.4 | 63.16 s | 1× |
| Natura-1 Agentic | 20-DoF 灵巧手 | 59.1 | 139.31 s | 2.2× |
| π0.5 | ARX AC One 夹爪 | 13.5 | 289.11 s | 4.6× |
| GPT-6-Astra | ARX AC One 夹爪 | 16.7 | 2,734.56 s | 43.3× |

Astra 的平均 2,734.56 s 是 45.6 分钟,已经是 60 分钟硬上限的 76%;这个均值被上限截断,真实的无上限均值很可能更长。SenseNova 自己也说,RoboDojo 的 10× episode 平均超过 3 小时;LIBERO-Long++ 每 episode 约 100 万 token、近 1 小时;Ultra-Long 约 900 万 token、近 2 小时(`papers/robotics/astra-embodied-2610.22471.md:213`)。这些数字与 Natura-1 的 63 s 放在一起,说明**"能力"与"可用性"之间隔着两到三个数量级的时间成本**。这正是 SenseNova 报告标题里"practical embodied intelligence"的含义:benchmark competence 已经跨过门槛,但 practical 还没有。

### C.4 张力在哪里

张力不在"100 ms vs 24.84 s"本身,而在两个隐含主张:

1. Natura-1 隐含主张"快速反应式策略是灵巧操作的必要条件";SenseNova 隐含主张"慢速 deliberation + 重试是当前通用 agent 获得高成功率的主要机制"。如果 Natura-1 的 100 ms 是真的,那么 SenseNova 的"用 10× 步数换成功率"在灵巧手场景里可能根本不成立,因为物理世界不允许你慢慢试。反过来,如果 SenseNova 的"不可逆失败"分析是对的,那么 Natura-1 的快速策略也必须解决"错了之后怎么办"的问题,而 100 ms 本身不解决这个问题。
2. 两份报告都没有给出"延迟 vs 成功率"的曲线。Natura-1 给的是固定 100 ms 下的成功率;SenseNova 给的是固定 60 min/10× 下的成功率。缺少这条曲线,就无法判断"多快才够"。

这直接连到下一节的 world clock。

---

## 5. 交叉审计 D:"世界时钟不停"与 πR²/staircase 实时策略战线

### D.1 SenseNova 的基准原型

SenseNova 的下一代基准有三条原则:任务来自真实人类需求而不是机器人能力;不提供同域演示,agent 必须在部署中自己探索;世界在 agent 思考时继续走,推理时间和它的物理后果都算进评分(`papers/robotics/astra-embodied-2610.22471.md:404-412,474`)。原型是一个 Isaac Sim 里的一居室,agent 通过 AgiBot G2 的本体相机观察,零样本执行"做早餐"这类只有需求、没有固定 goal state 的任务;评判看整条轨迹上的 Intent 和 Duty(例如冰箱有没有关、灶台有没有关、水有没有关),并且 world clock 在 agent 宣布完成后还继续走(`papers/robotics/astra-embodied-2610.22471.md:486-504`)。五个代表性失败里,有三个是时间/延迟直接造成的:冰箱门从 300 s 开到 765 s;水龙头开着、排水口关着,水溢出到柜子和地板;鸡蛋已经凝固后 agent 花了 8 s 决策,等关火时已经焦了(`papers/robotics/astra-embodied-2610.22471.md:594`)。

这不是一个完整的 benchmark,作者自己说 episodes 很少、是定性的,并且列出 judge evidence、simulation fidelity、real-time evaluation cost 三个开放问题(`papers/robotics/astra-embodied-2610.22471.md:606-610`)。但它的方向很明确:**把 inference latency 从"系统实现细节"提升为"评测变量"。**

### D.2 πR²/staircase 的延迟轴

我们 2026-07-31 的实时策略线程精读了 πR²(2607.26055):它保留大 backbone,用 per-position staircase noise level + last-minute finalization,把"观测延迟"作为调度变量;延迟低时 staircase 付质量税,延迟高时反超 sync(论文里还对比了 RTC)(`runs/2026-07-31-abotworld-umi-turbovla/realtime-policies/README.md`)。我们当时提出的猜想是"staircase 的优势是模态特异的"——对 proprio 可见扰动更强,对纯视觉扰动更弱。1D 玩具实验证伪了这个猜想:proprio 和视觉扰动的误差几乎重合,真正决定优劣的是观测延迟档,不是扰动模态(`memories/2026-07-31-memory.md:34-61`)。这个负结果留下了一个更准确的替代解释:**staircase 的价值来自"观测陈旧度管理",不是来自某个传感通道。**

SenseNova 的 world clock 正好把这个替代解释变成评测轴:世界不暂停,agent 的观测在推理期间变旧,动作必须适合"动作生效时"的世界状态,而不是"观测时"的世界状态。鸡蛋的例子就是最干净的版本:agent 的决策在观测时是对的(蛋已凝固,该关火),但动作生效时世界已经往前走了一步,蛋焦了。πR² 的 staircase 不能消除这个延迟,但它可以让策略在延迟下更稳;SenseNova 的 world clock 则让这个延迟的后果可见、可评分。

### D.3 三条设计点的光谱

把 Natura-1、πR²、Astra 放在同一条"延迟"轴上:

| 设计点 | 延迟策略 | 代表 | 代价 |
|---|---|---|---|
| 最小化延迟 | 100 ms policy loop,不做长 deliberation | Natura-1 | 需要极快的推理;Base 策略没有 test-time search/重试;错误恢复靠策略本身 |
| 适应延迟 | 根据观测延迟调整去噪/发射调度 | πR² staircase | 低延迟时付质量税;实现脆弱;需要 proprio/延迟估计 |
| 忽略延迟 | 慢速 agent 反复观察-推理-重试 | GPT-6-Astra | 物理世界不暂停;不可逆后果;成本高 |

这三者不是互斥的:一个系统可以在低层用 Natura-1 式的快速 policy,在中层用 πR² 式的延迟自适应调度,在高层的长程任务上用 Astra 式的 deliberation,但必须把"世界时钟"作为接口约束。SenseNova 的 prototype 实际上是在问:**当世界不暂停时,高层 deliberation 应该以什么频率、什么粒度介入?**这正是我们 2026-09-16 AGP 线程的"大脑-小脑接口"问题在评测侧的镜像:AGP 发现 Astra 会在需要高频反馈时自己合成 PD 控制器、把决策下沉到 shell 循环(`runs/2026-09-16-agp-syncworld-kdn/agp-2609.12541/README.md`);world clock 则把"下沉得够不够快"变成分数。

### D.4 一个可做的判别性实验

把 SenseNova 的 world-clock 原型简化成 3 个任务(鸡蛋/灶台、冰箱门、水槽),正交控制两个变量:推理延迟 d ∈ {0, 0.5, 2, 8 s} 和策略调度 {sync, staircase, 固定 chunk};测成功率、焦/溢/开门的后果、以及每个任务的世界时间。预测是:(a) 低 d 时 staircase 付质量税,与我们的 1D 结果一致;(b) 高 d 时 staircase 反超 sync,但反超的幅度取决于任务的可逆性——鸡蛋不可逆,冰箱门可逆,水槽介于两者之间;(c) 如果 Natura-1 式的 100 ms policy 可用,它在低 d 端会碾压 agent,但在需要长程规划的任务上仍需要高层介入。这个实验不需要 50B 模型,可以用我们已有的 1D/2D 玩具 + 真实 VLA checkpoint 做,成本可控;它能直接把"world clock"从概念变成一条可测的曲线。

---

## 6. 视频证据抽查

### 6.1 方法

从 Natura-1 博客的 `media.naturewill.com` 直链下载了 4 段代表性视频(共 57 MB),用 `ffprobe` 读时长/分辨率/帧率,用 `ffmpeg` 的 `scene` 分数逐帧检测硬切,再按每秒 1–2 帧抽帧拼成 contact sheet。脚本是 `code/video_audit.sh`;视频原件留在 `workspace/code/natura-videos/`(被 gitignore 的 lab bench),不进运行包;抽帧拼图在运行级 `../assets/`。

`scene` 分数是 ffmpeg 对相邻帧差异的归一化度量;硬切通常在 0.3 以上,渐变/快速运动通常 <0.1。四段视频的最大 scene 分数都 <0.04,说明没有硬切;目视检查也没有看到跳切、叠化或明显的帧重用。这不是"绝对没有剪辑"的证明(理论上可以用极低差异的叠化),但足以排除常见的跳切式剪辑。

| 视频 | 内容 | 时长/帧率/分辨率 | 最大 scene 分数 | 我看到的 |
|---|---|---|---:|---|
| `v002` | Making wontons(† 任务微调) | 72.56 s / 50 fps / 1920×1620 | 0.0075 | 白色人形上半身 + 两只五指 Wuji 手;双手在桌面工作区包馄饨;从拿皮、放馅到双手合拢;动作连续,无人类手入镜;结尾把折叠好的面皮送到圆盘 |
| `v034` | 杆子拉篮(基础权重) | 9.2 s / 30 fps / 1920×1080 | 0.036 | 两只黑色手套式灵巧手;右手先握杆,把高台上的篮子拉下来;左手伸进篮子取毛线;右手持续握杆;两步计划可见 |
| `v030` | 卡片边缘抓取(基础权重) | 9.1 s / 30 fps / 1920×1080 | 0.0088 | 卡片平放在黄色盒子上;左手把卡片推到边缘,卡片悬空;手指从下方插入;双手配合抬起 |
| `v026` | 柠檬干扰(基础权重) | 48.57 s / 30 fps / 1920×1080 | 0.018 | 一个穿白上衣的人站在机器人旁边,用手直接移动柠檬;机器人手臂跟随、伸手;32 s 时人把柠檬放进机器人张开的手掌;40 s 机器人握住柠檬 |

### 6.2 与文字描述一致/不一致的地方

**一致的地方。**四段视频都是真的双臂灵巧手(不是夹爪),动作连续,没有跳切;杆子拉篮的"两步计划"、卡片边缘的"新抓取"、柠檬的"跟随/伸手"都能在画面里看到;wontons 的"single continuous take"与 72.56 s 的单镜头一致。`v034` 和 `v030` 的 2× 速度、`v002` 的 1× 速度也与页面标注一致。`v002` 的机器人是固定在底座上的上半身,不是移动人形;这与博客说的"physical dual-arm robots"一致,不应误读成完整人形。

**不一致/需要打折的地方。**

1. **wontons 是六个 † 任务之一,不是基础权重。**博客明确说六个长程任务有额外微调,其他结果共享同一权重(`papers/vla/natura-1-blog.md:28-29`)。wontons 是最像"灵巧操作"的视频,但它恰恰不是基础模型的零样本能力。它是"任务微调后的上限",不能用来支持"单一基础模型零样本包馄饨"。基础权重的证据是 `v034`(杆子拉篮)、`v030`(卡片)、`v026`(柠檬)这三段,它们的动作更短、更简单。
2. **wontons 的成品没有被近距离验证。**72 s 的视频在结尾把折叠好的面皮送到盘子,但没有给馄饨的封口/形状一个特写;我无法从画面判断它是否真的"包成了馄饨",只能确认双手完成了折叠动作。文字说"包馄饨",视频支持"折叠面皮",不支持对成品质量的更强断言。
3. **杆子拉篮的视频从"已经握住杆"开始。**博客的章节写 0:00 grasp the rod,但下载的 9.2 s 片段在 0 s 时右手已经握住杆,没有展示"发现杆、伸手、握住"的过程。它支持"用杆子把篮子拉近"的两步计划,不支持"从零发现工具"的完整链条。篮子上的杆是实验者预先穿好的,这也是一个人工 scaffold。
4. **柠檬视频的文字说"用长杆移动柠檬和箱子",但画面里是人手直接移动柠檬。**博客写 "a researcher uses a long stick on three occasions to move the lemon and bin to different positions";我抽的 12 帧(0–44 s,每 4 s 一帧)里没有看到长杆,看到的是人手直接拿柠檬,并在 32 s 把柠檬放进机器人手掌。这可能是因为:(a) 我下载的 `v026` 与文字描述的不是同一个 take;(b) 长杆只用于移动箱子,柠檬是人手移动的;(c) 文字把"干扰"写得比实际更程序化。无论哪种,**"长杆干扰"这个具体描述没有被视频支持**;视频支持的是"人类实时移动柠檬,机器人跟随/伸手"。这不影响"机器人能在人类干扰下跟踪目标"的结论,但影响"干扰是用长杆、三次、程序化"的细节可信度。
5. **"autonomous" 的含义。**四段视频都标了 `autonomous 1x/2x`,但 `v026` 里人类在主动移动目标物,`v034`/`v030` 里实验者预置了场景(杆子穿好、卡片放在盒子上)。"autonomous" 在这里指"机器人策略自主",不是"整个试验无人干预"。这不算矛盾,但报告里应把"人类只做扰动/预置,不遥操作"说清楚。

### 6.3 视频能支持什么、不能支持什么

能支持:(a) Natura-1 确实驱动双臂五指灵巧手,不是夹爪;(b) 动作在 1×/2× 播放时连续,没有明显跳切;(c) 基础权重在杆子拉篮、卡片边缘抓取、柠檬跟踪上有多步闭环行为;(d) 任务微调的 wontons 能完成长程双手折叠。

不能支持:(a) "100 ms 控制环"——视频帧率(30/50 fps)和播放速度无法验证控制频率;(b) "单一基础模型零样本完成所有任务"——wontons 是 †,杆子/卡片/柠檬是基础,但样本只有 3 段;(c) "包馄饨的质量"——没有成品特写;(d) "长杆干扰"——画面是人手;(e) "完全无人干预"——人类在预置/扰动。

---

## 7. RoboRSI 仓库:实际放了什么

仓库 `https://github.com/OpenSenseNova/SenseNova-RoboRSI`(2026-10-09 创建,抓取时 133 star,Apache-2.0,约 63 MB,README 标注 Pre-Alpha)。README 的标题写着 "Code release coming soon",但实际内容远不止一个占位页:

- **运行时/评测框架**:`roborsi/` 下有 agents(composition、config、package)、benchmarks(registry、runtime_source)、evaluation(runner、batch、store、trajectory、Docker/HTTP service、media)、planner(API loop、Codex、Claude Code、Codex CLI)、runtime(context engine、cache、skills、trace)、session、tools、dashboard、CLI。`bench eval`、`bench status`、`bench compare`、`roborsi-trajectory` 都是可运行入口。
- **LIBERO 家族 benchmark**:`benchmarks/libero`(标准 LIBERO,CPU 镜像,MuJoCo 3.3.0/robosuite 1.5.2,600 steps 默认,15 步 settling,Panda OSC_POSE 20 Hz)、`benchmarks/libero_pro`(8 个 Task/Swap suites,每 suite 10 tasks × 10 init states = 800 official cases;migration subset 80)、`benchmarks/libero_lilo`(Long++ 12 configurations / Ultra-Long 9,各 10 次 reset;1× 预算 200 steps/skill;ordered-prefix 评分)。三个 benchmark 都有 Dockerfile、asset download、manifest、README。
- **任务目录**:`robots/libero/benchmark_catalog.py` 定义了 PRO 的 8 个 suite 名称、每个 10 task、每个 10 init state;Long++/Ultra 的 12/9 个配置、每个 10 次 bare random reset;PRO 的 1× 是 10,000 steps,migration subset 用固定 init state 3。具体的任务 instruction 不直接嵌入仓库,而是在构建镜像时从上游 LIBERO assets 的 manifest 生成;所以"任务清单"在仓库里是"case 结构 + 预算 + 评分协议",不是一份静态的自然语言列表。
- **Agent 配置**:`configs/agents/embodied/` 有 default(OpenAI Responses API)、codex、codex_cli、claude_code、short_context 等 preset;`examples/roborsi/` 有 hold agent 和 model agent。
- **测试**:约 60 个单元/集成测试文件,覆盖 agent 配置、benchmark registry、Docker/HTTP runner、batch resume、case status、trajectory、media、LiLo adapter 等。
- **报告 PDF**:仓库内的 `assets/has_gpt-6_achieved_embodied_intelligence.pdf`(约 20.7 MB),就是 arXiv 报告的镜像。
- **27B preview**:README 说 `SenseNova-RoboRSI-27B-Preview` 已在不同环境/真机上运行,开放权重预计 2026-10-20 发布,技术报告随后;当前仓库没有权重。

**没有放出的**:RoboDojo、RoboLab、Bench2Dex、DexJoCo、LHPR-VLN(VLN-CE)、SIMPLE、BEHAVIOR-1K 的 benchmark 代码,README 的 BenchZoo 表里都标 "Coming soon";Harness demo 也未发布。因此,**报告里 12 个 benchmark 的完整评测不能只靠这个仓库复现**;仓库目前能复现的是 LIBERO 家族三件套 + agent runtime,以及一套可扩展到其他 benchmark 的接口。这与"Natura-1 没有代码/权重、博客即技术报告"形成对比:Natura-1 的开放度更低(无代码、无权重、无原始 benchmark),RoboRSI 的开放度更高(有 runtime、有 LIBERO 家族、有测试),但两者的"可复现性"都不足以独立审计全部数字。

---

## 8. 对主报告叙事的建议

### 8.1 应该怎么写

把两份报告写成**协议光谱上的两个点**,而不是"谁更强"的擂台。主报告可以先给读者一个直觉:同样是"具身基础模型",一个把赌注押在"人类视频预训练 + 灵巧手 + 快速闭环",另一个把赌注押在"通用 agent + ICL + 大量试错";它们在最严格的真机零样本条件下都承认对方的一部分,在各自最宽松的协议下都拿到漂亮数字。然后逐项拆协议:训练制度、本体、时间预算、演示/ICL、任务来源、评分聚合。这样读者既不会被 53.4 vs 16.7 误导,也不会被 81.9 误导。

一个推荐的叙事链:
1. 两份报告在同一天出现,不是巧合,是具身智能两条路线的正面相遇:专用基础模型(端到端、快、灵巧)vs 通用 agent(慢、会试错、跨本体)。
2. 先把 Natura-1 的 12 维 rubric 和 SenseNova 的 12 个 benchmark 拆开,指出"12"不是同一个 12。
3. 用协议矩阵解释 16.7 vs 81.9:前者是 60 min + 灵巧手/夹爪跨本体 + 无 ICL;后者是 ICL + 10× + 仿真。
4. 在交集上,两边都同意:Astra 能做刚性抓放、双臂持物、部分工具使用;不能自主做高自由度灵巧手、精密插入、长程移动;两者都卡在"验证/恢复/长程进展"。
5. 把 scaling 作为"人类视频路线"的证据:Natura-1 的两点信号 + Dyna-2 的四点幂律,方向一致但强度不同;不要把 Natura-1 称为 scaling law。
6. 把 100 ms 与 world clock 作为"时间"这条隐藏轴:Natura-1 用速度换反应性,SenseNova 用 world clock 把延迟变成分数,πR² 用调度适应延迟;三者可以拼成一个完整的设计空间。
7. 收尾:真正的下一步不是"谁的 benchmark 更高",而是"在 world clock 下,多快的策略、多少 deliberation、什么粒度的恢复,才能把 benchmark 能力变成可用性"。

### 8.2 应该避免的说法

- 不要说"Natura-1 比 GPT-6-Astra 强 3.2 倍"——53.4 是灵巧手 + 机器人后训练,16.7 是夹爪 + 零样本 coding policy,跨本体 + 跨训练制度。
- 不要说"GPT-6-Astra 已经解决灵巧手"——SenseNova 的 Tianji 结果是 0/3(无人类脚本)vs 3/3(有人类脚本),而且只有 toy/bottle 两个任务。
- 不要说"Natura-1 建立了 scaling law"——只有两个点,没有 R²/误差棒/中间 rung,且算力与配方共变。
- 不要说"100 ms 已经验证"——博客没有给延迟分布、硬件、batch、action horizon;视频无法验证控制频率。
- 不要说"SenseNova 的 81.9 是 Astra 的真实水平"——它是 ICL + 10× 预算下的上界,标准预算 + ICL 直接控制是 27.6;没有 ICL 的直接控制是 15.56。
- 不要把 Natura-1 的 bimanual 0.0 当作"Astra 不会双臂"的证据,也不要把 dual Piper 14/14 当作"Astra 的双臂已经解决"的证据;两边的任务/评分/样本量都不透明。

### 8.3 建议用的图

报告用图与视频拼图都在运行级 `../assets/`(与主报告同目录);线程包只保留 `README.md` 与 `code/`。

- `natura_12dims.png`:12 维分数对比,一眼看出 Natura 的强项(空间/物体泛化、组合泛化)和 Astra 的强项(Recovery 75.0,但这与 SenseNova 的不可逆失败分析冲突,适合做讨论引子)。
- `astra_12benchmarks.png`:12 个 benchmark 的 1×/10× 对比,一眼看出"10× 把很多点推到 80–100",以及 RoboDojo/Bench2Dex 的标准预算并不高。
- `astra_harness_cost.png`:S1/S2/harness 的 token-success 散点,直接支撑"闭环比开环好、ICL+VLA 最好但最贵"。
- `astra_failure_modes.png`:三类失败的可视化,支撑"验证/不可逆/长程进展"三连。
- `astra_tianji_grasp.png`:0/3 vs 3/3,支撑"人类先验是灵巧手的开关"。
- `natura_rod_basket_sheet.png` / `natura_card_edge_sheet.png`:基础权重的多步闭环/新抓取,支撑"不是只会 replay"。
- `natura_wontons_sheet.jpg`:任务微调的双手长程折叠,支撑"上限",但要标注 †。
- `natura_lemon_sheet.png`:人类干扰下的跟踪,同时暴露"文字说长杆、画面是人手"的差异。
- `astra_household_pipeline.png` / `astra_icl_protocol.png`:world clock 和 ICL 协议,支撑"评测条件本身是变量"。

---

## 9. 复跑/复查路径

### 9.1 本地脚本

```bash
cd runs/2026-10-19-whyfar-linearmem-natura-astra/natura-astra-embodied
python3 code/scaling_fit.py          # 两点 scaling 换算,输出到 stdout
./code/video_audit.sh                # 下载 4 段视频到 workspace/code/natura-videos,抽帧,算 scene 分数,重建拼图
python3 code/make_figures.py         # 只重建 12 维图表/拼图;需要 NATURA_VIDEO_DIR 指向已抽帧目录
```

`video_audit.sh` 会自动调用 `make_figures.py`。如果只想重建 12 维图表,不需要视频:

```bash
python3 code/make_figures.py
```

依赖:`python3`(stdlib)、`ffmpeg`/`ffprobe`、`google-chrome`(headless 截图)。脚本里没有第三方 Python 包。单独跑 `make_figures.py` 时,如果 `NATURA_VIDEO_DIR` 下没有抽好的帧,它会只重建 12 维图表并跳过视频拼图;`video_audit.sh` 会先抽帧再重建,并把 wontons 拼图转成 jpg 以满足 2 MB/文件上限。

### 9.2 源文件与 URL

- Natura-1 博客正文存档:`papers/vla/natura-1-blog.md`
- Natura-1 博客原页:`https://www.naturewill.com/blog/natura-1`
- 12 维分数内嵌页:`https://www.naturewill.com/_releases/b45a07089dbe1f42ae5f8397aea3e0f54ded60c8c541e0c8fd62e5aff731ee5f/assets/natura-1/local/visualizations/score-embed.html`
- 视频直链前缀:`https://media.naturewill.com/research-blog/media/<asset-id>/20261012T210313Z-089ca3c35ad7/mp4/high.mp4`
- SenseNova 报告全文:`papers/robotics/astra-embodied-2610.22471.md`;arXiv:2610.22471v1
- SenseNova 报告主页:`https://sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence`
- RoboRSI 仓库:`https://github.com/OpenSenseNova/SenseNova-RoboRSI`
- Dyna-2 博客:`https://www.dyna.co/dyna-2`
- 本仓库第一方 Astra 测量:`runs/2026-09-16-agp-syncworld-kdn/agp-2609.12541/README.md`、`runs/2026-09-17-boundary-astra/astra-boundary/README.md`、`runs/INDEX.md:78`
- πR²/staircase 线程:`runs/2026-07-31-abotworld-umi-turbovla/realtime-policies/README.md`、`memories/2026-07-31-memory.md:34`

### 9.3 关键数字的原文位置

- Natura-1 基线 13.5/16.7/53.4/59.1、60 min、ROKAE/Wuji、ARX AC One:`papers/vla/natura-1-blog.md:39`
- Natura-1 固定训练集/非无机器人数据零样本:`papers/vla/natura-1-blog.md:41`
- Natura-1 数据规模/MPJPE/吞吐:`papers/vla/natura-1-blog.md:81-103`
- Natura-1 100 ms:`papers/vla/natura-1-blog.md:14,26,97,105`
- Natura-1 scaling 四点:`papers/vla/natura-1-blog.md:117`
- SenseNova 摘要/12 benchmark/ICL+10×:`papers/robotics/astra-embodied-2610.22471.md:17,33-37,61`
- SenseNova Table 2(12 benchmark 分数):`papers/robotics/astra-embodied-2610.22471.md:1150`
- SenseNova S2/S1 harness 表:`papers/robotics/astra-embodied-2610.22471.md:237-251`
- SenseNova 真机 8 配置 Table 14:`papers/robotics/astra-embodied-2610.22471.md:1473-1477`
- SenseNova Tianji 0/3 vs 3/3:`papers/robotics/astra-embodied-2610.22471.md:2100-2136`
- SenseNova G1 24.84 s/90% deliberation:`papers/robotics/astra-embodied-2610.22471.md:394`,App. E.9
- SenseNova world clock/鸡蛋/冰箱/水槽:`papers/robotics/astra-embodied-2610.22471.md:404-412,474,486-504,594`
- SenseNova 失败模式 Fig. 6:`papers/robotics/astra-embodied-2610.22471.md:199-225`
- SenseNova 开放挑战:`papers/robotics/astra-embodied-2610.22471.md:606-610`

---

## 10. 图清单

| 文件 | 尺寸 | 位置/来源 |
|---|---|---|
| `../assets/natura_12dims.png` | 1600×1100 | 用 score-embed 数据自绘的 12 维对比图 |
| `../assets/natura_12dims.svg` | 1600×1100 | 上图的矢量源 |
| `../assets/natura_wontons_sheet.jpg` | 1600×1040 | v002 的 6 帧拼图(† 任务微调) |
| `../assets/natura_rod_basket_sheet.png` | 1600×300 | v034 的 5 帧拼图(基础权重) |
| `../assets/natura_card_edge_sheet.png` | 1600×300 | v030 的 5 帧拼图(基础权重) |
| `../assets/natura_lemon_sheet.png` | 1600×748 | v026 的 6 帧拼图(基础权重;人类手移动柠檬) |
| `../assets/astra_12benchmarks.png` | 1316×408 | 报告 Fig. 1,12 benchmark 的 1×/10× 对比 |
| `../assets/astra_harness_cost.png` | 760×588 | 报告 Fig. 7,S1/S2/harness 的 token-success |
| `../assets/astra_token_cost.png` | 638×468 | 报告 Fig. 5a,每 benchmark token 成本 |
| `../assets/astra_time_cost.png` | 636×420 | 报告 Fig. 5b,每 benchmark 墙钟时间 |
| `../assets/astra_tianji_grasp.png` | 1311×418 | 报告 Fig. 18,Tianji 0/3 vs 3/3 |
| `../assets/astra_failure_modes.png` | 1289×670 | 报告 Fig. 6,三类失败模式 |
| `../assets/astra_icl_protocol.png` | 1295×260 | 报告 Fig. 2,ICL 协议 |
| `../assets/astra_household_pipeline.png` | 1280×554 | 报告 Fig. 11,world-clock 家庭原型 |

所有图最长边 ≤1600 px;视频拼图裁掉了底部空白;论文图只做了 2 px 边框裁剪和缩放,没有改内容。

---

## 11. 证据边界与未完成事项

- **没有 Natura-1 的代码/权重/原始 benchmark。**12 维 rubric 的任务数、试验数、聚合权重、Agentic 的定义、100 ms 的测量协议都没有公开;16.7 不能复算,53.4/59.1 也不能拆到题目级。
- **没有 GPT-6-Astra 的权重/原始日志。**SenseNova 的数字是自报的;报告给了大量 per-task 表和 token/time,但没有原始 rollout。我们只能做协议级交叉审计,不能重跑。
- **视频只抽了 4 段(约 45 段中)。**没有音频,没有逐帧的接触力/关节状态;scene 分数只能排除硬切,不能证明"没有叠化/加速";`v026` 的长杆描述与画面不一致,需要作者澄清。
- **scaling 只有两个点。**没有中间 rung、误差棒、R²;MSE/MAE 指数不一致,说明误差分布形状在变;不能称为 law。
- **RoboRSI 的 8 个非 LIBERO benchmark 尚未发布。**报告里 RoboDojo/RoboLab/Bench2Dex/DexJoCo/LHPR-VLN/VLN-CE/SIMPLE/BEHAVIOR 的数字不能从这个仓库复现;LIBERO 家族可以。
- **没有做真正的交叉实验。**本线程提出"把 Natura-1 放到 SenseNova 12 benchmark、把 Astra+ICL 放到 Natura-1 12 维 rubric"的建议,但没有执行;这是下一步最值得做的动作。
- **世界时钟的判别性实验没有跑。**本线程只做了概念映射和实验设计;πR² 的 1D 玩具负结果来自 2026-07-31 线程,不是本轮新实验。

如果只带走一句话:**Natura-1 和 SenseNova 报告不是互相反驳,而是把"具身能力"这个词拆成了两个问题——在什么本体上、用什么信息条件、允许多少时间。回答这两个问题之前,任何跨报告的数字比较都只是营销。**
