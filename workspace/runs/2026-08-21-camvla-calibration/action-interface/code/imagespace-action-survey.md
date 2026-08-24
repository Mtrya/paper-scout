# 图像空间动作表示谱系测绘:VLM/视觉友好的动作表示

日期:2026-08-24。用途:组会报告"相机标定 × 动作表示"的替代方案全谱系。
核实方式:逐篇 WebSearch/FetchURL 核对(截至 2026-08-24);标"未核验"处为查不到或未深核。
与本线程锚点(CamVLA 相机系 delta 动作)的关系:下表多数工作把动作/条件落在图像或点云/体素空间,
是"6-DoF 基座系 delta"的直接替代/对偶方案。

## 1. 总表

| 论文 (arXiv) | 年份 | 动作表示是什么 | 旋转怎么编码 | 相机假设 | 精度/成功率证据 | 开源 |
|---|---|---|---|---|---|---|
| Transporter Nets (2010.14406) | 2020 CoRL | 顶部正交 heightmap 上两个像素热图(pick、place);位移=像素 argmax | SE(2):输入按 k=36 bin(10° 步长)旋转、共享权重复用 FCN;SE(3) 扩展:rx/ry/z 三头 MLP 连续回归 | RGB-D 融合为 0.5×1m 正交俯视图(3.125mm/像素);需内外参;**论文自认"对相机-机器人标定敏感"**;真机单相机(Photoneo PhoXi / Azure Kinect) | 10 仿真任务 100 演示大多 >90%;真机 kit 组装+sweeping;推理 200ms;精度上界=像素分辨率 3.125mm | ✅ 代码+Ravens 基准 |
| PerAct (2209.05451) | 2022 CoRL | 下一个关键帧 EE 位姿;平移=体素分类(1m³→100³=1cm 体素) | 三轴欧拉角 5° bin 离散分类;夹爪二值 | RGB-D 点云体素化;RLBench 4 相机;真机 3 任务(RGB-D,相机数未核验) | RLBench 18 任务多任务 ~47%(RVT 口径 48%);精度上界=1cm 体素+5° bin | ✅ peract.github.io |
| RVT (2306.14896) | 2023 CoRL | 关键帧 EE 位姿;点云渲染 5 个虚拟视角,各视角图预测热图,反投影 3D 取最高分点 | 全局特征回归旋转 | RGB-D→点云→虚拟视角渲染;需内参与点云重建;RLBench 4 相机 | RLBench 18 任务 62.9%(比 PerAct 高 26%,训练快 36×) | ✅ 代码 |
| RVT-2 (2406.08545) | 2024 CoRL | 两级 coarse-to-fine:固定视角图定兴趣区→放大 4× 重渲染精定位;仍关键帧位姿 | **位置条件特征**(在 EE 位置池化)回归旋转,解决多物体朝向歧义 | RGB-D→点云→虚拟视角(3 视图);真机单第三人称相机 | RLBench 18 任务 **81.4%**(Act3D 65.0%→;摘要口径 65→82%);**真机毫米级插 peg/插 plug,10 条演示** | ✅ 代码+模型 |
| Act3D (2306.17817) | 2023 CoRL | 3D 特征场(自适应分辨率,coarse-to-fine);逐点热图回归 3D 平移 | 全局特征回归旋转 | RGB-D 点云;RLBench 4 相机(256² 输入) | RLBench 18 任务 ~65%(RVT-2 口径)/63.2%(3D-DA 口径) | ✅ act3d.github.io |
| 3D Diffuser Actor (2402.10885) | 2024 CoRL | 扩散策略直接生成连续 3D 轨迹(EE 位姿序列);条件=点云 per-point 特征+语言+本体感 | 连续 6-DoF 由扩散去噪生成,无离散化 | RGB-D 点云;RLBench 4 相机;真机单相机视角 | RLBench 18 任务 81.3%(+18.1% vs Act3D)、单视角 78.4%;**真机 12 任务单相机 15 条演示** | ✅ nickgkan/3d_diffuser_actor |
| PIVOT (2402.07872) | 2024 | 迭代视觉提示:候选动作(编号箭头)画在图上,VLM 逐轮挑选+细化(交叉熵式优化);输出映射为动作 | **图像里不表达旋转**;论文明说 VLM 无法可靠按深度/3D 旋转选择;操作动作空间=EE 相对 (x,y,z)+夹爪 | 单 RGB(头部/腕部/顶部相机)+相机矩阵投影;深度不可靠 | 真机导航+操作:成功率非零但"远未完美";离线指标=与 RT-X 演示动作的余弦相似度;RoboPoint 复测导航 2–4/5 | ✅ demo+代码 |
| MOKA (2403.03174) | 2024 RSS | 点式 affordance:抓取/功能/目标关键点+接触前/后 waypoint,全在 2D 图像,VLM 从标记多选;升 SE(3) 执行 | **旋转不靠 VLM**:解析 grasp sampler 从点云选最接近预测点的抓取姿态;物体朝向=VLM 文本多选(前/后/左/右…) | RGB-D 固定台架相机;关键点深度反投影;自由空间 waypoint 高度由 VLM 文本给 | 4 任务×2 子任务各 10 次;零样本优于 CaP/VoxPoser;in-context 与蒸馏(Octo,50 演示)再涨;失败分 reasoning/execution | ✅ moka-manipulation.github.io |
| RoboPoint (2406.10721) | 2024 | 2D 图像关键点 affordance(直接输出点坐标);点→3D 用深度反投影;下游=导航/操作/AR | 不覆盖旋转与夹爪(只给点,下游 planner 加偏移) | RGB+深度(执行阶段);训练全仿真合成(660K 对);声称跨视角一致 | 比 GPT-4o/PIVOT 高 **21.8%** grounding 准确率、**30.5%** 下游成功率;真机 7 任务各 10 次 **+39.5% vs GPT-4V**;精度指标=点在 GT mask 内比例 | ✅ robo-point.github.io |
| Set-of-Mark (2310.11441) | 2023 | 提示技术:分割区域编号标注,Gpt-4V 选号作答(**不产出动作**) | — | 无(纯 2D 图像) | RefCOCO 等指代定位基准显著提升 GPT-4V grounding(数值未逐项核验);被 PIVOT/MOKA 引为标注范式 | ✅ 提示代码 |
| ATM (2401.00025) | 2024 RSS | 任意点 2D 轨迹(相机帧内,32 网格点);轨迹模型从无动作视频预训练(CoTracker 造标签);策略=轨迹引导 BC | 轨迹本身无旋转;动作由策略从轨迹+观测学(旋转隐式) | **单 RGB;刻意不做深度/多视角/标定假设**(原文:"2D 轨迹在相机系,最小化对已标定相机的假设") | 130+ 语言条件任务平均 **63% vs 最强基线 37%**(+80%);LIBERO 全套;人类视频/异形机器人迁移有效 | ✅ xingyu-lin.github.io/atm |
| Track2Act (2405.01527) | 2024 | 2D 点轨迹(初始+目标图条件,DiT 扩散,web 视频预训练)→刚性变换→3D EE 位姿序列(开环)+残差策略(闭环) | 旋转含在 PnP 解出的每步刚体变换 Tt 中(2D 轨迹+首帧深度+内参 K 最小化投影误差) | 轨迹模型 RGB-only;部署需 **RGB-D+内参**(首帧深度,固定相机);Spot+RealSense | 真机 25 任务×5 场景×20 rollout/泛化层级(MG/G/CG/TG);优于目标条件 BC/affordance/video/mask 基线;无毫米级数字 | ⚠️ 项目页(代码未深核) |
| Im2Flow2Act (2407.15208) | 2024 RSS | **对象流**(2D 关键点轨迹,剔除本体/背景);AnimateDiff+GDINO+TAPIR 从人类视频生成;流条件扩散策略(纯仿真探索数据训练)执行 | 动作=6-DoF EE+夹爪,扩散策略输出(旋转隐式) | RealSense RGB-D(首帧 3D 坐标需深度);人类视频侧 RGB | 4 真机任务(刚体/铰接/柔性)**语言条件平均 81%**(90/80/85/70);零真机训练数据;启发式(GeneralFlow/Track2Act 式)在铰接/柔性上崩 | ⚠️ 项目页(代码未深核) |
| GeneralFlow (2401.11439) | 2024 | "general flow":场景点未来 3D 流(点云空间),作零样本 affordance 传给下游 | 流含 3D 运动(旋转体现为点位移);转动作需启发式/策略 | 深度相机(首帧点云);训练于人类视频 | 零样本操作任务;具体成功率数值未在本轮核验 | ✅ general-flow.github.io |
| LAPA (2410.11758) | 2024 | 隐式 latent action:VQ-VAE 对相邻帧量化出离散 latent;VLM 预训练预测 latent,小样本微调映射到 delta 动作 | 不显式编码(旋转在 latent 里) | RGB 视频(免动作标签) | 显著优于无动作标签训练基线、跨域转移好;数值未逐项核验 | ⚠️ 项目页(代码未核验) |
| RT-Trajectory (2311.01977) | 2023 | 2D 轨迹草图(EE 曲线+夹爪圆圈)作 RT-1 策略条件;训练用 hindsight 标签,测试用草图/人类视频/基础模型生成 | 草图不含旋转(2.5D 变体有高度暗示);动作由 RT-1 输出 | 单 RGB(头戴相机视角);草图在图像空间,须与执行相机视角一致 | 7 个新任务 **67% vs 11–26%**(2.5D 版);视觉提示工程可复现改变行为模式 | ⚠️ GitHub org 存在(内容未核验) |
| TraceVLA (2412.10345) | 2024-25 | 视觉 trace 提示:CoTracker 历史点迹叠加到 RGB 输入,微调 OpenVLA | 动作仍由 OpenVLA 输出(不改动作表示) | 单 RGB(CoTracker,免深度) | LIBERO 空间/时间任务优于 OpenVLA(数值未摘录;AxisGuide 报告 SmolVLA+TraceVLA 51.90) | ✅ umd-huang-lab/tracevla + HF tracevla_7b |
| UniPi (2302.00111) | 2023 NeurIPS | 视频生成即策略:文本+首帧→未来帧,逆动力学模型从相邻帧推动作;开环 | 逆动力学输出 7 维控制量(旋转隐式) | RGB 视频 | 仿真优于视频预训练基线(数值未逐项核验) | ❌ 未见官方仓库(google-research/uni-pi 404) |
| SuSIE (2310.10639) | 2023 ICLR'24 | 子目标图像:微调 InstructPix2Pix 从当前帧+指令生成子目标图;低层子目标条件策略执行 | 无(子目标是图像;动作由低层策略) | 单 RGB | CALVIN SOTA;真机多任务稳健(数值未逐项摘录) | ✅ rail-berkeley.github.io/susie |
| VoxPoser (2307.05973) | 2023 CoRL | 3D 价值图:LLM 写代码调 VLM(OwlViT+SAM)在体素空间组合 affordance/constraint;运动规划器合成轨迹 | 方向约束以价值图表达;旋转精度依赖规划器 | RGB-D 点云体素化(固定相机) | 真机零样本 28 任务,按任务报告成功率(数值未逐项核验;第三方复测口径 10%) | ✅ huangwl18/VoxPoser |
| ReKep (2409.01652) | 2024 | 关系关键点约束:LLM/VLM 生成作用于 3D 关键点的 Python 成本函数;约束优化求解 EE 轨迹 | 关键点+方向约束进优化(旋转由成本函数约束) | **标定 RGB-D 静态相机**(关键点 2D 检测后投影到世界系 R³) | 真机 10 任务;第三方复测口径 ReKep(Auto) 44.3% vs VoxPoser 10.0%、人工标注版 68.6%(来自 alphaxiv 摘录,待与原文核对) | ✅ rekep-robot.github.io |
| OmniManip (2501.03841) | 2025 | 物体中心交互原语(接触点+方向)作空间约束;单视角 3D 重建出 mesh;VLM 接地交互点,闭环 | 方向由 VLM 采样+几何滤波;结合 grasp 求解 | 单视角 RGB-D(3D 生成) | 真机多任务零样本/少样本(数值未逐项核验) | ⚠️ 项目页仓库为网站;未见正式实现代码 |
| kPAM (1903.06684) | 2019 ISRR | 类别级语义 3D 关键点;目标=关键点的几何成本/约束;MPC 执行 | 旋转通过关键点对/朝向约束进入 MPC | RGB-D(点云关键点检测) | 真机(杯子挂架、水瓶倒水等)按任务报告成功率(数值未逐项核验);kPAM 2.0 加反馈控制 | ✅ weigao95/kPAM |

## 2. 逐篇要点(重点:精度从哪来、相机假设是什么)

**像素热图/拣放点族**

- **Transporter Nets**:精度来自"空间一致"表示——把 RGB-D 反投影成正交俯视图,像素=固定 3.125mm 窗口,FCN 平移等变 + 36 个旋转 bin 的模板匹配。代价是整条链依赖标定:论文结论里自己承认"对相机-机器人标定敏感"。**这是"图像空间动作"精度由离散化+标定联合决定的教科书案例**。
- **PerAct**:动作被 1cm 体素 + 5° 旋转 bin 量化,精度上界由离散化给出(这也是它高精度任务弱的原因,被 RVT-2 点名)。
- **RVT / RVT-2 / Act3D / 3D Diffuser Actor**:同一谱系的连续化演进。RVT 把热图从原始相机图搬到虚拟渲染视角(点云→5 视图),Act3D 用自适应分辨率 3D 特征场,3D-DA 用扩散直接生成连续轨迹。**RVT-2 是唯一给出"毫米级精度"定量证据的**(真机插 peg/插 plug,10 条演示,单第三人称相机);它把精度来源拆解为多阶段 zoom-in + 位置条件旋转。全部假设标定 RGB-D(点云重建需要内外参),RLBench 用 4 相机、真机 1 相机。

**VLM 视觉提示/指向族**

- **PIVOT**:VLM 只做"从编号候选里选",动作由选中的图像内箭头映射;迭代细化(类似交叉熵法)把粗选变精。相机假设最轻(单 RGB+相机矩阵),但论文明说深度与 3D 旋转对 VLM 不可靠——**这是"图像空间动作"的精度天花板来自 VLM 自身局限而非几何的实例**。
- **MOKA**:把动作压成"点集+朝向多选",VLM 选点,深度反投影升 3D;**旋转故意外包给解析 grasp sampler**(点云几何),因为 VLM 预测 6-DoF 不可靠。精度来源=深度反投影精度+VLM 选点一致性;失败被分为 reasoning(选错点)与 execution(几何执行)两类,量化的正是这两类误差。
- **RoboPoint**:fine-tune 过的 VLM 直接输出 2D 点坐标;精度指标是"点落在 GT mask 内比例",并给出跨视角一致性证据。相机假设:执行需要深度反投影(论文图 2 明确),但训练数据纯仿真合成、不需要真机采集。
- **Set-of-Mark**:不是机器人系统,是"给图编号让 GPT-4V 选"的提示范式,被 PIVOT/MOKA 直接继承;本身无相机假设。

**点轨迹/光流族(动作代理)**

- **ATM**:2D 任意点轨迹做"子目标";**明确声明在相机系预测 2D 轨迹以最小化标定假设、不假设深度**——是这一族里"免标定"最彻底的一支。代价是 2D 轨迹不能直接执行,动作由另一条轨迹引导策略学(需要少量动作标签),旋转完全隐式。
- **Track2Act**:轨迹模型 RGB-only(web 视频预训练),但**执行时必须 RGB-D+内参 K**:2D 轨迹+首帧深度用 PnP 解刚体变换,再上残差策略闭环。精度误差路径最清楚:轨迹预测误差→PnP 求解误差→残差策略修正。
- **Im2Flow2Act**:把"轨迹"收窄为"对象流"(剔除本体/背景),跨本体/跨仿真迁移最干净;策略全仿真训练、真机 4 任务 81%。相机假设:RealSense RGB-D,首帧 3D 坐标需要深度。
- **GeneralFlow**:2D 流升级为点云空间 3D 流,作为零样本 affordance;需要深度相机出点云。
- **LAPA**:轨迹/流再进一步抽象为 latent action(VQ-VAE 量化帧间差);**旋转不存在于任何显式表示里**,是"表示即预训练"的极致。

**轨迹叠加/子目标图像族**

- **RT-Trajectory**:2D 草图只做**条件**(不预测动作),动作由 RT-1 输出;草图在图像空间,隐含"训练/执行相机视角一致"的假设;2.5D 变体比 2D 涨(67% vs 11–26%)。
- **TraceVLA**:CoTracker 历史点迹做**输入增强**(叠加到 RGB),微调 OpenVLA;免深度;是"表示只进训练/条件、不进动作参数"的典型。
- **UniPi / SuSIE**:子目标/未来帧是**图像本身**,动作由逆动力学或低层策略从图像对推出——表示与动作完全解耦,相机假设=单 RGB。

**3D 关键点/价值图 grounding 族(VLM agent 用)**

- **VoxPoser**:LLM 写代码在体素空间组合价值图,规划器合成轨迹;假设标定 RGB-D 点云。精度受体素分辨率与规划器控制。
- **ReKep**:3D 关键点约束(成本函数)+ 约束优化;**明确写"标定 RGB-D 相机"**;第三方复测口径 44.3% vs VoxPoser 10.0%,人工标注约束 68.6%——**这是"自动生成的图像空间接地"与"人工接地"之间差距的直接量化**。
- **OmniManip**:单视角 RGB-D 重建 mesh + 交互原语;方向由 VLM 采样+几何滤波。
- **kPAM**:最老的 3D 关键点方案(2019),关键点成本+MPC;RGB-D。

## 3. 谱系小结:内部的分化轴

1. **离散 vs 连续**:热图/体素/角度 bin(Transporter 36 bin、PerAct 1cm+5°、RVT 热图)→ 连续回归(Act3D)→ 扩散生成连续轨迹(3D-DA、Im2Flow2Act 的动作头)。离散给可解释性与等变性,连续给精度(代价是精度变成隐式)。
2. **2D vs 3D 承载**:2D 图像/像素(Transporter 俯视图、PIVOT/MOKA/RoboPoint 点、ATM/Track2Act/Im2Flow2Act 轨迹、RT-Trajectory/TraceVLA 草图)→ 3D 体素/点云(PerAct、Act3D、3D-DA、VoxPoser、ReKep、kPAM、GeneralFlow、OmniManip)。**2D 端免深度但执行有鸿沟(要再映射回 3D),3D 端可执行但要深度/标定**——这与本线程 5.1 节"2D 轨迹深度自由但与执行之间有鸿沟"的判断完全一致。
3. **表示即动作 vs 表示即条件/预训练**:Transporter/PerAct/RVT/3D-DA 的表示**就是动作参数**(热图/体素/轨迹直接执行);PIVOT/MOKA/RoboPoint/SoM 是"VLM 提示→选点"(表示是 VLM 的接口);ATM/Track2Act/Im2Flow2Act/RT-Trajectory/TraceVLA/LAPA/UniPi/SuSIE 是"表示做条件/子目标/预训练",动作由另一条策略学。**"表示即条件"一支几乎全部是 2D 的,因为 2D 是跨本体/跨域共享成本最低的通道**。
4. **旋转编码的谱系**:离散 bin(Transporter、PerAct)→ 特征回归(RVT/Act3D)→ 扩散连续(3D-DA)→ 几何求解(Track2Act PnP、kPAM MPC、ReKep 优化)→ 外部解析器(MOKA grasp sampler)→ **根本不表达**(PIVOT/RoboPoint/ATM/RT-Trajectory 等只给点/轨迹/草图)。VLM 提示族几乎都选择"旋转绕开",这与 CamVLA"旋转是标定误差唯一通道"的观察互为镜像:图像空间方法把旋转交给几何/规划器,基座系方法把旋转交给隐式学习。

## 4. 查不到的/待核验(诚实标注)

- **ReKep 44.3% vs VoxPoser 10.0%**:来自 alphaxiv 摘要摘录,未与原文 Table 核对;VoxPoser 原文真机各任务成功率也未逐项摘录。
- **UniPi 官方代码**:google-research/uni-pi 返回 404,未找到官方仓库。
- **RT-Trajectory / LAPA / Im2Flow2Act / Track2Act 代码**:项目页/GitHub org 存在但内容未逐项核验。
- **OmniManip**:GitHub 上只有项目页仓库(omnimanip.github.io),未见正式实现代码。
- **PIVOT 真机成功率具体数值**:原文 Table 1/2 未在抓取中展开(摘录只确认"非零但远未完美");RoboPoint 的导航复测 2–4/5 可用作旁证。
- **GeneralFlow / UniPi / SuSIE / kPAM / VoxPoser / PerAct 真机的具体成功率数值**:未逐项从原文表格核验,只核了方法与相机假设。
- **PerAct 真机相机数量**:未核验(RLBench 为 4 相机)。

## 5. 对本线程(相机标定 × 动作表示)的直接含义

- **谁给了"图像空间动作精度"的定量证据**:只有 RVT-2 给了毫米级(真机插 peg/插 plug);Transporter 的 3.125mm 像素分辨率与 PerAct 的 1cm 体素是"离散化即精度上界"的量化;RoboPoint 的"点在 mask 内"与 MOKA 的 reasoning/execution 失败拆分是 grounding 精度指标;ReKep 第三方复测给出自动 vs 人工接地的差距(44.3% vs 68.6%)。
- **谁明确写了相机标定假设**:Transporter(自认标定敏感)、ReKep("标定 RGB-D 静态相机")、Track2Act(部署需深度+内参)、Im2Flow2Act/GeneralFlow/3D-DA/RVT-2/VoxPoser(标定 RGB-D 点云)。**明确声明"免标定/免深度"的只有 ATM(刻意 2D 相机系)与 TraceVLA/PIVOT(RGB 提示)**。
- 对 CamVLA 的对照价值:图像空间家族普遍把"旋转"外包给几何(规划器/PnP/grasp sampler)或干脆不表达,而 CamVLA 用自估 hand-eye 把旋转留给网络——精度误差路径的可审计性排序:几何求解(Track2Act/ReKep)> 离散 bin(Transporter/PerAct)> 隐式(VLM 族、3D-DA、CamVLA)。
