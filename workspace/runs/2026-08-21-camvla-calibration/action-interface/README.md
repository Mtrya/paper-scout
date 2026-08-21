# 动作接口线程:跨本体/视角稳健的动作表示

锚点:CamVLA(2607.05396)——动作表示放在相机系 + 自估 hand-eye 矩阵合成基座系动作。
本线程三篇近作从三个不同位置攻击同一根因,为"相机中心动作表示"提供旁证与对偶:
**BARX**(2607.27549,Stanford,行为对齐中间表示)、**ContactFlow**(2607.26579,Bonn,3D 接触点轨迹)、**AxisGuide**(2606.06761,Korea U,图像内坐标轴提示)。

## 1. 问题与动机

标准 VLA 的动作空间定义在机器人基座系,视觉观测在相机系。网络被迫隐式学相机↔基座的变换(hand-eye),这个隐变量一旦在部署时改变(相机被碰、重装、手持),策略就崩。CamVLA 引用的例子:π0 在 RLBench 训练视角 65.3%,15° 旋转后 6.3%。

三篇的切口各不相同:

- **换动作空间**:CamVLA / OC-VLA 把动作直接写成相机系(OC-VLA 用已知外参变换标签,CamVLA 干脆让网络自估外参)。
- **换观测空间(对偶路线)**:AxisGuide 保持基座系动作不变,但把"基座系 +x/+y/+z 在图像里指向哪"渲染成额外通道喂给策略——不解决"相机在哪",而是把坐标含义摊开在像素里。
- **换中间表示(旁证)**:BARX 证明"图像内、行为对齐"的中间表示(EE 2D 轨迹等)对跨本体迁移是有效对齐通道;ContactFlow 把动作条件收缩为"actor 与物体接触点 3D 轨迹",证明接触几何是具身无关的动作信号,能统一人类演示与机器人执行。

共同点:都在把动作/条件从"本体运动学空间"搬到"图像可对齐的空间",只是承载物不同——相机系向量、图像内箭头、2D 轨迹文本、3D 接触点投影。

## 2. 三篇各自的方法与证据

### 2.1 BARX(2607.27549)——行为对齐表示是跨本体迁移的隐式对齐通道

**方法形式化。** 跨本体 IL:策略 πθ(a|o,l),辅助表示 z(i) ∈ Z(k) 要求对本体不变且对动作预测有信息。训练损失:

```
L_total = E[ ℓ(πθ(·|o,l,ẑ), a) + Σ_k λ_k ℓ_rep(πθ(·|o,l), z(k)) ],  ẑ ~ p_rep
```

实践中把 VLA 训练成"自回归预测一个表示子集(文本)→ 动作"的序列;用不同文本提示区分要预测哪种表示。三种表示:

- **Bounding boxes**:Grounding DINO 标注当前帧目标物体框(文本坐标)。
- **Language motions**:本体感阈值化得到"move left and down"式运动描述(RT-H 式)。
- **End-effector traces**:未来一段 EE 在图像帧中的 2D 位置序列,LLARVA 预训练检测模型标注。

**RoboCasa-X 基准。** MimicGen 生成 3 个源本体(IIWA/Kinova3/UR5e,均 Robotiq 夹爪)的跨本体先验数据 XP-900 / XP-3K(900 / 3000 条每任务);3 个目标本体(Panda 与 Jaco + Robotiq 2F-85,Panda-OG 用 Franka Hand——**先验数据中从未出现过的手**);每个目标 50 条人类演示 co-fine-tune;每配置 100 次 rollout、3 个 checkpoint 取最高成功率。仿真中 bbox/trace 用**特权真值**(论文明说)。相机位姿在源/目标间分布一致(这个"过度对齐"论文在 limitations 里自己承认了)。

**关键证据(都是仿真 RoboCasa-X,除最后一条):**

- 每种表示单独用都优于无表示;EE traces 最有用,其次 language motions,bbox 最弱。ECoT(bbox→trace→motion→action 固定链)与 Joint Reps(单模型多表示)在 Panda/Jaco 上优于单表示;但 **Panda-OG(未见过的末端执行器)上组合反而输给只用 traces**——轨迹对"末端执行器差异大"的数据最有对齐价值。
- 表示随先验数据规模更有效:无先验时 +5%,XP-900 +15%,XP-3K +19%(除 Turn On Sink Faucet 外全部任务);XP-3K+表示在 Panda/Jaco 上接近甚至超过同本体先验 SP-900,但 Panda-OG 仍显著落后——大本体间隙仍是短板。
- **推理时不必预测表示**:只预测动作(训练时也见过该模式)与"先预测表示再动作"几乎等价——与 Chen et al. CoRL'25(Efficient Embodied Reasoning)结论一致,表示的主要收益在训练期隐式产生。这是对"中间表示=执行时接口"叙事的重要修正。
- **action-free 数据可用**:只用表示预训练(无动作头)再 co-fine-tune,比从零 +14%、比"全先验带动作但无表示" +11%,大部分收益保留(除 Panda-OG)。
- **sim→real**:FR3 与 ViperX 300S 两个真机目标(50 演示/任务),真实动作空间存在控制频率、阻塞/非阻塞差异。无表示时跨本体先验只带来 +7%(vs 仅目标数据),**Joint Reps 时 +28% task progress**——真机增益大于仿真,因为真机域差更大。

**协议弱点:**

1. 仿真中各本体共享同一相机位姿分布、场景、任务集,动作空间统一为 delta Cartesian EE pose——limitations 自认"representations 因此被过度对齐",真实跨本体数据(Open X-Embodiment 的 1417 种相机视角)比这混乱得多。
2. 真机报告的是 task progress 且 10–30 rollouts/embodiment-task,规模小;数字是"最高 checkpoint"非均值。
3. ECoT 的链条顺序是固定假设,未消融顺序;bbox 弱可能因为它最依赖场景而非行为。
4. 无对照基线:没和"直接用 trace 作输入条件"(RT-Trajectory 式)比,也没和相机系动作(CamVLA/OC-VLA 式)比——它证明"表示有用",没证明"文本表示是最优载体"。
5. 真机 trace 靠 LLARVA 检测器,标注噪声未审计(仿真特权 vs 真机检测的差异没量化)。

### 2.2 ContactFlow(2607.26579)——接触点轨迹是具身无关的世界模型动作接口

**表示。** 每时刻一组稀疏 7 通道控制帧,投影到图像平面构成条件视频:

```
C_t = { c_t^(i) },  c_t^(i) = (x, y, z, Δx, Δy, Δz, w) ∈ R^7
```

(x,y,z) 为物体表面接触点在相机系的 3D 位置,(Δx,Δy,Δz) 为其到下一帧的位移(接触处主动动力学),w∈[0,1] 是置信度(几何一致性调制:邻域方向一致上加权、孤立点降权)。**刻意排除物体随后发生的被动运动**(如门把手松开后的开门)——主动动力学与被动动力学解耦。

**世界模型。** Wan 系 latent DiT + flow matching:

```
L_FM = E[ || v_θ(u_τ, τ, z0, C_1:T) − (z_1:T − ε) ||² ]
```

条件 = 首帧 z0 + 接触流 C_1:T;用 ControlNet(Wan2.2-5B/14B)与 VACE(Wan2.1-14B)两种注入机制验证与架构无关。

**数据管线(这是工作量的大头)。** 人类侧:HaMeR 手位姿 + MANO 拟合 + HACO 接触判定(只认右手,左手翻转输入;HACO 判定"是否接触"不可靠,须用几何距离 + mask 交叠过滤)+ SAM3 物体 mask,深度来自 FoundationStereo/MapAnything;机器人侧:URDF 渲染 + 3D bbox 定位(WildDet3D)+ SAM3(在 RoboEngine + 1000 手工图上微调)+ RobotInter 过滤。训练数据混合 DROID + Taste-ROB + TACO + OakInk + LIBERO。

**证据。** 指标全部在**遮掉 actor(手臂/手)的区域**上算(PSNR/SSIM/LPIPS/DreamSim/FID/FVD;DreamSim 为第一指标),25 条 held-out 片段 × 49 帧 @8FPS 832×480,stride 3(≈10s 预测):

- DROID:CF 全部配置 DreamSim 0.035–0.039 < Kinema4D 0.043(全机器人 4D pointmap 条件,同为 zero-shot)< CTRL-World 0.059(动作嵌入,非 zero-shot)< TesserAct 0.106(纯语言)。
- 跨数据集(训练集未见过的 RLBench/AgiBot/GenieSim ood,以及 TACO/TASTE-Rob/OakInk 人类手):混合训练 14B 在多数基准上 DreamSim 最优;**EgoDex ood 全线失败**(所有方法 0.123–0.218,egocentric 视角外推是共同的坑)。
- 真机闭环:Franka Panda 固定臂 + 单个外参 RGBD;场景一次重建(FoundationStereo + SAM3 + SAM3D + ICP + 可微渲染精修)成 symbolic twin,π0.5 在 twin 里提 EE 轨迹 → 转接触流 → 世界模型想象 rollout → Gemini VLM 判定 → 开环执行。**8/10 个未见场景预测正确;π0.5 直跑真机 0 成功。** 论文明确:验证步是部署能 work 的原因,π0.5 自己的轨迹太粗糙不能自证。

**协议弱点:**

1. **状态还停在投稿前**:Acknowledgments 是模板占位("If a paper is accepted...")。无代码、无数据链接。
2. 跨数据集表里 14B(mix)vs 14B(DROID-only)vs 5B 的差异解读模糊——mix 在人类数据基准上最好,但 AgiBot ood 上 14B(DROID)反而不如 5B;训练混合比例的作用没讲清。
3. 接触估计链很长(深度→点图→mask→接触),每环都依赖"标定的静态外参相机 + 可靠深度"(limitations 自认);真机只报 8/10,开环执行,成功标准只看终态;没有量化"跳过验证直接执行"的失败代价,因此"验证器增值"只有定性论证。
4. 置信度 w 是启发式组合,接触真值没有人工质量审计;HACO/URDF 渲染的接触点与真实接触的偏差没有数值报告。
5. 与策略类工作不同,这是世界模型的**条件接口**,不产出动作——"验证器"语义与"动作命令"语义是两回事,横向比较时需注意。

### 2.3 AxisGuide(2606.06761)——对偶路线:把基座系坐标轴渲染进图像

**动机实验。** SmolVLA 在"基座清晰可见"的简单 pick-up 任务上,物体换到未见位置就失败——模型把图像↔动作向量当记忆背下来,没学会"基座系 +x 在图像里指哪"。与 concurrent LIBERO-Plus 报告一致。

**方法。** 用相机内参 K、外参 T_c←w、EE 位姿 p_w,把基座系三个单位平移投影到图像:

```
Δu^(k) = Π(p_w + ε·e_k) − Π(p_w),  k ∈ {x,y,z},  Π = pinhole 投影
```

归一化方向,画在以 EE 投影像素为锚点的 3 通道图上(x/y/z 分别红/绿/蓝),与 RGB **通道拼接**成 6 通道输入(只改 backbone 首层卷积 3→6,其余不动)。不需深度。消融(Table V)证明最终设计 = **通道拼接 + EEF 锚定 + 归一化单位方向**(88.00% vs overlay 84.00 / center 80.00 / 不归一化 85.30,基线 74.67)。

**证据:**

- LIBERO-Spatial novel object position(210 rollouts,未见位置):52.38→65.71 (+13.33pp);SmolVLA+TraceVLA 51.90、+AimBot 52.38——**场景级空间 cue 无增益,动作坐标系 cue 才有**。真机 Pick Up(Pear,84 rollouts):30.12→50.00 (+19.88pp)。
- 单视图 DP:LIBERO Pick&Place/Drawer/Stove +13.34/+6.66/+8.00pp(优于 KYC +9.34pp on P&P);真机 3 任务 +10.00/+13.35/+6.67pp。
- 多视图(前 + 腕部相机):仿真 Pick&Place +13.3pp、Put Both +7.9pp;真机 Flip Pot **+20.00pp**、Close Pot +16.67pp——腕部相机视角连续变化时坐标提示最有用。
- SmolVLA 多任务 4 个 LIBERO suite:全一致增益,最好 LIBERO-Object +7.5pp(vs 全量微调 SmolVLA)。
- **标定鲁棒性(Table IV)**:推理时注入外参扰动 0–6°/0–3cm 平移、内参 0–10% 焦距/主点偏移,成功率 61.4–65.7% 基本平,**仍超无扰动 baseline 52.4%**——提示被当成有噪声的方向信息,不随标定误差数值化地进入动作。
- 视角泛化(Table VI):−45°~+45° 按 22.5° 间隔 5 个视角训练,10° 间隔测试,Avg 80.28 vs SmolVLA 71.94、+KYC 77.22;但 −30°、+10° 两个角度上 AxisGuide 反而低于 KYC——非一致优势。
- 开销:+0.590M 参数(+0.13%)、+5.41ms/推理——几乎免费。

**与 CamVLA/OC-VLA 的路线分歧(这是本线程最值得记的一条):**

| | CamVLA / OC-VLA | AxisGuide |
|---|---|---|
| 动作空间 | 换成相机系(执行时合成基座系) | 保持基座系 |
| 外参 | CamVLA 自估 / OC-VLA 必须已知 | **必须已知,但误差只进提示不进动作** |
| 网络负担 | 必须学会从 RGB 自估 hand-eye(新几何头) | 零架构负担,纯输入增强 |
| 免标定部署 | CamVLA 卖点 | 不是目标(但 Table IV 显示相当抗标定错) |
| 适用模型 | 需改动作头/训练目标 | DP/VLA/任意策略通吃 |

AxisGuide 是"免标定"家族的反命题:它接受已知标定,但把标定信息**降级为冗余提示**——对 VLM agent 而言,坐标箭头是"看得懂"的信息,而非藏在网络权重里的隐变量。缺点同样明显:它需要每帧 EE 位姿 + 内外参,在 DROID 级大规模数据上不可得;且提示依赖本体运动学,跨本体训练时不同本体的提示本身就不一致(论文没有跨本体实验)。

**协议弱点:**

1. 实验范围窄:仅 LIBERO(训练演示重渲染,过滤后 1711 条)+ 3 个真机任务各 50 演示;无大规模数据集验证(limitations 自认)。
2. SmolVLA 对比口径:官方默认冻结 backbone 训 action-expert,AxisGuide 必须训 backbone,论文同时给了两种 baseline——公平性取决于读者取哪个口径;Table 里与冻结版对比的增益会虚高。
3. 标定鲁棒实验只做推理期扰动、只在 novel position 任务、只测了 DP backbone;训练期标定漂移的情况没测。
4. 代码里有多种 cue 变体(全局消失点方向、Jacobian 像素/米缩放、局部投影),论文只报最终版,复现其他变体不可直接。
5. 提示是逐帧从 EE 位姿渲染的,EE 位姿本身错(运动学误差)会直接错位锚点;Table IV 的"机器人运动学误差"实际只测了投影链,没单独测 EE 位姿误差。

## 3. 横向比较

### 3.1 五条路线(含锚点家族)

| | CamVLA | OC-VLA | AxisGuide | BARX EE traces | ContactFlow |
|---|---|---|---|---|---|
| 动作/条件的载体 | 相机系 delta 动作 + 自估 hand-eye R | 相机系 delta 动作(已知外参变换) | 基座系动作 + 图像坐标轴提示通道 | 未来 2D EE 轨迹(文本) | 3D 接触点轨迹投影(7ch 视频条件) |
| 信息来源 | RGB(隐式)+ 基座系 EE 位姿 | 标签变换(已知 K/T) | K/T/EE 位姿 + 渲染 | 仿真特权 / LLARVA 检测(真机) | 深度 + 点图 + URDF/HaMeR/HACO + SAM3 |
| 外参依赖 | **无**(自估,误差只经 R_t) | 必须(训练标签 + 推理反变换) | 必须,但误差只进提示(Table IV 鲁棒) | 无(图像内 2D) | 必须(标定静态外参相机) |
| 深度依赖 | 无 | 无 | 无 | 无 | 必须(metric depth) |
| 跨本体性 | 视角稳健;本体间动作仍 delta EE pose,需各自 hand-eye | 同左 | 依赖本体运动学渲染提示,未验证跨本体 | **实证最好**(EE traces 最有用;真机 +28%) | **理论最强**(接触几何与 actor 无关;人类/机器人共享同一条件) |
| VLM agent 友好度 | 专用动作头 + 几何头,预测连续量 | 标签即插即用,但推理要外参 | 纯输入增强,零头改动,DP/VLA 通吃 | 文本 token,天然自回归/ECoT,可 action-free | 不是 VLM 输入,是世界模型条件(验证器) |
| 视角外推证据 | RLBench 密集 unseen 视角 +18.2pp;真机 15° 偏移 +13–18pp | ManiSkill2 随机相机 +8–14pp;真机 novel view −14% 下降 vs 基线 −20%+ | LIBERO viewpoint 平均 +8.34pp | 相机位姿源/目标同分布(未测视角外推) | 跨数据集 0-shot 强(egocentric EgoDex 除外) |

### 3.2 三条主轴

**信息从哪来:几何 vs 感知 vs 特权。** AxisGuide 与 OC-VLA 用几何(已知标定);CamVLA 用感知(RGB 自估);BARX 仿真用特权、真机用检测(LLARVA);ContactFlow 用感知+几何混合(深度估计、mask、URDF)。这是"免标定"的代价轴:CamVLA 免标定但把 hand-eye 塞给网络当回归任务(误差进 R);AxisGuide 要标定但误差只进提示。**越免标定,误差越往网络隐式部分挪;越几何化,误差越可审计。**

**跨本体性靠什么:行为对齐(图像内轨迹)vs 物理不变(接触几何)vs 渲染对齐(坐标提示)。** BARX 证明图像内 EE 轨迹是三种表示里跨本体最有效的——它是"行为"的图像投影,对末端执行器外观/运动学不变;ContactFlow 更进一步,只保留"接触处"的几何,连非接触运动都丢掉,换来的代价是需要在接触估计上花大量工程(并且只在世界模型验证器场景里被验证);AxisGuide 的提示逐本体渲染,跨本体时提示集合不一致,这一支的跨本体性尚未被实证。

**对 VLM agent:表示放哪一层。** BARX 把表示做成文本(序列里先"说"表示再"说"动作,ECoT),这是 VLM 原生形态,且推理时可省(BARX/Chen et al. 两篇独立结论)——表示主要做训练期正则/课程,不是执行时必需接口。AxisGuide 放在输入通道,模型架构零改动。CamVLA 放在输出头(需要专用几何头)。ContactFlow 放在世界模型条件(验证器),与策略接口正交。**文本化(BARX)与输入增强(AxisGuide)是最"VLM 友好"的两端,它们恰好都不要求模型学会几何。**

## 4. 代码核验(2026-08-21)

| 论文 | 仓库 | 放出内容 | 抽查结论 |
|---|---|---|---|
| BARX | `ajaysridhar0/barx`(MIT,2026-08-03,sim-only) | 训练/评估全代码 + RoboCasa-X 基准;HF collections:预训练模型(Apache-2.0)、RLDS 数据集、raw HDF5(CC BY 4.0) | ✅ 与论文一致 |
| AxisGuide | `JiyunJang-24/AxisGuide-code`(Apache-2.0,2026-07-19,lerobot fork) | 训练/评估脚本 + LIBERO 依赖;`visual_cue_mode=basis_concat` | ✅ 与论文一致 |
| ContactFlow | 无专门仓库(作者 samiazirar 放出全部管线组件:Kinema4D、TASTE-Rob、TACO-Instructions、HACO_RELEASE、SAM3/sam-3d-objects、FoundationStereo、WildDet3D、RoboInter、RobotSeg、ContactHands、AnyHand) | 整合代码/数据**未放出**;论文仍为投稿前状态(Acknowledgments 模板占位) | ❌ |
| CamVLA(锚点) | `alibaba-damo-academy/CamVLA` | **只有项目页**(index.html/images/videos),无代码 | ❌ |
| OC-VLA(锚点) | 无公开仓库(论文仅"code will be publicly available") | — | ❌ |

**BARX 抽查(2 点):**

1. **表示确实以文本 QA 形式进模型序列**:`policy/prismatic/vla/datasets/datasets.py` 的 `AUX_TASK_QA_FUNCTIONS` 含 bbox / low_level_motion / obj_pose / ee_pose_2D 四类,提示词 `"predict the end-effector's future 2D trace to ..."`,答案 `[(x, y), ...]` 保留 2 位小数;`ChainedTransform` 注释明写 "Chains aux and action prediction in a CoT way"(ECoT),JointReps 用 `aux_task_types` 列表实现。配置 `future_2D_trace_window_size=40, ee_pose_2D_stride=8` → 5 个未来 2D 点。动作用 OpenVLA 式 `ActionTokenizer`(vq/ 下逐任务 tokenizer 是动作的,不是 trace 的)。
2. **仿真 trace 是特权真值**:`dataset/rlds/robocasa_x_dataset_builder.py:86` 的 `ee_pose_2D` 直接取自仿真 `ee_positions_2d`;`barx/names.py` 中 `"end_effector_trace": "ee_pose_2D"`。与论文"仿真用特权真值、真机用 LLARVA"一致。

**AxisGuide 抽查(2 点):**

1. **渲染几何与论文一致**:`src/lerobot/datasets/visual_cue_utils.py` 的 `_get_motion_dynamics_basis` 把基座系单位方向经 R_cw、K 投影成图像方向(消失点法);`project_world_point_to_pixel_cam_to_world` 把 EE 世界位姿投影为箭头锚点;x/y/z 三色箭头在独立画布(非叠加)上绘制——通道拼接语义。
2. **模式开关**:README 明示 `--policy.visual_cue_mode="basis_concat"` 才是论文变体,否则 "you are **not** running the AxisGuide variant"。代码内含多种实验变体(全局消失点方向、Jacobian 像素/米物理缩放箭头、局部投影),与论文 Table V 消融(overlay/concat、center/EEF、归一化)对应,论文只报最终版——**这解释了为什么消融表要特意做**:渲染方案本身就有很多自由度。

## 5. 开放问题

### 5.1 相机系动作与点轨迹表示能否统一?

三条线其实都在把动作"图像对齐化",但承载物不同维度:CamVLA 是 3D 相机系向量(自由向量,线性变换),BARX trace 是 2D 图像轨迹(丢深度),ContactFlow 是 3D 相机系接触点(带深度,逐点流)。关键张力:

- **2D 轨迹深度自由但与执行之间有鸿沟**(2D 轨迹不能直接执行,还要再映射回 3D);**3D 相机系向量可执行但要深度或自估几何**(CamVLA 用自估 hand-eye 避开了深度,但 Δp_c 本身是 3D 的,只解决"相机↔基座",不解决"图像↔3D"——它靠网络隐式学后者)。
- 一个自然的统一候选:**把相机系 delta 动作投影成图像内方向**(AxisGuide 渲染的就是这个东西的"含义"),或反过来**把 2D trace 提升为带置信度的 3D 相机系点流**(ContactFlow 的接触点 + Δ 就是局部实例)。BARX 的 EE trace 其实与 CamVLA 的 Δp_c 投影在数学上同源:前者是 EE 未来位置序列的图像投影,后者是单步位移的相机系向量——**差在一个是"未来点集"、一个是"增量向量",一个是"给世界模型/给序列"、一个是"给执行器"**。
- BARX 的教训要先吸收:表示收益主要在训练期隐式产生,推理时可不预测(Chen et al. 独立佐证)。所以"统一"如果只是把两种表示拼进一个序列,可能只是加正则;真正值得试的统一是**同一份图像对齐表示同时充当训练期中间表示与执行期动作参数**(如:策略输出相机系 delta,但训练时同时监督其 2D 投影 trace 与接触点),让 CamVLA 的几何头与 BARX 的表示监督互相增强。
- ContactFlow 与 BARX trace 的实质区别也要摆正:接触点编码**主动动力学**(接触处的驱动),EE trace 编码**执行器轨迹**(含非接触运动)。世界模型场景要前者(结果判别力),策略接口场景要后者(如何到达)——"统一"不应该是选一个,而是分清两种语义各在哪一层用。

### 5.2 外参误差沿哪条路径进入执行?

这是本线程最可操作的问题,五条线的答案各不相同:

1. **OC-VLA:误差走全部 T**。训练标签变换与推理反变换都含外参,外参错则动作整体错(平移直接进动作数值)。它是五条线里最"标定敏感"的。
2. **CamVLA:误差只走 R_t**。Δp_b = R_t·Δp_c,平移 τ 对相对动作零物理影响(论文明确利用自由向量性质把 τ 的误差隔离)。所以误差路径唯一:旋转误差 × 相机系位移。实测:15° 视角偏移下自估 R 误差达 9.4°,trans 误差 27cm(但无物理后果),成功率仍 29–33%(基线 15–16%)——**9.4° 是闭环策略能容忍的旋转误差上限量级**。这是"自估 hand-eye"路线的核心卖点:把 6-DoF 标定误差问题降成 3-DoF 旋转误差问题。
3. **AxisGuide:误差只进提示,不进动作数值**。箭头方向/锚点位置错 → 感知噪声;Table IV 显示 6°/3cm 内成功率几乎不掉。它的外参是"解说词"不是"公式"。代价:提示本身依赖运动学,EE 位姿误差会直接错位锚点(未单独测)。
4. **BARX trace:无外参依赖(2D 图像内)**。仿真特权,真机检测器,误差路径走检测器而非标定;但 2D→执行要另一次映射,误差在映射里。
5. **ContactFlow:误差走整条感知链**。外参 → 深度/点图 → mask → 接触点 3D 位置 → 投影与置信度,链上任何一环的错都进条件信号;8/10 真机表现意味着链条总误差在开环验证预算内,但没有分环节误差报告。

**总结论**:误差路径 = "动作数值里是否含 T(OC-VLA 最敏感) → 是否只含 R(CamVLA) → 是否只含提示(AxisGuide) → 是否不含外参(BARX trace) → 是否整链含(BARX 真机/ContactFlow)"。对相机系动作家族,外参误差经旋转进入执行;对图像内表示家族,外参误差经投影进入但方向性提示有冗余容错。**AxisGuide 的 Table IV 与 CamVLA 的 Table 3/4 合起来给出一个经验上界:方向提示与自估旋转都容忍 ~10° 级误差,而直接数值化外参(OC-VLA)的容忍度由 T 的平移项主导。**

### 5.3 其他

- BARX 在 Panda-OG(未见过的末端执行器)上表示失效:行为对齐表示的边界在哪?相机系动作在大本体间隙下是否同样失效?→ 值得拿 CamVLA 的设定(跨本体 + 视角)复测 BARX 的表示。
- AxisGuide 与 CamVLA 的融合点:**用 CamVLA 的自估 hand-eye 生成 AxisGuide 坐标轴提示**——免标定家族与输入增强家族最自然的交叉,提示从"已知标定渲染"变成"自估标定渲染",误差仍是提示级。
- ContactFlow 的 egocentric 全崩(EgoDex):接触点/轨迹类表示对视角外推的脆弱性是否普遍?BARX 的 trace 检测器(LLARVA)在 egocentric 视图上同样可能崩,值得交叉验证。
- 世界模型验证器(ContactFlow)与策略(其余)用同一接口语义的问题:接触点/轨迹能否同时充当"策略的动作条件"与"验证器的结果判据",统一到 GR-1/lingbot 式的视频-动作联合模型里?

---

**一句话收束**:三篇共同指向"把动作从本体运动学空间搬到图像可对齐的空间",但搬法分两族——**数值族**(相机系向量,误差经旋转进入,免标定靠自估)与**提示族**(图像内方向/轨迹/接触点,误差经投影进入,有冗余容错)。BARX 给数值族补了"表示收益在训练期"的警醒,ContactFlow 给提示族补了"只留接触"的极致化,AxisGuide 证明提示族对标定误差天然鲁棒——CamVLA 的"自估 hand-eye + 相机系动作"恰好站在两族之间:动作数值化(误差进 R)但几何自估(免标定)。
