# GreenWAM(arXiv 2610.18774)线程包:论文—代码—榜单三方核验

**一句话结论**:GreenWAM 的方法侧**真实存在且经得起独立核验**——79 槽解剖学动作空间(legs 12 / body 23 / arms 18 / hands 26)、43 个已注册布局、21 个族簇、1217 个数据 root、per-layout 符号翻转(114/1178 通道)与有效性掩码都在 `greenwam/registry/` 里逐条落成 JSON,`semantic_space.py` + `shared_projectors.py` 是可运行代码(我做过 native→slot→79 槽的 round-trip,误差 0.0);"policy = generator + action expert"也是可测的架构增量:**四个 HF checkpoint 真实存在(35.45 GB)**,policy 比 world model 多 281 个 `*_action_expert*` 张量、多 **1.409 B 参数(= generator 塔 1.409 B 的完整孪生,占 policy 4.780 B 的 29.5%)**。榜单侧**口径基本诚实但覆盖面有选择性**:WorldArena 的每个数字我都能用公开榜单文件逐一复算命中(112/127、113/127、UNIS 73.64→63.65、JEPA 98.41 第 2 名只输 BLM 98.49),但 63.31 在加了运动帽的完整 127 条目榜上只排 **#35**,且已有 11 个条目在 trajectory、11 个在 depth 上高于它(除 BLM 外全部闭源);RoboTwin 的 67.54 按榜单协议排 **#5/20**——论文 Table 3 **漏掉了两个排在它前面的榜单条目**(PatchWAM 79.14,2026-10-02 上榜;ViGAR 74.51,2026-10-11 上榜,都早于论文 10-13 的日期);真实人形 95.6% vs π0.5 84.4% 是 45 条指令上的 43:38,**Fisher 精确检验 p=0.16、差值 95% CI 跨 0,不可分辨**。LIBERO-plus 零样本 86.8 是"所比方法中最高",我查到的公开零样本数字里确实没有更高的(次高 CKT-WAM 86.1、π0.5 85.7;PatchWAM 的 91.8 是**在 LIBERO-plus 增强数据上训练**,其零样本只有 80.2)。

---

## 0. 五分钟背景

论文(Sber Robotics Center,2026-10-13)问一个问题:能不能让**同一个机器人世界模型既当世界模型又当策略,并且跨本体**?做法是拿 NVIDIA Cosmos 3 Edge 4B 当父模型,冻结 reasoner 与视频 VAE,robotics-stage 只训 generator、共享动作投影与动作模态嵌入(reasoner 与视频 VAE 冻结):(1) 把 20 个机器人族的原生关节状态映射进一套**共享解剖学动作空间**(79 槽,legs/body/arms/hands 四块,布局专用 codec 负责选取、重排、翻符号、对齐零点、并在有效位掩码里排除缺失坐标);(2) 用**七种训练模式**(text→video、image→video、image→video+history、forward dynamics、forward dynamics+history、inverse dynamics、world–action WAM)在同一套 generator 权重上联合训练,数据是 1.44 M 个"subtask 原子"、1688.5 小时(清洗后 1.43 M / 1674.7 h,其中 1.19 M 有可用指令);(3) 要策略时把动作 token 搬进一个**从 generator 复制出来的 action expert**,两层 token 仍走同一套联合注意力。四点主张:一个模型两种角色、跨 20 个族共享一个动作接口、WorldArena 1.0 上 trajectory/depth/3D 顶级、真实人形上超过 π0.5 适配版。受控实验给出三条"为什么管用":机器人专门化同时提升预测与控制(Q1)、开始时切出 action expert 提升成功率(Q2)、策略适配期保留世界模型模式提升随机场景鲁棒性(Q3)。

三个承重术语(报告里首次出现需要定义):
- **subtask 原子(atom)**:一段有语义边界、带同步多视角视频、实测状态、文本与来源信息的子任务片段;训练与采样都以它为单位(不是固定长度窗口)。
- **EWMScore**:论文自定义的 WorldArena 1.0 track-1 聚合分 = 15 个榜单指标的均值,其中 Dynamic Degree / Flow Score / Motion Smoothness **按每个视频截到真值视频的值**(WorldArena 2.0 的运动帽规则),满分 100。
- **解剖学槽位**:一个编号 0–78 的共享关节坐标表;每个槽位是一个具体关节角色(如 `arms.left.elbow`、`hands.right.gripper`),布局 codec 负责把原生通道放进对应槽位并给出符号与零点。

## 1. 论文关键数字(便于写报告时取用)

| 项目 | 数字 | 出处/口径 |
| --- | --- | --- |
| 起点 | Cosmos 3 Edge 4B(冻结 reasoner + VAE) | §3.1 |
| 数据 | 1.44 M 原子 / 1688.5 h → 保留 1.43 M / 1674.7 h;1.24 M 有 caption,1.19 M 有命令 | §4.2、A.2 |
| 训练 | 64×H100,global batch 256,36 500 步 = 9.34 M 样本;λ=10(video+state 等权) | §4.3、Table 5 |
| 模式占比 | t2v 20 / i2v 10 / i2v+history 10 / forward 15 / forward+history 20 / inverse 10 / WAM 15(%) | Table 2 |
| WAM horizon | ≈1.07 s(17 个未来状态 @16.67 Hz) | Table 5、B.2 |
| WorldArena | EWMScore 63.31(帽后)/ 63.35(帽前);trajectory 56.0、depth 98.2、3D 93.4、JEPA 98.41;真值视频自评 67.23 | §5.2、Table 11 |
| RoboTwin 2.0 | clean 86.98 / randomized 48.10 / 平均 67.54(50 任务 ×100 试次) | §6.3.1 |
| CALVIN | 4.64(1000 条链) | §6.3.2、Table 12 |
| RoboDojo | 16.12 分 / 10.49%(榜第 1 是 36.27 / 31.38%) | Table 13 |
| LIBERO / LIBERO-plus | 97.4 / 86.8 零样本(同一策略) | §6.3.4、Table 14 |
| 真实人形 | Green 机器人打包任务:ID 95.6%、OOD 84.0%;Cosmos 3 Edge 88.9/80.0;π0.5 84.4/76.0;蒸馏学生 91.1/80.0 | §6.5、C.2 |
| 延迟 | RTX 4090,435.9 ms → 优化 134.6 ms → 蒸馏一步 35.4 ms(28.3 Hz) | §6.4、Table 10 |
| 受控对比 | Q1 专门化 +5.4%(clean +7.9);Q2 action expert +3.3%;Q3 保留世界模型模式:randomized +8.0%、clean 不变 | §7、Figure 18 |

## 2. 代码审计(一):79 槽解剖学动作空间是真的

发布仓:`code/greenwam-repo`(GitHub `greenvla/GreenWAM`,基于 NVIDIA cosmos-framework 2026-07-20 版,license OpenMDW-1.1)。审计脚本 `code/audit_action_space.py`(纯读;带 torch 的解释器可跑 round-trip)。

| 论文说法 | 代码证据 | 核验 |
| --- | --- | --- |
| 79 槽,legs 12 / body 23 / arms 18 / hands 26 | `greenwam/registry/semantic_space_v1.json`:`semantic_dim=79`、`body_parts` 块 0–12/12–35/35–53/53–79、79 个唯一槽名 | PASS(脚本逐条打印) |
| 20 个机器人族 | 注册表实际是 **21 个族簇**(C01–C21),`family_projectors_v2.json:counts.families=21`;Table 8 本身列了 21 行(含 OpenArm,<0.01 h) | 计数不一致(见 §7) |
| 43 个注册布局、1217 个 root | `layouts.json:n_layouts=43`;`shared_projectors.py:196` 硬校验 43 布局 / 1217 roots | PASS |
| 布局 codec:选取、重排、翻符号、对零点 | 每个布局的 `recipe` 里有 `source_indices` / `destination_slots` / `scale` / `offset`;`shared_projectors.py:103-143` 执行 `value = x[idx]*scale + offset`;全库 **114/1178 个通道 scale<0(26 个布局有符号翻转)**,**只有 4 个通道 offset≠0**(AgibotWorld G2A 的第 14/15 通道 =1.0;ManipArena EX001 第 12/13 通道) | 机制真实,"零点"自由度几乎没被用到 |
| 有效性掩码把缺失/不可解坐标排除出损失 | 每个布局有 `availability_mask`(静态)+ 逐帧有效位;`validity_contract.clip=false`("reject invalid channels, never clip");语义层还有 `shared_mask`(全库排除 42 个坐标不进共享监督) | PASS |
| 缺失坐标永远不当 0 目标学 | `semantic_space.pack()` 对缺席槽位写 0 并返回 bool mask;损失用 mask 排除 | PASS |
| 一个共享投影器、本体没有自己的动作参数 | `cosmos_framework/model/generator/mot/semantic_action_projector.py`:`SharedSemanticActionInput/Output`,输入 = `value(79槽值) + mask(有效性掩码) + presence(该槽在本布局是否存在) + context(布局 id 嵌入) + representation(joint / 相对 EEF / 绝对 EEF 嵌入)`,五个加性项都是共享权重 + 静态查表。**权重实测**(worldarena checkpoint header):`value/mask/presence` 各 `[2048,79]`、`context [44,2048]`、`representation [3,2048]`,两个方向合计 **0.788 M 参数** | 基本成立,但**有每布局的 context 嵌入**(44×2048=90 k,占投影器 11%、占 3.37 B 模型的 0.0027%),严格说法是"没有每本体的投影矩阵,只有每布局的小偏置表" |
| 新机器人只需 codec | `shared_projectors.add_layout_extensions` + 语义扩展文件:RoboTwin 走 `benchmarks/robotwin/layout_extensions.json`(声明复用语义布局 42 = dual_arm_b/ARX5),Franka 走 `benchmarks/semantic_extensions/franka_eef_v1.json`(新增 contract 76/90 + 布局 44,把 xyz+rot6d+gripper 放进 `arms.right.*` 与 `hands.right.gripper` 槽位) | PASS,且扩展只增不改:扩展文件必须匹配父空间 sha256,复用的槽位集合逐条比对 |
| 符号/零点由 codec 负责 | `anatomical_maps_v2.json` 每条通道带 `identity_level: anatomical_role` 与 `limitation: "Role from URDF zero-pose kinematics or producer schema; **units, sign, zero and calibration remain source-codec dependent**"`(630 条同类说明) | 论文 Eq.(1) 的 ρ 与 o 在发布物里不是"解剖学身份"的一部分,而是数据侧拟合出来的 `scale/offset`;注册表明确拒绝为它们背书 |

可运行的 round-trip(用带 torch 的解释器驱动发布代码;我用 `code/onewm-env/bin/python`,CPU 即可):

```bash
python3 code/audit_action_space.py --torch-python ../../../../code/onewm-env/bin/python
# ROBOTWIN target_dim 14 / available slots 14 / round-trip max abs error 0.0
# semantic layout id 42 / semantic_dim 79 / transport_dim 128
```

## 3. 代码审计(二):七种训练模式与数据编解码

- 模式表在 `greenwam/configs/greenwam_modes.py`:`DEFAULT_MODE_WEIGHTS = {t2v 20, i2v 10, i2v_ar 10, forward 15, forward_ar 20, inverse 10, wam 15}`,**与论文 Table 2 逐个数字一致**;另有第 8 个 `policy` 模式(策略适配期的 WAM 变体)。
- 每个模式由"哪些流是干净条件、哪些是加噪目标"定义:`mode_spec()` 给出 `condition_frame_indexes_vision`(t2v 无、inverse 用整段视频、AR 模式固定四条历史锚 `[0,1,2,3]`、其余用首帧)与 `condition_frame_indexes_action`;AR 模式带 `vision_prefix_frame_count: 4`,与论文"四个固定位置 RGB 锚、训练与推理同位置"一致(`B.2` Eq.11 的位置公式在 markdown 里,代码只保留"四anchor"这一契约)。
- 相对量编码在 `greenwam/data/joint_deltas.py`,即论文 Eq.(2):`encode_joint_deltas` 只对 `joint_mask` 选中的槽位做 `q_k − q0`,夹爪/手指保持绝对;`delta_target_slots` 允许布局显式声明"整段状态相对"(RoboTwin 声明前 12 个臂关节为相对),解码是 `q0 + δ` 而非累积和(`decode_joint_deltas`);q0 作为干净前缀走同一个共享投影器,不走单独的 state2llm。
- WAM 的正样本门控:`wam_outcomes.py` 的 `successful()` 要求 `outcome=success` 且带 `evidence_ref` 与 ledger sha256,与论文"WAM 只用正成功标签"一致。WAM 帧数由物理时间下取整:`wam_future_frames(1.0667 s, 16.667 Hz)=17`。
- 损失权重:发布 config 里 `rectified_flow_training_config.action_loss_weight=10.0`、`loss_scale=10.0`(论文 λ=10);世界模型 config 用 `action_loss_reduction: mean_over_valid_body_parts_v1`(论文 Eq.6 的"体内先平均再跨体平均"),RoboTwin/libero/calvin 策略 config 覆写为 `mean_over_valid_coordinates_v1`(论文 B.5"14 个坐标等权"),两处在代码里是可切的 schema 开关而非描述。
- 值得记录的代码注释:`greenwam_modes.py:14` 写着 `# Starter sample shares, not an empirically optimized or compute-matched mixture.`——论文把同一组比例呈现为"The fixed mixture gives three quarters…"(见 §7)。

## 4. 代码审计(三):action expert 怎么从 generator 切分(附权重实测)

- 机制(`cosmos_framework/model/generator/mot/unified_mot.py`):`_action_expert_rows()` 把 generation 流里**动作行**的 q/k/v/o 输出换成 `*_action_expert` 孪生模块的计算(其余 token 仍走原塔),`init_action_expert_from_gen()` 用 `*_moe_gen → *_action_expert` 逐张量 copy 初始化;两层 token 在同一个联合注意力里互相看得到(`PackedAttentionMoT.forward`,2026-07-20 版 cosmos-framework 的 MoT 实现)。config 开关 `action_expert: bool` 的注释写明"Part of the checkpoint schema",即它是 checkpoint 层面的新阶段。
- 权重侧实测(不下载权重,只读 safetensors header;脚本 `code/audit_checkpoints.py`):

| checkpoint | 张量数 | 参数 | `*_action_expert*` |
| --- | --- | --- | --- |
| robotwin(policy) | 852 | 4.780 B | 281 张量 / **1.409 B(29.5%)** |
| worldarena(世界模型) | 571 | 3.370 B | 0 |
| 差 | +281 | +1.409 B | = 1.409 B |

- 逐层解剖(28 层):每层孪生 10 个张量 = `input/post_attention layernorm` + `q/k/v/o_proj` + `q/k_norm` + `mlp.up/down`(各 18.87 M),**每层 50.34 M × 28 = 1.409 B,与生成通路 `*_moe_gen` 完全等大**——论文"expert 增加与 generator 塔一样多的参数"精确成立;整份 policy 4.780 B = 理解通路 1.409 B + 词嵌入/lm_head 0.537 B + 生成通路 1.409 B + 动作 I/O 头 0.015 B + expert 1.409 B。
- 唯一能验证"两个角色来自同一个模型"的部分:world-model checkpoint 的 3.370 B 就是"减掉 expert 的 policy";worldarena 的 config 与 robotwin 的差异**只有** `action_expert`、`action_loss_reduction`、`data_parallel_replicate_degree`、`sampling` 四类项(`diff` 实测)。训练阶段的"expert 只加 8% 单步时间"无法核验(需要训练代码/机器)。

## 5. 代码审计(四):评测代码与论文数字口径

- **WorldArena**:`benchmarks/worldarena/` 自带 rollout(`rollout.sh`)+ 打补丁的官方 2.0 评估器 + 打分脚本。rollout 口径与论文一致:无文本、段长 ≤15 s、35 采样步、guidance 1.0、seed 0、每段用四条自身生成帧当历史;`scoring/ewmscore.py` 精确实现 Eq.(7)(逐视频截帽 + 15 指标均值)。README 里主动写明:**论文报的 63.31 在发布评估器下是 63.29**(VLM 裁判竞态使 19/1000 视频缺 3 个 VLM 指标),未截帽(榜单原公式)为 63.35。
- **RoboTwin**:`benchmarks/robotwin/evaluate.py` 走官方评测路径(官方种子 100000+,先做 expert 可解性检查,不可解/不稳定/超时的 seed 只记为 skip 不计失败),`tally.py` 只统计每个任务前 100 个"有记录"的 seed;`skip_seeds.txt` 预登记了 69 个在模拟器里挂死的 seed(挂点发生在第一个策略步之前,属环境问题)。README 同时坦白两点:**指令生成器用未播种的 Python 全局 random**(两次评估的指令可能与 seed 集不同),**expert 检查不可复现**(两 checkpoint 各 100 集、共 16 495 个共同 seed 中有 1 095 个(6.6%)判定翻转),所以配对比较只对共同 seed 做。相机帧以红蓝交换顺序送出(training_colours):训练数据就是 BGR,发布策略继承了这一点。
- **LIBERO / LIBERO-plus**:`benchmarks/libero/` 给到逐集计数:LIBERO **1948/2000**(spatial 493/500、object 497/500、goal 481/500、libero_10 477/500),LIBERO-plus **8704/10 030**,并给七个扰动轴的分解(Camera 81.74、Robot 67.55、Language 94.27、Light 97.37、Background 90.33、Noise 95.57、Layout 84.39)。口径细节(10 步 warm-up 计入步数上限、`--send_state`、chunk 32 执行 16、4 UniPC 步、shift 5、guide 1)都写清了。
- **CALVIN**:官方 1000 条 canonical 序列、每子任务 ≤360 步、链在首次失败处停止、`seed_everything(0)`;报 4.64。
- **人形评测没有代码、没有数据、没有视频**:发布仓只覆盖四个 benchmark,真实机器人的 14 个场景配置 ×5 条指令、超时判定、split 定义都只在论文 C.2 里。

## 6. Claim 三角核验

### 6.1 WorldArena 1.0:EWMScore 是什么、63.31 在什么水平

EWMScore 不是榜单官方分数,而是论文按 WorldArena 2.0 的运动帽规则重算的 15 指标均值(§0 有定义)。我把 HF Space `WorldArena/WorldArena` 的 **127 个榜单条目全部下载,按同一规则独立重算**(`code/verify_worldarena_board.py`,纯 stdlib):

| 论文声称 | 我的独立重算 | 结论 |
| --- | --- | --- |
| 112/127 条目在 dynamic degree 上超过真值 | **112/127**(最高 AccioWM 86.14 = 真值 3.7 倍) | 精确命中 |
| 113/127 在 flow score 上超过真值 | **113/127**(最高 SisyphusWorld 99.82 = 8.0 倍) | 精确命中 |
| UNIS 截帽后 73.64 → 63.65 | 榜单均值 73.64 → 63.65 | 精确命中 |
| Motubrain 63.16、FlowWAM-FiveAges 62.99 | 63.16 / 62.99 | 精确命中 |
| JEPA 相似度 98.41,全榜第二,仅次于 BLM 98.49 | 第二,唯一更高的是 BLM 98.49 | 精确命中 |
| "103 个条目在 image quality 上超过真值视频;instruction following / interaction quality 上分别有 41 和 67 个超过" | 用真值行反推(Visual 59.5 = mean(IQ, AQ, JEPA 99.9) ⇒ IQ≈39.3;Physics 79.6 = mean(Interaction, Traj 86.9) ⇒ Interaction≈72.3):image quality 上方 **104** 个、interaction quality 上方 **66** 个,instruction following 第 41 名的阈值是 83.7 —— 与论文的 103/67/41 差 ±1(差在真值行只给了分类均值) | 命中(±1) |
| trajectory 56.0 / depth 98.2 / 3D 93.4,是"所比模型"中最高 | 在论文 Table 11 的选择集内确实最高;但**完整 127 条目榜上有 11 个条目 trajectory 更高**(WorldScape v0.2 69.38、FAW 64.61、EonWorld 62.93、GentleWM 62.19、Z-WM 61.69、BLM 59.79…),**11 个 depth 更高**(EonWorld 99.11…),除 BLM 外全部标注 open_source=no | 说法成立,但"所比"范围是论文自选 |
| —(论文未声称) | GreenWAM 63.31 在截帽榜排 **#35/127**;未截帽 63.35 排 **#75/127**(榜首原始分 73.64+)。六个分类(截帽)里的位置:3D **#23**、Physics **#26**、Content **#27**、Visual **#75**——它的优势集中在几何/物理保真类,画质类本来就落后于真值 59.5(它 57.2),论文自己也承认这两类不看"是否忠于记录" | 报告应写明"顶级"是三指标口径而非合成分口径 |
| —(论文未声称) | 论文表 11 里的 BWM-Fast 63.80、UNIS 63.65、ABot-PhysWorld 63.65 三项都高于 GreenWAM 的 63.31(它们的帽是上界,真实值更低)——所以"ahead of every world–action model … in our comparison"只能读作"在那三个保真指标上" | 判读要点 |

### 6.2 RoboTwin 2.0:67.54 在榜单位置

官方榜单页把条目以内嵌 JSON 发出,我直接解析并按 Average(c2c+c2r) 重排(`code/verify_robotwin_leaderboard.py`):

| # | 方法 | clean | randomized | 平均 |
| --- | --- | --- | --- | --- |
| 1 | PatchWAM(2026-10-02 上榜) | 91.56 | 66.72 | **79.14** |
| 2 | ME-Dex-1.0 | 89.58 | 68.12 | 78.85 |
| 3 | ViGAR(2026-10-11 上榜) | 82.00 | 67.02 | **74.51** |
| 4 | UniWAM(原 OLA-Sem,10-10 改名) | 75.12 | 67.56 | 71.34 |
| — | **GreenWAM(论文值)** | **86.98** | **48.10** | **67.54 → #5/20** |
| 5 | GigaBrain-0.7 | 66.80 | 67.90 | 67.35 |

即:论文"among the top entries"成立(第 5),但**Table 3 的上半块恰好漏掉两个排在它前面的榜单条目 PatchWAM 与 ViGAR**,二者上榜时间(10-02 / 10-11)都早于论文日期(10-13),且 PatchWAM 是"专门挑战 action expert 必要性"的工作——正是 GreenWAM 第三条主张的直接竞争者。表内其余 17 个名字都能和榜单对上(仅 OLA-Sem 已被官方改名 UniWAM)。

### 6.3 "真实人形 95.6% vs π0.5 84.4%"的协议与强度

协议(C.2):一个打包任务(把瓶子和盖子放进袋子再提起来)写成**五条连续指令**;14 个场景配置(9 个 ID + 5 个 OOD),每配置五条指令 ⇒ **每个策略 45 条 ID + 25 条 OOD 指令,每条一次尝试**;四个策略(π0.5 公开 base、Cosmos 3 Edge、GreenWAM、蒸馏学生)用**同一套相机、同一批受控关节、同一演示划分**,但各自保留自己的动作表示与观测预处理,且预测视野不同(π0.5 每次 50 个动作,GreenWAM 16)。

把百分比还原成整数计数(`code/verify_humanoid_claim.py`):GreenWAM 43/45 vs π0.5 38/45(vs Cosmos 3 Edge 40/45);OOD 21/25 vs 19/25。**双比例检验:ID 差值 +11.1 pp,95% CI [−1.1, +23.3],Fisher 精确 p=0.157;OOD +8.0 pp,CI [−14.1, +30.1],p=0.725。** 论文对这一组对比没有给区间(受控的实验对比 Figure 18 给了 bootstrap 95% 区间)。结论:排序在两个 split 上一致、蒸馏学生保持优势(41/45、20/25),但**"超过 π0.5"这句话的统计强度不足**,报告引用时必须带这套协议描述而不是裸数字。

### 6.4 LIBERO-plus 86.8 零样本"最高"

论文 Table 14 的对比集里它最高(CKT-WAM 86.1、π0.5 85.7、OA-WAM 83.9、Cosmos Policy 82.2、Faster-WAM 73.6/Fast-WAM 49.1)。我在文献里检索到的其它零样本数字都低于它:W²-VLA 8424/10030=84.0、World Pilot 84.7、ABot-M0.5 83.4、ImageWAM 83.1(其项目页另有一处 84.7 的说法,均 <86.8)、PatchWAM 零样本 80.2。**唯一更高的公开数字 PatchWAM 91.8% 是在 LIBERO-plus 扰动数据上训练**(非零样本,其零样本是 80.2,论文自己也写明"reflects training on perturbed data rather than zero-shot robustness")。所以"highest zero-shot among the compared methods"是站得住的,但要写成"零样本"限定,并注明它与 CKT-WAM 只差 0.7 pp。

### 6.5 其它

- **RoboDojo**:16.12 分 / 10.49% 对榜第 1(PhysicalRSI 36.27 / 31.38%)、人类遥操作 80.42 / 76.03%——论文没夸大,只是"高于 π0.5/X-VLA/X-WAM"这类低位对比。
- **CALVIN**:4.64 在表 12 里排第三档(低于 MMaDA-VLA 4.78、Xiaomi-Robotics-0 4.75,高于 FLOWER 4.53),论文措辞为"competitive",与数字相符。
- **发布物**:HF `SberRoboticsCenter/GreenWAM` 四目录(robotwin/libero/calvin/worldarena)齐全,含 bf16 两分片 + config.yaml/json + 统计量 + SHA256SUMS,**总 35.45 GB**;四个 benchmark 的权重确实"随论文发布",与 README 表格一致。**训练配方、数据管线、robotics-stage 基座 checkpoint、训练数据不发布**(README 明说),所以只能复现评测、不能复现训练。

## 7. 论文—代码出入清单(报告的"审计"素材)

| # | 出入 | 证据 | 影响 |
| --- | --- | --- | --- |
| 1 | "20 个机器人族" vs 注册表 **21 个族簇** | `family_projectors_v2.json:counts.families=21`;论文 Table 8 本身列 21 行(最后一行 OpenArm <0.01 h) | 计数口径;不影响机制 |
| 2 | Table 2 的"固定混合" vs 代码注释"starter shares, not empirically optimized or compute-matched" | `greenwam_modes.py:14` | 论文把未调优的先验当结论陈述,削弱"混合策略"的实证地位 |
| 3 | Eq.(1) 的"符号 ρ 与零点 o"在发布物里被注册表**明确让给 source codec**;实际只有 4 个通道 offset≠0 | `anatomical_maps_v2.json` limitation 字段;`family_projectors_v2.json` scale/offset | 解剖学对齐的"零点"部分在数据里几乎未被行使;跨本体零点对齐更像承诺而非已验证事实 |
| 4 | "no robot has action parameters of its own" vs 共享投影器里有**每布局 context 嵌入(90 k,`context.weight [44,2048]`)+ 每表示类型嵌入** | `semantic_action_projector.py:174-182, 210-225`;worldarena checkpoint header 实测 | 严格说法是"每本体只有 ~90 k 的偏置参数,没有每本体的投影矩阵";新布局要新开一行 context(扩展禁止复用 context id) |
| 5 | 论文 Figure 9/摘要的"ahead of every world-action model and video generator" vs Table 11 里三个模型 EWMScore 更高 | 我的 127 条目重算 | 只能读作"三个保真指标上";合成分上它是 #35/127(截帽) |
| 6 | Table 3 漏掉两个排在 GreenWAM 之前的榜单条目(PatchWAM、ViGAR) | 官方榜单页内嵌 JSON,上榜日期 10-02 / 10-11 | 最实质的一处选择性引用;报告可用作"排名要看完整榜单"的例子 |
| 7 | 真实人形对比没有区间;45/25 条指令下差异不可分辨 | §6.3 的 Fisher 检验 | 摘要措辞("succeeds more often")强于证据 |
| 8 | WorldArena 报的 63.31 与发布评估器 63.29 不同 | `benchmarks/worldarena/README.md`(论文自己的说明) | 属主动披露,不算隐瞒,但引用时要用 63.29/63.31 的区分 |
| 9 | RoboTwin 策略"预测 16 步"vs 发布说明"预测 17 步、执行前 16 步" | 论文 B.5 vs `benchmarks/robotwin/README.md` | 措辞;由 WAM horizon 1.07 s×16.67 Hz=17 帧决定 |
| 10 | RoboTwin 布局在注册表里是 `adapter_readiness: source_adapter_available_calibration_unverified`、`runtime_calibration_fully_validated: false` | `benchmarks/robotwin/layout_extensions.json` | 论文 A.3 的 FK 校验(5.2/7.1 mm)在发布物里只是 requirement 文本,没有校验代码与结果文件 |
| 11 | 论文称 RoboTwin"复用 Worker 族",代码复用语义布局 42 = `dual_arm_b`(ARX5 common 14) | `layout_extensions.json:semantic_layout_id=42` | 同一族的不同命名(论文 Table 8 把 Worker/ARX5 列为一行),不构成矛盾 |
| 12 | 受控对比 Q1 把"解剖学接口"与"20 族机器人数据"绑在一起测 | §7.1 的配方说明:父模型"必须从零学关节空间投影" | 与我们的判据直接相关:没有一个 arm 能分离"接口归纳偏置"与"机器人专门化数据"的贡献(见 §8) |

## 8. 好奇心清单

能追的(本轮已追,证据在上文):解剖学注册表能不能用代码跑通(能)、action expert 是不是真增量(是,1.409 B 精确等大)、WorldArena 的分数能不能第三方复算(全部命中)、67.54 在榜单的位置(第 5,且表 3 漏人)、人形对比的统计强度(不显著)。

追不动、留给下一轮的(按价值排序):

1. **接口效应与数据效应的分离实验**。Q1 是"接口 + 20 族数据 + 联合训练"的打包对比。要拆开,最小实验是:父模型 + 同一套解剖学接口(不喂机器人语料)vs 父模型,在 WorldArena/RoboTwin 同预算下测。需要训练代码与 64 卡,发布物没有;但可以在小规模替代(parent = 更小的视频模型 + 同接口)上做机制验证——这是我们主线("接口归纳偏置 vs 预训练先验")最直接可造的跟进。
2. **零点的真实作用**。全库只有 4 个非零 zero-offset,而论文把"fixes sign and zero"列为接口三大职责之一。可造问题:如果人为把某族的零点错位(比如把 elbow 的 home 偏移 10°),模型能否靠 scale/offset 拟合回来,还是必须靠额外微调?这决定"解剖学槽位"是硬约束还是软先验。
3. **掩码是输入特征而非只做损失屏蔽**。代码里 `mask(validity) + presence(该槽是否存在)` 作为加性特征进入投影器(小初始化)。值得测:去掉 mask/presence 特征、只保留损失屏蔽,会不会掉点?这正是"硬约束管记账"的边界问题。
4. **人形对比的重做**。45/25 条指令不足以支撑摘要级结论;把 14 个场景扩到 40+、每配置多次尝试,或至少给配对(同指令)统计,才能判断 5 条指令的差距是能力还是噪声。
5. **长程漂移与本征误差**。论文承认长 rollout 从自身生成帧续推、误差累积;WorldArena 只做了 ≤15 s 段。可测:把段长从 15 s 拉到 30 s,trajectory 指标如何衰减(需要 checkpoint + 评测器,约 1 天机时,可做)。
6. **机器人专门化是否可迁移到别的父模型**。论文的 Q1 只做了 Cosmos 3 Edge。把同样的 20 族数据+接口接到 Wan / 其它视频 DiT 上,增益是否同量级?这决定"解剖学动作空间"是一个 Cosmos 特定技巧还是通用中间表示。
7. **"一个模型两角色"的确切含义**。发布的是四个**分别后训练**的 checkpoint,不是一个权重同时干四件事;推理服务也是四个入口。报告里要把"one foundation"写成"一个 robotics-stage 基座 + 四条后训练线",并指出基座本身未发布。
8. **算力账**。论文给了 64×H100、36 500 步、9.34 M 样本,但没有墙钟时间与 GPU-hours;对复现/扩展成本的估算缺一块。

## 9. 报告素材(图,定位符已核过)

图例约定:`doc:f38ca8c` 是本文档 id;导出命令
`mineru read "doc:f38ca8c/tier:basic/page:N/block:M" -f image -o out.png`
(`drafts/images/greenwam-2610.18774/img-NN.png` 的序号是解析完成顺序,不是文档顺序,必须按 locator 反查。)我把精选图导出并**逐张肉眼核对**放进 `runs/2026-10-16-greenwam-onewm-xtrace-cortex/assets/`,已裁掉白边、单文件 33–566 KB。

| 首选 | 定位符 | 资产文件 | 承载的叙事 |
| --- | --- | --- | --- |
| 图 1(海报图) | `doc:f38ca8c/tier:basic/page:2/block:1` | `greenwam-fig1-teaser.png` | 一张图讲完"机器人经验 → 79 槽共享接口 → 一个模型七种任务 → 两种角色 → 六角雷达对比":上半是数据与接口,中间是 robotics-stage 的训练任务清单,右侧是世界预测与策略(生成器 + action expert),右下是六个 benchmark 的雷达。适合做线程开场图 |
| 图 4 上半(codec) | `doc:f38ca8c/tier:basic/page:7/block:1` | `greenwam-fig4-codec.png` | 承重机制特写:Worker/ARX5 的 3 个原生通道 → codec(选取/重排/符号零点) → slot 36 `arms.left.shoulder_roll`、47 `arms.right.elbow`、65 `hands.left.gripper` → 一个共享输入投影 → state tokens。用它解释"同一槽位在任何机器人上意味着同一个关节运动" |
| 图 4 下半(覆盖) | `doc:f38ca8c/tier:basic/page:7/block:2` | `greenwam-fig4-coverage.png` | 8 个布局在 79 槽上的填充矩阵(legs 12 / body 23 / arms 18 / hands 26),灰格=未解通道、浅格=缺席槽;一眼看出"族间只在手臂块重叠,腿/手指只覆盖部分语料",与"缺席坐标永不变成零目标"对应 |
| 图 3(架构) | `doc:f38ca8c/tier:basic/page:5/block:4` | `greenwam-fig3-architecture.png` | 三列结构:冻结 reasoner(指令)/ 训练的 generator(画布 + 视频)/ 训练的 action expert(机器人状态 + 动作),中间一条"共享多模态注意力";底部 `q̂ = q0 + Δq̂` 与状态输出投影回原生关节。用它讲 action expert 与 Eq.(8) 的联合注意力 |
| 备用:图 18 | `doc:f38ca8c/tier:basic/page:21/block:6` | `greenwam-fig18-controlled.png` | 受控对比森林图(Q1 专门化 / Q2 action expert / Q1+Q2 / Q3 世界模型模式),带 50 任务 ×20 试次的 bootstrap 区间;讲"为什么管用" |
| 备用:图 17 | `doc:f38ca8c/tier:basic/page:21/block:2` | `greenwam-fig17-humanoid.png` | 真实人形四个策略的 ID/OOD 成功率条图(95.6/88.9/84.4/91.1;84.0/80.0/76.0/80.0)。配 §6.3 的计数与检验一起用,避免裸引用 |
| 备用:图 9 | `doc:f38ca8c/tier:basic/page:13/block:2` | `greenwam-fig9-worldarena.png` | WorldArena 七个面板(JEPA 98.4、EWMScore 63.3、Trajectory 56.0、Depth 98.2、3D 93.4 等)与图标图例;讲"保真指标顶级"的位置 |
| 备用:图 11(RoboTwin 面板) | `doc:f38ca8c/tier:basic/page:17/block:1` | `greenwam-fig11-robotwin.png` | RoboTwin 柱图(67.5 vs StarVLA 24.8 / Fast-WAM 39.9 / X-VLA 44.5 / π0.5 58.4 / GigaBrain 67.3):五个对照全部低于 GreenWAM,榜单上高于 67.54 的四个条目(PatchWAM 79.1、ME-Dex 78.9、ViGAR 74.5、UniWAM 71.3)一个都没画。如要用,建议并排贴 §6.2 的完整榜单排名,并说明"榜单是滚动更新的" |

其它可能用到的图(未导出,locator 已确认属于对应图):图 8 动作跟随 `page:12/block:6`;图 10 长 rollout `page:14/block:4`、`page:14/block:6`;图 12 RoboTwin 回合 `page:18/block:2`;图 16 人形连续回合 `page:20/block:3`、`block:5`…;图 22 策略想象 `page:24/block:2`、`block:8`、`block:9`;图 24 随机化轴 `page:25/block:1`;图 31 失败案例 `page:43/*`。

## 10. 本线程脚本(全部可重跑,只读外部世界)

| 脚本 | 作用 | 依赖 | 我跑出的结果 |
| --- | --- | --- | --- |
| `code/audit_action_space.py` | 79 槽/43 布局/21 族/1217 root/符号与零点/掩码/codec provenance 审计 + 可选 round-trip | 纯 stdlib;`--torch-python` 指向带 torch 的解释器(我用 `code/onewm-env/bin/python`) | 全部 PASS,round-trip 误差 0.0 |
| `code/audit_checkpoints.py` | 四个 HF checkpoint 的存在性、config 开关、safetensors header 参数统计(不下权重) | 纯 stdlib(HTTP Range) | 全部 PASS:policy 4.780 B = world model 3.370 B + expert 1.409 B |
| `code/verify_worldarena_board.py` | 下载 127 个榜单条目,重算截帽/未截帽 EWMScore 与各项排名 | 纯 stdlib | 全部 PASS(112/113、UNIS 73.64→63.65、JEPA #2/127;63.31 → #35/127) |
| `code/verify_robotwin_leaderboard.py` | 解析官方榜单内嵌 JSON,重排 Average(c2c+c2r) | 纯 stdlib | GreenWAM 67.54 → #5/20;列出被漏掉的两个上游条目 |
| `code/verify_humanoid_claim.py` | 人形协议的整数计数还原 + Fisher 精确检验/差值 CI | 纯 stdlib | ID p=0.157、OOD p=0.725,CI 均跨 0 |

运行示例(工作区根目录):

```bash
cd runs/2026-10-16-greenwam-onewm-xtrace-cortex/greenwam-2610.18774/code
python3 audit_action_space.py --torch-python ../../../../code/onewm-env/bin/python
python3 audit_checkpoints.py
python3 verify_worldarena_board.py
python3 verify_robotwin_leaderboard.py
python3 verify_humanoid_claim.py
```

脚本输出留档在 `code/outputs/*.txt`(五次运行的完整 stdout);两份小体量原始证据在 `code/data/`:`robotwin_leaderboard_entries.json`(2026-10-16 抓取的 25 条榜单条目)与 `worldarena_board_ranking.tsv`(127 条目的截帽/未截帽 EWMScore 与三个保真指标,末行附 GreenWAM)。

## 11. 未竟事项与障碍

- **没有跑发布评测本身**:四个 checkpoint 共 35.45 GB,加上 WorldArena 评测器(SAM3/V-JEPA/Qwen3-VL 等 12 组权重与 3 个 Python 环境)与 RoboTwin/LIBERO/CALVIN 模拟器,单机预算超出本轮;所以所有 benchmark 数字都是**第三方复算 + 榜单核对**,不是端到端重跑。要重跑,最便宜的一路是 LIBERO(发布脚本 + 一个 GPU,约半天)+ WorldArena rollout(需先下 6.74 GB world-model 权重与 12 组评测权重)。
- **真实人形不可复现**:无机器人、无数据、无代码;"95.6% vs 84.4%"只能按 §6.3 的统计口径呈现。
- **训练侧未发布**:模式混合的采样器、max-entropy 联合分布(论文 Eq.13)、captioner 管线、混合损失的具体实现都不在发布物里;第 3 节的模式/损失核验止于"契约与开关存在",不能核验"训练确实这么采样"。
- **FK 校验(5.2/7.1 mm)无证据**:注册表只有 requirement 文本与 `calibration_unverified` 标记。
- **"expert 加 8% 单步时间"无法核验**(需训练)。
- 顺带修复:该文档的 mineru 源 PDF 曾被删除,`mineru read` 报 `No active source file found`;我从 `blobs/f3/8ca8c9…` 恢复了一份到 `drafts/greenwam-2610.18774.pdf`(drafts 为忽略区)。其它线程(cortex/xtrace/onewm/diffusionart/dreamtrue/score-centering/multimodal-flow/cyfa)的源 PDF 同样缺失,如需导出图片需同样从 blobs 恢复(blob 名前缀即 docId 前 7 位)。
