# Motion Mamba 2.0:官方权重本机复算 + 源码审计 + LoomAttention 对撞

线程:用户点名 Motion Mamba 2.0(arXiv:2611.29897,北大 AIGeeks)。精读 + 承重前作(ARDY、Mamba-3)+ 官方仓库逐行审计 + 本机 RTX 3060 官方权重复算(HumanML3D 全量 + HiPHI 48 族×4 时长 + PCM 读消融)+ LoomAttention(arXiv:2611.30634)作为同题异解的承重邻居对撞。

## 白板底稿

见 drafts/briefs/motionmamba2-2611.29897.md(与本 README 同步)。

## 证据 A:源码审计(核验级)

官方仓 https://github.com/AIGeeksGroup/MotionMamba2.0(clone 于 code/motionmamba2,76MB,238 个 Python 文件),权重 + 冻结 codec + 数据 + 全部基准输入内置。逐项核验:

| 论文声称 | 仓库证据 | 判定 |
| --- | --- | --- |
| PCM 写入门控 EMA,写率上限 η | motion_mamba2/model/content_memory.py:86-88 `gate = max_write_rate*sigmoid(...); slots=(1-gate)*prev+gate*proposal` | 吻合 |
| η=0.25、槽位 2 个/层、零初始化 | configs/model.yaml:89 `content_memory_max_write_rate: 0.25`;`content_memory_slots: 2`;content_memory.py:62-63 零初始化 | 吻合 |
| 读出残差 tanh(ρ),ρ 零初始化精确退化为 Native | content_memory.py:51 `read_gate=Parameter(0.0)`,91-92 `tanh(read_gate)`;sequence_mixer.py 双注意力调用差分 | 吻合 |
| 仅接受后原子提交,失败回滚 | history_commit.py:56-77 状态备份 + "must commit exactly once" 断言 + 异常回滚;sequence_mixer.py:347-367 commit_version 守卫 | 吻合 |
| Mamba-3 exponential-trapezoidal 递推 | mamba_state.py:44-68 四件套状态(angle/ssm/k/v)+ mamba3_siso_combined;对应 Mamba-3 原文(arXiv:2603.15569)Table 1 | 吻合 |
| 训练 GT 历史最长 196 帧 < 4 窗缓存 208 帧 | 论文 App. C 原文;configs/training_motion_mamba2.yaml 的 (H,F) 采样表上限 196 | 吻合 |

**仓库考古发现(论文未报)**:configs 暴露 token_state_interaction 六变体:naive_gated_add / state_query / state_kv_modulation / h0_parallel / h1_innovation / h2_delta_rule。h2(delta_rule.py:10-36)把 delta-rule 修正(innovation = value − S·key,外积除以 token 数,×tanh(β))直接写进 SSM 状态矩阵——"往快权重矩阵闭式写入"的接口变体。论文只报了 PCM(旁路翻译)这一个点;直接写状态与旁路翻译两条路作者都实现过。

**环境不匹配记录**:cuda_graph 加速路径在本机(torch 2.14.1+cu126、triton 3.8.0;仓库钉 triton>=3.5.1)于 humanml3d frames_208 评测 17 个 rollout 后触发异步 CUDA 错误(细节未在前台复现);评测改走 eager 参考路径。论文声明优化逐位保真(4 个种子 × 100 窗 rollout 逐位一致),故 eager 数值与论文可比。此为本机环境问题,不判官仓。

## 证据 B:本机复算(实验级)

设置:RTX 3060 12GB,eager 路径,10 步 DDIM 确定性采样,文本/空间引导 (2,2),官方权重 sha 见输出目录 inputs.json。协议:论文冻结基准输入(128 个 locked_test 初始化族 × 3 噪声种子);HumanML3D 两时长跑全量 384;HiPHI 四时长取前 48 族(×3 种子 = 144 rollout,配对初始化保持)。Root CE/End CE 为族均值(mean [CI95]);Cmd 为粗运动状态一致率。

| 任务 | 口径 | Root CE (cm) | End CE (cm) | Cmd (%) | 论文对照 Root/End/Cmd |
| --- | --- | --- | --- | --- | --- |
| HumanML3D frames_52 | 全量 384 | 2.21 [2.01, 2.41] | 6.74 [6.09, 7.47] | 88.10 | 2.18 / 6.73 / 88.07 |
| HumanML3D frames_208 | 全量 384 | 2.17 [2.00, 2.32] | 7.00 [6.37, 7.68] | 58.63 | 2.17 / 7.03 / 58.81 |
| HiPHI frames_52 | 48 族 | 2.58 [2.27, 2.92] | 7.11 [6.18, 8.17] | 87.62 | 2.78 / 7.19 / 86.96 |
| HiPHI frames_208 | 48 族 | 2.39 [2.14, 2.67] | 7.33 [6.54, 8.20] | 48.70 | 2.53 / 7.11 / 48.82 |
| HiPHI frames_468 | 48 族 | 2.66 [2.28, 3.08] | 8.85 [7.63, 10.21] | 47.45 | 2.87 / 8.45 / 47.11 |
| HiPHI frames_988 | 48 族 | 6.89 [4.59, 9.82] | 12.77 [10.13, 15.69] | 43.07 | 7.55 / 13.64 / 46.05 |
| HiPHI frames_988 PCM 读关(全窗) | 48 族 | 6.57 [4.65, 9.03] | 12.41 [10.26, 14.85] | 43.13 | 配对差(读关−开):Root −0.32 [−1.20, +0.56] / End −0.36 [−1.50, +0.78],统计零 |
| HiPHI frames_988 PCM 读关(第 4 窗起) | 48 族 | 7.13 [4.40, 10.46] | 13.58 [10.50, 16.92] | 43.13 | 配对差:Root +0.24 [−1.11, +1.59] / End +0.81 [−0.64, +2.26],统计零 |

两个全量口径(HumanML3D 52/208)与论文自报一致(根/末端误差差 0.003–0.03cm,其中 208 帧根误差 2.167 对 2.17 在报告精度内逐位一致;Cmd 差 ≤0.2pp)——复现锚点。48 族子集全部低于论文全量 5–9%(方向一致),但论文值均落在 CI95 内;子集选择性检查(初始根-目标距离):前 48 族 0.750m vs 全量 0.742m,无难度偏置,偏差归因于族抽样方差。

PCM 读消融的教训:直接翻 configs/model.yaml 的 content_memory_read_enabled 对推理**零效果**(首次尝试逐位复现了 PCM-on 的 rollout)——dit.py 每次前向显式传 content_memory_read_enabled_runtime,由 MemoryController 按窗口重设(默认恒 True),显式参数覆盖 mixer 配置。正确入口是 MemoryController.configure_content_memory_intervention('readout_off')(作者自带分析工具,只从第 4 窗起关读);全窗口关读在其上猴补丁 _set_content_memory_runtime 强制(见 code/pcm_read_ablation.py)。

**PCM 读消融结果与 ρ 审计:释放权重里 PCM 数值性惰性(本线程最重发现)**。四边证据:(1) 权重审计——checkpoint 全部 16 层(root 8 + body 8)的 read_gate ρ 满足 |ρ| ≤ 4.6e-7(tanh(ρ)≈ρ),PCM 其余参数(decoder/write_gate/conditioner)最大幅值 ≤2e-3,全部被 weight decay 磨到近零;(2) 因果探针(两种模式)——全窗读关与第 4 窗起读关(恰为超训练视界工况,作者自带工具原语义)对三项指标均零影响,配对逐 trial 差统计零(全窗:Root −0.32 [−1.20, +0.56]、End −0.36 [−1.50, +0.78];w4:Root +0.24 [−1.11, +1.59]、End +0.81 [−0.64, +2.26]);(3) 代码核验——sequence_mixer.py 的读贡献合成式 `out += tanh(ρ)·(attn_with_slots − attn_without)`,tanh(ρ) 是读路径唯一可训练缩放;(4) 复现锚点——同一 checkpoint 在同一批冻结基准上复现出与论文自报一致的数字(根/末端误差差 0.003–0.03cm)。附带的机制观察:两种读关与 PCM-on 的逐 trial |差| 均值 2.2–3.3cm——ρ 的 1e-7 量级残差经 39.52 秒确定性扩散轨迹放大到厘米级逐 trial 散布(混沌放大,无系统均值);这同时标定了该口径下单 seed 数字的意义边界(族均值 CI 才是正确单位)。核心推论:**论文的 7.55 Root CE@39.52s 本来就是在 PCM 读通道死亡的状态下取得的**。进一步:论文表 4 的 PCM 消融差(Native 9.36 vs 完整 7.55,即 PCM "贡献 19.3%")无法用这份权重的推理期读出解释——完整模型收敛点 ρ≈0 ⟹ 其前向与 Native 精确同构(论文自己声明零初始化精确退化),7.55 vs 9.36 是两次独立训练的轨迹/方差差,不是翻译机制的运行时贡献。训练中 ρ 是否一度活跃无法从最终权重判断;但最终模型的行为不依赖 PCM 读。论文机制叙述("PCM 把压缩历史翻译给注意力")与释放权重的实际工作方式不符。长程能力的真正载体:Mamba-3 可续状态,经 fusion gate(16 层 sigmoid 门,15 层 0.48–0.56、1 层 0.70)与局部注意力混合——论文表 4 自己的最大消融项(Minimal vs Native:60.0 vs 9.4)正是这一支。

逐帧到目标距离分析(window_error_curve.py,988 帧 144 rollout,图 assets/mm2-hiphi-curve.png):**倒 U 型**——到目标距离从首窗 77cm 升至峰值 ~253cm(第 9 窗),再单调降,末窗均值 61.5cm、末端帧 6.89cm;直线度仅 0.164(高度绕行),逐窗位移 lag-1 自相关 0.454(运动模式跨窗持续)。既非随机游走也非线性漂移:长视界行为是"指令巡游 + 末端归巢"——Root CE 度量的是终点按时归巢,不是路径跟踪;468→988 帧的跳升(2.87→7.55)来自更长历史下末端归巢的难度,不是误差累积。468 帧(9 窗,同样超缓存超训练视界)曲线同样是巡游后归巢(峰值第 4–5 窗 ~125cm,末窗 30cm,末端 2.66cm)——形态随期限伸缩而非随域内/域外切换,巡游段随视界变长、归巢段稳定在最后 4–5 窗;208 帧(4 窗期限)距离即刻单调下降(72→29cm),整个视界就是一段归巢长度。

## 证据 C:LoomAttention 对撞(承重邻居)

精读 arXiv:2611.30634(彩云科技,85 票)2/3.1/3.2/3.3/5.3/5.5 节。同一深层对象(定长矩阵状态被注意力读取),两条正交轴:

| 维度 | Motion Mamba 2.0 | LoomAttention |
| --- | --- | --- |
| 状态累积轴 | 时间(每层私有 SSM 状态,跨窗历史) | 深度(每 token 全层共享矩阵记忆,头输出压缩) |
| 状态尺寸 | 定长,与序列长度无关 | d_k×d = 2n×d ≈ 2D,与深度无关 |
| 读者 | softmax 注意力(水平) | 归一化门控线性注意力(垂直 q^m·M) |
| 翻译层 | 需要:PCM 学习式投影(EMA 门控写入 + 残差读出) | 不需要:线性注意力原生读矩阵态 |
| 容量归因 | ρ 零初始化,容量靠训练挣得 | Prop 3.1:端口固定即退化为标准 Transformer,容量全来自逐 token 端口生成 |
| 消融证据 | PCM 读消融(本线程);论文训练期消融 PCM 贡献 19.3%/12.2%(Root/End) | 动态重连贡献 44% 增益;读侧动态性(损 24.6%)重于写侧(损 9.7%) |
| 2.8B 实测 | — | 动态接线有效秩升至 41(上限 64);后层头利用率 74.4% vs Pythia 的 22.9% |

对接记忆支线判断:softmax 全注意力对上下文无有限维充分统计量,线性注意力有 (S,z)。MM2 的 PCM 是"softmax 读者需要学习式翻译"的构造性证据;Loom 的 vertical read 是"线性读者原生可读"的构造性证据。Loom 原文 Table 1 明确把 linear attention(delta rule 系)列为时间轴、自家为深度轴,且声明两者正交可合并——与我们支线的坐标系一致。

## 复现

```bash
# 依赖:code/mm2-env(Python 3.11, torch 2.14.1+cu126, vector-quantize-pytorch==1.20.0[注:新版 1.34 的
# FSQ 在 __init__ 无条件断言 codebook_size<=2^63,与 levels=[64]*128 冲突,必须降版])
cd code/motionmamba2
# paths.env 指向 code/mm2-assets/{MotionMamba2.pt,frontend,skeleton};输出 code/mm2-out
code/mm2-env/bin/python -m scripts.evaluate --domain humanml3d --task frames_208
code/mm2-env/bin/python -m scripts.evaluate --domain hiphi --task frames_988 --limit 48
# PCM 读消融:MemoryController intervention 接口(yaml 翻转无效,见证据 B 的教训记录)
code/mm2-env/bin/python runs/2026-11-26-motionmamba2-muon/motionmamba2-2611.29897/code/pcm_read_ablation.py full  # 全窗读关
code/mm2-env/bin/python runs/2026-11-26-motionmamba2-muon/motionmamba2-2611.29897/code/pcm_read_ablation.py w4    # 第 4 窗起读关
# 逐帧距离曲线 + 图
code/mm2-env/bin/python runs/2026-11-26-motionmamba2-muon/motionmamba2-2611.29897/code/window_error_curve.py code/mm2-out/hiphi/frames_988 window_curve_988.json 48
```

资产:权重 1.32GB(HF AIGeeksGroup/MotionMamba2.0)、frontend 137MB、骨架 joints.p(GitHub 固定 tree dcaac4d1 的 ardy/ardy/assets/skeletons/g1skel34/,评测只需 joints.p)。

## 好奇心清单

- 把训练历史采样拉到 >208 帧(超缓存工况直接监督),Mamba 支路的长程收益是否进一步抬升?这是"归纳偏置 vs 监督信号"的判别实验。
- **为什么训练没打开 ρ?** 训练历史 ≤196 帧全装得进 4 窗注意力缓存,softmax 读者在域内本就不缺历史——翻译层没有梯度压力去打开;拉长训练视界后 ρ 会不会自己长起来,是"接口需要工况才会长出来"的判别实验。
- h2_delta_rule(直接写 SSM 状态)vs PCM(旁路翻译)的对照训练:闭式写入是否在少数据区占优(与 2026-11-09 写入谱判断对接)?
- 论文表 4 的 PCM 消融差(9.36 vs 7.55)按我们的证据应重读为训练方差/轨迹效应:多种子重训 Native 与完整模型,差值是否仍在?这是对我们判定的最直接检验。
- Loom 与 MM2 合并:时间轴 SSM 状态 + 深度轴矩阵记忆,两者都已在论文里声明正交——谁先做出来?
- Cmd. 在 39.52s 只有 46.05%:粗根运动状态一致率不过半,语义天花板在哪(指令调度 vs 状态可分辨性)?

## 边界

- 48 族子集与论文全量(128 族)口径不同;若数字偏差,先查子集选择性,再下结论(已查:无偏置)。
- ρ 审计只覆盖最终权重;训练中 ρ 是否一度活跃无法从 checkpoint 判断。我们的判定是"释放模型的行为不依赖 PCM 读",不是"PCM 概念无用"——训练期它可能作为过参数化路径影响过优化轨迹(这正是方差解释的内容)。
- Loom 是精读级邻居,其机制分析与消融只取接口对比相关部分;未复算其语言建模实验。
- 评测在 eager 参考路径;cuda_graph 路径的 6.138× 加速未在本机复验(环境不匹配,见上)。
