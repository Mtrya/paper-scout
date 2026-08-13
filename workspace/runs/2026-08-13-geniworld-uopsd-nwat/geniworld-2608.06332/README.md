# GeniWorld 线程 — 世界模型与动作的接口:第四种形态

- 论文: GeniWorld: A Generalizable Interactive World Model for Robotic Manipulation via Visual Actions (arXiv:2608.06332, Tsinghua SIGS + Tencent Robotics X)
- 项目页: https://chenghaogu.github.io/GeniWorld/ — 代码 "Coming soon"(2026-08-13 核验)
- 论文全文: papers/world-models/geniworld-2608.06332.md

## 线程问题

世界模型与动作之间的接口是我们 07-31 以来的主线。已有形态:符号化指令(ABot-World)、
物理 token(PhiZero)、蒸馏动力学轨迹(ShadowDancer)。GeniWorld 提出第四种:**把数值动作
经 URDF 渲染成"视觉动作"(只含机器人本体、不含场景),与视频潜变量通道拼接**注入
Wan2.2-TI2V-5B 流匹配主干。论文声称由此获得:(1) 空间对齐的动作控制;(2) 本体运动学
与场景动力学的解耦 → 仅用固定场景演示即可零样本泛化到随机场景;(3) few-step 采样鲁棒
(50→5 步,FVD 仅退化 ~2%,数值动作退化 ~22%);(4) 8Hz 交互闭环。

## 研究动作

### 1. 代码三角验证(Ctrl-World,官方代码开源)

`code/ctrl-world`(ICLR 2026,Ctrl-World 官方仓库)核验数值动作基线设计:
- `models/ctrl_world.py:71-107` `Action_encoder2`:(B,T,7) 数值动作 → 三层 MLP → 1024 维 →
  作为 `encoder_hidden_states` 帧级注入 SVD UNet 的 cross-attention。
- 当前帧通道堆叠作图像条件;EDM 式 lognormal sigma 采样 + x0 预测 MSE(未来帧);DROID 训练。
- 这是 GeniWorld 消融表里 "ours w/ numerical actions" 所模拟的范式。

接口谱系定位(GeniWorld 论文 related work + 各自代码/论文核验):
- Ctrl-World:数值动作 + MLP + cross-attn(无像素级对齐)
- EnerVerse-AC:投影末端位姿(稀疏显式像素对齐条件)
- BridgeV2W:URDF 渲染 embodiment mask + ControlNet 式支路(掩码对齐)
- GeniWorld:稠密运动渲染 + 通道拼接(运动对齐)

GeniWorld 自己的消融已覆盖 numeric/EE/skeleton/ControlNet-style 四档,但**没有做过
"静态掩码 vs 运动渲染"的对照**——即空间对齐(BridgeV2W 路线)单独贡献多少、运动信号
本身贡献多少,文献里无人隔离过。这是本线程玩具实验的判别性问题。

### 2. 玩具实验(远端 4090,真机制、受控合成数据)

`code/geniworld-probe/`(gen_arm_data.py 生成 3200 段合成双连杆臂搬运视频:
16 帧 × 64×64,三种训练背景 + 两种 OOD 背景,附末端轨迹/关节角/逐帧渲染/静态渲染/
打乱渲染五类动作表示,带脚本化真值立方体位置)。

`toy_wam.py`:12M 参数 3D-UNet 流匹配模型,同一主干的四条件接口消融:
- `numeric`:末端轨迹 (T,3) → MLP → 帧级 cross-attn(Ctrl-World 范式)
- `concat_static`:平均姿态渲染重复 T 帧(只有空间对齐,无运动)
- `concat_shuffle`:逐帧渲染但帧序打乱(边缘统计不变,时间运动摧毁)
- `concat_motion`:逐帧运动渲染(GeniWorld 范式)

问题:
- Q1 论文主张:运动渲染是否比数值动作收敛更快、预测更好?
- Q2 论文盲区:static vs shuffle vs motion——增益来自空间对齐还是运动信号?
- Q3 OOD:渲染条件在未见背景上是否泛化更好?
- Q4 few-step:渲染条件在 10/5 步欧拉采样下退化是否更小?

结果:`code/` 下 results_v1.json / analysis.json / plots/(convergence.png, ood_fewstep.png, cube_err.png)。
关键数字(50 步采样,像素 MSE;64×64 玩具):

| 条件 | 域内 val | OOD stripes | OOD dots | 退化比 ood2/val | 立方体误差(域内) |
|---|---|---|---|---|---|
| numeric | 0.095 | 0.263 | 0.392 | 4.1× | 2.9px |
| static | 0.082 | 0.344 | 0.540 | 6.6× | 1.7px |
| shuffle | 0.079 | 0.363 | 0.563 | 7.2× | 1.0px |
| motion | 0.080 | 0.361 | 0.558 | 7.0× | 1.0px |

few-step:四条件 5/50 步 MSE 比均 ≈0.85(空结果)。

## 结论(已写入报告)

1. **域内质量与收敛:论文方向成立**——渲染 0.078–0.082 vs 数值 0.095;运动渲染
   500 步 0.076 vs 数值 0.097;立方体误差 1.0–1.7 vs 2.9px(空间对齐是真的)。
2. **论文盲区:对齐 ≫ 运动**——static ≈ shuffle ≈ motion 在所有指标上打平,唯一
   运动胜出是域内立方体误差(1.0 vs static 1.7)。掩码路线拿走大部分收益。
3. **OOD:论文主张在玩具尺度反向**——渲染条件相对退化 6.6–7.2× vs 数值 4.1×。
   "解耦→OOD 泛化"不是接口的归纳偏置,是预训练先验的功劳。机制解释:通道拼接
   教会模型的是"把渲染机械臂合成进背景"的函数,合成函数本身过拟合训练背景。
4. **few-step:空结果**——论文的 22% vs 2% FVD 退化是大模型+特定采样器现象,
   玩具尺度不出现。

## 初步解读(结果回传后补全)
(已并入结论;见 README 更新)

## 局限与不确定

- 玩具模型从头训练,隔离的是接口的归纳偏置,不含预训练视频先验与真实物理的交互;
  结论以受控机制证据的强度进入报告。
- GeniWorld 无代码/权重:无法直接复现 RoboTwin/真机数字;其"仅固定场景训练"的
  OOD 结论只能以玩具实验 + 消融表解读支撑。
- Dyna-2(2026-08-10 新闻发布,1M 小时人类视频 WAM,宣称首个纯人类数据 scaling law)
  技术报告尚未放出——本线程是其在接口问题上的学术侧镜像,报告纳入背景。
