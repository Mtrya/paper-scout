# 动作表示对照实验:相机系 vs 基座系 + 外参误差时间结构

Blob world 上的玩具级、机制干净实验,检验 CamVLA(2607.05396)的两个核心论断:
① 基座系动作表示把 hand-eye 变换隐式背进权重("geometric entanglement"),视角一动就崩;相机系动作 + 显式旋转合成则稳健;
② 其附录噪声鲁棒性实验每步独立重采样误差,而真实失准(相机被碰、漂移)是时间相关的——本实验检验三种误差时间结构(每步独立 / 整集静态 / AR(1) 慢相关)对闭环执行的影响。

## 环境

```
uv venv code/camcalib-exp/.venv --python 3.11
uv pip install --python code/camcalib-exp/.venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python code/camcalib-exp/.venv/bin/python numpy matplotlib
```

## 跑法

```
cd code/camcalib-exp/action_repr
../.venv/bin/python run_all.py            # 完整运行(约 30-45 分钟,CPU)
../.venv/bin/python run_all.py --resume   # 复用已存在的 checkpoint,只重跑评测/出图
```

产出:
- `figures/fig1_test1_singleview_yawsweep.png` — 测试1:单视角训练,视角偏移闭环评测
- `figures/fig2_test2_multiview_yawsweep.png` — 测试2:多视角训练对比
- `figures/fig3_test3_errorstruct_K1.png` / `K5` — 测试3:误差时间结构(K=1 每步重规划 / K=5 开环段)
- `figures/fig4_sample_views.png` — 示例渲染(绿=EE,红=目标,灰=固定干扰斑)
- `results.json` — 全部数值(成功率、终距,均值±标准误)
- `run_log.txt` — 运行日志

## 文件

| 文件 | 职责 |
|---|---|
| `blob_world.py` | 世界动力学、针孔相机 rig(yaw 绕基座 z 旋转)、渲染、oracle 演示生成 |
| `models.py` | 小 CNN(3 conv + MLP)→ 3 维 delta 动作 |
| `train.py` | 演示数据集构建(渲染缓存为 uint8)+ 行为克隆(两个头共用同一数据集) |
| `evaluate.py` | 闭环 rollout、视角扫描、误差时间结构注入(iid/static/AR(1),同边缘分布) |
| `run_all.py` | 编排:训练 4 个模型(T1/T2 × Base/Cam)、三组测试、出图、结果 JSON |

## 实验设计要点

- **任务**:基座系 3D 空间随机 EE/目标点,oracle 比例控制器(基座系朝目标走,步长裁剪到 0.08,加小噪声)生成演示;≤50 步,成功=终距 < 0.08。闭环评测时策略每一步用自身输出执行(裁剪 + 小执行噪声);EE 位置被限制在工作空间内(模拟关节限位),策略方向性错误会停靠在边界而非逃逸。
- **相机**:针孔,内参已知;整个相机 rig(位置+姿态)绕基座 z 轴旋转构成不同视角(rig 模型保证相机始终看向原点)。64×64 RGB,高斯 splat:EE 绿斑、目标红斑、3 个固定灰斑干扰。斑表观大小 ∝ 1/深度,深度可弱观测(这是单目逆投影误差的主来源)。
- **两个输出头**(相同 CNN 架构、相同数据集、相同种子,只有输出坐标系不同):
  - Base:直接预测基座系 delta,执行时直接用;
  - Cam:预测相机系 delta,执行前用 R_bc(yaw_est) 合成回基座系(Δp_b = R Δp_c)。
- **测试1**:仅 yaw=0 训练,在 yaw∈[-45°,45°] 每 5° 闭环评测(≥50 集/点,报均值±标准误)。
- **测试2**:yaw∈{-30°,0°,30°} 稀疏训练,同一网格评测(插值/外推)。
- **测试3**(核心):Cam 头在训练视角评测时注入旋转误差,三种形态边缘分布严格同为 N(0,σ),σ∈{0..20}° 扫描:
  - iid:每步独立重采样(CamVLA 附录式);
  - static:整集固定(失准/被碰);
  - AR(1):ε_{t+1}=ρε_t+η,ρ∈{0.5, 0.9}(滤波/漂移)。
  另测重规划间隔 K=1(每步重规划)与 K=5(5 步开环段,模拟低帧率 VLA)。T1-Cam 与 T2-Cam 两套都跑。
