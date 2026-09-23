# toy-parts — PARTS 算法的玩具复现与 verifier 噪声扩展实验

对 arXiv:2609.21788(PARTS, Policy Adaptation with RL on Targeted Subtasks)的
纯 CPU 玩具复现:2D 点质量末端执行器执行三段任务链,冻结 BC base + 瓶颈段
残差 RL。外加论文未做的研究问题:**success verifier 标签噪声对 PARTS 的影响**。

## 环境(env.py)

- 平面语义:x = 桌面横向,y = 竖直。30Hz 离散,EE 速度控制(≤0.5 m/s),
  夹爪一维连续开合(速率 3.0/s)。观测 = [ee(2), ee_vel(2), obj(2), obj_vel(2),
  grip(1), slot(2)] = 11 维,加 3 维阶段 one-hot 作为策略输入(共 14 维)。
  **EE/物块位置观测带 σ=2mm 高斯噪声**(视觉伺服噪声;判据 φ 用真实状态)。
- 任务链(事件驱动推进,每段 200 步上限,episodes 600 步上限):
  - σ1 抓取:夹爪在物块 2.5cm 内闭合 → 抓住(物块刚性跟随 EE)。
  - σ2 绕障搬运:墙段 x=0.35, y∈[0,0.35] 挡住直线;须从墙顶(y>0.35)绕过,
    把物块运到 staging 点 (0.70, 0.30) 半径 5cm 内。
  - σ3 精密插入(瓶颈):槽中心 (0.70, 0.10),槽宽 = 块宽+6mm(单侧 3mm 间隙)。
    释放时 |横向误差| ≤ 3mm 且 |竖向误差| ≤ 8mm 才算成功。提前释放/超时 = 失败。
- episode 成功 R = φ1·φ2·φ3(稀疏二值,只在 episode 末端)。

## base 策略(base.py / expert.py)

2 层 MLP(隐藏 128),输入 14 维,输出 10 步动作 chunk(30 维);每 5 步重规划,
预测 10 步执行前 5 步(π0.5 式 chunking,E≤C)。BC 训练:
σ1/σ2 各 200 条脚本专家演示(乘性噪声 15%),σ3 仅 8 条(噪声 20% +
**槽位标定偏差**:演示者以为槽在 slot+5mm 处,偏差逐条 N(5mm, 2.5mm))。
按阶段等权重加权,消除段长差异造成的样本淹没。
共享 base 只训练一次(seed 0)并缓存于 `cache/base_shared.pt`,所有方法臂与
RL 种子共用;测得 σ1=100%, σ2=100%, **σ3≈27%**(500 episodes)。

## 方法臂(run.py)

| 臂 | 残差激活 | 奖励 | 其它 |
|---|---|---|---|
| A base-only | 无 | — | 纯评估 |
| B 全任务 RL | 所有段,全维 | episode 末端稀疏 R(人工标注,无噪声) | 对应 DSRL/EXPO-FT 类 |
| C PARTS | 仅 σ3(规则 selector:φ2 满足即激活) | 局部 r3=φ3(带 verifier 噪声 ε) | 每 15k 步 success-reweighted 重训,ρ=0.25 |
| D PARTS 消融 | 同 C | 同 C | 去掉周期性重训 |

残差结构(论文式 2):a = clip(nominal + B⊙M⊙u),M=[1,1,1],
B=[0.2, 0.2, 1.0](速度维 ±0.2 = 0.1m/s,夹爪维可翻转)。残差 actor 末层零初始化
(未训练 = 零残差),训练时探索 = chunk 级共享高斯偏移 δ~N(0,σ),σ: 0.05→0.03
线性衰减 + 0.02 逐行抖动;评估时 σ=0。

训练预算:60k 环境步/run(restaging 同论文,不计环境步);3 个 RL 种子;
每 10k 步快照评估 100 episodes(greedy)。

**PARTS 训练循环(论文 III-C/D 的玩具对应)**:σ3 失败后,若 verifier(带噪)判失败
且 episode 步数有余,restage 到本 episode σ3 入口附近(±8mm 窄分布)重试——
对应论文的 restage 机制;判成功则交接结束 episode。

## RL 算法(rl.py)— 与论文的偏差说明

论文用 chunk 级 TD3+BC(式 3-4);此处按任务约定用**尝试级 REINFORCE-with-baseline**
替代 critic 项(λ_Q),保留论文式 4 的另外两个结构成分:成功 BC(λ+=0.3,
向成功尝试执行过的残差克隆)、失败锚定(λ−=0.05,失败尝试状态上的残差向零拉),
以及 reference dropout(25% 概率置零 nominal chunk)。两个有意的工程选择:

1. **正优势截断**:REINFORCE 项只用 max(A,0)。毫米级精度下尝试结局主要由探索
   噪声运气决定,从失败尝试(U≈好纠正+倒霉噪声)上排斥会推毁已学到的好纠正——
   实测对称优势会把 93% 的策略打到 0%。负样本信号由 λ− 锚定项承担。
2. **重训步数 30 步**(fresh actor,Adam lr=3e-4):重训过长(≥150 步)会在小的
   成功加权数据集上过拟合探索噪声并摧毁策略(实测 150 步 → 0%)。论文中由
   "operator 选择 checkpoint"承担这一稳定性职能。

重训 = 论文 III-D:D̃ = D⁺ ∪ Sample_ρ(D⁻)(attempt 级,ρ=0.25),全新 actor 在
D̃ 上训练后部署,curated buffer 继续 online。

## verifier 噪声(扩展实验,超出论文)

PARTS 的局部奖励来自 success verifier(论文中耳塞任务的 VLM verifier 不准、需人检)。
实验:σ3 的局部奖励以概率 ε ∈ {0, 0.05, 0.15, 0.3} 翻转(成功↔失败),
外加规格外扩展点 ε=0.5(标签零信息量,锚定曲线终点);噪声标签
用于学习(优势、BC+/锚定分桶、重训 curation)与 restage/交接决策;评估始终用真值。
B 臂的终端奖励按论文设定为人工标注,不受 ε 污染。

## 复现

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python numpy matplotlib torch
# 单次:python run.py --method C --seed 0 --eps 0.0 --budget 60000
# 冒烟:python run.py --method C --seed 0 --smoke
python run_all.py --out results --workers 6   # 全部 24 个 run(约 6 分钟,12 核)
python plot.py --results results              # fig_training.png / fig_noise.png
python summarize.py --results results         # summary.md
```

种子:base 训练 seed=0(共享,缓存);RL 种子 0/1/2(控制环境采样、探索与评估)。
全部 CPU,单 run 约 15–30s,无需缩小规模。

## 文件

- `env.py` 环境与阶段判据;`expert.py` 脚本专家与演示生成;`base.py` BC base;
- `rl.py` 残差 actor + 学习器(online 更新 + success-reweighted 重训);
- `run.py` 单 run;`run_all.py` 网格编排;`plot.py` / `summarize.py` 产物;
- `results/` 全部 JSON、图与 `summary.md`。
