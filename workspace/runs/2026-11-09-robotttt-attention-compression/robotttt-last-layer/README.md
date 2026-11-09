# RoboTTT 重访：把内环限制在 fast MLP 的末层

## 研究问题与结论

上一轮 RoboTTT 玩具探针发现：全参数 TTT 在 2×训练 horizon 上逐步退化，gated delta 状态更稳。TTTLA 给出了一个具体替代方案：只让 fast network 的线性输出矩阵在序列内更新，隐藏核函数保持静态；去掉 weight norm 后可用并行 prefix scan 实现。

本轮在同一任务上问：只更新末层能否保留上下文学习、改善长 horizon 表现并降低内环成本？配对结果给出带冷启动代价的肯定回答。末层版本在前 32 步显著较弱，在第 32–64 步追上，在 896–1024 步以 R² 0.860 对 0.726 明显优于全参数版本。它的训练耗时减少 36.3%，顺序推理延迟减少 16.8%。

结果同时否定了“末层通过减小参数漂移获得稳定性”的简单解释。末层第 0 层 W2 的相对漂移在 1024 步达到 2.923，高于全参数整体状态的 0.773，动作精度却更好。末层静态特征更新的完整瞬时雅可比谱范数在所有受测位置均为 1；局部更新增益、状态幅度与任务函数精度应分别分析。

## 配对设计

- 沿用旧合成任务：每个 episode 有未知目标与控制器增益，多阶段重复到达；输入 `[o_t,a_{t-1}]`，预测非线性专家动作。每 48 步重新放置点质量，目标与增益不变。
- 两臂各有 50,372 参数，共享架构、逐参数初始化、训练数据 seed、outer optimizer 与留出 episodes。初始化最大逐参数差为 0。
- `ttt`：序列内更新 fast MLP 的 `W1,b1,W2,b2`。
- `ttt_last`：序列内只更新线性输出矩阵 `W2`；`W1,b1,b2` 是序列内静态、外层可训练的 slow parameters，输出 bias 不作 fast update。无 weight norm。
- 配置：seed=0，T=512，2500 iterations，batch=32，TBPTT segment=64，d=64，两层，outer AdamW lr=3e-4。
- 主要读出：训练曲线、inner reconstruction loss、可更新状态的相对漂移；相同 64 个留出 episodes 上 T512/T1024 的非饱和动作 R²。T512 是同一批 T1024 轨迹的前缀。
- R² 使用 `1-Σ(a_hat-a)²/Σa²`，只统计 `max|a|<0.95` 的非饱和位置，并略去前 8 步。1 为完美预测，0 与全零预测同误差。
- 成本：训练 wall-clock；d=256、T=2048、batch=1、单线程、3 次重复的顺序推理 per-step latency，逐位置取三次最小值再按窗口平均。
- 资格检查：`ttt_last` 的 `W1,b1,b2` 在完整序列内变化严格为 0；这些静态参数在首次 outer backward 中均有非零梯度。`eta_h` 是末层版本不使用的参数，梯度为 None。

这是 RoboTTT 机制玩具，不是真实机器人复现；RoboTTT 官方实现和权重仍未发布。`ttt_last` 借鉴 TTTLA 的末层语义，但使用本线程的顺序 PyTorch 内环，不是 LaCT 官方 chunk-parallel 实现。末层臂的 slow parameters 在每个 TBPTT 分段重新绑定以保留外层梯度，全参数臂沿用 fast-state detach；训练曲线差异因而同时包含内环参数集和外层梯度路径。

## 训练与留出评测

| 指标 | `ttt` 全参数 | `ttt_last` 末层 |
|---|---:|---:|
| 最后 10 次训练日志平均动作 MSE | **0.004751** | 0.005404 |
| R² 8–16 | **0.864** | 0.508 |
| R² 16–32 | **0.066** | −2.830 |
| R² 32–64 | **0.896** | 0.886 |
| R² 384–512 | 0.891 | **0.906** |
| R² 896–1024 | 0.726 | **0.860** |
| 综合 R² T512 | 0.884 | **0.891** |
| 综合 R² T1024 | 0.841 | **0.882** |
| 训练耗时（秒） | 5,339.05 | **3,401.62** |
| 1536–2048 步顺序延迟（ms/step） | 1.166 | **0.969** |

末层第 0 层 inner MSE 在第 8 步为 3.384，全参数为 0.355；第 64 步为 0.687 对 0.110，解释了冷启动缺口。第 1024 步则变为 0.027 对 0.114，末层在训练窗之后继续改善重构。第 1 层同期为 0.008171 对 0.002950，全参数在深层保持更低重构误差；报告没有把单层趋势泛化成所有层都占优。

可更新状态相对漂移在第 1024 步为：

| 层 | `ttt` 全参数 | `ttt_last` 末层 |
|---|---:|---:|
| 第 0 层 | 0.773 | 2.923 |
| 第 1 层 | 0.742 | 0.660 |

两臂漂移各按实际更新参数的初值范数归一化。末层第 0 层的大幅漂移与稳定精度并存，说明该范数不能独立充当稳定性指标。

## 谱诊断与资格检查

对输出通道 `j`，静态特征上的差分更新雅可比块为 `J_j=I-(2η_j/d)φ^Tφ`。沿 `φ^T` 的径向特征值为 `r_j=1-(2η_j/d)||φ||²`，正交方向为 1；隐藏宽度大于 1 时，完整雅可比谱范数为 `max(1,max_j |r_j|)`。

`spectral_diag.py` 以 seed=999、batch=16 另行生成同分布留出轨迹，逐位置计算该值，同时记录非正径向比例、放大通道比例和完整雅可比非收缩比例。所有时间桶的完整谱范数均为 1，非正径向与放大通道比例均为 0。第 0、1 层全轨迹最小径向特征值分别为 0.816402 与 0.877620。该结果是当前状态邻域的局部证据，闭环全局稳定还由逐层、逐步映射共同决定。

`jacobian_qualification.py` 用 `d=4, hidden=3` 的实际 `_inner_update` 逐坐标构造完整 autograd Jacobian。最大奇异值为 1.0000000000000002，闭式结果为 1.0，绝对误差 2.22e-16。

## 保留证据

- `pair_T512_2500.json`：完整训练历史、资格检查、留出 R²、逐位置 inner loss 与各状态张量漂移。
- `latency_d256.json`：三次重复后的逐位置与逐窗口延迟。
- `spectral_ttt_last_T1024.json`：逐位置径向特征值、精确完整谱范数和聚合桶。
- `../assets/robotttt-paired-results.png`：训练曲线、冷启动、长 horizon、inner loss、漂移与径向谱的六联图。
- `train.py`、`models.py`、`task.py`、`probe_eval.py`、`latency.py`：配对训练与评测实现。
- `spectral_diag.py`、`jacobian_qualification.py`：谱诊断及其独立自动微分资格检查。
- `run_remote.sh`、`run_spectral_when_ready.sh`、`plot_pair_results.py`：远端执行与绘图。

checkpoint 不进运行包，实验结束后也不下载到本机。

## 重跑

远端环境：Workspace `可上网GPU资源`，Project `前沿课题探索`，Compute Group `4090-2-cuda13.2`，quota `1,10,100`，镜像 `ngc-pytorch:25.02-cuda12.8.0-py3`，torch 2.7.0a0。训练使用 CPU 8 线程，延迟使用单线程。远端根目录为 `/inspire/qb-ilm/project/exploration-topic/ky26021/attention-1109`，下文用 `$ROOT` 指代。

```bash
# 配对训练与顺序延迟
bash $ROOT/code/robotttt-1109/run_remote.sh $ROOT

# 等训练与延迟完成后执行谱诊断
bash $ROOT/code/robotttt-1109/run_spectral_when_ready.sh $ROOT

# 谱公式的完整 autograd 资格检查
cd $ROOT/code/robotttt-1109
$ROOT/.venv/bin/python jacobian_qualification.py

# 本地用已晋升的小型 JSON 重绘
uv run --with matplotlib --with numpy python \
  runs/2026-11-09-robotttt-attention-compression/robotttt-last-layer/code/plot_pair_results.py \
  --thread runs/2026-11-09-robotttt-attention-compression/robotttt-last-layer \
  --assets runs/2026-11-09-robotttt-attention-compression/assets
```
