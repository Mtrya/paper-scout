# CMD 线程:教师信息集错配的玩具定量(2608.13391)

## 研究问题

Context-Matched Distillation(CMD,NVIDIA,2608.13391)的核心声明:AR 视频蒸馏里,双向教师给目标帧的 score 依赖未来帧与未来控制,与因果学生的部署信息集错配;修复=因果教师 + Prefix Scoring(用学生生成的前缀)+ Prefix Corruption。CMD 未放代码("Code soon")。本线程在可解析的线性高斯世界里把这个声明变成三个可测的量,独立验证其机制方向。

## 玩具设定

世界:`x_t = A x_{t-1} + B c_t + ε_t`,ε~N(0,σ²I),d=8,T=32。控制 `c_t = sign(0.7·c_{t-1} + w_t)`(相机轨迹式平滑控制,w_t 是"未来控制中不可预测的增量")。

- **教师条件均值**(DMD 监督在低噪声极限的形式):
  - 因果教师:`μ_f = A x_{t-1} + B c_t`(只依赖前缀)
  - 双向教师:`μ_b = (Q⁻¹ + AᵀQ⁻¹A)⁻¹[Q⁻¹(A x_{t-1} + B c_t) + AᵀQ⁻¹(x_{t+1} − B c_{t+1})]`(固定区间平滑器,显式含未来控制 c_{t+1})
- **学生**:一步线性因果映射 `x̂_t = W x̂_{t-1} + U c_t`,部署永远只用前缀。蒸馏=迭代自回归 rollout + 对教师条件均值的最小二乘拟合(学生结构允许的最优解)。
- **Prefix 变体**:base-style(拟合样本前缀=真值序列)vs prefix-style(前缀=学生自己的 rollout;即 CMD 的 Prefix Scoring)。
- **部署**:用学到的 (W,U) 因果在线 rollout,只观察 c_1..c_t,测量一步 MSE(噪声底 σ²=0.09)。

## 结果(cmd_probe_fig.png,cmd_probe_data.json)

1. **acausal 梯度比** `||∂μ_b/∂c_{t+1}|| / ||∂μ_b/∂c_t||`:ar=0.2 时 0.084,ar=0.95 时 0.379——双向教师监督中未来控制的权重随动力学记忆单调增长,最高达当前控制的 38%。因果教师此量为 0(恒等)。
2. **错配的代价是训练瞬态,不是稳态偏差**(本轮最重要发现,cmd_iter_budget.py):W=A 是双向监督的**良性不动点**——当学生完全复刻真值动态时,x̂_{t+1}=A x̂_t+B c_{t+1} 代入平滑均值,未来项恰好与当前项合并,μ_b 坍缩成滤波均值 A x̂_t+B c_t,未来不再携带新信息。因此充分训练下两类学生都收敛到真值。但**相同训练预算下**,双向学生被 acausal 监督拖慢:ar=0.9 时因果学生 16 轮即到 A(||W−A||=0),双向学生 32 轮才到(0.019);部署误差差在 iter=4/8 时最大(0.34/0.42,噪声底 0.09 的 4 倍以上),随预算增大消失。解析推导(cmd_analytic_fold.py,稳态线性化)与样本投影解一致(|W*−Wsp|=0.006),确认机制;迭代蒸馏 8 轮停在收敛途中,正是真实 DMD 训练预算有限的对应物。
3. **前缀上下文错配**:base-style(教师见真值前缀)vs prefix-style(教师见学生 rollout),ar=0.7 差 0.0135,ar=0.9 差 0.272(prefix-style 误差仅为 base 的 38%)。Prefix Scoring 的收益在高记忆强度下与信息边界修复同量级。
4. **Prefix Corruption** 在干净前缀世界无益甚至有害(||W−A|| 增大):其收益机制是教师对"学生早期漂移伪影"的鲁棒,不是信息集;本玩具前缀无伪影,如实记录为 null。

## 结论与边界

CMD 的三个组件在机制方向上都得到独立确认:信息边界(1/2)与上下文保真(3)是部署信息集对齐的两个正交维度,corruption 是针对漂移伪影的稳定化措施而非信息集修复。边界:线性高斯世界,学生为线性映射,监督取教师条件均值极限;不承诺非线性 DMD 的数值级结论。泄漏度量选用了梯度比与部署误差(学生状态中不可预测增量 w_{t+1} 的可解码性恒为 0——线性投影把它消掉了,这一负结果本身说明"泄漏"在可表达性受限的学生上表现为映射偏差而非显式编码)。

## 重跑

```bash
code/scout-exp/bin/python cmd-teacher-mismatch/code/cmd_teacher_mismatch.py
code/scout-exp/bin/python cmd-teacher-mismatch/code/cmd_prefix_ablation.py
code/scout-exp/bin/python cmd-teacher-mismatch/code/plot_cmd_v2.py
```

依赖 numpy+matplotlib(本机 code/scout-exp venv)。运行时间 <1min。
