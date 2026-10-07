# 预登记预测(2026-10-07 巡航,实验开始前登记)

本文件在任何实验结果产生之前写就。三个实验的预测按"论文复测 + 自有猜想"区分;自有猜想证伪与证实同等交付。

## 实验 A:astra 混合控制 × 小脑先验质量网格(复测 2609.38537 C2)

设置:2D 桌面 sim(gpt-policy-sim),T1(易)/T2(精密)× {direct, hybrid×oracle/noisy/biased/degenerate} × 3 seeds = 30 格。hybrid 臂:脚本小脑提出动作 chunk,astra 逐决策点审查(接受/替换)。

- P1(复测论文 RoboDojo 方向):hybrid-oracle 成功率 ≥ direct,且 astra 决策数显著更少。
- P2(复测论文 RoboLab 方向的对偶):hybrid-degenerate ≈ direct(astra 学会无视退化提案,退化提案不拖垮系统)。
- P3(**自有猜想"危险中间态"**):hybrid-biased < direct——貌似合理但系统性错误的提案(平滑轨迹 + 固定偏移/轴互换)会锚定 astra,比明显垃圾的 degenerate 提案更危险。若成立,则论文的"先验质量决定正负号"需要修正:质量轴不是单调的,中间态比两端更差。
- P4(锚定签名):接受率随提案质量下降,但 biased 臂的接受率顽固偏高(显著高于 degenerate)。
- 附带测量:astra 介入比例,对照论文 RoboDojo 的 14.4% / 85.6% 步骤走策略。

## 实验 B:在线 TTT 臂对照(同场检验 ASCENT 2610.05303 + ouroboros 2610.05076)

设置:真实小模型(Qwen3-0.6B,Inspire 4090)在程序化生成、程序化验证的文本任务流上做在线 TTT。臂:frozen / 闭环硬模仿 / verified-only 硬模仿(RFT) / ASCENT 式(冻结 hindsight 教师软蒸馏进 LoRA) / RFT+Settlement 门控。

- B1(复测 ASCENT Fig 1):RFT 臂后段失稳(成功率跌破 base、熵坍缩、非法格式率升)。
- B2(复测 ASCENT 主结果):ASCENT 臂稳定提升,不失稳。
- B3(**结合断言,本轮核心**):Settlement 修复稳定性但不带来 ASCENT 的提升——门控管稳定、hindsight 目标管提升,两者是可分离的机制。若 B3 成立,两篇论文各自只拿到了一半。
- B4(复测 ouroboros 的 settlement 探针):独立真实文本 NLL 只在硬模仿臂退化;Settlement/ASCENT 臂保持不变。
- B5(复测 ouroboros Fixed Generation):用冻结 base 生成训练数据即可消除大部分伤害,即使不做软蒸馏。

## 实验 C:flow 策略 RL 低秩机制复测(玩具,本机 CPU)

设置:2D 点质量任务,MLP 动作专家显式带 TimeMLP + AdaRMS(d=128),BC 预训 → PPO 微调。

- C1(复测论文 Fig 2-3):RL 更新密集但低秩(r95 ≪ BC),且集中在时间步调制模块。
- C2(复测论文成因断言):只在 RL 见过的离散去噪步上做 BC,同样产生低秩更新(成因是训练信号稀疏,不是 RL 本身)。
- C3(复测论文 scale/shift 分解):AdaRMS 的 scale/gate 沿原方向缩放(|cos| 高),shift 长出新方向(|cos| 低)。
- C4(复测模块替换):只把 RL 后的时间步模块移植进 BC 模型,保住大部分 RL 增益;反向替换则掉增益。
- C5(**自有延拓**):若 shift 方向探针能预测成败(论文 ROC-AUC ≤99.6%),则在玩具设置里 steering 沿 shift 方向也应给出免费增益;若玩具设置复现不出 steering 增益,则论文的 steering 结论可能依赖大模型的高维几何。
