# WCM（arXiv:2607.29613）— 线程记录

## 这条线做了什么

全文审读 World Critic Model 的 POMDP 动机、联合目标、149-task 仿真与 7-task
真机实验；clone 官方 repo 后追踪 causal Transformer、action-conditioned dynamics、
value head、SIGReg 与 DDP loss；核验公开 checkpoint / RL 资产范围。

## 关键发现

- 机制不是“多看几帧”，而是联合优化 value regression 与动作条件的下一 latent
  prediction；只用历史、令预测权重 λ=0 的 ViT critic 没有稳定增益。
- λ=0.3–0.5 最佳；OOD 对 λ 的波动 10.6pp，IND 仅 2.7pp，支持预测目标主要
  抑制静态相关性过拟合的解释。
- 真机：π0.5 炉灶清洁 4/50→RECAP 27/50→WCM 33/50，寿司 13→18→24；
  OpenVLA-OFT 毛巾 16→35→40，布料 15→29→38。
- 代码确认 history trunk 为标准 self-attention，随 K 是 O(K²)，论文摘要的
  “exponential complexity”是过度表述；默认只做一步 latent prediction，也不等于
  可长程 rollout 的完整世界模型。
- repo 有核心 critic 代码、小型真机资产与 checkpoint；149-task 全规模 RL 集成和
  全部论文 checkpoint 并未完整打包，论文结果不能由公开小 checkpoint 复现。

## 线程产物

- 报告图：`../assets/wcm-method.jpg`（论文 Figure 2，已人工检查）。
- 本线程无自研实验代码；代码审计证据来自本轮 clone 的官方仓库。
