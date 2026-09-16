# KDN(Kalman Delta Networks):循环记忆的不确定度

- 论文:Kalman Delta Networks: Uncertainty-aware Associative Memory(arXiv 2609.07816),papers/control/kdn-2609.07816.md
- 代码:github.com/ngocbh/kalman-delta-networks(完整 oracle + Möbius chunk 实现,已克隆 code/kdn-repo 核对)
- 研究动作:玩具探针(code/kdn-toy/)——什么条件下"逐方向不确定度"是决定性的

## 是什么

把循环联想记忆重述为线性高斯状态空间模型:潜记忆 S̃ 经 D_t 漂移(遗忘=过程模型),token 提供单键方向上的带噪观测 v_t = S̃^T k_t + e_t。Kalman 滤波是最优递归估计;delta 规则 = 用各向同性替代协方差、不追踪协方差的特例(DeltaNet D=I,GDN D=αI,KDA D=diag(α))。两个 scan 兼容近似:Diagonal KDN(逐通道 p_i,Möbius 2×2 结合扫描,均值保持精确单步后验)与 Isotropic KDN(标量 b_t)。750M/1.3B 受控预训练超 KDA/Mamba-3/GDN-2。

## 我们的问题

论文证明 KDN 在语言预训练上更好,但没回答一个机制边界:**逐方向的不确定度到底在什么条件下是决定性的?** 这直接连着我们的接口线:大脑读小脑 h_t 时,置信度该不该逐方向?

## 玩具探针结论(code/kdn-toy/probe.py → probe_v3.py → probe_v4.py)

- v1(稠密随机键+各向同性):Kalman 输给固定小 beta —— 没有方向性置信度可用时机制无意义。
- v2(one-hot 同质通道):iso 始终略优于 diag —— 池化 32 通道证据,逐通道追踪反而滞后;同质统计下逐方向不确定度没有存在理由。
- v3(异质:一半稳定可靠/一半噪声漂移):iso 仍整体最优,diag 居第二 —— 池化被可靠多数锚定反而占优。
- v4(漂移组 100% 噪声观测):**iso 在漂移方向崩溃(0.41,比 fixed-0.1 还差),diag 成为唯一全制度近最优** —— 池化置信度被可靠多数锚定后,持续噪声的方向被"饿死"增益,永远追不上漂移。

机制陈述:逐方向不确定度的价值不在单一制度内,而在**跨制度的鲁棒性**;池化在同质制度内占优,在制度外静默失败。对语言流的直觉:键方向的可靠性天然异质,这正是 KDN 的设定。

## 文件

- probe.py(v1+v2 与 omega 扫描)、probe_v3.py、probe_v4.py,曲线 *_curve.json
