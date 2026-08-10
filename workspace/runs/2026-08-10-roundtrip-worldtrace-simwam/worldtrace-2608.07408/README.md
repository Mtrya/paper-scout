# WorldTrace / Addressable Memory (2608.07408) — RoPE 相位相消核验

论文:NVIDIA,ICML'26 F2S Best Paper。KV cache 记忆的两个失效(RoPE 位置出界→不可寻址;旋转空间朴素平均→相位相消),解法是槽位秩虚拟位置 + canonical(反旋转→平均→再旋转)。无代码。

## 我们的研究动作

用 Qwen3-0.6B 真实权重(28 层 GQA,head_dim 128,(i, i+P) 半劈 RoPE)对"相位相消"机制做三组直接测量:

1. **逐频率对存活率**(rope_probe.py):真实文本前向取中层 K,M 个位置的 key 旋转后朴素平均,逐频率对量"平均后范数/平均前范数"。θ·Δt≥2 的高频对存活 0.27-0.37(最差 0.007),低频 0.6-0.8;canonical 全频段 0.7-0.9。
2. **logit 平凡性教训**:mean(K)·q ≡ mean(K·q),logit 对比测不出相消;真正的失败在向量范数收缩→softmax 读权重坍缩。测量必须测分配后的权重。
3. **聚合 softmax 读权重**(rope_probe2.py):8 KV 头 × 16 query × 3 层,摘要槽读权重 / 源位置读权重。M=32:naive 0.003 vs canonical 0.016(中位,5 倍);M=8/stride=4:0.105 vs 0.125(几乎无差)——相消是参与平均的相位个数的函数,低压缩比下问题轻。

## 核心发现

- 论文的机制声明在真实权重里定量成立,形状与理论预期吻合。
- 论文没强调的边界:相消随参与平均的相位数恶化;设计含义是"要么 canonical,要么别平均(Landmark 逐字)",朴素平均在高压缩比下等于删掉记忆的高频通道。
- 未复现 LoopBench 端到端(需 Matrix-Game-2 完整推理),论文 +15.5%/+19.5% 下游增益未独立验证。

## 内容

- `code/`:rope_probe.py(存活率 + 单点 softmax)、rope_probe2.py(聚合 softmax)
- `code/results/`:rope_probe.json(逐配置 survival 曲线)、rope_softmax.json(聚合读权重)
