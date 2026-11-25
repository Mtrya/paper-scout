# 线程：Loom Attention 的逐 token 线路重连（2611.30634）

论文：Rewirable Circuits in Transformers with Loom Attention（Da Xiao 等，BUPT + 彩云科技；arXiv 2611.30634v2，ICML 格式）。代码 https://github.com/Caiyun-AI/LoomFormer （JAX 训练实现 + PyTorch HF 推理移植，均含测试）。

## 论文核心主张

1. Transformer 的头间组合（线路）在垂直方向（层间）是静态的：pre-LN 下每 token 只能做秩 1 的节点门控（Prop. 2.3），要实现任意布线模式需要 Θ(w) 个头/读者（Thm. 2.4，cover-free family 下界）。
2. Loom attention 把"对更早头输出的垂直注意力"与"对 token 的水平注意力"交织：垂直注意力用门控线性注意力在逐 token 矩阵记忆 M（d_k×d，默认 d_k=2n）上实现——即把线性注意力从时间轴换到深度轴。M 是第二残差流（矩阵流），向量流出端口（控制），矩阵流出内容。
3. Prop. 3.1：固定端口时等价于普通 node-gated Transformer（对 (x, vec(M))）；端口逐 token 生成时，单头/读者即可实现至多 d_k 个写者的任意布线模式。
4. 实验：Pile scaling（432M/778M/1.4B）每档损失最低，等效 Transformer++ 的 3.19–3.25× 训练计算；300B token 下 LoomPythia-2.8B 零样本平均追平 Pythia-12B，5-shot 反超，FLAN ppl 6.94 全表最好；FLOPs 开销仅 1.95–3.7%，训练吞吐保持 89–92%。
5. 机制分析：动态布线有效秩随深度升至 41（均值 26.4，2.8B、d_k=64），71% 动态布线能量在秩 1 之外；深层"开着的"（非 sink）头占 74.4%，是 Pythia（22.9%）与 mHC（17.7%）的 3.3–4.2×；消融显示静态化双端口后仍保留 56% 增益（静态双流骨干 ≈ 一半增益，逐 token 重连 ≈ 另一半），去掉 FFN_M 写支路伤害最大（剩 80.7%）。

## 本轮做了什么与看到了什么

### 1. 全文精读
全文（含附录目录）通读，缓存于 `papers/architecture/loom-attention-2611.30634.md`。方法部分逐式核对（Eqs. 4–11），实验与机制分析数字见上。

### 2. 官方仓库源码审计（code/LoomFormer，浅克隆）
- **PyTorch 移植**（pytorch/loomformer/modeling_loomformer.py）与论文逐条吻合：MRead 双臂（动态 Norm+门 + 静态 q̄ 门）、MShift（σ 门混合 M[t-1]）、MNorm 对矩阵整体 RMS、严格因果掩码排除当前 token（`allowed = keys < queries`，第 339 行）、自身内容走 bypass o^m、FFN_M 是 embedding 之后向量流入 M 的唯一通道、末层不写 M。
- **一个疑似差异的裁决**：论文 Eq.6 把写门乘在归一化 key 上，PyTorch 移植乘在归一化 content 上。核对 JAX 训练实现（jax/loomformer/loomformer.py 第 293 行）确认门在 key 上；但门是**每头标量**，标量乘在外积的任意一侧数学恒等（g·(k⊗o) = (g·k)⊗o），故论文、JAX、PyTorch 三者无矛盾——这是实现层面的因式分解自由度，不是论文-代码不一致。
- **配置核对**（jax/configs/loomformer_medium.yml）：D=1200、n=16、d=75、d_k=32=2n、d_r=256=n²、rope_dim=18=⌊75/4⌋，全部与论文一致。
- **一个实现细节洞见**：水平注意力的 logit 是 q·k = (q̃·k̃) + (qr·kr)——M 读出的槽路由项与 RoPE 的 token 路由项相加；即"选哪个槽"与"选哪个 token"在同一次点积里求和。

### 3. 机制探针（code/loom_rewiring_probe.py，CPU torch，输出 probe_output.txt）
按 Eqs. 4–11 自建 6 层玩具栈（D=64, n=4, d=16, d_k=8），验证论文的三个结构性主张：
- **逐 token 重连真实存在**：读/写端口跨 token 余弦距离 0.49–0.62（动态端口），与论文训练后 2.8B 模型测得的 0.33–0.75 同量级。
- **动态布线能量有效秩随深度增长且被 d_k 钉住**：均值 2.98→5.05（max 6.39），上界 min(d_k=8, w)；论文 2.8B 里是 26.4→41（上界 64）。秩界是结构性的，不是训练出来的。
- **静态端口消融**：把 q^m、k^m 冻结为 token 均值后，动态变化量降至 1.2e-7（浮点噪声底），验证 Prop. 3.1(i)——静态端口 = node-gated Transformer，重连能力来自逐 token 端口生成，不来自矩阵记忆本身。
- **秩一节点门控不可能性**（Thm. 2.4 最小实例）：目标布线 {W1→R1, W3→R3}，rank-1 最优逼近相对残差恰为 1/√2≈0.707，rank-2（即 d_k=2 个槽）精确实现——"加头"救不了节点门控，"换轴"可以。
- 产物图：assets/loom_probe_rank.png（动态秩随深度 vs 静态=0，d_k 上界线）。

### 4. 图片素材
arXiv HTML 只嵌入了 Fig.1（SVG 草图）、Fig.4（arch.png，已下载 drafts/images/arch.png）与附录 Fig.8/9；Fig.2/3/5/6/7 未进入 HTML（LaTeXML 丢弃）。报告配图用 arch.png + 我们的探针图，机制数字以正文引用。

## 如何重跑
```
python3 runs/2026-11-26-loom-motionmamba/loom-attention-2611.30634/code/loom_rewiring_probe.py
```
（依赖 torch 与 numpy；CPU 即可，约 10 秒。绘图产物写在本目录 loom_probe_rank.png。）

## 对报告的含义
- 可信的核心判断：Loom 的增益机制是"把线性注意力换轴到深度方向"，让每 token 的头间布线从秩 1 门控解放为秩 ≤ d_k 的自由矩阵；静态双流骨干与动态重连各贡献约一半增益。
- 与本实例长期主线（线性注意力/常数复杂度机制）的连接：这是"线性注意力机制本身"的一篇正作——它不省时间轴的 KV 缓存，而是用同一套外积写/读原语在深度轴上做头间路由；与 TTT/GDN 线（2026-11-09 记忆的写入收缩 vs 读出几何）互为正交轴。
- 边界与不确定：所有训练级证据来自论文自报（Pile/Pythia 协议），本轮未复训；玩具探针验证的是机制容量与秩界，不是端到端收益；d_k 的干扰行为（>d_k 个写者共享槽时互相干扰，论文 Prop. 3.1(ii) 边界）未独立验证。
