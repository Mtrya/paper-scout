# DreamX × LARYBench:冻结视频扩散先验里的动作信息量(先验归因探针)

**论文**:DreamX: Video Diffusion Model is a Universal Latent Action Learner(arXiv:2611.05377,KAIST+NVIDIA)
**问题**:DreamX 在 LARYBench 动作回归上报 0.17 avg MSE(CALVIN 0.15),比最强基线 DiLA(0.38)低一半以上。它的潜动作由"冻结 Cosmos-Predict2.5-2B + LoRA query 读出 + 未来帧重建监督"训出。这个数字里有多少是**冻结世界模型先验本身**自带的动作信息,有多少来自 DreamX 的生成式监督训练?
**方法**:不训练任何 LoRA/query。直接取冻结 Cosmos-Predict2.5-2B 的特征,用 DreamX 自己的探针协议(B.4 attentive probe)在 LARYBench 官方发布数据上回归动作,与论文锚点对比。

## 论文侧方法论发现(审计)

1. **DreamX Table 1 裁掉了 LARY Table 2 的原始编码器基线**。LARY 原文 Table 2 有冻结通用编码器的两帧回归:DINOv3 CALVIN 0.22(avg 0.19)、V-JEPA2 0.27(avg 0.25)、Wan2.2 像素 VAE 0.39(avg 0.30);分类 V-JEPA2 avg 76.62 高于 DreamX 的 69.67。DreamX Table 1 这些行全部消失,只留潜动作模型(LAPA/UniVLA/villa-X/MoTo/AdaWorld/VideoWorld2/DiLA)。"SOTA 一半提升"的参照系是被裁剪过的。
2. **同表混协议**。LAPA/UniVLA/villa-X/LAPA-* 各行与 LARY Table 2 逐位相同(照录,两帧协议),而 DreamX 自己(B.4)"用元数据从源数据集恢复了中间帧"——回归基准原本每样本只有两帧。视频类模型(MoTo/DiLA/DreamX)用恢复出的完整片段,帧对模型用两帧,同一张表里两种输入信息量。
3. **LARYBench 发布版只有 stride-5 端点帧**(本机磁盘证实:images/ep_1/ 只有 frame_000065, frame_000070, ...),恢复中间帧需要原始 CALVIN/DROID 等全量数据集(数百 GB 级)。即:第三方无法仅凭 LARYBench 发布包复现 DreamX 自己的评测条件。
4. **DreamX 未开源**(2026-11-05 项目页只有 "Code soon",无仓库链接),无法读码核验其实现细节。
5. **DreamX 自己的零化对照暴露了先验自带运动**(论文 C.3 Table 8):固定首帧与解码噪声,把潜动作 query 全部置零后,生成视频的运动量仍有原版的 **45.6%**(motion ratio 0.456)——先验会自己"脑补"运动,潜动作负责的是把运动导向任务相关方向。这正是本探针要量化的东西。

6. **LARY Table 2 锚点数字的探针协议与 DreamX 的读出协议完全不同**(读 LARY 官方仓库 `regression/main.py` + `get_latent_action` 证实):官方回归探针是把**整段 token 图全部拉直**(DINOv3 一对帧 = 2×256×1024 = 524,288 维)喂给一个 hidden 4096、两个残差块的 MLP——仅第一层就 21 亿参数;Huber δ=1、lr 1e-4、batch 256、50 epochs。而 DreamX B.4 的读出是 768 投影 + learned-query 池化的 attentive probe(约 4M 参数)。两者差着近三个数量级的读出容量:Table 2 的 0.22/0.27/0.39 量的是"巨型回归器能从特征里榨出多少",不是"轻量读出器能直接读到多少"。**对照这两类协议下同一组冻结特征的表现,本身就是本线程要给的答案之一。**

## 实验设置

- **特征**:冻结 Cosmos-Predict2.5-2B(官方 post-trained EMA 权重),输入 [src, tgt×4] 5 帧(因果 VAE 要求 T≡1 mod 4;潜帧 0=src,潜帧 1=静态 tgt 尾),空文本。
  - VAE 臂:两潜帧的 token,(2, 900, 16) fp16。
  - DiT 臂:层 {6, 13, 20, 27} 各 (2, 225, 2048) fp16;三种约定:pilot 对比 A(σ=1e-4 近似 clean, cond_mask=1)/B(σ=1.0 训练噪声区, mask=1)/C(σ=0, mask=0)。pilot 结果 B 一致最优 → 全量采用 B。
- **探针**:两套协议。(a) DreamX B.4 attentive probe(投影 768 + LN + GELU + 时间嵌入 + learned-query 池化 + 线性头,约 4M 参数),Huber 训练,报归一化 MSE;动作归一化常数取自 LARY 官方探针代码。(b) LARY 官方 ActionExpertMLP(token 拉直 + LN→Linear(·,4096)→ReLU→2 残差块→LN→Linear(·,35),Huber δ=1、lr 1e-4 常数调度 + 10% warmup,与官方 `regression/main.py` 逐项对齐)。另有 pool4 臂(15×15→2×2=4 token/潜帧,DreamX 的 300× 压缩预算)与 mean-pool 臂(每潜帧 1 向量)。
- **数据**:LARYBench 发布版 CALVIN regression(train 19 卷 = 19,000 样本 / val 11,969;全量 train 198 卷在启智远端续下)。
- **参照锚点**(CALVIN 归一化 MSE):DreamX 0.15(恢复中间帧+训练过的 query)、冻结 DINOv3 0.22 / V-JEPA2 0.27(LARY Table 2,两帧)、Wan2.2-VAE 0.39(LARY Table 2,像素 VAE)、DiLA 0.43(视频,LARY 原数)、LAPA 0.96(≈均值地板)、A0 均值地板 0.987(本机算)。

## 结果

### Pilot(2,000 train / 1,000 val,25 epochs,seed 42;过拟合区间,只看相对序)

| 臂 | l6 | l13 | l20 | l27 |
|---|---|---|---|---|
| A: σ≈0, mask=1 | 1.2015 | 1.2065 | 1.2223 | 1.1828 |
| B: σ=1.0, mask=1 | 1.1441 | 1.1572 | 1.2077 | **1.1260** |
| C: σ=0, mask=0 | 1.2168 | 1.2188 | 1.2330 | 1.1990 |

VAE 臂 1.2548。均值地板(A0)= 0.987(pilot val 同分布)。
发现:σ=1.0(模型训练噪声区)特征一致优于 clean 特征;mask=1(标记为观测帧)优于 mask=0。

### 全量(19,000 train / 11,969 val,40 epochs)

attentive probe 臂(best val MSE / 3 seeds;均值地板 A0 = 0.987):

| 臂 | seed42 | seed0 | seed1 |
|---|---|---|---|
| dit_l27(σ=1.0) | 1.0513 | 1.1203 | 1.0415 |
| dit_l27_pool4 | 1.0615 | 1.1177 | 1.0531 |
| vae tokens | 0.8576 | 0.8358 | 0.9428 |

观察:(1) 所有臂晚期严重过拟合(final 最高到 3.08),数字一律取 best epoch;(2) VAE 潜变量三种子全部低于地板(0.836–0.943),DiT σ=1.0 特征在轻量探针下只到地板附近(1.04–1.12);(3) pool4 把 token 数压到 1/56 后精度几乎不掉,说明瓶颈不在 token 预算。

层曲线(σ=1.0,seed 42):l6 = 0.9937,**l13 = 0.8976**,l20 = 1.1849,l27 = 1.0513。非单调:第 13 层是四个采样层里唯一低于地板的 DiT 特征,最深的 l20 反而最差——动作信息可读性并不随深度单调增长。

MLP 臂(LARY 官方读出协议,seed 42):

| 臂 | best val MSE | best epoch | position | orientation | gripper |
|---|---|---|---|---|---|
| vae 拉直(28,800 维,152M 探针) | **0.8176** | 21 | 0.252 | 1.257 | 1.195 |
| dit_l27 pool4 拉直(16,384 维,101M) | 1.0751 | **1** | 0.895 | 1.097 | 1.551 |
| dit_l27 均值池化(4,096 维) | 1.2160 | **1** | 1.242 | 1.247 | 1.043 |

MLP 臂判读:(1) 同一组 VAE 特征上,巨型 MLP 把 position 读到 0.252(小探针 0.81 的三分之一),却把 orientation 读差(1.26 vs 0.72),总分仅 0.858→0.818——读出容量改变的是"能读出哪些维度",不是均匀的增益;(2) DiT σ=1.0 特征上巨型 MLP 的最佳点出现在 epoch 1 即开始过拟合,容量找不到更多东西——先验的 DiT 隐状态里没有可被回归器直接榨出的动作信息,与读出器大小无关;(3) 与锚点的差距(VAE 0.82 vs Wan2.2-VAE 0.39 / DINOv3 0.22)主要来自特征本身,不是探针容量。

**总结论**:冻结 Cosmos-Predict2.5 先验的动作信息集中在 VAE 潜空间(可破地板),DiT 训练噪声区隐状态对任何规模的冻结读出都接近地板。DreamX 的 0.15 与先验自带读出(0.82–0.94)之间约六倍差距,其间是生成式监督训练(LoRA 调整特征朝向)与恢复中间帧的输入增益——先验给了矿脉,矿不是现成的。

## 复现

```
# 特征(需 cosmos-venv,见 extract_cosmos.py 头部补丁说明)
bash run_extract_full.sh   # ~3h on RTX 3060
# 探针
bash run_full_probes.sh
```

关键文件:`extract_cosmos.py`(特征提取,含 18 通道输入约定与 cond_mask 语义)、`probe.py`(B.4 attentive probe 复刻)、`patches/cosmos_patch.py`(mock transformer_engine/megatron 以便本地加载)。
