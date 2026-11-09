# Attention 压缩的两条路线：写入损失与读出损失

## 研究问题

《将Softmax Attention线性化为Gated DeltaNet》（科学空间 11823）从固定 query 的精确递归出发，只对 Softmax 对角元作一阶展开，再以 centered key 替代残余 query，得到 GDN 形态。《平均之路：从Attention到TTT》（科学空间 11993）先学习 K→V 的均值模型，再把注意力加权平均的 key 换成 query。两条路线都把全注意力压成定长状态，但近似发生在不同位置。

本线程用真实模型的 Q/K/V 分解误差，而不是只比较最终任务分数：

1. 对角元一阶展开与 centered-key 替代各自造成多大输出误差？centered-key 的尺度是否是稳定性的关键？
2. 在线性均值模型中，平均场是恒等式；保留精确注意力读出时，K→V 的写入拟合误差有多大？再把读出输入换成 query 后增加多少误差？
3. TTTLA 的官方实现怎样把 last-layer-only TTT 写成学习型线性注意力？这对 RoboTTT 的机制解释改变了什么？

最终结论有三层。第一，原文形式的 GDN 递推在真实 Q/K 上首先遇到的是谱稳定性，而不是近似精度。第二，一个因果谱钳制能消除发散并一致改善 VaLA，但不能宣称优于所有压缩基线。第三，从 Softmax 到 TTT 的主要可测损失在读出方向；这支持 hybrid 作为压缩既有 attention 的方案，却不否定端到端训练的 TTT，因为后者可以学习不同的表示与算子。

## 实验设置与资格检查

- 模型与数据：Qwen3-0.6B 官方权重，WikiText-2 raw validation。构造段从 token 0 开始取 4×512，确认段从 token 131072 开始另取不重叠的 4×512。全部下载和计算在启智，本机不保存模型或数据。
- 捕获层：0、13、27；使用 post-QK-norm、post-RoPE 的 Q/K 与 V，重复 GQA 的 KV heads。Q 乘 `1/sqrt(d_head)`，与真实 attention logits 对齐。模型前向为 bf16，误差分析转为 float64。
- 真实模型资格检查：手工重建的 Q/K 与 Qwen 同批同长度 attention 概率对比，构造段三层最大误差为 0、3.10e-6、6.10e-4；确认段为 0、9.54e-6、0。误差来自 bf16 计算，未用跨批或跨形状比较冒充模型不一致。
- 数学资格检查：κ=0 的 GDN 形态与直接 VaLA 的最大绝对误差在构造段为 4.88e-15、1.78e-14、4.97e-14；线性平均场 `W(Σa_ik_i)=Σa_i(Wk_i)` 的误差不超过 5.2e-14。确认段同量级通过。
- 对照：精确 causal softmax、历史 V 均值、VaLA、一阶递归 GDN 形态。主网格 κ={0,0.25,0.5,1,2}，后续尺度网格见下。
- 在线 ridge/Mesa：`G_t=G_{t-1}+v_tk_t^T`，`H_t=H_{t-1}+k_tk_t^T`，`W_t=G_t(H_t+λ_tI)^{-1}`。λ_t 取 H_t 平均对角线的 {1e-4,1e-3,1e-2,1e-1} 倍，完整保留网格。
- 读出分解：精确 attention 输出、`W_t(Σa_ik_i)`、`W_tq_t`、以及每个 token 用最小二乘投影系数校准后的 `W_t(λq_t)`。最后一项是对方向假设更宽松的上界，不是可部署算法。
- 指标：时间桶 {1–16,17–32,33–64,65–128,129–256,257–512} 的 NMSE、cosine、输出范数比。每层先在 4×16 heads 上聚合，再对三层做等权宏观平均；全部位置共 98,304 个输出向量。
- 持久 JSON：`qkv_full.json`、`qkv_gdn_scales.json`、`qkv_gdn_repair.json`、`qkv_ridge_preupdate.json` 及对应 `_heldout` 文件（需要确认的三项）。

## GDN 路线：先解决谱稳定性

令 `u_t=k_t-k̄_t`，原文替代 `c_t=κu_t` 后，状态线性映射在 `u_t` 方向的乘子为

```text
m_t = 1 - (1 + κ||u_t||²) / t.
```

主网格中 κ=0 的 VaLA 有限且通过等价校验，κ={0.25,0.5,1,2} 全部指数发散并产生 inf/nan。这些溢出不是任务误差，不能用于排名。追加网格 κ={0,1/512,1/256,1/128,1/64,1/32,1/16,1/8} 显示：

| κ | 全位置 NMSE | 1–16 桶非收缩比例 | 1–16 桶最小乘子 |
|---|---:|---:|---:|
| 0 | 2.362 | 0 | 0 |
| 1/512 | 122.7 | 6.12% | -14.66 |
| 1/128 | 2.28e23 | 16.08% | -60.65 |
| 1/8 | inf | 63.74% | -980.37 |

固定的 1/d 不足，因为 learned QK norm 的通道增益并不产生单位范数。构造段第 0/13/27 层 key 平均范数为 203.94/21.46/30.87，centered key 均方为 5077.25/270.07/266.84。早期少量非收缩步骤已经足以造成后续数量级爆炸。

### 因果谱钳制

构造修复：

```text
κ_t = min(1, (t-1) / ||u_t||²)
```

它只用当前及历史可得的量，使 `0≤m_t≤1`；安全时回到原文 κ=1。由于状态转移的其余方向乘子为 `1-1/t`，这也限制了线性状态转移的谱范数；外部写入项仍决定实际状态幅度，所以报告不把该条件说成一切任务的精度保证。

| 文本段 | 指标 | V 均值 | VaLA | 谱钳制 |
|---|---|---:|---:|---:|
| 构造段 | NMSE | 1.690 | 2.362 | **1.624** |
| 构造段 | cosine | 0.467 | **0.578** | 0.519 |
| 确认段 | NMSE | **1.810** | 2.490 | 1.858 |
| 确认段 | cosine | 0.497 | **0.603** | 0.551 |

两段的每个时间桶上，钳制都降低 VaLA 的 NMSE，cosine 则始终低于 VaLA、高于 V 均值。两段的非收缩比例均为 0。构造段上钳制略优于 V 均值，但确认段反转，因此只保留“稳定且一致改善 VaLA”的结论，不声称它普遍优于均值压缩。

对应图：`../assets/attention-gdn-stability.png`。完整逐层、逐桶数值与有效 κ 在 `qkv_gdn_repair*.json`。

## TTT 路线：读出方向是主要误差

记 `r_t=Σ_i a_{t,i}k_i`，其中 `a` 是精确 causal softmax。构造段 `cos(r_t,q_t)` 从早期 0.201 到后期 0.225，最优投影残差 `||r-λq||/||r||` 从 0.969 到 0.958；确认段分别为 0.198→0.223 与 0.970→0.960。即使允许每个 token 独立选择最优标量，query 方向仍遗漏约 96% 的 `r` 幅度。

ρ=0.001 时的整体结果：

| 文本段 | 保留注意力读出 `Wr` | 直接 `Wq` | 最优投影后 `W(λq)` |
|---|---:|---:|---:|
| 构造段 | **0.329** | 1.899 | 4.545 |
| 确认段 | **0.276** | 1.821 | 4.486 |

投影校准只匹配 `r` 的方向投影，既不能恢复垂直分量，也可能被 W 的各向异性放大，所以 NMSE 反而更高。ρ 网格显示正则化不能消除这个几何问题：

| ρ | 构造段 `Wr` | 构造段 `Wq` | 确认段 `Wr` | 确认段 `Wq` |
|---|---:|---:|---:|---:|
| 1e-4 | **0.294** | 7.803 | **0.240** | 8.219 |
| 1e-3 | **0.329** | 1.899 | **0.276** | 1.821 |
| 1e-2 | **0.476** | 1.024 | **0.422** | 1.014 |
| 1e-1 | **0.803** | 0.964 | **0.762** | 0.964 |

### 当前样本自拟合与 pre-update 对照

post-update 的 `W_t` 包含当前 `(k_t,v_t)`，早期近零误差可能只是插值。另用 `W_{t-1}` 做 pre-update，仍保留精确 attention 的读出输入：

| 文本段 | pre `Wr` | post `Wr` | pre `Wq` | post `Wq` |
|---|---:|---:|---:|---:|
| 构造段 | 0.639 | **0.329** | 1.916 | **1.899** |
| 确认段 | 0.590 | **0.276** | 1.838 | **1.821** |

构造段前 16 个 token 上，post `Wr` 的 NMSE 为 0，pre `Wr` 则为 0.581；pre `Wq` 已为 0.979。当前样本拟合确实贡献明显，但 query 读出的高误差在两种时序下都存在。第 0 层后期 self-attention 平均权重约 0.378，也解释了为什么 update-then-read 能直接改善输出。

对应图：`../assets/attention-ttt-readout.png`。这里评价的是“压缩一个已经给定的 Softmax Attention”，不是训练后模型的 perplexity，也不是说端到端 TTT 的 query 没有作用。

## TTTLA 与 Muir 提供的边界

TTTLA（arXiv 2602.21204）的官方入口是 https://github.com/nv-tlabs/tttla 。LLM/NVS 实现位于 https://github.com/JunchenLiu77/LaCT 的 `main`（核对 commit `1c2109f`）；旧 `tttla` 分支链接重定向到 `main`，代码并未缺失。

- `lact_llm/lact_model/_ttt_operation_impl/only_w1.py` 只在内环更新内层 SwiGLU 的输出矩阵 `w1`；`w0/w2` 在序列内静态，但仍由外层训练。“last layer” 不是 Transformer 的最后一层。
- `only_w1_no_wn_parallel.py` 去掉逐行 weight norm，以静态核函数、末层更新和 prefix sum 得到并行形式。公开仓库只有 Variant 2 的独立并行文件，其他 Variant 的并行性来自数学形式，不能都当成已有 kernel。
- 顺序与并行实现都是“当前块先用旧状态读出，再更新”，与 RoboTTT 逐步 update-then-read 不同；本线程的 pre-update 对照因此也回答了块时序会改变哪部分误差。
- 论文 Variant 1 在 LLM PPL、NVS PSNR、图像准确率上为 15.93、25.97、79.63%，baseline 为 16.43、25.94、79.34%；Variant 2 并行实现吞吐 30.18M tokens/s，顺序实现 11.02M。数字支持简化与并行价值，但不证明任意任务只更新末层都更好。
- TTTLA 的 Replace Q with K 在三项任务上为 16.18、25.95、79.18%，与 baseline 16.43、25.94、79.34% 接近。这与本探针不矛盾：训练好的 TTT 不必模仿冻结 Softmax 的检索读出，而能学习新的有效 query/key/value。

Muir（arXiv 2611.08376）是同期直接邻居。它明确保留单位范数 key 以维持 rank-one erasure 的收缩性，低秩 erase 还抑制会把谱范数推过 1 的斜向分量；matched-filter readout 处理状态行能量不均。其静态 readout 在 1M passkey retrieval 上达到 99.4% post-cooldown，说明读出校准可以转化为长程收益。论文附录 J 给出 Muir token mixer 及 boundary-pivoted chunkwise 前后向的 JAX 参考实现；当前未关联独立训练仓库或权重，本巡航未复现这些结果，只把它作为与“写入收缩 + 读出校准”同向的论文与实现证据。

## 重跑

远端环境：Workspace `可上网GPU资源`，Project `前沿课题探索`，Compute Group `4090-2-cuda13.2`，quota `1,10,100`，镜像 `ngc-pytorch:25.02-cuda12.8.0-py3`，torch 2.7.0a0、transformers 4.51.3、datasets 3.6.0。模型经 ModelScope、WikiText 经 hf-mirror 下载到启智共享盘，显式取消代理。共享根目录为 `/inspire/qb-ilm/project/exploration-topic/ky26021/attention-1109`，下文用 `$ROOT` 指代。

```bash
# 幂等准备模型、文本与完整性检查
bash $ROOT/code/attention-1109/attention-1109/prepare_remote.sh $ROOT

# 构造段与确认段：默认参数即本文网格
CODE=$ROOT/code/attention-1109/attention-1109
COMMON=(--model-dir $ROOT/models/Qwen3-0.6B \
  --text-file $ROOT/data/wikitext-2-raw-validation.txt \
  --layers 0 13 27 --sequences 4 --length 512 --device cuda)
for OFFSET in 0 131072; do
  SUFFIX=""; [ "$OFFSET" = 0 ] || SUFFIX="_heldout"
  for SPEC in qkv_probe:qkv_full gdn_scale_probe:qkv_gdn_scales \
              gdn_repair_probe:qkv_gdn_repair ridge_preupdate_probe:qkv_ridge_preupdate; do
    SCRIPT=${SPEC%%:*}; NAME=${SPEC##*:}
    QKV_TOKEN_OFFSET=$OFFSET HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      $ROOT/.venv/bin/python $CODE/$SCRIPT.py "${COMMON[@]}" \
      --out $ROOT/results/${NAME}${SUFFIX}.json
  done
done

# 报告图
uv run --with matplotlib --with numpy python plot_attention_results.py \
  --thread runs/2026-11-09-robotttt-attention-compression/attention-compression \
  --assets runs/2026-11-09-robotttt-attention-compression/assets
```

完整参数保存在每份 JSON 的 `config` 与 `metadata`。运行包不保存 Q/K/V、attention 矩阵、模型、数据或 checkpoint。

## 后续好奇心

- 把同一谱约束作为末层 TTT 的因果学习率，而不是事后归一化状态；另一线程的玩具诊断直接检查这一点。
- 在真实机器人流上比较“读出方向误差”与“状态行能量不均”哪一个先出现；Muir 的 matched filter 与 hybrid attention 可能解决的是不同层面的读出损失。
- 将谱钳制扩展到更长文本和非语言流，并测量它相对 learned gate 的额外成本；本轮只证明两组 4×512 文本上的机制可行性。
