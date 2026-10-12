# Multimodal Flow(arXiv 2609.40362)文本潜变量 round-trip 保真探针

**一句话总结论:MF-1 的"8-token T5-small 连续潜 → 文本"通道在真实权重下**几乎无损**——236 个样本 / 7,641 个内容 token 上位置对齐还原率 100.00%,样本级完全还原率 100%,teacher-forced 交叉熵 7.0e-5 nats(PPL 1.00007);字符级编辑距离 0.91% 与"只用 tokenizer 自己的往返"基线完全相等,即所有字符偏差都来自 T5 detokenizer,没有一丝来自潜变量通道。通道对扰动的容忍度也远超预期(归一化空间里加 std=1.0 的各向同性噪声,还原率仍 98.85%),原因不是 decoder 弱,而是**块内 8 个位置是一份联合码**:把某个位置自己的潜向量整个换成无关块的同位置潜向量,该 token 仍有均值 93.0% 能从其余 7 个位置还原(leave-one-out)。因此"全连续统一建立在一个有损通道上"这个担心在 decoder 侧不成立;MF-1 文本质量的真正瓶颈只能是 flow backbone 的端点预测误差,而本探针给出了它的容忍预算:归一化空间误差 std ≲1.0(约等于潜向量自身尺度的 1.24 倍)时 token 还原率仍 ≥98.8%,σ_e=1.5 才降到 86.7%。**

## 1. 复现的管线(一律以代码为准)

| 步骤 | 做法 | 代码位置(commit `c84caf5`) |
| --- | --- | --- |
| 1 分词 | T5 sentencepiece,`add_special_tokens=False`,剥掉 EOS(1)/PAD(0) | `src/mf/data/text.py:_content_ids` |
| 2 切块/补齐 | 每条记录末尾追加 EOS,再把最后一块用 EOS 填满到 8 的倍数 | `tokenize_block_aligned_document_ids`;由 `flow.text_block_causal.packing=block_aligned_eos` + 默认 `whole_record_defer_pad` 策略选中(`src/mf/data/physical.py`) |
| 3 编码 | **每个 8-token 块独立过一次冻结 T5-small**,注意力只在块内 | `src/mf/codecs/online.py:_encode_target_rows`(`target_encoding=block_local`) |
| 4 归一化 | 逐位置 `(h − 0.0) / 0.2`,标量常数、跨维度共享 | `src/mf/latents/stats.py:normalize_text` + `configs/pretrain.yaml:latent_stats.text_normal` |
| 5 解码 | `LatentTextDecoder`:6 层双向、宽 512、8 头 × 64 维、RMSNorm、RoPE、SwiGLU(隐层 1365)、512→128→512 瓶颈、tanh-GELU 头投射到 32128 维 T5 词表;每个输入位置输出一个 token;推理不吃 attention mask | `src/mf/codecs/text_decoder.py` + `configs/pretrain.yaml:model.text_decoder` |
| 6 还原 | 逐位置 argmax → token id → T5 detokenize | 同上 |

论文描述与代码的对照(有出入的地方都在代码一侧,结论仍然一致):

- **"每个 8-token 块独立编码"**:代码确认。`_encode_target_rows` 把 `[rows, tokens]` reshape 成 `[-1, 8]` 后再批量过 T5,块与块之间没有注意力。
- **σ_text = 0.2、μ_text = 0**:代码里是**标量**(`latent_stats.text_normal: {std: 0.2}`),不是逐维统计。实测(本探针 1082 个块)冻结 T5-small 原始潜向量整体 std = 0.1616,逐维 std 落在 0.0706–0.2874(中位 0.1397);除以 0.2 之后逐维 std 落在 **0.35–1.44(中位 0.70)**。也就是说"归一化空间"并不是单位方差空间,这一点论文没写,而它决定了下面 λ/σ_e 的物理含义(σ_e=1.0 相当于给一个自身 std≈0.81 的量加同等量级的噪声)。
- **λ = σ(0.8 + 0.8z)、噪声强度 s**:代码确认(`decoder_noise.py` + `decoder_noise_scale: 1.0`、`decoder_p_mean/std = 0.8/0.8`)。按代码算,**z = −3/−1.5/0/1.5/3 对应 λ = 0.168/0.401/0.690/0.881/0.961**(任务描述里给的 0.12/0.30/0.55/0.83/0.97 是近似值,与 σ(0.8+0.8z) 并不严格对应)。本探针用代码公式,并把实现与发布函数做了逐位相等校验(`corruption_replication_exact: true`)。
- **EOS 与 padding**:记录末尾的 EOS 和终端块的 EOS 填充位置**同样是 decoder 的监督目标**(不 mask、不置零),PAD 只出现在整流序列尾部;decoder 推理时 `use_attention_mask: false`,块内 8 个位置互相全可见。
- **归一化发生在 decoder 之外**:`LatentTextDecoder` 内部没有 RMSNorm 之外的可学习缩放/平移,也没再做一次输入归一化;它的第一个算子就是把 512 维潜向量线性压到 128 维瓶颈(`text_proj.proj1`,`bias=False`)。所以"喂进 decoder 的到底是原始还是归一化潜向量"完全由 `input_space` 决定,并已用第 4.5 节的对照实测确认(喂原始潜向量掉到 79.5%)。
- **推理配置交叉核对**:从 HF 仓库单独取了 `MF/pretrain/config.json`(2,702 B),确认 `model.text_decoder.input_space = "normalized"`、`max_length = 8`、`flow.text_block_causal.block_size = 8`,以及 `flow.text_noise_scale = 2.0`(视觉为 1.0)——即文本分支 flow 的噪声尺度被刻意放大到归一化潜尺度的 2 倍,视觉分支是 1 倍。

## 2. 权重、代码与来源

| 资产 | 来源 | 大小 / 哈希 | 说明 |
| --- | --- | --- | --- |
| 文本 decoder 权重 | HF `hustvl/Multimodal-Flow` → `Text Decoder/text_decoder.safetensors` | 71,580,384 B,sha256 `86ba2911fad4b18e497ae4135d5841d6ecee9dbee7e6b4abe47c5c1e06a0a922` | safetensors 元数据 `{source: pretrain-step-160000, weights: ema, format: multimodal-flow-hf}`;80 个 BF16 张量,合计 35,786,364 参数;键集与 `LatentTextDecoder.state_dict()` 完全一致,`load_state_dict(strict=True)` 直接通过 |
| T5 编码器/分词器 | HF `google-t5/t5-small` | `model.safetensors` 242,043,056 B(sha256 前缀 `bd944e5f`);`spiece.model` 791,656 B | 只当编码器用;下载后删掉了 tf/flax/rust/onnx 四份重复权重(2.1 GB → 234 MB) |
| 推理配置 | HF 同上 → `MF/pretrain/config.json` | 2,702 B | 用于第 1 节的交叉核对 |
| 代码 | 本地克隆 `code/Multimodal-Flow` @ `c84caf50fc3da57e30d5984a4d99dee81f8b0c43`(2026-10-08) | `src/mf/codecs/text_decoder.py` sha256 `a4a86a0e…`;`src/mf/latents/decoder_noise.py` sha256 `d2b471f5…` | 两者已**逐字复制**进本目录(`mf_text_decoder.py` / `mf_decoder_noise.py`,仅加 5 行来源注释;`diff` 校验 body 完全一致),使探针不依赖本地克隆即可运行 |
| **未下载** | `MF/pretrain/model.safetensors`、`MF/sft/model.safetensors` | 各 3,277,220,832 B(3.28 GB) | 按任务要求不下载 1.6B backbone → **flow 自身的端点预测误差无法测量**,见第 7 节 |

## 3. 方法与样本

- 语料:`papers/` 下 6 篇论文正文(agents / robotics / vla / perception / control / embodied-data 各一),按仓库自己的句子切分规则 `split_complete_text_units` 取句级单元,过滤掉表格/公式/图注行,得 2,799 条 4–200 token 的候选;按长度分 6 档(4–12 / 13–24 / 25–40 / 41–70 / 71–120 / 121–200)分层抽取 **200 条**。
- 另加 **36 条手工句**,覆盖标点(破折号、问号、引号、分号)、数字与百分数、日期、专有名词与版本号、缩写 `e.g.`、URL、连字符、大小写、CJK、emoji。
- 合计 **236 个样本 → 1,082 个 8-token 块 → 7,641 个内容 token**(不含追加的 EOS 与填充)。
- 指标:`token_acc_real`(内容 token 上位置对齐的还原率)、`sample_exact`(整条样本全对)、块级 8/8 全对率、字符级归一化编辑距离。**关键对照是 "tokenizer-only 基线"**:`decode(encode(text))` 与原串的距离——它剥离了 T5 detokenizer 本身的往返损失,不能让 decoder 背这个锅。
- 计算:CPU、float32、单线程(BF16 权重载入 fp32 参数);全流程 457.6 s(同一 seed 跑两次逐项数字完全一致,图也逐字节相同)。

## 4. 结果

### 4.1 实验 A:干净 round-trip —— 通道几乎无损

| 指标 | 值 |
| --- | --- |
| 内容 token 还原率 | **100.00%**(7,641/7,641) |
| 样本级完全还原率 | **100.00%**(236/236) |
| 块级 8/8 全对率 | **100.00%**(1,082/1,082) |
| 逐位置还原率 | 位置 0–7 全部 1.0000 |
| teacher-forced CE / PPL | 6.98e-5 nats / **1.00007** |
| 平均 top-1 概率 | 0.99994 |
| 字符级编辑距离 | 0.00911 = **tokenizer-only 基线 0.00911** |
| ─ 论文正文组(200 条) | 0.00043 |
| ─ 手工组(36 条) | 0.0573(全部来自 detokenizer:CJK/emoji/罕见标点) |
| 追加 EOS 与填充位置 | 也 100% 正确(`token_acc_all_targets = 1.0`)→ 生成时的 EOS 停止条件在干净潜向量下不会误触发 |

"全对"不是指标太粗:同一套度量下,位置滚动负对照只有 0.84%(§4.5)。

### 4.2 实验 B:按论文加噪规则扫描 λ = σ(0.8 + 0.8z)

z 固定、全块加噪(λ 越大越干净):

| z | λ | token 还原率 | 块 8/8 | 样本全对 | PPL |
| --- | --- | --- | --- | --- | --- |
| −3.0 | 0.168 | 0.0742 | 0.0139 | 0.0000 | 8160.6 |
| −2.5 | 0.2315 | 0.2345 | 0.0213 | 0.0000 | 284.6 |
| −2.0 | 0.310 | 0.6341 | 0.0970 | 0.0000 | 6.60 |
| −1.5 | 0.401 | 0.9295 | 0.6488 | 0.1568 | 1.308 |
| −1.0 | 0.500 | 0.9931 | 0.9538 | 0.7924 | 1.022 |
| −0.5 | 0.599 | 0.9995 | 0.9963 | 0.9831 | 1.002 |
| 0.0 | 0.690 | 0.9999 | 0.9991 | 0.9958 | 1.000 |
| +0.5…+1.0 | 0.769…0.832 | 0.9999 | 0.9963–0.9972 | 0.9958 | 1.000 |
| +1.5…+3.0 | 0.881…0.961 | 1.0000 | ≥0.9972 | ≥1.0000 | 1.000 |

按训练时的真实分布抽样(z ~ N(0,1) 逐位置,3 次):**99.83% ± 0.02%,平均 λ = 0.668**。也就是说 decoder 被训练时的加噪比它实际需要的强度高得多,而这些"训练噪声"几乎不构成损失。

### 4.3 实验 B2:各向同性误差预算(给 flow 的容忍额度)

在干净归一化潜向量上加 `σ_e · ε`(ε ~ N(0,I)):

| σ_e(归一化空间) | 折算成原始 T5 尺度 | token 还原率 | 块 8/8 | PPL |
| --- | --- | --- | --- | --- |
| 0.00 | 0.000 | 1.0000 | 1.0000 | 1.000 |
| 0.10 | 0.020 | 0.9999 | 0.9991 | 1.000 |
| 0.25 | 0.050 | 0.9999 | 0.9991 | 1.000 |
| 0.50 | 0.100 | 0.9997 | 0.9982 | 1.001 |
| 0.75 | 0.150 | 0.9980 | 0.9843 | 1.006 |
| **1.00** | 0.200 | **0.9885** | 0.9113 | 1.043 |
| 1.50 | 0.300 | 0.8668 | 0.4002 | 1.808 |
| 2.00 | 0.400 | 0.6371 | 0.0767 | 6.237 |

归一化潜向量的整体 std 是 0.808,所以 **σ_e = 1.0 相当于给信号本身加 1.24 倍标准差的噪声,仍能还原 98.85% 的 token**。这就是 flow backbone 的容忍预算;超过这条线才开始断崖(σ_e=1.5 → 86.7%,2.0 → 63.7%)。

### 4.4 实验 C:把"带上下文"的潜向量喂给冻结 decoder

把整条记录过一次 T5(全上下文,而非块内 8 token)再切成块送进 decoder:**99.10%**(vs block-local 的 100%),块 8/8 88.5%,样本全对 77.5%,PPL 1.046。说明冻结 decoder 对这一点分布偏移几乎不敏感,生产路径的块内独立编码在解码侧几乎没有代价。

### 4.5 实验 D:负对照(证明上面 100% 不是评测脚本在自证)

| 对照 | token 还原率 | 含义 |
| --- | --- | --- |
| 块内位置整体滚动 1 位 | 0.0084 | 通道是严格位置对齐的,不是"读整块内容" |
| 整块潜向量换成无关块 | 0.0114 | **没有标签泄漏**;decoder 输出的是它真正读到的内容 |
| 喂未归一化的原始潜向量(错误输入空间) | 0.7951 | 反向验证 `input_space="normalized"` 的判断;同时也说明 decoder 对整体尺度有相当容忍,但不是无损 |

### 4.6 实验 E:leave-one-out —— 8 个位置是联合码,不是 8 条独立信道

把某个位置**自己**的潜向量换成无关块同位置的潜向量(其余 7 个位置保持干净):

| 被替换的位置 | 该位置 token 还原率 | 其余位置还原率 |
| --- | --- | --- |
| 0 | 0.8895 | 0.9992 |
| 1 | 0.9087 | 0.9939 |
| 2 | 0.9221 | 0.9979 |
| 3 | 0.9475 | 0.9991 |
| 4 | 0.9609 | 0.9990 |
| 5 | 0.9642 | 0.9991 |
| 6 | 0.9804 | 0.9986 |
| 7 | 0.8676 | 0.9970 |
| 均值 | **0.9301** | 0.9980 |

一个 token 的潜向量整个被拿掉/换掉,它仍有 93% 的概率被还原,而其他位置毫发无损。这种"缺失自身潜向量"的损伤里,块边界(位置 0 与 7)最弱(0.890 / 0.868,中段可达 0.96–0.98);而在整体加噪(§4.2)下位置差异并不这么规律(位置 1、7 略低,中段略高),所以边界劣势主要体现在"自身位置失效"这一类损伤上。这解释了为什么加噪曲线这么平:**单个位置甚至整块的各向同性噪声,都只是在一份高度冗余的联合码上做局部扰动**。

## 5. 失败案例长什么样(取自 `code/results/roundtrip.json` 的 exemplars)

- **λ = 0.401(z = −1.5,还原率 93%)**:错误全是"语义近邻 + 功能词"级别的替换,句子仍然通顺——`Unified temporal and spatial calibration for multi-sensor systems.` → `Unified temporal and spatial compensation for multi-core systems.`;`An internal-function node records…` → `An external-function node records…`;`GPT-4o, Llama-3.1-405B, …` → `GPT IVo, Llama (3.1-405C, …`;问号被吞掉。
- **λ = 0.310(z = −2.0,还原率 63%)**:骨架还在,拼写开始崩——`Grommelt, P., Weiss, L., Pfreundt, F.J., Keuper, J.: Fake or jpeg?` → `Lobnss, P., Webeis, L., Cfreanda, F.J., Keausbed, J.. be across or j French`;长句里人名/符号大面积失真但句法结构保留。
- **σ_e = 1.0**:选出的样例仍然逐字全对(98.85% 的整体还原率意味着错误集中在少数 token 上)。
- **换块对照**:输出的是另一段同样流畅、但内容完全无关的英文——这正是它该有的行为,也说明 100% 的还原率来自真实信息而非先验补齐。

## 6. 给报告的三条可引用结论

1. **"8-token 连续潜是一个有损通道"不成立**(至少在 decoder 侧):真实权重、真实英文正文、1,082 个块上干净 round-trip 逐 token 100% 还原,CE ≈ 7e-5 nats。补充证据:字符级编辑距离与"只用 tokenizer 往返"的基线**完全相等**(0.91%)。
2. **瓶颈在 flow backbone,不在 decoder**:decoder 能容忍归一化空间误差 std ≈1.0(信号尺度的 1.24 倍)而仍 98.85% 还原,σ_e=1.5 才跌到 86.7%。所以任何"文本生成质量差"的归因都应落在 flow 的端点预测精度上,而这条曲线给出了定量的容忍预算。
3. **机制是联合编码**:8 个位置共同编码 8 个 token,单点(甚至整块的一部分)信息由其余位置补足(leave-one-out 93%),块边界位置最脆弱。这也意味着报告的"容量"讨论不应按"每 token 512 维"来算,而应按"每 8 token 一份冗余的 4096 维联合码"来理解。

## 7. 局限(读数字前请先读这一节)

- **没有 flow backbone 权重**(按要求未下载 3.28 GB ×2),所以 flow 的实际端点误差**没有被测量**;§4.3 给的是"decoder 能容忍多少误差"的上限预算,不是"flow 实际有多少误差"。要把两者接起来,需要单独下载 backbone 跑一次生成并统计 `‖x̂₀ − x₀‖`(明确不在本次任务范围内)。
- 各向同性高斯误差**不等于** flow 的误差结构:flow 的误差通常是有偏、低维、贴着数据流形方向的,可能比各向同性噪声更致命(落在判别方向上)也可能更友好(落在 decoder 忽略的方向上)。这条曲线只能当预算上限。
- §4.3 的扰动是块内 8 个位置同时加独立噪声;若 flow 误差在块内高度相关,真实情形会与曲线不同。
- §4.6 的解读混合了两件事:"T5 的上下文编码把 token i 的信息写进了兄弟位置"与"英文本身从 8 token 窗口就可预测";用发布的这个 decoder 无法把二者分开(要分开需要另训一个只吃上下文的对照 decoder)。
- 样本域是论文正文句子 + 手工句;decoder 的训练域是 ultrafineweb 混合语料,域不完全一致(这只会让结论偏向保守)。
- 样本最长 181 token,没有触发 512-token 记录的截断分支;块大小只测了论文主配置的 8(代码里还允许 16,未测)。
- 权重是 BF16 EMA 快照载入 fp32 参数,CPU 单线程;没有做数值精度敏感性分析。
- `special_predictions` 少量出现在内容位置上,是 T5 的 `<unk>`(id 2)被正确预测所致,不是异常。

## 8. 复现

```bash
# 环境(独立 venv,不动其它子代理的 .venv-probe)
cd code && uv venv .venv-mmf --python 3.12
uv pip install -p .venv-mmf torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv-mmf transformers safetensors numpy matplotlib huggingface_hub

# 权重
hf download hustvl/Multimodal-Flow "Text Decoder/text_decoder.safetensors" --local-dir code/mmf-assets/hustvl-mf
hf download hustvl/Multimodal-Flow "MF/pretrain/config.json"            --local-dir code/mmf-assets/hustvl-mf
hf download google-t5/t5-small                                         --local-dir code/mmf-assets/t5-small

# 探针(全量约 11 分钟,CPU 单线程)
code/.venv-mmf/bin/python runs/2026-10-12-cyfa-mmflow-scorecenter-dreamtrue-art/multimodal-flow-2609.40362/code/roundtrip.py
# 只重画图(从已存的 json,不加载模型)
code/.venv-mmf/bin/python .../code/roundtrip.py --figure-only
```

产物:`code/results/roundtrip.json`(全部指标 + 236 条逐样本记录 + exemplars)、`code/fig-roundtrip.png`(四联图,另有一份同样的图放在运行包 `assets/mmf-roundtrip.png` 供报告引用)。`--max-paper-samples` / `--seed` / `--papers-root` / `--decoder-weights` / `--t5` / `--threads` 均可覆盖。
