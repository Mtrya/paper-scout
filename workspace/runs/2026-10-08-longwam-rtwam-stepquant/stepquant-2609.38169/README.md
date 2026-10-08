# 真实 GDN 权重上的 STEPQuant 机制核验(arXiv 2609.38169)

本轮巡航 STEPQuant 线程的实验记录。目标:在一个真实训练的 Gated DeltaNet(GDN)checkpoint 上,用 CPU 逐条核验论文 Appendix B 的三个机制性论断——(A) 门控半衰期与 INT6 状态累积误差强相关(论文 ρ≈0.80,最长寿 1/4 头占 52.5% 误差);(B) delta 规则的自纠反馈有实作用:换成 exact-read oracle 后读出误差放大 26.82×;(C) 均匀 INT6 状态量化在长解码下崩盘(论文 Qwen 长生成均值 80.60→45.04),而寿命感知位宽分配能救回。

一句话结论:**探针 A 精确复现且误差集中度比论文更强(ρ=0.809 vs 论文 0.8004;最长寿 1/4 头占 97.2% vs 52.5%);探针 B 方向复现但倍数偏弱(7.55×/7.73× vs 论文 26.82×/18.65×);探针 C 未复现 INT6 崩盘(excess NLL≈+0.001)——而这与论文自身理论自洽:本模型门控寿命太短(中位 2.8 token),量化误差不跨步累积,INT6 噪声被门控即时冲刷,既崩不了,寿命感知分配也无收益可捞。**

---

## 1. 设置

### 1.1 模型与 CPU 推理路径

测试模型:`m-a-p/340M-20B-GatedDeltaNet-pure-baseline`(HF, commit 14661a9, FP32 safetensors)。这是 fla 框架(flame)训练的纯 GDN 基线:24 层全 GDN、hidden 1024、4 头 × head_dim 256(每头状态 256×256,共 96 头)、short conv 核 4、vocab 32000(Mistral 词表)、SwiGLU 中间层 2816、无偏置、untied lm_head。权重布局是 fla 标准(`attn.{q,k,v}_proj`、`a_proj`、`b_proj`、`A_log`、`dt_bias`、`g_proj`、`o_norm`、`o_proj` 等),不需要 trust_remote_code(transformers 5.19 内置 `gated_deltanet`)。checkpoint 里的 `attn.D` 在全部 24 层精确为 0(死参数),忽略。

fla 的 Triton kernel 在无 GPU 机器上跑不了(import 后初始化即崩),所以手写了一个纯 torch 的逐 token 递推实现 `code/gdn_model.py`,语义逐条对照 fla 0.5.2 源码(并核对过 0.1/0.3.0 的历史差异)。忠实性证据:`code/cross_check.py` 用 monkeypatch 把 fla 官方模型的全部 triton 算子换成 naive 后加载官方权重,与手写模型在相同输入上 logits 最大差 **1.1e-5**(均值 1.1e-6)、NLL 完全一致。所有探针只依赖这套验证过的递推;纯探针脚本不 import fla(triton 崩),只有 cross_check.py 需要。

### 1.2 文本与协议

文本:WikiText-2 test(Salesforce/wikitext),3 段 × 2048 token(`prepare_data.py` 固化)。`run_capture.py` 先做一遍 FP32 teacher-forced 前向,缓存每层每步的 q/k/v/β/g 与参考 NLL(缓存 ~1.7GB,不进包,可重跑再生)。探针 A/B 离线重放缓存输入流(FP32 参考与量化路径吃**完全相同**的输入,与论文 App.B 的 replay 协议一致);探针 C 在完整模型前向上 monkeypatch 递推、每步量化状态。

量化器与论文 App.B.5 一致:逐 key-row 对称整型量化,scale = absmax/(2^{b-1}−1),clamp 到 [2^-14, 65504],码值域 ±(2^{b-1}−1)。

与论文协议的出入(逐条标注):

| | 本实验 | 论文 |
|---|---|---|
| 模型 | 340M 纯 GDN,96 头 | Qwen3.8-27B(混合,2304 个 GDN 头)/ Kimi-Linear-48B(KDA) |
| 文本 | WikiText-2 test,3 段 × 2048 步,全部 96 头 | 校准 WikiText、误差流 C4;B.1/B.5 用 4 条 C4 轨迹 × 8192 步 × 32 个固定头;B.2 用 2048 预填充 + 6144 强制解码 |
| 探针 B 注入 | step 256 单点注入,与论文一致 | 同 |
| 探针 C | 零初始状态、教师强制 2048 步 | 主表是 65K token 自回归生成;B.2 是教师强制但带原生预填充 |

### 1.3 必须先交代的意外:这个 checkpoint 是弱 LM

该 checkpoint 的语言建模质量远低于同规模预期:WikiText-2 test 教师强制 NLL 6.75–7.07/段(PPL 857–1177),fineweb-edu NLL 5.94(PPL 378),贪婪生成退化为复读。两种独立实现(手写 + fla 官方 naive)给出逐位一致的 logits,说明这是 checkpoint 本身的性质(或 flame 训练侧与公开 fla 之间某种不可考的差异),不是实现 bug。备用候选 linear-moe-hub/Gated-Deltanet-340M 更差且权重布局不同(fused gate_up),弃用。

这不妨碍三个探针的有效性:A/B 是**输入条件化重放**——重放的是模型真实权重产生的激活流,测的是门控/键/值的统计结构,不要求模型是好 LM;C 是相对的 excess NLL,只要 NLL 差异可测(实测段间差异 ±0.01–0.03,可测)就有效。但它是解读探针 C 的必要背景,故单列。

---

## 2. 逐探针判定

| 探针 | 判定 | 一句话 |
|---|---|---|
| A 半衰期 ↔ 累积误差 | **复现(更强)** | ρ=0.809 vs 论文 0.8004;误差集中度 97.2% vs 52.5% |
| B exact-read oracle | **部分复现** | 方向对(去掉自纠误差放大),倍数 7.5×/7.7× vs 论文 26.8×/18.7× |
| C INT6 崩盘 / 寿命感知 | **未复现(但与论文理论自洽)** | 均匀 INT6 excess NLL +0.001,寿命感知 +0.006,都在噪声级 |

### 探针 A:门控半衰期 ↔ INT6 累积状态误差 —— 复现,且集中度比论文更强

对每个 (层, 头) 重放 FP32 参考与每步 INT6 量化的递推(相同输入),单元 u = 一个头(GDN 每头标量门),半衰期 τ = ln2 / (−E[log α]),误差度量 D_u = Σ_t ||E_t||_F²(3 段 × 2048 步累计)。

| 量 | 本实验(96 头) | 论文(Qwen,2304 头) |
|---|---|---|
| Spearman(τ, D_u) | **0.809**(p=2.2e-23) | 0.8004(KDA 0.8017) |
| Spearman(τ, 末端相对误差) | 0.724(p=8.0e-17) | — |
| 最长寿 1/4 头占累积误差 | **97.2%** | 52.5%(KDA 78.8%) |
| 最短寿 1/4 头占比 | 0.2% | — |
| τ 分布(min/中位/max) | 0.46 / 2.79 / 1582 token | — |

末端相对状态误差按 τ 分位组:最短寿组均值 7.5%、中间组 10.7%、最长寿组 24.8%(单头最大 77.6%)。图:`code/figures/fig_halflife_error.png`(τ-D_u 散点,与论文 Fig.1b 同构)、`fig_error_growth.png`(分组误差增长曲线)。

**解读**:单调相关几乎一比一复现;集中度比论文高近一倍,原因是本模型的 τ 分布尾部极端(中位 2.8、最大 1582),少数长寿头吸走了几乎全部误差质量。论文核心论断"误差集中在长寿记忆"在本模型上不但成立,而且更锋利。

### 探针 B:exact-read oracle —— 方向复现,倍数偏弱,且逐头倍数与 τ 只有弱相关

严格按论文 App.B.1 的 matched single-injection:两条路径吃相同 FP32 输入流,step 256 对状态注入**一次**完全相同的量化误差,之后不再量化。native 路径保留 delta 回缩(误差参与自纠);oracle 路径把 delta 残差里的读出换成对参考状态的精确读出,S̃_t = D_t S̃_{t−1} + βk(vᵀ − kᵀ D_t S_ref_{t−1}),于是注入误差只能经门控衰减(E_t = D_t E_{t−1})。指标 = 注入后所有步的平方读出误差总和之比(96 头 × 3 段)。

| | INT6 | INT8 |
|---|---|---|
| oracle/native 总比值 | **7.55×** | **7.73×** |
| 论文(Qwen) | 26.82× | 18.65× |
| 逐头比值范围 | 1.05–21.4× | 1.04–22.3× |
| Spearman(τ, 逐头比值) | 0.271(p=7.7e-3) | 0.295(p=3.5e-3) |
| 分位组中位比值(短/中/长) | 1.04 / 1.15 / 1.26 | 1.04 / 1.13 / 1.27 |

图:`code/figures/fig_oracle.png`(注入后累积读出误差曲线,oracle 明显高于 native 且都不回落——门控中位寿命 2.8 步,两条曲线的误差都在几十步内被衰减到 plateau)。

**解读**:机制方向复现——拿掉 delta 自纠,读出误差显著放大;但总比值只有论文的 ~1/3.5,且 INT6≈INT8(论文是 INT6>INT8)。两个本模型特有的结构事实:(i) 总比值由少数绝对误差大的头主导(如层 17 头 3,τ=44,单头贡献了 native 总误差的大头),而不是长寿头;(ii) 逐头比值与 τ 只弱相关(ρ≈0.27),最大比值 21.4× 出现在 τ=106 的头(层 3 头 1)——自纠的"收益倍数"在本模型里并不主要分布在最长寿的头。**实现陷阱记录**:初版 oracle 误用了更新后的 S_ref_t 而非 S_ref_{t−1},比值虚高到 42742×;修正后落在 7.5×。这个 off-by-one 极易犯,论文公式里下标稍不留意就会踩进去。

### 探针 C:均匀 INT6 vs 简化寿命感知分配的 excess NLL —— 崩盘未复现,分配无收益

monkeypatch 全模型前向,每步状态更新后量化(逐行 scale)。寿命感知变体按探针 A 的 τ 全局排序:最长寿 24 头→8 bit、中间 48 头→6 bit、最短寿 24 头→4 bit(均值恰 6.0 bit)。**简化掉了** STEPQuant 的 key-row 双轴拟合、FP16 pivot 与失真加权优化分配——这里测的是"寿命轴"这一条思想,不是完整方法。教师强制,WikiText-2 test 3 段,零初始状态。

| 变体 | NLL | excess NLL(相对 FP32 6.8987) |
|---|---|---|
| 均匀 INT6 | 6.8999 | **+0.0012**(逐段 −0.015 / +0.030 / −0.011) |
| 寿命感知 8/6/4 | 6.9044 | **+0.0057**(逐段 +0.013 / +0.006 / −0.001) |

对照论文:Qwen INT6 的压缩读出 excess NLL 从前 256 步的 0.102 涨到末 256 步的 2.102(App.B.2),长生成主表 80.60→45.04;STEPQuant@6bit excess 仅 0.0011。

**判定:INT6 崩盘在本模型上没有复现;寿命感知分配也无可见收益(两者差异在段间噪声 ±0.01–0.03 之内)。** 图:`code/figures/fig_nll.png`(三条 NLL 曲线几乎重合)。

**解读(与论文理论自洽)**:论文自己的机制链条是"长寿记忆 ⇒ 误差跨步累积 ⇒ 读出损伤 ⇒ NLL 涨"。本模型门控衰减快(τ 中位 2.8 token),探针 A/B 都显示注入误差在几十步内被门控冲刷掉——误差根本不累积到能伤 NLL 的量级,INT6 自然不崩;没有可捞的损伤,寿命感知分配自然无收益。一个不协调点值得记下:探针 A 显示最长寿 1/4 头的状态相对误差达 25%,但 NLL 纹丝不动——**(猜想)** 该模型的长寿头对下一 token 预测的贡献弱,状态误差大不等于读出损伤大。另一个诚实的保留:探针 C 是 2048 步教师强制、零初始状态,而论文崩盘场景是 65K token 自回归生成(更长 horizon + 误差反馈进输入流),horizon 差 30 倍,自回归反馈可能放大误差——本轮没有测生成场景。

---

## 3. 综合解读

1. **论文的时间轴机制(τ ↔ 误差累积)是硬的、可移植的**:换一个模型族(混合 Qwen → 纯 GDN 340M)、换文本(C4 → WikiText-2)、头数缩到 1/24,Spearman 几乎不变(0.809 vs 0.8004),集中度甚至更强。STEPQuant 的寿命感知位宽分配的物理基础成立。
2. **但"误差累积 ⇒ NLL 崩盘"这一段是模型条件依赖的**:同一个机制在 Qwen3.8-27B(有大量长寿头)上推出 INT6 崩盘,在门控寿命中位 2.8 的 340M 上推出"INT6 无损"。两个结果都被论文自己的理论预言。实践含义:想把 STEPQuant 用在某个模型上前,先量 τ 分布——中位寿命短的模型根本不需要状态量化之外的任何花活,均匀 INT6 即可。
3. **delta 自纠的存在性复现、量级未复现**:oracle 比值 7.5× vs 26.8×,且本模型里比值与 τ 弱相关、总比值被少数大误差头主导。"自纠多大程度上撑起低比特状态量化"可能像崩盘一样依赖模型的门控/头结构,论文的 26.82× 不宜当作普适常数引用。

## 4. 局限

1. **弱 LM checkpoint**:绝对 NLL 高(WT2 PPL ~990),机制探针是相对测量故仍有效,但探针 C 的"零损伤"结论外推到强 LM 需谨慎。
2. **探针 C 协议与论文崩盘场景不同**:2048 步教师强制 + 零初始状态 vs 论文的 65K 自回归生成;horizon 短、无误差反馈回路。
3. **寿命感知变体是简化版**:只有按 τ 排序的三档分配,没有 key-row 双轴 scale 拟合、FP16 pivot、失真加权优化——测的是寿命轴思想,不是完整 STEPQuant。
4. **规模差**:96 头 × 3 段 × 2048 步 vs 论文 2304 头 × 4 轨迹 × 8192 步;单模型单文本。
5. 探针 A 的 τ 与误差在同一文本(WikiText-2)上估计,论文则校准(WikiText)与误差流(C4)分开;同文同测可能轻微高估 τ-误差对应(方向不影响单调性结论)。
6. 量化是仿真的(store-after-quantize),非 kernel 级;全部 CPU FP32 递推。

## 5. 产物清单

### 代码(`code/`)

| 文件 | 作用 |
|---|---|
| `gdn_model.py` | 纯 torch GDN 340M 推理 + 递推 + 量化器(语义对照 fla 0.5.2 naive 路径) |
| `cross_check.py` | 忠实性证据:monkeypatch fla 官方模型为 naive 算子后与本实现逐 logit 对比(max diff 1.1e-5) |
| `prepare_data.py` | WikiText-2 test 3 段 × 2048 token 固化 |
| `run_capture.py` | FP32 teacher-forced 前向,缓存每层每步 q/k/v/β/g 与参考 NLL |
| `probe_a.py` | 探针 A:半衰期 vs INT6 累积误差 |
| `probe_b.py` | 探针 B:matched single-injection oracle 对比 |
| `probe_c.py` | 探针 C:全模型均匀 INT6 vs 寿命感知 8/6/4 的 excess NLL |
| `make_figures.py` | 出 4 张图 |

### 图(`code/figures/`)

| 图 | 内容 |
|---|---|
| `fig_halflife_error.png` | τ-D_u 散点(论文 Fig.1b 同构),含 ρ 与集中度对照 |
| `fig_error_growth.png` | 按 τ 分组的误差随步数增长曲线 |
| `fig_oracle.png` | 单点注入后 native vs oracle 累积读出误差(INT6/INT8) |
| `fig_nll.png` | FP32 / 均匀 INT6 / 寿命感知的 NLL 曲线与 excess 柱状图 |

### 数据(`code/results/`)

`probe_a_summary.json` / `probe_a_perhead.csv`(96 头逐头 τ、D_u、末端相对误差)/ `probe_a_tau.npy` / `probe_a_errcurve.npy`(逐头逐步误差曲线)/ `probe_b_summary.json` / `probe_b_perhead_int{6,8}.npy` / `probe_b_curves_int{6,8}.npy` / `probe_c_summary.json` / `probe_c_nll_{fp32_ref,uniform6,lifetime_8_6_4}.npy`(三条 NLL 曲线)。激活缓存 `caps/*.npz`(~1.7GB)不进包,`run_capture.py` 可重跑再生。

### 复现

环境:`code/stepquant-probe/.venv`(uv 创建;torch 2.14.1+cpu、transformers 5.19.0、fla 0.5.2、numpy、scipy、matplotlib、datasets)。顺序:`prepare_data.py` → `run_capture.py` → `probe_a.py` → `probe_b.py` → `probe_c.py` → `make_figures.py`,全部 CPU 可跑。模型与 WikiText-2 路径从本地 HF cache 通配解析(模型取 `m-a-p/340M-20B-GatedDeltaNet-pure-baseline` 的本地快照,实测为 commit 14661a9),先 `hf download` 对应 repo 即可。注意纯探针脚本不要 import fla(triton 在无 GPU 时初始化即崩);只有 `cross_check.py` 需要 fla。
