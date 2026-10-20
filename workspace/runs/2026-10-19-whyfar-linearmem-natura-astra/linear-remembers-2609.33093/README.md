# 线性注意力记忆的因果干预复现:HLAR(arXiv 2609.33093)

本线程用**完全独立的实现**在论文指定的 GDN-340M checkpoint 上复现 "How Linear Attention Remembers" 的核心因果链,并追加一个论文没有回答的问题:因果 hub 头是否就是门控半衰期最长的头。

**一句话结论:事实确实被写进了循环状态,也按门控半衰期保留了下来,但这个 checkpoint 的 LM head 几乎读不出状态内容。** 在 512 filler 之后,最长寿的 L11H2 仍保留 6.8% 的相对 donor−recipient 状态差,而论文选的 L20H2 只剩 1.1%(96 头中排第 49);把整个状态换成 donor,四选一 logit 只移动 0.0023。native 四选一准确率在 dev 与 confirmation 上都是 25.0%,模型对每个 source 都预测同一个词 "green"。因此论文的 J 热图、block/donor 行为效应、L20H2 的选择都无法在本 checkpoint 上复现;可复现的是 write→retain 两段,断在 read 段。

---

## 1. 设置

### 1.1 模型

`m-a-p/340M-20B-GatedDeltaNet-pure-baseline`,commit `14661a98768e1f5d5493ff7b18f2d6809717a4c8`,FP32,399.5M 参数,24 层 × 4 头 × 256 维 key/value(每头状态 256×256,共 96 头)。纯 torch 逐步递推(`code/model.py`),语义沿用 2026-10-08 STEPQuant 线程已对 fla 0.5.2 验证过的实现(logits 差约 1.1e-5)。本线程把 prefix 与 query token 拆开计算,拆分开销与整段前向的 logits 差 6e-6,保证"查询前状态置换"不需要重跑 prefix。

环境:RTX 3060 12GB,FP32,batch 16;batch 16 的 626-token 全序列前向约 10 秒。不 import fla。

### 1.2 数据与任务构造

WikiText-103 raw(`Salesforce/wikitext`):validation 流做 development,test 流做 confirmation。offsets 为 2000 + 1100i(i=0..15),每段先取 64 个自然 token 做 prefix。

每个 source 的布局:

```text
[64 token prefix] [K 个 key-value 绑定 × 3 次呈现] [512 filler] [ " {key}" ":" ]
```

绑定写成 `" {key}: {value}."`,4 个 token;查询写成 `" {key}:"`,最后一个 `":"` 就是论文所说的 one-token query。K=4 时四条绑定的 key 固定为 apple/table/river/mountain,value 从单 token 的 {red, blue, green, yellow} 中取。**每个 source 的 target value 按 red/blue/green/yellow 轮换(16 个 source 各 4 次),其余三条绑定拿到剩余三个 value 的随机排列**;这一点很关键——早期版本让 target 恒为 red,导致线性探针可以直接背下标签,所有"完美解码"都是假象。donor 序列只把 target 绑定三次出现中的 value 换成另一个随机 value,长度、key、其余绑定完全不变;recipient 与 donor 的 token 序列只在 3 个 value token 上不同。

四选一分数 = 最后一个 `":"` 位置上四个 color token 的 logit;连续量 = donor value 与 recipient value 的 logit margin。seed 20260910 + i(逐 source 加 i),offset 协议与论文一致。

论文没有给出绑定模板和 value 词表,这个格式是我们自己的合理重建;论文 D.2 明确提到 "colon template",所以它在论文用过的模板族里。我们另外试过 " The {k} is {v}."、" The color of the {k} is {v}."、" Q: {k} A: {v}." 等 5 种模板和多种查询位置,模型在所有格式下都给出同一个常数预测,因此 native 失败不是模板选择造成的。

### 1.3 计算与对照口径

所有实验在 dev(validation)上做;baseline、screen、retention 另在 confirmation(test)上重跑。由于 native LM head 在 chance 水平,行为指标(四选一准确率、donor 转移率、J)没有可测效应;我们另外报告**状态层面的测量**,它不依赖 LM head:

- `rel`:查询前状态相对 donor−recipient 差,`||S_donor − S_recipient||_F / ||S_recipient||_F`,逐头、逐 source 平均。
- 写干预后的 `d_recipient = ||S_after − S_recipient||`、`d_donor = ||S_after − S_donor||`、`ratio = d_recipient / (d_recipient + d_donor)`。ratio > 0.5 表示干预后的状态离 donor 更近。

这些量测的是 write/retain,不是 read;它们能回答"事实有没有进状态、留了多久",不能替代论文的行为读出。这一点在结论里反复标注。

---

## 2. 逐项实验与论文对照

| # | 实验 | 论文(GDN-340M) | 本复现(dev / test) | 判定 |
|---|---|---|---|---|
| 1 | native 四选一准确率 | 46.9% | **25.0% / 25.0%**,16/16 预测 "green" | 未复现;chance 25% |
| 2 | 头筛选 J(查询前状态置换) | L20H2,16/16 source 最大 | max\|J\| = 0.0013 / 0.0038 logits;L20H2 J = −0.0012 / −0.0005;dev 第一 L7H2(+0.0013),test 第一 L11H2(+0.0038) | 未复现;L20H2 不是 hub |
| 3 | fact-span 写置零 | 召回掉 36–79pp | native acc 25.0%,margin +0.07553→+0.07567(Δ+0.0001);L11H2 状态 rel 0.1096,ratio 0.524(与 recipient/donor 等距) | 行为 null;状态层面必要性可见 |
| 4 | 写入换成 donor 写 | donor 答案率 68–90% | native acc 25.0%,donor 答案率 43.8%(= donor 恰为 "green" 的 base rate,不是转移),margin Δ+0.0011;L11H2 状态 d_donor 0.056 vs d_recipient 0.209,ratio 0.795 | 行为 null;状态层面内容特异性很强 |
| 5 | 查询前状态置换 | 纯 GDN 里 essentially full donor effect | 全 96 头状态置换:margin +0.0755→+0.0779;donor 自己跑一遍也是 +0.0779;四候选 logit 的 mean\|Δ\| = 0.0095 | 状态置换忠实恢复 donor 轨迹,但 donor−recipient 本身只有 0.0023 logit |
| 6a | 负载 K=1→16(定长) | 掉 30–40pp | native acc dev 全 25.0%,test 除 K=8 的 31.3% 外全 25.0%;L11H2 状态 rel 0.1045→0.0625(−40% 相对) | 行为 null;状态层面有负载效应 |
| 6b | 距离 128→1024(K=4) | 仅温和下降 | native acc(dev)每个距离都是 25.0%;L11H2 状态 rel 0.1828(0)→0.0683(512)→0.0426(1024) | 行为 null;状态层面距离衰减明显 |
| 7 | τ 交叉(本线程新增) | 未做 | L20H2 τ=6.17(96 头第 22);L11H2 τ=1582(第 1);ρ(τ, rel)=0.606 / 0.631 @512,0.840 / 0.844 @0 | 长寿命头确实承载事实,但不是 L20H2 |

### 2.1 native 读出的失败是决定性的

16 个 dev source 的 target value 是 red/blue/green/yellow 各 4 次,而模型的 argmax 在 16 个 source 上全部是 "green",所以准确率恰好 25.0%。test 同样 16/16 预测 "green",准确率 25.0%。这不是"偶尔答错":模型对上下文 value 的改变几乎无感。

量化"无感"的程度:

```text
recipient 干净 margin   +0.07553
donor 干净 margin       +0.07786      → donor−recipient = +0.00233
整状态全 96 头置换       +0.07786      → 与 donor 干净值完全一致
四候选 logit 的 mean|recipient−donor| = 0.00955(logit 量级)
```

也就是说:**状态置换是忠实的**——把 recipient 的状态换成 donor 的状态,logits 精确回到 donor 自己跑出来的值;但 donor 与 recipient 的 logits 本身只差 0.0023。记忆在状态里,读出增益接近零。这不是 swap 实现的问题,而是 LM head 对循环状态内容不敏感。

作为交叉检查,模型在简单 induction 提示上也不复制:`The apple is red. The apple is` 的 top token 是 "a"/"the"/"not",不是 "red";`red red red` 之后预测 "." 而不是 "red"。该 checkpoint 的 WikiText-2 PPL 约 990(STEPQuant 线程已记录),是一个弱 LM。论文的 46.9% 很可能来自与我们的模板不同的任务构造,或来自该 checkpoint 的另一个训练状态;我们无法用公开 commit 复现。

### 2.2 头筛选 J:没有集中 hub

按论文式(6)的文字("donor-versus-recipient logit margin 的平均变化"),我们取

```text
J(l,h) = mean_i [ (z_d − z_r)_after_swap − (z_d − z_r)_clean ]
```

dev 上 96 个头的 J 全部落在 ±0.0013 内(std 0.0003),test 上最大也只有 +0.0038(L11H2)。作为参照,干净 margin 的绝对值是 0.0755,单头 swap 能改变的量不到它的 5%。论文的 L20H2 在 dev 上 J = −0.0012(排在中下),test 上 −0.0005,不是 hub。

论文式(6)字面上是一个**乘积**,而不是差分。我们照字面也算了一遍:`mean[(z_d−z_r)_after × (z_d−z_r)_clean]`。96 个头全部落在 2.005–2.007(dev)/2.477–2.480(test),头部之间的差异约 0.1%,不构成有意义的排序(JSON 里的 `J_mult` 字段)。这个 ≈2 的常数可由逐 source 统计完全解释(`probe_margins.py`,CPU 复算,与 GPU 原始结果差 <2e-6;逐 source 值在 `code/results/probe_{validation,test}.json`):干净 margin 逐 source 散布在 −2.6~+1.8(dev:均值 0.076、标准差 1.41),其平方均值恰为 2.006(test:0.147/1.57/2.479);分解 mean(m·m₀)=mean(m₀²)+mean((m−m₀)·m₀) 里,唯一带头信息的第二项 ≤0.002。乘积口径量到的是逐 source margin 的能量而非头效应,因此正文采用差分口径,并在 `code/results/screen_*.json` 里保留两种口径。

### 2.3 写干预:状态层面是干净的,行为层面是零

在 L11H2(状态层面的事实载体)上,把三次呈现的 12 个 token 的写项做四种处理:

| 条件 | 状态 rel | d_recipient | d_donor | ratio | native acc | native margin |
|---|---|---|---|---|---|---|
| clean | 0 | 0 | 0.2164 | 0 | 25.0% | +0.0755 |
| block(置零) | 0.1096 | 0.3470 | 0.3147 | 0.524 | 25.0% | +0.0757 |
| donor(换 donor 写) | 0.0659 | 0.2088 | **0.0561** | **0.795** | 25.0% | +0.0766 |
| random(等范数随机) | 0.2112 | 0.6687 | 0.6523 | 0.506 | 25.0% | +0.0608 |

读法:block 把 recipient 的写拿掉,状态停在 recipient 与 donor 的中间(ratio 0.524);donor 写把状态送到 donor 附近(d_donor 0.056,ratio 0.795),而且总扰动(rel 0.066)只有等范数随机对照(0.211)的三分之一——这是内容特异性的直接证据。random 把状态推到离两者都很远的地方(两个距离都约 0.65),ratio 回到 0.5。

同样的干预在 L20H2 上**几乎不动状态**(d_recipient ≈ 2.5e-10,d_donor 停在 clean 的 0.0177):换掉 fact-span 的写对查询边界状态的影响小于 1e-8。L20H2 的 donor−recipient 状态差本身是 0.0177,但其中来自 fact-span 写的部分已经不可测;它的 τ_eff 在 512 filler 的序列上约 147,说明平均半衰期不能当作保留时间用,门控在 filler 段的局部值、以及后续写对目标方向的重叠,才是决定保留的因素。

行为层面四个条件的 native accuracy 都是 25.0%,margin 的变化只有 +0.0001(block)和 +0.0011(donor),远小于 0.0023 的 donor−recipient 干净差。写干预在状态里成立,在 logits 上不成立。

零位移 sanity:把 donor 状态换成 recipient 自己的状态(L11H2/L20H2/L17H3),logits 与 clean 的最大差是 0.0;把 recipient 自己的写注入 fact span,状态(相对 prefix 边界)与 logits 的最大差也是 0.0(`code/results/sanity.json`)。干预代码没有引入 off-target 位移。

### 2.4 负载 vs 距离:状态层面距离的衰减不比负载弱

- 定长 626,增加 K:K=1 时 L11H2 的 rel = 0.1045,K=2/4/8/16 为 0.0869/0.0683/0.0633/0.0625。负载让 target 的相对状态签名下降约 40%,但在 K=4 之后基本平台。
- 固定 K=4,增加 filler 距离:0/64/128/256/512/1024 对应 0.1828/0.1461/0.1236/0.0961/0.0683/0.0426。距离让签名掉到原来的 23%。

在**状态层面**,距离的衰减比负载更强,这与论文"负载而非流逝距离主导遗忘"的方向不一致。但这不是对论文的反驳:论文测的是四选一召回,而我们的 native 读出在 chance,无法测召回;状态相对差只是"记忆还在不在"的代理,不等于"读不读得出"。更稳妥的表述是:本 checkpoint 的写/留两段支持论文的"负载会压缩记忆",但在状态签名上距离同样是强一阶因素,不能支持"距离影响很小"。这个差异应在报告里标注为"代理指标与行为指标的不可比"。

### 2.5 预查询状态置换:状态对了,读出没跟上

全 96 头、全 24 层的状态置换让 logits 精确等于 donor 自己跑出的 logits(+0.07786 对 +0.07786),说明 swap 没有实现错误。但 donor 与 recipient 的干净 logits 只差 0.0023,四候选的 mean|Δ| 是 0.0095。单头 swap 的 J 在 ±0.004 内,与"没有任何单头是 hub"一致。论文的 "essentially full donor effect" 在本 checkpoint 上退化为 "full swap 恢复了一个本来就只差 0.0023 的轨迹"。

---

## 3. τ 交叉:记忆确实住在慢头里,但不是论文选的那颗

τ 取 2026-10-08 STEPQuant 线程在同一个 checkpoint 上算出的逐头门控半衰期(`probe_a_tau.npy`,WikiText-2,24×4)。同时我们也在本轮序列上算了 `τ_eff = ln2 / (−mean g)`(只作对照,因为门控是输入依赖的)。

| 头 | τ (WikiText) | τ 排名/96 | τ_eff @512 | rel @0 | rel @512 | rel 排名 @512(dev/test) |
|---|---|---|---|---|---|---|
| L11H2 | **1582.46** | **1** | 50883 | 0.1828 | **0.0683** | **1 / 1** |
| L7H2 | 989.42 | 2 | 25220 | 0.1287 | 0.0423 | 2 / 2 |
| L14H2 | 94.76 | 4 | 2934 | 0.1770 | 0.0210 | 3 / 4 |
| L17H3 | 44.40 | 6 | 1061 | 0.1757 | 0.0183 | 5 / 5 |
| L20H2(论文 hub) | **6.17** | **22** | 147 | 0.0350 | 0.0111 | 49 / 53 |

- **ρ(τ, rel)= +0.840(d=0)、+0.684(64)、+0.650(128)、+0.635(256)、+0.606(512)、+0.615(1024)**(dev);test 为 +0.844 / +0.630 / +0.654 / +0.637 / +0.631 / +0.602。全部 p < 1e-9。
- 用 τ_eff 替代 τ,ρ 略高(+0.610 @512 dev),方向一致。
- 距离 0 时,L11H2/L14H2/L17H3 三者几乎并列(0.183/0.177/0.176),短延迟记忆可以由较快的头承载;到 64 token,L17H3 已经掉到 0.066,到 512 只剩 0.018。**长延迟记忆集中在最慢的头**。
- 论文的 L20H2 的 τ 排名是 22/96(第 77 百分位),不算极端长寿;它在 512 的保留排名是 49/96(dev),在 96 个头里属于下半区。**"记忆住在遗忘慢的头里"这个命题成立,但论文选的 L20H2 不是那个头。**

这回答了本线程新增的问题:因果 hub 与 τ 的交叉不是"hub 恰好长寿",而是"长寿头才是长延迟的事实载体";L20H2 的 τ 和保留量都不支持它是 512-filler 任务的主载体。论文的 L20H2 更可能是短/中延迟或不同任务格式下的选择,而不是长延迟 recall 的普遍 hub。

---

## 4. 偏差来源

1. **Checkpoint 本身是弱 LM,且 native 读出在 chance。** 这是最大的偏差。WikiText-2 PPL ~990,16/16 预测同一个词,简单 induction 提示不复制;四选一 25.0%,donor−recipient logit 差 0.0023。论文的 46.9% 无法用公开 commit 复现。任何依赖 LM head 的行为指标(J、block/donor 转移率、load/distance 准确率)在本 checkpoint 上都退化为 null。另一个可能的解释:如果论文的 target value 在 source 间不是均匀的、而是偏向模型先验(本模型是 "green"),一个常数预测器就能拿到接近 46.9% 而不需要 recall。论文说候选值是 balanced,但没有说正确值在 source 间 balanced;附录没有给出这一点,我们无法验证。
2. **任务格式未被论文完全指定。** 绑定模板、value 词表、query 位置、512 filler 与绑定的先后顺序都需要重建。我们试了 5 种模板 + 3 种查询位置,模型都给出常数预测,所以 native 失败不是单一模板的锅;但换一个模板/词表可能改变绝对准确率。colon 模板是论文 D.2 明确用过的。
3. **式(6)的乘积口径。** 按字面算是乘积,值是由逐 source 干净 margin 的平方均值决定的头无关常数(全头 2.005–2.007;mean(m₀²)=2.006,逐 source 标准差 1.41 远大于均值 0.076,见 `probe_margins.py` 与 `code/results/probe_*.json`),不能当排序分数;正文用差分。两种口径都保存在 `code/results/screen_*.json`。
4. **状态层面指标是本线程的适配,不是论文的行为读出。** `rel`、`d_recipient/d_donor`、`ratio` 测的是"事实在不在状态里",不能证明"模型能不能答对"。论文的 write→retain→read 链条里,我们只复现了前两段。
5. **只测了 GDN-340M。** 没有 GLA-340M、GDN-1.3B、hybrid 或 Qwen3.5;跨模型泛化未验证。
6. **seed 与 offset。** 用 20260910 + i(逐 source 加 i)而不是单一 20260910;offset 2000+1100i 与论文一致。target value 按 source 轮换,避免标签泄漏;其余 value 的排列由 seed 决定。
7. **负载/距离的布局。** 负载扫描固定总长 626,通过缩短 filler 给新增绑定腾位置;距离扫描固定 K=4、增加 filler。论文的精确布局未给出,因此 6a/6b 的数值不能逐点对照,只能看方向。
8. **L20H2 的写干预状态差为零。** 这不是 bug:同一套干预代码在 L11H2 上产生了 0.11 的相对状态变化,在随机短序列上也能把状态清零;L20H2 的 fact-span 写对查询边界状态的可测贡献小于 1e-8。它说明"平均 τ"不能预测保留,门控在 filler 段的局部值与后续写的重叠才是关键。

---

## 5. 复跑

### 5.1 环境

下面的命令在 lab bench 的 `code/hlar-probe/` 里执行;线程包 `code/` 是同一批脚本的快照,把 `data/` 放进去、在包内 `uv venv .venv` 并安装同一组依赖后,命令相同(只把 `cd code/hlar-probe` 换成 `cd runs/2026-10-19-whyfar-linearmem-natura-astra/linear-remembers-2609.33093/code`)。

```bash
cd code/hlar-probe
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch transformers safetensors numpy scipy matplotlib pyarrow pillow
```

模型从 HF cache 读取:

```text
~/.cache/huggingface/hub/models--m-a-p--340M-20B-GatedDeltaNet-pure-baseline/snapshots/14661a98768e1f5d5493ff7b18f2d6809717a4c8/model.safetensors
```

如果本地没有,先 `hf download m-a-p/340M-20B-GatedDeltaNet-pure-baseline --revision 14661a98768e1f5d5493ff7b18f2d6809717a4c8`。

WikiText-103 的 validation/test parquet 放在 `code/hlar-probe/data/`:

```text
wt103_validation.parquet
wt103_test.parquet
```

### 5.2 跑实验

```bash
cd code/hlar-probe
.venv/bin/python run_experiments.py --step baseline  --split validation
.venv/bin/python run_experiments.py --step baseline  --split test
.venv/bin/python run_experiments.py --step screen    --split validation
.venv/bin/python run_experiments.py --step screen    --split test
.venv/bin/python run_experiments.py --step retention --split validation
.venv/bin/python run_experiments.py --step retention --split test
.venv/bin/python run_experiments.py --step write --head 11,2
.venv/bin/python run_experiments.py --step write --head 20,2
.venv/bin/python run_experiments.py --step load --split validation
.venv/bin/python run_experiments.py --step load --split test
.venv/bin/python run_experiments.py --step distance --split validation
.venv/bin/python run_experiments.py --step load_state --head 11,2
.venv/bin/python tau_cross.py
.venv/bin/python make_summary.py
.venv/bin/python make_figures.py
```

`probe_margins.py` 是事后补的裁决探针(CPU 即可,两个 split 约 11 分钟):复算 screen 的逐 source 干净 margin 与两种口径,验证乘积口径的平台值等于 mean(m₀²)。

所有实验都是确定性的;单卡 3060 上整套约 20 分钟。`code/results/` 里的 JSON/CSV/NPY 是脚本直接产物;图与运行级 `assets/hlar_*.png` 是同一批,线程目录不再另存。

### 5.3 读结果

- `code/results/head_table.csv`:96 行,逐头 τ、τ 排名、τ_eff@512、rel@0、rel@512、rel 排名(dev/test)、J(dev/test)。
- `code/results/summary.json`:所有实验的原始指标。
- `code/results/readout_gain.json`:clean/donor/全状态置换的 margin 与四候选 logit 差。
- `code/results/state_retention_{validation,test}.json`:逐距离的 24×4 rel 与 τ_eff。
- `code/results/write_validation_L11H2.json` / `..._L20H2.json`:写干预的状态与行为指标。
- `code/results/probe_{validation,test}.json`:逐 source 干净 margin(16 浮点)与 96 头 J/J_mult 的 CPU 复算,含与 GPU 原始结果的最大绝对差。

---

## 6. 产物

```text
runs/2026-10-19-whyfar-linearmem-natura-astra/
├── linear-remembers-2609.33093/
│   ├── README.md
│   ├── code/            # model.py, task.py, run_experiments.py, tau_cross.py,
│   │                    # probe_margins.py, make_summary.py, make_figures.py; data/ 放 parquet
│   │                    # results/ 在 code/ 下:JSON / CSV / NPY(含 probe_a_tau.npy)
└── assets/
    ├── hlar_baseline.png
    ├── hlar_j_heatmap.png
    ├── hlar_retention_distance.png
    ├── hlar_tau_cross.png
    ├── hlar_load.png
    └── hlar_write.png
```

图注:

- `hlar_baseline.png`:native 四选一准确率,chance 25%、本 checkpoint 25.0%、论文 46.9%。
- `hlar_j_heatmap.png`:dev/test 的 96 头 J 热图;L20H2 用方框标出,L11H2 用星号标出;色标范围 ±0.004 logit。
- `hlar_retention_distance.png`:五个代表头的相对状态差 vs filler 距离;L11H2 在 512/1024 仍最高,L17H3 在 64 后快速衰减,L20H2 全程最低。
- `hlar_tau_cross.png`:96 个头的 τ vs rel,d=0(短延迟)与 d=512(长延迟)两栏;ρ=+0.84 与 +0.61。
- `hlar_load.png`:native 准确率 vs K(平在 25%)与 L11H2 状态 rel vs K。
- `hlar_write.png`:L11H2 的写干预状态距离与 native 行为;donor 写落到 donor 附近,行为仍是 25%。
