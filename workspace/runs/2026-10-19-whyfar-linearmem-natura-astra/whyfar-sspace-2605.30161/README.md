# WFLU × S-Space:同一 Qwen2.5-VL-3B 上的空间表征交叉检验

本线程把两篇"探针式理解 VLM 空间表征"的论文放在同一个模型上对撞:

- **WFLU**(*Why Far Looks Up*, arXiv 2605.30161):把图像里"更靠上"当作"更远"的透视捷径。它测的是**关系判断的位移**:同一张图、交换问题里两个物体的角色,取最后一个 prompt token 的 hidden state 差 Δ,再看同一空间轴上的 Δ 方向是否一致(Coh),以及垂直轴与距离轴的 Δ 方向是否按透视规律耦合(VD-EI)。
- **S-Space**(arXiv 2610.13527):中间层存在一个三维线性子空间,物体名 token 的激活里可以线性读出水平/垂直/距离坐标。它测的是**单个物体 token 的绝对坐标**:对答案分数关于物体 token 激活求梯度得到三条轴,再用 `v_g^T(h_q - h_r)` 读关系。

表面张力是:WFLU 说距离轴在表征里最弱、且与垂直轴纠缠;S-Space 说距离坐标在线性子空间里清晰可读。两者可以同时为真,因为它们测的不是同一个东西——**绝对编码**与**关系读出**、**物体 token** 与**最后 token**、**全向量方向一致性**与**标量坐标可读性**。

## 1. 两篇论文的方法,精确版

### 1.1 WFLU 的 contrastive probing

对每个空间关系题,构造最小对照对:把问题里的两个物体互换,答案随之翻转。设 `h_q` 是问题 q 在层 L 的最后一个 token 的 hidden state,定义

```
Δ = h(swapped) - h(original)
```

对每条轴,把相反方向的样本 Δ 取负,使所有 Δ 指向同一方向,然后算平均成对余弦:

```
Coh_axis = 1 / C(n,2) * Σ_{i<j} cos(δ_i, δ_j)
```

VD-EI(论文 Eq. 3)用四个类别均值 Δ 的余弦:

```
VD-EI = 1/4 [ cos(μ_above, μ_far) + cos(μ_below, μ_close)
              - cos(μ_above, μ_close) - cos(μ_below, μ_far) ]
```

WFLU 在 EmbSpatial-Bench 上取每个方向最多 200 题,模型相关层 L* 由三条准则选出(三条轴 Coh 接近平台、VD-EI 稳定、避开最后几层)。Qwen2.5-VL-3B 的 L* = 28(/36),论文报告 CohH 0.367、CohV 0.293、CohD 0.043、VD-EI 0.457。

### 1.2 S-Space 的梯度轴与线性读出

对每条轴 g,定义答案对比分数 `s_g`(水平为 logit(right)-logit(left),垂直为 logit(above)-logit(below),距离为 logit(close)-logit(far))。对 query 和 reference 两个物体名 token 的激活求梯度,取角色反对称差:

```
a_p,g = (∇_{h_query} s_g - ∇_{h_reference} s_g) / 2
v_g   = unit(mean_p a_p,g)
```

在 5 个 prompt 模板、原序/交换序、6000 张 COCO 图上平均,得到每层三条单位轴。读出时只做前向:

```
c_t = V^T h_t,  V = [v_x, v_y, v_z]
m_g = v_g^T (h_q - h_r)
```

`m_g` 的符号给出 right/above/close 或 left/below/far。S-Space 选层标准是 COCO 验证集上三轴总体读出准确率最高。论文的逐轴证据(Figure 3,Spearman 相关)是:绝对坐标 H 0.789、V 0.774、D 0.521;成对坐标差 H 0.871、V 0.872、D 0.688。**距离轴在 S-Space 自己的数字里也是最弱的**,但仍有可观的相关性;而在 EmbSpatial/SpatialTunnel 的逐方向读出里,far 方向经常显著低于 close(例如 Qwen3.5-4B 在 EmbSpatial 上 far 47.5%、close 90.0%)。

这两组数字已经提示:两篇论文在"距离最难"上其实一致,分歧在于**距离信息是"不存在/被垂直替代"还是"存在但读出/使用得不好"**。

## 2. 实验设置

| 项 | 设置 |
|---|---|
| 模型 | Qwen2.5-VL-3B-Instruct(HF commit `66285546d2b821cf421d4f5eb2576359d3770cd3`,bf16) |
| 硬件 | RTX 3060 12 GB,单卡,prefill-only,batch 1;WFLU 前向峰值约 10.3 GB,S-Space 梯度峰值约 10.9 GB |
| 数据 | EmbSpatial-Bench TSV(`ch-min/EmbSpatial-Bench-tsv`,SHA256 前缀 `21e87cea7d30dd76`) |
| WFLU 采样 | 每个方向 200 题,seed 42(与官方 `extract_swap_features` 同规则) |
| S-Space 轴识别 | 主结果每个方向 2 题,seed 7,与 WFLU 200 题集合不相交;2 个模板 × 原序/交换序;bootstrap 200 次给轴不确定性 |
| 图像预处理 | `max_pixels=401408`(512×28×28),`min_pixels=256×28×28`;覆盖多数 640×480 原图,1296×968 图被降采样 |
| 层 | WFLU 全 36 层;S-Space 轴在 0,4,8,12,15,18,21,24,28,32,35 层识别 |
| 复现对照 | 官方 `probing.py` 在 2 题/方向的子集上另跑一次,验证 forward-only 与 generate+prefill hook 的 Δ 一致(详见 `code/results/official_delta_check.json`) |

## 3. 结果

### 3.1 WFLU 复现:方向一致性

1200 个对照对(每个方向 200 题),物体名 token 定位 0 失败。层 L*=28 上的结果与论文对比如下:

| 指标(L28) | 论文 Qwen2.5-VL-3B | 本次复现(200/方向) |
|---|---:|---:|
| CohH | 0.367 | **0.298** |
| CohV | 0.293 | **0.251** |
| CohD | 0.043 | **0.044** |
| VD-EI(论文 Eq. 3) | 0.457 | **0.439** |
| VD-EI(官方仓实现) | — | 0.463 |

距离轴的一致性几乎逐位复现(0.0436 vs 0.043),VD-EI 也在 0.02 以内。水平/垂直一致性比论文低 0.04–0.07,可能来自 512-token 的像素预算上限(论文用模型默认的更大分辨率)与采样噪声;方向排序、距离轴最弱、VD-EI 为正这三件事都成立。

这里有一个复现时容易踩的坑:官方仓 `compute_vd_ei_per_layer` 实际算的是 `cos(垂直符号校正均值, 距离符号校正均值)`,不是论文 Eq. 3 的四项平均。L28 上它给出 0.463,比 Eq. 3 的 0.439 高 0.024。两者排序一致,但报数时要注明用的是哪一个;本线程两个都存了。

把论文 Eq. 3 拆开看,L28 的四个耦合项是:

```
cos(μ_above, μ_far)   = +0.449
cos(μ_below, μ_close) = +0.436
cos(μ_above, μ_close) = -0.419
cos(μ_below, μ_far)   = -0.450
```

这正是透视捷径的符号模式:above 与 far、below 与 close 同向,交叉项反向。水平轴与另外两轴基本正交(cos ≈ 0.11–0.27),而垂直与距离的均值方向相差约 45°(cos ≈ 0.45),不是完全重合,但远不是独立。

同层 6×6 均值 Δ 余弦还暴露了一个关键事实:

```
cos(μ_left, μ_right)  = -0.993
cos(μ_above, μ_below) = -0.985
cos(μ_far,  μ_close)  = -0.808
```

**相反方向的对立性在距离轴上明显更弱**(-0.81 vs -0.99)。再看每个方向均值 Δ 的范数:水平 32.2、垂直 23.7、距离只有 6.0。距离不是没有方向,而是方向弱、样本噪声大。

这一点可以用一个不依赖 S-Space 的标量信噪比刻画。把每个样本的 Δ 投影到该轴的 canonical 均值方向(符号校正后),算均值/标准差(d′)与残差 RMS:

| 轴 | d′ = mean/std | 均值方向范数 | 残差 RMS |
|---|---:|---:|---:|
| 水平 | 1.75 | 32.2 | 43.1 |
| 垂直 | 1.78 | 23.7 | 36.5 |
| 距离 | **0.74** | **6.0** | 25.1 |

水平和垂直的单样本方向一致性接近(d′≈1.75),距离只有 0.74:**均值方向存在,但单样本几乎被残差淹没**。Coh_D 低是低信噪比的结果,不是"距离方向不存在"。这给 S-Space 的"绝对坐标可读"留出了空间——只要绝对编码走的是另一条更稳的通路。

PCA(L28,所有 Δ)也支持这个图像:PC1 解释 26.0% 方差,基本是水平轴;PC2 解释 14.8%,基本是垂直轴;距离类别均值在 PC2 上的载荷只有 3.5(垂直是 21.8),在 PC3 上只有 -1.8。距离位移没有自己的主成分,而是寄生在垂直主成分附近。

#### 复现保真度检查

用官方 `probing.py`(generate + prefill hook)在同一 2 题/方向的子集上另跑一次,与我们的 forward-only 提取对比:修正 `min_pixels` 后,L28 的 Δ 余弦相似度为 **0.998**,相对 L2 差 5.8%;L24 为 0.996,早期层差异较大(L0 约 0.73),但所有指标都在 L24–L28 的平台上计算。forward-only 没有改变被测量的对象。

### 3.2 S-Space 式检验:同一模型的绝对坐标读出

用同样的 EmbSpatial 题面构造 S-Space 的梯度轴:对答案对比分数关于 query/reference 物体名 token 的激活求梯度,取角色反对称差,在 2 个模板 × 原序/交换序上平均。受 GPU 竞争限制,主结果用 2 题/方向的 8 条梯度/轴(seed 7,与 WFLU 的 200 题集合不相交);另外用 bootstrap 重采样梯度记录,给出轴估计不确定下的读出置信区间。10 题/方向的扩展估计尝试过,但被另一线程的显存占用反复打断(见第 5 节),主结果保留 2 题/方向 + bootstrap。

在 11 个候选层上,读出准确率在 **L24** 达到峰值(总体 0.842):

| 层 L24 | 读出准确率 | bootstrap 95% CI(2 题/方向轴) |
|---|---:|---:|
| 水平 | **0.902** | [0.810, 0.915] |
| 垂直 | **0.958** | [0.918, 0.963] |
| 距离 | **0.665** | [0.552, 0.693] |
| 距离(对垂直/水平正交化后) | **0.608** | [0.522, 0.668] |
| 垂直(对水平/距离正交化后) | 0.943 | — |

距离读出显著高于 0.5,但比水平/垂直低 0.24–0.29。把距离轴对垂直/水平正交化后,读出从 0.665 降到 0.608:点估计说明**距离轴有一部分独立于垂直的信号**,但正交化后的 95% CI 下界只有 0.522,所以不能排除"距离读出里有相当一部分是垂直泄漏"。轴几何也支持这一点:L24 上 `cos(v_y, v_z) = -0.293`,`cos(v_x, v_y) = -0.068`,`cos(v_x, v_z) = +0.162`——水平轴与另两条近似正交,垂直与距离有约 17° 的负相关;距离轴与水平轴也有可辨的 9° 级分量。

同一层的 WFLU 指标是 CohH 0.183、CohV 0.161、**CohD 0.032**。也就是说:**绝对坐标读出 D=0.665,而关系位移的全向量一致性只有 0.032。** 两者在同一模型、同一层、同一批题上同时成立。

### 3.3 桥接:距离信号在关系位移里还剩多少?

把 WFLU 的 Δ 投影到 S-Space 基上,在 L24 得到:

| 量 | 水平 | 垂直 | 距离 |
|---|---:|---:|---:|
| Δ 的 S-Space 分量符号准确率 | 0.933 | 0.948 | **0.688** |
| Δ 在 S-Space 内的方差占比 | — | — | **2.5%** |
| 全向量 Coh | 0.183 | 0.161 | 0.032 |
| 投影后 Coh(3 维) | 0.585 | 0.526 | **0.063** |
| 均值 Δ 的 S-Space 投影占比 | — | 0.242(above) | 0.158(far) |

均值层面的方向也能量化(对样本做 1000 次 bootstrap):`cos(μ_far, v_z) = +0.137`,95% CI [0.097, 0.161];`cos(μ_far, v_y) = -0.112`,95% CI [-0.147, -0.061]。距离 Δ 确实更靠近距离轴而不是垂直轴,但两个余弦都很小——它是一个"有距离分量、也有垂直分量、整体很弱"的方向。

三点很关键:

1. **距离关系并非不可读**。Δ 在 S-Space 距离轴上的投影符号有 68.8% 正确,且该投影与真实坐标差的相关系数 r=0.81(R²=0.66)。关系位移里确实带着距离信息。
2. **但关系位移的绝大部分不在三维 S-Space 里**。单个 Δ 只有 2.5% 的方差落在 S-Space;均值 Δ 也只有 16–24% 落在里面。投影到 S-Space 后,水平和垂直的 Coh 从 0.18/0.16 跳到 0.59/0.53,说明高维 nuisance 被去掉后轴结构显现;距离的投影 Coh 仍只有 0.063,几乎没有改善。
3. **距离的低 Coh 不是"没有方向",而是"方向弱且样本间不一致"**。L24 的均值 Δ_far 与 Δ_above 的余弦只有 0.149,到 L26/28 才升到 0.465/0.449;也就是说,垂直-距离耦合在关系读出层(L26–L28)才显著,而绝对坐标读出在 L24 最好。L24 的 d′ 仍是 0.74(H/V 约 1.75);投影到 S-Space 后距离方向仍然不稳定。距离信息在均值/投影层面存在,在单样本全向量层面被噪声淹没。

## 4. 机制图像

把两篇论文放在同一个模型上,最自然的图像是**两级表征**:

1. **绝对坐标级(S-Space)**:在 L15–L24,物体名 token 的激活里有一个近似正交的三维子空间,水平/垂直坐标高度可读,距离坐标也可读但更弱,并且距离轴带有约 17° 的垂直分量。这个阶段的信息是"物体在哪里"。
2. **关系读出级(WFLU)**:到 L26–L28,模型要把两个物体的坐标差转成一个关系判断。最后的 Δ 是高维的,只有约 2.5% 的方差落在 S-Space;距离轴的均值位移只有水平轴的 1/5,单样本 d′ 0.74(H/V 约 1.75),而且均值 Δ_far 与 Δ_above 的余弦约 0.45——关系读出把距离判断部分折叠到了垂直方向上。

这解释了两篇论文为什么可以同时为真:

- S-Space 测的是**低维投影**:`v_z^T(h_q - h_r)` 把 2048 维压到 1 维,噪声被平均掉,所以即使绝对距离信号不强,读出仍能到 0.66;它回答"距离信息是否以线性坐标形式存在"。
- WFLU 测的是**全向量方向一致性**:`cos(Δ_i, Δ_j)` 对高维 nuisance 极其敏感。距离 Δ 的均值方向虽然存在,但单样本方向被 97.5% 的 S-Space 外分量主导,所以 CohD 只有 0.03;它回答"关系位移是否稳定地指向同一方向"。
- 两者在"距离最难"上其实一致:S-Space 自己的逐轴相关也是 D 最低(0.521 vs H 0.789/V 0.774),EmbSpatial 上 far 方向的读出经常接近 chance。分歧只在解释:WFLU 把它归因于垂直-距离纠缠,S-Space 把它归因于读出/计算瓶颈。我们的数据支持**两者都有**:距离轴在正交化后仍有 0.608 的读出(CI 下界 0.522),说明存在独立分量;同时垂直泄漏是真实的(cos_yz=-0.293;正交化损失约 6 个点);关系读出层的垂直耦合(cos(μ_above, μ_far)=+0.45)比绝对轴层更强。

换言之:**模型知道物体有多远,但它把这个知识读成一个关系判断时,用的是一个低信噪比、部分垂直化的方向。** 这不是"表征里没有距离",也不是"垂直捷径完全替代了距离",而是"绝对编码 → 关系读出"这一步的瓶颈。S-Space 论文自己的 perception-computation decomposition 也指向同一类结论:表征足够,计算/读出不够。

层级上的分离更具体:L24 上绝对坐标读出最好(0.842),但 VD-EI 只有 0.143;到 L26/28,VD-EI 跳到 0.45 左右,关系位移的垂直-距离耦合才出现。与此同时,CohD 从 0.032 只升到 0.044–0.048。这说明**纠缠不是从绝对轴里直接继承的,而是在更晚的关系读出层被"旋转"进去的**;而距离的关系位移即使在耦合增强后仍然低相干,因为它的均值位移太小、单样本噪声太大。这个"先有可读坐标,后有垂直化读出"的时序,是两篇论文能同时成立的最直接机制证据。

一个可检验的推论:如果给模型显式的 S-Space 坐标读出(而不是让它自己生成答案),距离判断应显著好于它的原生回答。S-Space 论文在 SpinBench 上报告了 +30 个点的读出增益;我们的 L24 读出 D=0.665,论文报告同一模型在 SpatialTunnel 上的 logit 正确率 v=0.570、counter=0.360。题面与评价协议不同,不能把这两个数字直接相减,但方向与 S-Space 的"表征够、读出不够"一致。


## 5. 不确定性与障碍

- **轴估计样本量**:主结果每条轴只有 8 条梯度(2 题 × 2 模板 × 2 序)。bootstrap 显示 L24 的距离读出 95% CI 为 [0.55, 0.69],水平/垂直的 CI 很窄;所以"距离可读但弱"是稳的,"具体是 0.665 还是 0.62"不稳。距离轴在模板间的余弦只有 0.25–0.36(H/V 为 0.92–0.95),轴本身比 H/V 难估。尝试过 10 题/方向,但被 GPU 竞争打断;在空闲机器上建议至少 20–50 题/方向。
- **层不匹配**:S-Space 读出峰值在 L24,WFLU 的 L* 是 L28。L28 上 2 题/方向的梯度范数比 L24 小约 20 倍,轴估计不可靠(读出接近 chance),因此我们不以 L28 的 S-Space 数字做结论;L35 的梯度直接退化为 0,图中省略。两篇论文本来也选了不同层(S-Space 的 Qwen3.5-4B 选 L15/32,WFLU 的 Qwen2.5-3B 选 L28/36);"同一模型"不等于"同一层"。
- **预处理**:`max_pixels=401408`(512²)、`min_pixels=200704`。约 1/3 的 EmbSpatial 图是 1296×968,被降采样;300×300 的图被放大到 256 token。官方 `probing.py` 在默认预处理下的 L28 Δ 与我们的 forward-only 提取余弦 0.998,说明 forward 等价性没问题;但绝对 Coh 会随分辨率移动(论文 H/V 比我们高 0.04–0.07)。不要把我们的绝对值当成论文的逐位复现。
- **数据域**:轴在 EmbSpatial 上识别、在 EmbSpatial 留出题上读出,是同分布检验;没有跑 COCO 轴,也没有跑 SpatialTunnel 的跨域读出。S-Space 论文的跨域结论(CV-Bench/EmbSpatial/SpatialTunnel)本轮没有复现。
- **没有逐轴 R²**:EmbSpatial TSV 只有关系类别(左/右、上/下、远/近),没有连续 x/y/z 或深度标注;我们无法在同一数据上算线性探针的逐轴 R²,改用 S-Space 的符号读出准确率,并引用论文在 COCO 上的逐轴 Spearman 相关(0.789/0.774/0.521)。若要做 R²,需要 COCO 的 bbox 加 Depth Anything 的连续深度,本轮没跑。
- **没有测原生行为**:我们没有在同一批题上重跑模型的直接回答,因此"读出 0.665 vs 论文报告的原生 v=0.570/counter=0.360"不是同题面对比;原生行为数字引自 WFLU 论文的 SpatialTunnel 协议。
- **GPU 竞争**:另一条线程间歇占用 3–6 GB 显存,S-Space 全量 80 题/方向的运行在 37% 时 OOM;断点续跑的 10 题/方向版本也多次被打断。若最终只有 2 题/方向轴,这是资源约束下的最小可行估计,不是方法本身的样本量。
- **S-Space 轴未正交化**:与论文一致,我们保留原始梯度轴;`cos_yz` 与"正交化后读出"是我们加的诊断,不是论文指标。正交化后的 0.608 应理解为"去掉垂直分量后仍可读的距离信息"的下界(线性去相关,不保证因果)。


## 6. 复跑

```bash
PY=python  # 任一带 torch/transformers/PIL 的环境;lab bench 里是 code/.venv/bin/python
cd runs/2026-10-19-whyfar-linearmem-natura-astra/whyfar-sspace-2605.30161/code

# 1) WFLU 特征(约 20-25 分钟)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY run_wflu.py \
    --n-per-category 200 --max-pixels 401408 --min-pixels 200704

# 2) S-Space 梯度轴(主结果 2 题/方向约 1-2 分钟;
#    空闲机器上建议 --n-per-category 20 以上,约 10-20 分钟)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY run_sspace_id.py \
    --n-per-category 2 --max-pixels 401408 --min-pixels 200704 \
    --out code/probe_out/sspace_rerun

# 3) 指标与桥接分析
$PY analyze.py --sspace-dir code/probe_out/sspace_rerun \
    --wflu-dir code/probe_out/wflu_qwen25 \
    --out code/probe_out/summary_rerun
$PY wflu_only.py

# 4) 图
$PY plot_results.py --sspace-dir code/probe_out/sspace_rerun \
    --summary-dir code/probe_out/summary_rerun \
    --out code/probe_out/plots_rerun
```

原始数组在 `code/probe_out/`(gitignored),精选汇总与本 README 在线程包内。

## 7. 文件

线程包内:

- `README.md` — 本文件
- `code/` — 全部探针与分析脚本(见下)
- `code/results/wflu_metrics.csv` — 逐层 Coh / VD-EI / d′ / 均值范数
- `code/results/metrics_by_layer.csv` — WFLU × S-Space 逐层联合指标
- `code/results/summary.json` — L24 / L28 的完整指标与 bootstrap 几何
- `code/results/key_numbers.json` — 报告用的核心数字汇总
- `code/results/axis_bootstrap.json` — 轴重采样下 H/V/D 读出的 95% CI
- `code/results/axis_diagnostics.json` — 每层三条轴的两两余弦与梯度范数
- `code/results/official_delta_check.json` — 官方 `probing.py` 与 forward-only 的 Δ 对照
- `../assets/whyfar_*.png` — 本线程的图,在运行级 assets 目录

脚本:

| 文件 | 作用 |
|---|---|
| `common.py` | 模型/processor、prompt 构造、物体 token 定位、Δ/Coh/VD-EI 工具 |
| `run_wflu.py` | WFLU 全量提取(原序 + 交换序,保存 last-token Δ 与物体 token 状态) |
| `run_sspace_id.py` | S-Space 梯度轴识别(角色反对称差,支持断点续跑) |
| `analyze.py` | 逐层联合指标、S-Space 读出、桥接、正交化、bootstrap |
| `wflu_only.py` | 仅 WFLU 的逐层指标与 d′ |
| `plot_results.py` | 生成本线程的图 |
| `bootstrap_axes.py` | 从逐样本梯度重采样轴并重算读出 CI |
| `run_official_smoke.py` | 官方代码的 2 题/方向保真度对照 |
| `smoke_test.py`, `test_mem.py` | 环境/显存/定位冒烟测试 |

原始数组(`delta.npy`, `hs_q.npy`, `hs_r.npy`, `axes.npy`, `grad_records.npz`)在 `code/probe_out/`(gitignored,约 1.5 GB),不在线程包内。

## 8. 图清单

本线程实验图:

- `../assets/whyfar_coexistence.png` — 2×2 总览:A WFLU Coh/VD-EI,B S-Space 读出,C 桥接散点,D 均值 Δ 方向
- `../assets/whyfar_wflu_layerwise.png` — 逐层 Coh(H/V/D)与 VD-EI(两个定义)
- `../assets/whyfar_wflu_heatmap_L28.png` — L28 6×6 均值 Δ 余弦(透视耦合的直接图像)
- `../assets/whyfar_sspace_readout.png` — 逐层 H/V/D 绝对坐标读出(轴来自 2 题/方向)
- `../assets/whyfar_bridge_scatter.png` — Δ 的 v_z 投影 vs S-Space 距离坐标差(r=0.81, R²=0.66)
- `../assets/whyfar_delta_axis_cosines.png` — 均值 Δ_far/Δ_close 与 v_y/v_z 的余弦 + bootstrap CI

论文原图(已裁白边、≤1600 px):

- `../assets/whyfar_wflu_teaser.png` — WFLU Fig. 1(透视捷径与反转样本)
- `../assets/whyfar_wflu_probing.png` — WFLU Fig. 5(contrastive probing 流程)
- `../assets/whyfar_wflu_pca.png` — WFLU Fig. 7(五个模型的 Δ PCA)
- `../assets/whyfar_sspace_formulation.png` — S-Space Fig. 2(三维线性子空间与读出)
- `../assets/whyfar_sspace_axis_correlations.png` — S-Space Fig. 3a/b 拼合(逐轴 Spearman 相关)

