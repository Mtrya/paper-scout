# cortex-2610.20399 线程:Cortex 精读 + 「预期查询是否真有增量」的玩具探针

锚点论文 **Cortex: Consolidating Context into Parametric Memory**(arXiv 2610.20399v1,ByteDance Seed × 浙大,2026-10)。本线程命中支线「快权重/慢权重双通道与经验的低成本固化」,也是 MEMORY.md 边界④(闭式最优巩固只对快权重/线性读出成立,固化进深权重仍需蒸馏训练)的第一个大规模实证样本。

- 论文缓存:`papers/llm-agents/cortex-2610.20399.md`(99 页,含附录 A 理论、B 协议、C 全表、D 提示词)
- 探针代码:`code/cortex-probe/`(probe_data.py 数据、probe.py 主程序、plot.py 出图、run_all.sh 一键重跑;`.venv` 与权重不进库)
- 结果:`code/cortex-probe/artifacts/`(生成的问题、教师答案、分类标签、5 个训练日志、eval_*.json、summary.json)+ `code/cortex-probe/figures/probe_summary.png`)

## 1. 论文本体

### 1.1 形式化:Memory Consolidation(MC)

论文把一个一直散落在若干子领域里的目标写成了一条式子。给定模型 $M$ 与一段临时可用的上下文 $C$,记 $\operatorname{Perf}(M,Q,C)$ 为在上下文 $C$ 存在时模型对查询分布 $Q$ 的期望得分,则**上下文增益(context gap)**为

$$\Delta_{\text{ctx}}(M,Q_C^+,C)=\operatorname{Perf}(M,Q_C^+,C)-\operatorname{Perf}(M,Q_C^+).$$

$\Delta_{\text{ctx}}>0$ 意味着模型"看得见、用得上、一撤就没"——作者把它命名为 **memory failure**(区别于 capability failure,即带上下文也做不好)。**记忆固化(Memory Consolidation, MC)**于是定义为:求 $M'=\mathcal{A}(M,C)$,使 $M'$ 在**推理时没有 $C$** 的情况下仍恢复增益,同时不破坏非目标能力。两个度量:

$$ \mathrm{CR}=\frac{\operatorname{Perf}(M',Q_C^+)-\operatorname{Perf}(M,Q_C^+)}{\Delta_{\text{ctx}}(M,Q_C^+,C)},\qquad \mathrm{RR}=\frac{\operatorname{Perf}(M',Q_{C'}^-)}{\operatorname{Perf}(M,Q_{C'}^-)} .$$

值得注意的三个设计决定:(1) MC 是**结果式(outcome-based)表述**——$C$ 只提供"行为参考",不规定训练信号,学习数据怎么造属于方法的一部分;(2) 它显式要求保留**初始模型能力**(SELF-PARAM 只要求保留先前注入的上下文);(3) 支持 single / batch / **sequential**(逐个到达、旧上下文不可再访问)三档,其中 sequential 还要保住此前已固化的记忆。由此,上下文指定的知识编辑、技能学习、行为引导、知识抑制被统一成同一个带 CR/RR 的问题。

五任务族测试台(Fig. 4):agent 经验(ALFWorld / τ²-Bench)、新 API 学习(15 个 2025-11-01 之后发布的 Python 包,150 道带可执行测试的题)、人格引导(Power-seeking / Psychopathy)、知识抑制(RWKU)、事实编辑(50 条 zsRE 事实)。覆盖"陈述/程序/行为"三种记忆类型 × "新增/修订/抑制"三种操作 × 局部/全局作用域。

### 1.2 方法:三个设计问题

**① 学什么 —— 预期查询(prospective queries)。** 预更新模型自己充当生成器:$q_i\sim M(\cdot\mid I_q,C)$,$y_i\sim M(\cdot\mid C,q_i)$。$I_q$ 要求"预判这段上下文将来会被怎么用",而不是复述它。作者的动机是认知科学的未来模拟(remembering the past to imagine the future)与 CLS 理论中的 replay。回答由**模型自己**产生(Cortex 声称这让目标分布不引入额外错配);附录 B.3.7 用外部模型(Gemini 3 Pro)替换答案来源做对照:GPT-OSS-20B batch 编辑上 efficacy 从 89.5% 塌到 21.4%,MMLU 从 75.8% 塌到 15.9%——**答案来源**这一项比很多结构设计更致命。agent 任务例外:查询来自环境回放,只保留成功轨迹。

**② 写在哪 —— FFN 内的两种可塑性。** 沿用 FFN≈key-value memory 的证据,更新限制在 MLP:
- **Synaptic**:$W' = W + B_W A_W$,对 $W_G,W_U,W_D$ 加秩 ≤ r 的增量(原矩阵冻结),部署时并进原矩阵;
- **Structural**:$F'(x)=F(x)+D_s[(U_sx)\odot \mathrm{SiLU}(G_sx)]$,加 $k$ 个新非线性通道,原权重块不动,部署时等价于把 FFN 加宽。

MoE 上两者都冻结 router;Structural 用**一条共享分支**而非每专家独立扩展——因为 $k_u$ 宽度的独立扩展只为被路由到的 token 提供梯度,而共享分支每个 token 都走。附录 A.4 给出梯度连通性刻画(独立扩展的梯度带 $a_e(x)$ 因子),实测 GPT-OSS-20B 上 768 个 layer-expert 分支里有 32 个在三个种子里曝光率都低于该层均值的 1%。

**③ 怎么写 —— context-to-memory distillation。** 教师 = 冻结的更新前模型 $M_{t-1}$,输入带 $C$;学生 = 只训练新引入的 FFN 参数 $\psi_t$,输入不带 $C$。损失(式 8):

$$\mathcal{L}_t(\psi_t)=\mathbb{E}_{\mathcal{B}}\frac{1}{|\mathcal{J_B}|}\sum_{i\in\mathcal{J_B}}\Big[\beta T^2\, \mathrm{KL}_{K_v}(p_i^T\Vert p_i^S)+\frac{\alpha}{Ld_h}\sum_{\ell=1}^L\|h_{\ell,i}^S-h_{\ell,i}^T\|_2^2\Big].$$

KL 只保留教师 top-$K_v$(1024)词表以省存储,学生仍在全词表归一。MSE 监督的是 **post-FFN residual** 的对应位置——附录 A.1 证明了它为什么在这里特别:若该 block 的 FFN 输出直接加进残差流,则

$$\|h_i^S-h_i^T\|_2^2=\|R(u_i)-\delta_i\|_2^2,\qquad \delta_i=h_i^T-\bar h_i,$$

即"hidden-state 对齐"**恰好就是对所需修正量的回归**。含 post-norm 时同一恒等式对 $R_{\text{eff}}=P(F+R)-P(F)$ 成立,但闭式最小二乘只对 raw branch output 成立(原文明确说明)。

sequential 设定:每个新上下文训一个新的 adapter,旧的全部冻结并保持激活;可选 **query replay**——只存历史**查询**,答案与教师目标在每轮用 $M_{t-1}$ 重新生成(不存历史上下文、不存历史答案串);另有 16 条通用问题(固定 132 题库)由 $M_0$ 缓存监督。

### 1.3 结果与声称

- **单/批量**:Fig. 5 的 15 个面板中,每个面板至少一个 Cortex 变体在 CR–RR 经验前沿上;steering 95–100% CR / 99–100% RR,batch 编辑 80–91% CR。
- **持续编辑**(50 条 zsRE,Qwen2.5-14B-Instruct,3 seed):Structural $k{=}1$ 最终 **CR 88.9±3.5 / RR 99.8±0.8**,每次编辑只加 **0.74M** 参数;Synaptic $r{=}1$ 为 80.6±3.7 / 97.2±0.8(每次 2.73M);最强基线 ROME 28.5±1.9 / 94.4±0.5,AlphaEdit 22.7±3.1 / 96.0±0.6。
- **容量反直觉**:$k\in\{1..16\}$、$r\in\{1,2,4\}$ 扫下来,**最小容量 CR 最高**(88.9% → 84.3%;80.6% → 75.6%)。
- **位置**:FFN-only 比 FFN+Attn(同秩)高 +3.8 CR / +4.0 RR。
- **replay**:去掉历史查询回放后,再经历 25 次编辑,Structural 保持 40.5%、Synaptic 只剩 8.0%(有 replay 时两者 >96%)——**冻结参数不等于冻结行为**。
- **代价面**:知识抑制上 CR 97–100% 但 RR 掉到 76–83%(邻居知识 ROUGE 掉 9–20 分);GPT-OSS-20B ALFWorld 上 Synaptic CR 101% / RR 23%;τ²-Bench 上 Synaptic CR −92.3%。
- **反例面**(作者自己报的):Qwen2.5 batch 编辑上,普通 LoRA-CE 用 Cortex 的数据就能拿到 91.9% CR,高于 Synaptic 90.8 / Structural 88.6——写作机制不是所有任务上都重要。

### 1.4 附录 A 的理论:它到底证明了什么

- **Prop A.1**(特征固定):$\min_{D_s}\|D_s\Phi-\Delta\|_F^2=\|\Delta(I-\Phi^\dagger\Phi)\|_F^2$,取 $D_s=\Delta\Phi^\dagger$;恰好插值 ⟺ $\operatorname{row}(\Delta)\subseteq\operatorname{row}(\Phi)$。
- **Thm A.2**(特征可学):输入两两不同且不互为相反数时,$k\ge n$ 个通道可以**插值任意** $n$ 个目标(构造:沿一个方向 $a$ 排序 $z_i=a^\top u_i$,用饱和门造出上三角特征矩阵)。GPT-OSS 的原生通道亦可,只需输入两两不同。
- **Cor A.3**:若最后一个 FFN 输出直连残差流且共享输出头,则在有限状态集上可以对上教师 logits。
- **Prop A.4**:down-only 子类要精确匹配 $n$ 个状态,需 $\operatorname{row}(\Delta)\subseteq\operatorname{row}(H)$ 且 $\operatorname{rank}(\Delta)\le r$;单状态时就是秩 1 的 $\Delta W_D=\delta h_0(u)^\top/\|h_0(u)\|^2$。
- **A.4**:共享分支的梯度不带路由因子;归一化路由下可等价地把共享分支折叠进每个专家。

## 2. 邻里坐标(与谁像、与谁不像)

| 邻居 | 与 Cortex 的真实差异 |
| --- | --- |
| **ROME / MEMIT / AlphaEdit**(闭式编辑) | 它们解的是**单点/多点 key→value 的最小二乘**(ROME 是单层 down 投影的秩 1 写入;MEMIT 把多个编辑分摊到多层;AlphaEdit 加零空间约束保护旧 key)。Cortex 的 A.3 说明:只更新 down 投影时它确实退化成同一类闭式解。差别有三:Cortex 的目标不是存下来的 key-value,而是**行为参考**(教师在全词表上的分布 + 各层残差修正);它同时更新 gate/up,特征随参数变化,闭式不再存在;它的"保留"由 rehearsal + 通用数据承担,而不是零空间约束。 |
| **LoRA 顺序微调 / 持续学习** | 同一优化器与数据下的对照(RQ1/RQ2):FFN-only + KL + replay 明显优于 LoRA-All + CE;CE 版在 50 次编辑中 MMLU 从 81 掉到 69,而 Structural KL 基本平线。差别不在"要不要训练",而在**监督信号(分布匹配 vs 答案拟合)+ 更新位置 + 历史行为约束**三件套。 |
| **context distillation(DCD / SELF-PARAM / SDCD / OPCD)** | 共享"教师带上下文、学生不带"的骨架。差异:(a) MC 不接收任务输入分布,数据构造属于方法——预期查询是它的答案;(b) 它把**保留初始能力**写成硬要求并逐轮评测;(c) 更新参数化被限制为可折叠的 FFN 低秩/结构增量;(d) 它用离线 teacher forcing(OPCD 用 on-policy 轨迹)。 |
| **TTT / 在线记忆**(δ-Mem、Locas) | 它们把状态留在推理期;Cortex 要把效果**写进权重**且此后不再需要任何外挂。实验里 TTT 系 RR 接近满分但 CR ≈ 0 甚至为负——"守住能力"与"拿到知识"是两件事。 |
| **Hypernetwork**(Doc2LoRA、SHINE) | 用元训练把"上下文→LoRA"摊销成一次前向;Cortex 不摊销,直接对每个上下文做梯度优化。代价是每个上下文一次训练,收益是单样本上更强(单条编辑 Doc2LoRA 95% CR 超过 Cortex 的 84–89%,但 batch 上掉到 24%)。 |
| **NoteBank(2610.15715)** | 同月的另一端:线性注意力的**稀疏情景记忆**,训练-free、把选中的原文存外挂、用原生前向重放,1M token 上把 Gated DeltaNet 的 RULER 检索从 1.7% 提到 70.3%。它不固化任何参数,RR 天然 100%,代价是推理期的存储与检索——正好印证 MEMORY.md 边界③(softmax/固定状态没有有限维充分统计量时,要么外挂、要么近似)。 |
| **MaLoW(2610.22070)** | "Memory as Weights":循环 hypernetwork 把流式视频历史**内化进定长 LoRA**,历史进权重、当前观测进输入。与 Cortex 同向不同构造:定容量(而非每上下文增量增长)、元学习摊销(而非逐上下文优化)、以及**没有**保留非目标能力的要求;381K QA 对的训练集是它的入场券。 |

一句话定位:**Cortex 是"把上下文固化成 FFN 参数"这一族里,第一个把"保留"与"恢复"同时做成可比较指标、并把测试台、更新参数化、监督信号三件事拼成统一配方的工作**。它的新颖度集中在"预期查询"与可折叠的两种 FFN 可塑性上;方法与损失本身是已有构件(Snell 式 context distillation + LoRA/加宽 + replay)的重组。

## 3. 机制推导:与 MEMORY.md 边界②③④对表

**边界③(softmax 全注意力没有有限维充分统计量)决定了问题的形状。** 线性注意力可以把上下文压进状态 $S$;softmax 注意力不能。Cortex 完全不去构造充分统计量:它退一步,用**采样出来的输入状态** $\{u_i\}$ 上的修正量 $\{\delta_i\}$ 来近似上下文的效果。于是"上下文的效果"被替换成"在一个查询分布上的效果",**rehearsal 分布从细节升格为主设计轴**——这就是"预期查询"在体系里的位置:它是对缺失的充分统计量的经验替代物。论文的 Cor A.3 一律带"on this set"的限定,Thm A.2 是有限状态集上的插值定理,都说明它没有、也不声称有"精确搬迁"。

**边界④(闭式最优只对快权重/线性读出成立)直接对应 A.1–A.3。** 把"蒸馏时我们在做什么"三件套拆开:散度 = forward KL(输出) + ℓ²(post-FFN residual);函数类 = 低秩增量或 $k$ 通道加宽;数据分布 = 预期查询。三件套里唯一有闭式的组合是"特征固定 + 只解最后一层线性读出",即 Prop A.1 的 $D_s=\Delta\Phi^\dagger$。Cortex 把特征也放进优化(U_s/G_s/gate/up 都训练)并叠加输出 KL,于是**联合目标非凸、无闭式**,只能梯度下降。它的 MSE 项可以读作"逐层把闭式回归问题的损失写出来,但把特征留给优化器"——一个很干净的折中形态。

**边界②(精确搬迁可行但无压缩)在容量上的体现。** Thm A.2 的 $k\ge n$:要精确钉住 $n$ 个状态,结构通道数至少要 $n$——一个通道一条命,没有压缩。这与我们从 $\Delta S$ 的 SVD 得到的判据同构:可压缩性取决于修正量矩阵 $\Delta$ 的谱(或 $\operatorname{rank}(\Delta)$),而不是状态个数。Prop A.4 的 $\operatorname{rank}(\Delta)\le r$ 正是同一个判据在 Synaptic 一侧的版本。**工程上二者都走了"每个上下文加一小块、累积增长"的路线**($k{=}1$ 每次 0.74M,50 次编辑后 Structural 累计 +36.9M 参数),这是"无压缩"在部署上的代价;而 Synaptic 的低秩侧则是在谱给面子时**主动压缩**的那一半。

**一个由此推出的判据/猜想**:Cortex 的现象(容量越大 CR 越低、$k{=}1$ 最好)与"修正量张成低维子空间"的解释一致——如果 rehearsal 覆盖的修正方向确实低维,加大通道只是给了优化器在 rehearsal 集上过拟合的余地。可检验形式:对每个上下文的逐层修正矩阵 $\Delta_\ell$ 做 SVD,若谱衰减快(比如 90% 能量在前 r 个奇异值),那么"低秩 Synaptic 足够、加宽无益"是可预期的;谱平坦的上下文应当反过来受益于 Structural。我们这轮没有做这个测量(见 §4 障碍),这是留给下一轮最干净的实验。

## 4. 探针:预期查询到底有没有增量

### 4.1 问题与设计

论文声称的核心机制是**查询内容**——"预判将来怎么用"的查询优于复述式查询。这条声称在论文里几乎没被单独立过检验(RQ1 只对"数据 vs 监督"做了 LoRA 对照,而"预期"两个字藏在 $I_q$ 的措辞里,且附录没有给出 $I_q$ 的原文,官方代码仓 404)。我们把它做成一个受控的 2×2:

- **上下文 $C$**:一段虚构的内部工程文档 `meridian`(约 300 token,含默认值、错误码、行为规则、诊断命令 16 条事实)。
- **两条被测轴**:查询种类 R(restate,复述)/ P(prospective,预期);监督 KL(教师分布)/ CE(教师强制答案)。
- **生成器**:Qwen2.5-3B-Instruct,两条提示词只差"你要预判将来怎么用"还是"你要考别人记不记得"。各取 32 条,条数、生成器、解码参数、教师、优化器、步数全部一致。**唯一变量是问题内容**。
- **回答**:由被测模型自己带上下文生成(Cortex 式 4,保证答案分布同源)。
- **学生/教师**:Qwen3-0.6B(稠密 SwiGLU,28 层)。LoRA 只挂在 FFN 的 gate/up/down(秩 8,2.75M 参数),教师是关掉 adapter 的同一模型——严格复刻"FFN-only + 只训新增参数 + 教师=更新前模型"。
- **评测(全部 held-out,训练从不接触)**:20 条复述题 + 20 条应用题,四选一强制选择(候选答案字符串的 log-prob 排序,选项顺序固定随机);保留面板 = MMLU test 150 题(4 选 1)。CR 用 NoCtx / Ctx 两种参考打分归一。
- **操纵检查**:用同一个 3B 模型把每条 rehearsal 查询判为 RECALL / USE,验证两条提示词确实造出了不同种类的问题。

### 4.2 结果

**上下文增益是真的、且很大**:NoCtx 25%(复述)/20%(应用),Ctx 80%/65%(自由续写读数),字母读数 32.5%→85% / 30%→60%。**固化机制也真的在工作**:交叉诊断显示,两个 KL 臂在**各自被训练的那 32 条问题**上,无上下文时的答案对数似然已经完全追平带上下文的教师(恢复率 1.00/1.00)。

**但泛化到没见过的问法很弱**:held-out 上的 CR 只有 0–44%(论文在 14B 上是 80–90%)。三个 KL 臂、每臂 3 seed、每格 20 题的均值:

| 臂(rehearsal 内容) | 应用题 CR(自由读数) | 复述题 CR(自由读数) | 字母读数(应用/复述) | MMLU RR |
| --- | --- | --- | --- | --- |
| R-KL(32 条复述) | +22% | +36% | +25% / +2% | 96% |
| P-KL(32 条预期) | +39% | +38% | +25% / +17% | 104% |
| M-KL(16 复述 + 16 预期) | +44% | +33% | 0% / +13% | 99% |
| R-CE(复述 + CE,1 seed) | +33% | 0% | 0% / 0% | 106% |
| P-CE(预期 + CE,1 seed) | +22% | +9% | +25% / +5% | 108% |

按题配对 bootstrap(KL 臂,3 seed 平均到每题再重采样):

- 应用题上 **M−R = +10.0pp,95% CI [−1.7, +25.0],P(>0)=0.91**;P−R = +8.3pp,CI [−11.7, +30.0];
- 复述题上 M−R = −1.7pp、R−P = −1.7pp,即**没有差别**。

也就是说:**方向与论文的机制声称一致——把"将来怎么用"写进 rehearsal,恢复的增量落在应用题上而不落在复述题上;但在这个规模上 8–10pp 的差异只有 ~1σ,不能算证实。**换成字母读数(结构化 4 选 1)后差异消失甚至反向,说明这一效应还小于读数选择带来的方法噪声。

**交叉迁移诊断**(seed 0,同一文档的两组问题上互测,均值对数似然口径):

| 训练的查询 | 在 R 问题上的恢复 | 在 P 问题上的恢复 |
| --- | --- | --- |
| 32 条 R(复述) | 1.00 | 0.61 |
| 32 条 P(预期) | 0.49 | 1.00 |

两个方向都不完整,而且**复述臂迁到预期问题上(0.61)略好于反向(0.49)**——与"预期查询是更强的监督"这一直觉相反。一个合理解释是:在 32 条样本上,LoRA 基本是在**记住自己被训练的那些问答对**(恢复率恰好 1.00 就是证据),跨类的迁移主要由词面重叠驱动,而我们的"预期"生成器写出来的问题(见 `artifacts/rehearsal_questions.json`)很多其实是"带情境外壳的复述题",内容上与复述臂的差距小于名义差距——操纵检查也显示:3B 分类器把 R 臂的 16/32、P 臂的 29/32 判为"应用型",两臂并非完全分离。

### 4.3 结论与障碍

可以下的结论:

1. **"rehearsal 内容决定恢复范围"这一点在玩具尺度上成立且很锐利**——同一文档、同一教师、同一优化器,只换 rehearsal 的问题种类,恢复的落点就跟着换;这支持论文把"学什么(预期查询)"列为第一设计轴的前提。论文 RQ1 只做了"数据 vs 监督"的对照,而"数据内部按用途分种类"这一层没有被单独做实验,我们补的是这一格。
2. **"预期查询优于复述查询"在 0.6B / 32 条 / 单文档上没有得到显著支持**:应用题上的增益为 +8~10pp(~1σ,且在字母读数下消失),跨类迁移上甚至是复述臂更强。
3. **保留侧没有差异**:所有臂 MMLU RR 在 96–108% 之间,与论文"小容量 FFN 写入不伤非目标能力"一致;0.6B + rank-8 + 32 样本的写入太小,造不出可测的损伤。

障碍(如果要把这个探针做成能给出≥3σ 结论的实验,需要的是):

- **更强的固化**:当前 CR 上限太低(泛化到 held-out 问法只剩几十个百分点),效应被压在噪声地板附近。可选路线:更大骨干(7B 级)、把 rehearsal 扩到每事实 8–10 条问题、或把评测改成"文档内事实的检索式问答"降低对问法泛化的要求。本机 12GB 显存 + 与另一条线程共享 GPU 的现状,限制了这三条同时上。
- **更干净的操纵**:需要让"预期"提示词真的产出任务型问题(我们的生成器大量输出"带疑问句壳的复述"),例如先给未来任务模板再让其提问;或反过来用**真实的下游任务**(Cortex 的 agent/code 族)来定义"用",那就需要环境与执行器,超出本轮预算。
- **更多的题**:每格 20 题 ⇒ 每题 ±11pp 的二项噪声;配对 bootstrap 已用上,但要让 10pp 显著需要 ~60–80 题/格,意味着评测集要人工扩写 3–4 倍。

这一节没有做的、但下一轮最值得做的:对每个上下文的逐层修正量 $\Delta_\ell$ 做 SVD,直接测"可压缩性判据"(§3 末),把"低秩 vs 加宽"的选择从经验调参变成先验判断。

### 4.4 好奇心清单

1. **预期查询的 prompt 原文没公开,代码仓(论文里写的 github.com/zjunlp/cortex)404**——这套方法最独特的构件因此不可复现。若作者放出代码,第一件该做的事是把 $I_q$ 换掉做我上面这一格的对照。
2. **Cortex 在 sequential 上把"每轮训一个新 adapter"当作正交设计**:50 次编辑后 Structural 累计新增 36.9M 参数(每次 0.74M),部署时等价于每层 FFN 加宽 50 通道。它与 LoRAHub / adapter-merging 一族的差别只是"加了就不合"——那么当编辑到 500 次、1000 次时这条曲线怎么走?论文没有做超过 50 步的持续实验。
3. **RR 的口径漂移**:单编辑用 locality、steering 用 fluency、持续编辑用 locality+MMLU+SimpleQA+GSM8K 等权平均。"99.8% RR"里 MMLU 与 GSM8K 权重最大,而 SimpleQA 从 3.9% 到 4.8% 这种量级根本不构成信号。做横向比较时应把 RR 拆开看。
4. **论文 A.1 的恒等式 $\|h^S-h^T\|^2=\|R(u)-\delta\|^2 把隐状态对齐变成逐层回归**,但只对"FFN 输出直连残差流"成立;含 post-norm 时闭式最小二乘不再对应这个损失(原文自己指出)。那么 Llama/Qwen 系(带 pre-norm + post-attention norm)上 MSE 项的"修正量语义"到底还剩多少?值得用一个小模型做逐层 δ 的可达性测量(我们这轮只做了机制推理)。
5. **"保留"与"恢复"的耦合**:RQ4 显示 MSE 对 Synaptic 是 −4.0 CR / +2.7 RR 的交换,对 Structural k=16 却是双赢。这说明两个损失项的相互作用依赖参数化,那么理论上是否存在一个"参数化无关"的保留-恢复前沿?或者前沿本身由参数化选择决定(即存在严格占优)?
6. **与 NoteBank/MaLoW 的三点对比**:外挂 vs 固化 vs 摊销固化,在"何时该固化"上有没有一个定量判据?比如上下文复用次数 × 每次推理的上下文成本 > 一次固化的训练成本时固化才划算——Cortex 没有从成本角度给这个判据,而它的每上下文一次训练对大模型并不便宜(论文未来工作也承认"learning them still requires computation through the backbone")。
7. **Cortex 的 agent 任务用的是环境查询而不是自生成查询**——这恰好是它的"预期查询"最该起作用的地方却换了数据源,说明"自生成的预期查询能否覆盖真实任务分布"仍未验证。

### 4.5 论文图/素材清单

放在 `runs/2026-10-16-greenwam-onewm-xtrace-cortex/assets/`(文件名与图片定位符的对应见 `assets/cortex-figures.json`):

| 文件 | 内容 | 来源定位符 / 提取图 |
| --- | --- | --- |
| `cortex-fig2-workflow.png` | 工作流三段:rehearsal 构造 / 三种固化设定 / 蒸馏 | `doc:6123bd4/tier:basic/page:5/block:1`(img-58) |
| `cortex-fig3-plasticity.png` | Synaptic vs Structural + MoE 共享分支 | `doc:6123bd4/tier:basic/page:6/block:1`(img-80) |
| `cortex-fig11-retention.png` | 50 次编辑中 MMLU/SimpleQA/GSM8K 轨迹(k=1/r=1) | `doc:6123bd4/tier:basic/page:45/block:1`(img-59) |
| `cortex-fig5-pareto.png` | 15 面板 CR–RR 经验前沿(备选) | `doc:6123bd4/tier:basic/page:11/block:1`(img-01) |
| `cortex-fig4-testbed.png` | 五任务族测试台构造(备选) | `doc:6123bd4/tier:basic/page:8/block:1`(img-86) |
| `cortex-fig1-mc-objective.png` | MC 目标示意(备选) | `doc:6123bd4/tier:basic/page:1/block:3`(img-22) |
| `cortex-exp-probe.png` | 我们的探针三面板图 | `code/cortex-probe/figures/probe_summary.png` |

注:本机 mineru 文档库的源 PDF 已被清理(`mineru read ... -f image` 报 "No active source file found"),上述图片是从解析时导出的 `drafts/images/cortex-2610.20399/img-NN.png`(提取顺序≠文档顺序)中逐一目视比对认出的。裁剪已去掉白边。

## 5. 如何重跑

```bash
cd code/cortex-probe                 # 工作副本;包内副本见 runs/.../cortex-2610.20399/code/
bash run_all.sh
for s in 1 2; do for a in R P M; do ../cortex-env/bin/python probe.py train --arm $a --loss kl --seed $s; done; done
../cortex-env/bin/python probe.py eval --runs R:kl:0 P:kl:0 M:kl:0 R:ce:0 P:ce:0 \
    R:kl:1 P:kl:1 M:kl:1 R:kl:2 P:kl:2 M:kl:2 --tag main
../cortex-env/bin/python probe.py diag          # 交叉迁移诊断
../cortex-env/bin/python probe.py report
../cortex-env/bin/python analysis.py            # 逐题配对 bootstrap
../cortex-env/bin/python plot.py
```

环境:`code/cortex-env`(uv venv,Python 3.12;torch 2.6.0+cu124 / transformers 5.20.1 / peft 0.21.2 / datasets / matplotlib),RTX 3060 12GB。全流程约 35 分钟(生成 4 min、训练 11×(1.5–2) min、评测 5 min、诊断 1 min、其余秒级)。训练种子、问题生成种子、评测选项打乱种子分别固定在 `probe.py` 的三个位置,重跑可复现;唯一未固定的是 transformers 版本升级带来的数值差异。


