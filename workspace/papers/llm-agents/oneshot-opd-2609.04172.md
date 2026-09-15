# 2609.04172

ID: 2609.04172
Source: https://arxiv.org/html/2609.04172v1
Authors: Zixuan Fu Bingxiang He Yuxin Zuo Haohuan Huang Affiliation: University of Chinese Academy of Sciences Jinqian Zhang Affiliation: Tsinghua University Ruhang Xiao Affiliation: Northeastern University Cheng Qian Affiliation: University of Illinois Urbana-Champaign Qinyu Luo Affiliation: Johns Hopkins University * Equal Contribution. † Project Lead. ‡ Corresponding Authors. Code: https://github.com/Thinking-Space/One-Shot-OPD . hebx24@mails.tsinghua.edu.cn, {dingning,xcj}@tsinghua.edu.cn Huan-ang Gao Affiliation: Tsinghua University Yudong Wang Affiliation: Tsinghua University Zhiyuan Liu Affiliation: Tsinghua University Ning Ding Chaojun Xiao



\setheadertext

One-Shot On-Policy Distillation \pretitlefigure

[element]
Figure 1 : A single query recovers most of full-data OPD’s gain on mathematical reasoning (left), and the effect holds across domains and model families. Under multi-teacher OPD, 16 queries per domain match full-data training (right).
[/element]

# Rethinking On-Policy Distillation of Large Language Models II: One Training Example

Zixuan Fu

  

Bingxiang He

  

Yuxin Zuo

  

Haohuan Huang

Affiliation: 
University of Chinese Academy of Sciences

  

Jinqian Zhang

Affiliation: 
Tsinghua University

  

Ruhang Xiao

Affiliation: 
Northeastern University

  

Cheng Qian

Affiliation: 
University of Illinois Urbana-Champaign

  

Qinyu Luo

Affiliation: 
Johns Hopkins University
*
Equal Contribution.
†
Project Lead.
‡
Corresponding Authors.

Code:
https://github.com/Thinking-Space/One-Shot-OPD
.

hebx24@mails.tsinghua.edu.cn, {dingning,xcj}@tsinghua.edu.cn

  

Huan-ang Gao

Affiliation: 
Tsinghua University

  

Yudong Wang

Affiliation: 
Tsinghua University

  

Zhiyuan Liu

Affiliation: 
Tsinghua University

  

Ning Ding

  

Chaojun Xiao

###### Abstract

On-policy distillation (OPD) combines student-generated rollouts with dense token-level supervision from a teacher. Existing work has mainly studied its algorithmic behavior, leaving the role of training data unclear. We examine this role at the data-minimal limit by training on a single query. One-shot OPD keeps improving for hundreds of steps and recovers most of full-data OPD’s gain across task domains and model families. We explain this result through the states visited during training and the rate at which the student aligns with the teacher. We measure state coverage , the fraction of the states full-data OPD visits that a query set’s rollouts reach. A single query already reaches $71.5\%$ , most of it within the first 100 steps. Adding semantically distinct queries raises coverage and validation accuracy together, until 16 queries reach $98.9\%$ and match full-data training. Yet alignment slows at a similar pace whether OPD trains on one query or the whole dataset, and even a fixed set of states takes hundreds of steps to absorb. OPD is therefore data-overfed but algorithm-starved. Its rollouts quickly expose broad supervision, while the student absorbs that supervision increasingly slowly. The state-coverage result extends to multi-teacher OPD, where 16 semantically diverse queries per domain match full-data MOPD. As a further stress test, content-light templates and off-domain WildChat queries also approach the real-query baseline. Task content and induced state coverage can therefore come apart. We hope these findings direct future work toward the step efficiency of OPD, and prompt a re-examination of the data and the mechanisms behind its recent successes in frontier post-training.

## 1Introduction

On-policy distillation (OPD) is becoming increasingly common in frontier LLM post-training.
Qwen3 [[Yang et al., 2025]] , MiMo [[Xiao et al., 2026]] , GLM-5 [[Zeng et al., 2026]] , DeepSeek-V4 [[Xu et al., 2026]] , and Kimi K3 [[Team et al., 2026]] all use OPD alongside supervised fine-tuning (SFT) and reinforcement learning (RL).
What makes OPD distinctive is its combination of on-policy state visitation and dense distillation supervision [[Gu et al., 2024,Agarwal et al., 2024]] .
The student samples its own rollouts, while the teacher provides the full next-token distribution at every visited prefix, yielding dense, token-level supervision rather than the single outcome-level reward typically used in reinforcement learning with verifiable rewards (RLVR).

A growing line of work has systematically investigated OPD’s training dynamics and mechanisms, largely from an algorithmic perspective, to explain how and why the method works [[Li et al., 2026,Fu et al., 2026,Cai et al., 2026,Zhu et al., 2026]] .
Yet no prior work has studied how training data shapes OPD, or how the data and the algorithm interact.
Closing this gap is essential to a complete understanding of OPD.
In RLVR, [Wang et al. [2026a]] introduce one-shot RLVR as an extreme controlled experiment to isolate the role of training data.
We bring the same experimental lens to OPD by training on a single query, a setting we call one-shot OPD.
To our surprise, one-shot OPD produces a learning curve strikingly similar to that of one-shot RLVR as shown in Figure 1 : despite repeatedly training on just one query, the model continues to improve for hundreds of steps and ultimately achieves a substantial gain.
We therefore devote this work to answering a single question:

Why can OPD, when trained on just one example, keep learning for so long and improve by so much ?

[element]
Figure 2 : OPD is data-overfed but algorithm-starved. Data: one query already covers most of the states full-data OPD visits. Algorithm: the student absorbs an ever smaller proportion of the remaining teacher–student gap, no matter how many queries it trains on.
[/element]

This question exposes two aspects of OPD efficiency that standard training entangles: keep learning for so long concerns how fast the algorithm absorbs supervision, whereas improve by so much concerns why a small amount of data supply sufficient supervision.
By reducing the data supply to its minimum, one-shot OPD disentangles the two, and the phenomenon itself proves robust: the gain holds across four task domains and three model families, and persists even on queries the student never solves ( Section 3 ).
Our answer is that OPD is data-overfed but algorithm-starved .
From the data perspective , a query acts on the student through the states its rollouts reach, each state being a prefix $s=(x,y_{<i})$ at which the teacher supplies token-level supervision, so a query set is worth the part of that space it covers. We quantify this by state coverage : we group the states full-data OPD visits into semantic clusters and report the fraction a setting’s rollouts reach. A single query already supplies enough states on its own, covering $71.5\%$ of what full-data OPD visits, and 16 semantically diverse queries cover $98.9\%$ and match full-data training ( Section 4 ). This is why so little data improves the student by so much. From the algorithm perspective , what falls as training proceeds is the absorption rate , the proportion of the remaining teacher–student gap that one update closes, and it declines in much the same way on one query as on all 17k ( Section 5 ), so the pace of a run is a property of OPD rather than of the training set.
In other words, the supervision a single query supplies remains largely unexploited, rate-limited by how quickly an on-policy student can absorb it.

The same mechanism informs data design beyond the one-shot setting.
In multi-teacher OPD (MOPD) [[Xiao et al., 2026,Xu et al., 2026]] , where a single student is trained across several domains and each query is routed to its domain teacher, we find that 16 semantically diverse queries per domain suffice to match full-data training ( Section 6 ).
Taking the data side to its extreme, we further show that even a training input that states no problem can remain effective: off-domain WildChat prompts and content-free templates drive OPD nearly as effectively as the full training set ( Section 7.1 ).
Therefore, an input is useful largely because it starts the student reasoning.
Furthermore, we conduct a controlled comparison with one-shot RLVR on the same query to make the algorithmic contrast concrete: the outcome reward is exhausted once the query is solved on nearly every rollout, whereas OPD’s token-level signal persists throughout training, and OPD’s validation gain over $1000$ steps is more than twice that of RLVR ( Section 7.2 ).

In short, OPD is supplied with more supervision than its algorithm can absorb.
What data curation has to settle is therefore no longer how many problems to collect, but which states an input induces.
We hope this state-level view of the training data can guide future work on selecting queries by the states they induce, and on raising the rate at which a student absorbs them.

## 2Preliminaries

### 2.1Notation

Let $x\sim\mathcal{D}$ denote a training input and $y=(y_{1},\ldots,y_{L})$ a response, with $y_{<i}=(y_{1},\ldots,y_{i-1})$ the prefix up to token position $i$ . We consider two LLMs: a student $\pi_{\theta}$ and a teacher $\pi_{T}$ , each defining a next-token distribution over a shared vocabulary. A trajectory is a pair $(x,y)$ with $y$ sampled autoregressively from a policy given $x$ . We use $i$ to index token positions and $t$ to index optimization steps, writing $\theta_{t}$ for the student parameters at step $t$ . A state is the autoregressive context $s_{i}=(x,y_{<i})$ , the object on which the OPD objective is defined.

### 2.2On-Policy Distillation

On-policy distillation (OPD) samples trajectories from the student and aligns the student to the teacher on the prefixes the student actually visits [[Agarwal et al., 2024]] . OPD can also be viewed as a special case of dense KL-constrained reinforcement learning, where the teacher distribution induces a token-level reward and the KL regularizer has a fixed relative weight [[Yang et al., 2026]] .

In the full-distribution form, OPD minimizes a per-token KL divergence on visited states:

[element]
$\mathcal{L}_{\mathrm{OPD}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_{\theta}}\left[\sum_{i=1}^{L}\operatorname{KL}\!\left(\pi_{\theta}(\cdot\mid s_{i})\,\|\,\pi_{T}(\cdot\mid s_{i})\right)\right].$ (1)
[/element]

In practice, we estimate it with a per-token advantage:

[element]
$A_{i}^{\mathrm{OPD}}=\log\pi_{T}(y_{i}\mid s)-\log\pi_{\theta}(y_{i}\mid s),$ (2)
[/element]

where $s=(x,y_{<i})$ . Two properties follow from this formulation: (i) the signal is local in that it depends only on the prediction problem at state $s$ , not on the trajectory’s outcome; and (ii) it is dense in that the teacher supplies a distributional signal at every visited state instead of a terminal scalar reward.

Equation 2 evaluates the correction only at the token the student emitted. A second estimator of the same divergence keeps the student’s $k$ most likely tokens at each visited state and weights them by the student’s probability:

[element]
$A_{i}^{\text{top-}k}=\sum_{v\in\mathcal{V}_{i}}\tilde{\pi}_{\theta}(v\mid s)\left[\log\pi_{T}(v\mid s)-\log\pi_{\theta}(v\mid s)\right],$ (3)
[/element]

where $\mathcal{V}_{i}=\operatorname{TopK}(\pi_{\theta}(\cdot\mid s),k)$ and $\tilde{\pi}_{\theta}$ is $\pi_{\theta}$ renormalized over that set. It truncates the divergence to $k$ terms rather than estimating it from one sampled token, trading the distribution’s tail for lower variance. Section 3.1 states which form each run uses. The alignment metrics below are diagnostics computed from the two policies’ logged distributions, so they are available under either form.

### 2.3Dynamic Metrics

#### Gap recovery.

Because evaluation criteria and initial teacher–student performance gaps vary across domains and model pairs, raw score improvements are not directly comparable. We therefore report improvement as a ratio to the initial teacher–student gap. Let $M_{0}$ , $M_{T}$ , and $M_{t}$ denote the evaluation scores of the initial student, the teacher, and the student at optimization step $t$ , respectively. The gap recovery ratio at step $t$ is $\frac{M_{t}-M_{0}}{M_{T}-M_{0}}\times 100\%.$ A value of $0\%$ , $100\%$ , or above $100\%$ indicates no improvement toward the teacher, matching the teacher, or surpassing the teacher, respectively.

#### Full-data recovery.

When the reference is full-data OPD rather than the teacher, we normalize by the gain of full-data OPD instead. Let $M_{F}$ denote the full-data OPD score at the same optimization step. The full-data recovery ratio is $\frac{M_{t}-M_{0}}{M_{F}-M_{0}}\times 100\%.$ A value approaching $100\%$ means that a reduced query set nearly matches the gain of full-data OPD while a value above $100\%$ means it surpasses it.

#### Top-$k$token overlap ratio.

This metric measures agreement between the two policies’ high-probability token sets. Let $S_{t}^{\theta}(s)=\operatorname{TopK}(\pi_{\theta_{t}}(\cdot\mid s),k)$ and $S^{T}(s)=\operatorname{TopK}(\pi_{T}(\cdot\mid s),k)$ . The overlap at optimization step $t$ is

[element]
$O_{t}^{(k)}=\frac{1}{|\mathcal{T}_{t}|}\sum_{i\in\mathcal{T}_{t}}\frac{|S_{t}^{\theta}(s_{i})\cap S^{T}(s_{i})|}{k},$ (4)
[/element]

where $\mathcal{T}_{t}$ indexes the non-padding response positions of the rollouts in the batch collected at step $t$ [[Li et al., 2026]] .

#### Overlap-token advantage.

Inside that agreement, we average the teacher–student log-probability difference over the shared tokens $S_{t}^{\theta}(s_{i})\cap S^{T}(s_{i})$ , weighting each shared token by the probability the student assigns it. The result reports how far apart the two policies remain on the tokens they already rank highly. Because of that weighting, and because it averages over top- $k$ entries rather than over sampled tokens, it sits on a much smaller scale than the teacher–student distance of Section 5.1 , and the levels of the two are not comparable.

## 3The One-Shot Phenomenon

This section presents the experimental setup and results of One-Shot OPD. We investigate how much of full-data OPD’s gain a single query recovers, and how robust that gain is .

Takeaways

One query is enough to induce robust OPD gains.

A single query recovers most of the teacher–student gap across task domains and model families. The gain is insensitive to query difficulty, response length, and sampling temperature, and a query the student never solves is as effective as one it always solves.

### 3.1Experimental Setup

We establish one-shot OPD across math, code generation, instruction following, and agentic tool use for controlled analysis to test generality.

#### Models.

[element]
Table 1: Student–teacher pairs used in our experiments. Setting Student Teacher Mathematical reasoning DeepSeek-R1-Distill-Qwen-1.5B (R1-Distill-1.5B) JustRL-DeepSeek-1.5B (JustRL-1.5B) Llama-3.2-3B-Instruct (Llama-3B-It) GT-Llama-3.2-3B-Instruct-MATH (GT-Llama-3B-Math) OLMo-3-7B-Instruct-DPO (OLMo-7B-It-DPO) OLMo-3-7B-Instruct (OLMo-7B-It) Code generation DeepSeek-R1-Distill-Qwen-1.5B (R1-Distill-1.5B) Nemotron-Research-Reasoning-Qwen-1.5B-v2-RLVE (Nemotron-1.5B) Instruction following DeepSeek-R1-Distill-Qwen-1.5B (R1-Distill-1.5B) UltraData-IF-1.5B Agentic tool use Qwen2.5-Coder-1.5B-Instruct (Qwen-Coder-1.5B) Hammer2.1-1.5b (Hammer-1.5B)
[/element]

For each domain, we pair a student with a post-trained teacher from the same family. Table 1 lists the full identifiers and the short names used throughout. Math, code generation, and instruction following share the student R1-Distill-1.5B [[Guo et al., 2025]] , whose teachers are, respectively, JustRL-1.5B [[He et al., 2025b]] , Nemotron-1.5B [[Zeng et al., 2025]] , and an instruction-following variant we post-train from the same student (Appendix A.1 ). For agentic tool use, the student is Qwen-Coder-1.5B [[Hui et al., 2024]] and the teacher is Hammer-1.5B [[Lin et al., 2024]] . To test whether the phenomenon is specific to Qwen-based pairs, we add two mathematical-reasoning pairs from other families: Llama-3B-It [[Grattafiori et al., 2024]] with GT-Llama-3B-Math [[Zhang et al., 2025]] , and OLMo-7B-It-DPO with OLMo-7B-It [[Olmo et al., 2025]] .

#### Datasets.

The domain-specific training sets comprise DAPO-Math-17K [[Yu et al., 2026]] for mathematical reasoning, Open-R1 Codeforces [[Penedo et al., 2025]] for code generation, a sampled subset of UltraData-SFT-2605 [[OpenBMB, 2026]] for instruction following, and xLAM-function-calling-60K [[Liu et al., 2024]] for agentic tool use. For One-shot OPD in mathematics, we select three queries spanning easy, medium, and hard initial difficulty, scored by the student’s pass rate over 8 rollouts before training ( $8/8$ , $4/8$ , and $0/8$ , respectively). In each of the other domains, the one-shot query is sampled randomly from the corresponding training set. Appendix A.1 details the dataset preprocessing and the selected one-shot queries.

#### Training.

We implement OPD in veRL [[Sheng et al., 2025]] .
The mathematical-reasoning runs of this section optimize the top- $k$ advantage of Eq. 3 with $k=16$ ; the code, instruction-following, and agentic runs optimize the sampled-token advantage of Eq. 2 .
Every update uses a batch of $64$ rollouts. We use AdamW with a learning rate of $10^{-6}$ , a rollout temperature of $1.0$ , and a gradient clip norm of $1.0$ ; remaining hyperparameters are listed in Table 3 .

#### Evaluation.

For math, we evaluate on MATH-500 [[Hendrycks et al., 2021]] , AMC 2023 [[Li et al., 2024]] , and AIME 2025 [[Balunovic et al., 2025]] , sampling 16 responses per problem and reporting avg@16 accuracy.
For code generation, LiveCodeBench v6 (LCB v6) [[Jain et al., 2025]] uses 3 sampled solutions per problem and reports avg@3 with the official execution-based evaluator.
For instruction following, Multi-IF [[He et al., 2024]] evaluates three-turn conversations and reports the final-turn score averaged over its eight languages.
For agentic tool use, BFCL v3 [[Patil et al., 2025]] reports avg@8 over the evaluated subsets. And the response caps are $31{,}744$ tokens
for mathematics, $65{,}536$ for LCB v6, $16{,}384$ per turn for Multi-IF, and $4{,}096$ for BFCL v3. Unless stated otherwise, mathematics figures report validation accuracy macro-averaged over MATH-500, AMC 2023, and AIME 2025, and a dashed line marks the teacher.

[element]
Figure 3 : Validation accuracy and Training dynamics of one-shot OPD on mathematical reasoning.
[/element]

[element]
Figure 4 : Validation accuracy of one-shot OPD across 3 student–teacher families, on mathematical reasoning averaged over MATH-500 and AMC 2023. AIME 2025 is excluded because the Llama pair scores near zero on it.
[/element]

[element]
Figure 5 : Validation accuracy of one-shot OPD across three task domains.
[/element]

[element]
Figure 6 : Validation Accuracy of one-shot OPD under varying query difficulty, response-length cap on the hard query, and sampling temperature on the medium query.
[/element]

### 3.2Experimental Results

#### One-shot OPD recovers most of full-data OPD’s gain in mathematics.

Figure 3 shows that training on a single query improves accuracy on MATH-500, AIME 2025, and AMC 2023, approaching the full-data OPD on all three benchmarks. Averaged over them, one-shot OPD reaches $68.5$ against $69.8$ for full-data OPD, recovering $69\%$ of the teacher–student gap and $87\%$ of full-data OPD’s gain at step 300. Beyond step $300$ both curves stay within a band of about $3$ points, and the recovered fraction ranges from $62\%$ to $89\%$ through step $1000$ , where one-shot OPD reaches $68.4$ against $72.1$ and recovers $72\%$ of full-data OPD’s gain (Figure 1 ). The training dynamics follow the same alignment process under both settings: the top-16 overlap ratio climbs to the full-data level, the overlap-token advantage approaches zero, and the absolute entropy gap nearly closes. Thus, even when all rollouts originate from a single query, OPD improves accuracy while progressively aligning the student distribution with the teacher on the visited states.

#### The one-shot effect is robust across families and task domains.

Figure 4 shows that one-shot OPD improves mathematical reasoning across all three student–teacher families. At the final checkpoint, the averaged scores increase from $77.1$ , $28.2$ , and $70.8$ for the respective student baselines to $85.5$ , $40.2$ , and $82.4$ for R1-Distill-1.5B, Llama-3B-It, and OLMo-7B-It-DPO. Figure 5 further shows that the effect extends beyond mathematical reasoning: on code generation, instruction following, and agentic tool use, one-shot OPD recovers $73\%$ , $66\%$ , and $64\%$ of the corresponding teacher–student gaps, respectively. Together, these results show that the one-shot effect is robust across both model families and task domains.

#### The one-shot effect is robust to query and rollout properties.

We next examine whether one-shot OPD remains effective under varying query difficulty, response-length budgets, and rollout sampling temperature. As shown in Figure 6 , one-shot OPD works across easy, medium, and hard queries, even though the fraction of correct rollouts evolves very differently in the three settings (Appendix A.1 ): the easy query is solved on nearly every step, the medium query becomes largely solvable during training, and the hard query is never solved. Tightening the response-length cap and lowering the rollout temperature both preserve the one-shot gain. Together, these results show that one-shot OPD is robust to query difficulty, response-length budget, and rollout sampling temperature.

## 4Data Perspective: Abundant States

Section 3 showed that One-Shot OPD recovers much of full-data OPD’s gain robustly. This section asks why training on a single query produces such a large gain, and whether the same explanation also covers the cases where n-shot OPD can match full-data OPD . 1 1 1 The runs analyzed here and in Section 5 optimize the sampled-token advantage of Eq. 2 , rather than the top- $k$ advantage used by the mathematics runs of Section 3 , so their absolute accuracies differ slightly from those reported there. Every comparison below is between runs that share this form.

Takeaways

•

One query already reaches a large part of the full-data state space.

What OPD consumes is primarily states rather than queries. In practice, repeated rollouts from a single query can reach 
$71.5\%$
 state coverage, with most of it within the first 
$100$
 steps.

•

Diverse but few queries reach sufficient states and rival full-data performance.
 Adding semantically distinct queries raises both state coverage and validation accuracy, and 16 queries already match full-data OPD. The value of an extra example lies in whether it covers states missed by earlier ones.

### 4.1State Coverage of One-Shot OPD

#### Setup.

OPD trains on states rather than queries. A query $x$ and a sampled response $y$ produce one state $s=(x,y_{<i})$ at every token position $i$ , each paired with a target distribution from the teacher. With $64$ rollouts per update, even a single query yields tens of thousands of supervised states, so query count can substantially understate the amount of supervision available to OPD.

Hypothesis

A single query can provide enough supervision to recover most of full-data OPD’s gain if its rollouts cover a broad part of the state space visited by full-data OPD.

To test this hypothesis, we measure the breadth rather than the raw number of visited states, since every generated token creates a distinct prefix. We call this measure state coverage : we represent states in a shared representation space, partition that space into clusters, and report the fraction reached by a setting’s rollouts.

•

Representation. We represent each state by its teacher signature $h_{T}(s)$ , the teacher’s final-layer hidden vector at the state’s last token.

•

Reference space. We pool states visited by full-data OPD over its entire run on DAPO-Math-17K, sampling $8$ uniformly spaced positions from each rollout. A held-out portion of these rollouts takes no part in defining the clusters and is measured as a setting of its own, full data (held-out) . 2 2 2 Three of every five collected rollouts form the pool; the other two are the held-out setting, so it is what full-data OPD reaches on a fresh sample under the same budget.

•

Clusters. After PCA, $K$ -means partitions the reference signatures into $K=200$ clusters. Each state falls in the nearest one, denoted $c(s)$ .

The state coverage of a set of states $S$ is then the fraction of the $K$ clusters it reaches:

[element]
$\mathrm{Cov}(S)=\frac{1}{K}\left|\left\{\,c(s)\;:\;s\in S\,\right\}\right|.$ (5)
[/element]

Coverage records which clusters a setting reaches, not how often it visits them. We fit the clusters once and measure every setting over the same $300$ steps. Full data (held-out) reaches $100\%$ on this budget, so the top of the scale is a level full-data OPD attains rather than a maximum the construction guarantees. Appendix B.1 gives the construction and schedule, and shows that the comparisons below are stable across choices of $K$ , reference set, and state positions.

#### One-shot OPD reaches$71.5\%$of the state space.

[element]
Figure 7 : State coverage and validation accuracy of one-shot OPD on the medium query, against full-data OPD. Coverage is measured against the state space full-data OPD reaches, which is therefore the $1.0$ level rather than a separate curve.
[/element]

Figure 7 shows that one-shot OPD reaches $71.5\%$ state coverage by step $300$ . Most of this coverage appears early: the run reaches $65.9\%$ by step $100$ , then adds only $5.6$ percentage points over the next $200$ steps. Repeated rollouts from the same query therefore continue to discover new clusters, but at a sharply diminishing rate.
The same run raises validation accuracy from $59.1$ to $66.9$ , compared with $70.8$ for full-data OPD at step $300$ . These results show the same pattern on the data side: a single query already covers most of the state-space clusters reached by full-data OPD and produces a substantial validation gain. One-shot OPD is therefore small in query count, yet broad in the supervision it generates.

### 4.2Diversity Expands State Coverage

The analysis above shows that one query already reaches most of the full-data state space, but it leaves the causal question open: a run that trains well might simply visit more states along the way, with the extra states doing none of the work. We therefore ablate state coverage directly, varying how many distinct states the training data reaches while holding the rest of the setup fixed.

#### Setup.

We raise data diversity from two directions, each with its own control. (i) Response diversity (off-policy). We sample $64$ trajectories once from the initial student on the one-shot query and keep that pool fixed, which also removes a confound of on-policy training, where the states keep changing as the student does. Nested subsets retain $1$ , $4$ , $16$ , or all $64$ trajectories, and the retained ones are repeated to fill each batch of $64$ . Every condition shares the query, the batch size, and the optimization budget, so the number of distinct trajectories, and hence of states, is the only quantity that changes. (ii) Query diversity (on-policy). These runs instead change the query set, and all follow the mathematical configuration. Starting from the medium query used by one-shot OPD, we build 4- and 16-shot sets by clustering DAPO-Math-17K with BGE-M3 [[Chen et al., 2024a]] and taking one representative per semantic cluster, so each query added to the ladder is semantically distinct from those already in it, and we compare these sets with full-data OPD.
We report validation accuracy for every run, and state coverage for the on-policy ones, where the rollouts are those the student actually generates during training. Off-policy scores are averaged over the five evaluation checkpoints from step $300$ to $500$ . On the coverage side, the full-data bar is the held-out split that sets the $100\%$ ceiling.

[element]
Figure 8 : Validation accuracy and state coverage as the training data is made more diverse, on the response side by off-policy training on distinct trajectories and on the query side by on-policy training on distinct queries.
[/element]

#### Data diversity is state diversity, and a few queries match full-data OPD.

Figure 8 shows off-policy validation accuracy rising monotonically with the number of distinct trajectories. Everything else is held fixed, so the only thing the extra trajectories add is states. The on-policy runs reach the same conclusion from the query side. Semantically distinct queries raise validation accuracy until 16 of them match full-data OPD, and they match it on state coverage at the same point, which rises from $71.5\%$ for one query to $98.9\%$ for 16. Every query on the ladder is a new semantic cluster, so it raises diversity and count together, and two controls in Appendix B.2 separate them: holding the count at 16, drawing those queries from 16 semantic clusters rather than from one raises state coverage and validation accuracy sharply, whereas the order in which the student trains on a fixed query set does not matter. What an added query is worth is therefore set by whether it reaches new states, and a few diverse queries already reach nearly all the states full-data OPD does.

## 5Algorithm Perspective: Slow Alignment

Section 4 analyzed the one-shot gain from the data side: what a query is worth to the student is set by the states its rollouts reach. Turning to the algorithm side, this section asks why one query keeps producing gains over so many steps .

Takeaways

•

Alignment slows over time, and this trend is insensitive to query count.
 The student continuously narrows the teacher–student gap throughout the run, but the rate of progress diminishes steadily, which is why a run requires hundreds of steps rather than tens. Increasing the number of queries does not accelerate or decelerate this process.

•

Even a fixed set of states takes hundreds of steps to learn from.

A run that trains on the same states from beginning to end still improves over hundreds of steps, so a steady supply of fresh states is not required to keep a run going that long.

### 5.1Alignment Metrics

OPD aligns the student with the teacher. Two quantities describe the state of that alignment at any moment: how much teacher–student disagreement is left, and how fast it is falling. We measure both on the states the student actually visits.

•

Distance. The teacher and the student disagree at each visited position, by the per-token advantage of Eq. 2 . We take the distance $d_{t}$ to be the average size of that disagreement over the positions $\mathcal{T}_{t}$ the student visits at step $t$ :

[element]
$d_{t}=\frac{1}{|\mathcal{T}_{t}|}\sum_{i\in\mathcal{T}_{t}}\left|\log\pi_{T}(y_{i}\mid s_{i})-\log\pi_{\theta_{t}}(y_{i}\mid s_{i})\right|.$ (6)
[/element]

Here $s_{i}=(x,y_{<i})$ is the state at token position $i$ , as defined in Section 2 . Averaging magnitudes keeps positive and negative per-token gaps from cancelling, so $d_{t}$ falls to zero only as the two policies come to agree at the positions the student visits.

•

Absorption rate. The absorption rate $v_{t}$ is then the proportion of that distance one update absorbs,

[element]
$v_{t}=\frac{d_{t}-d_{t+1}}{d_{t}}.$ (7)
[/element]

We track both metrics from step 30, after gradient clipping no longer binds. Since each run measures distance over the states it visits, these runs have settled at different levels by then. We therefore report each run against its own value at step 30, which we refer to as the distance left $d_{t}/d_{30}$ . 3 3 3 Late in training a single pair of consecutive steps is too noisy to read, so $v_{t}$ is estimated over the geometric window $[t/1.4,\,1.4t]$ , which is why it stops short of step $300$ . Appendix C.1 gives the estimators and the evidence behind both conventions.

### 5.2Alignment Slows Similarly Across Data

We track $d_{t}$ and $v_{t}$ for OPD trained on 1, 4, 16, and all DAPO-Math-17k queries, over the same $300$ steps and under the mathematical configuration of Section 3.1 .

[element]
Figure 9 : Teacher–student alignment for OPD trained on 1, 4, 16, and all DAPO-Math-17k queries. Distance (left) and absorption rate (right) are both on logarithmic vertical axes against linear steps.
[/element]

#### The student aligns ever more slowly.

The distance falls for the whole run, so the student is never stuck. What slows is the absorption rate, which falls throughout the run in the right panel of Figure 9 : on the logarithmic distance axis at left a constant rate would trace a straight line, and all four curves bend flatter instead. Each update thus absorbs less of what is left than the last, which is why a run takes hundreds of steps rather than tens.

#### The absorption rate falls in much the same way at every training-set size.

The four runs do this alike: each removes $78\%$ to $84\%$ of its step- $30$ distance by step $300$ , and all four slow by a similar factor during training (Appendix C.1 gives the per-run numbers and the conventions these readings are taken under). One query therefore does not make alignment slower or faster: the pace is a property of OPD rather than of the training set. The learning rate changes how many steps that takes, but not the way the distance falls (Appendix C.2 ).

### 5.3Ablation on Fixed States

On-policy training keeps producing new states as the student changes, and one reading of the hundreds of steps is that the run lasts as long as that supply does. Holding the training states fixed tests that reading directly: if a fixed set of states is exhausted quickly, the off-policy run should finish early.

#### Setup.

We compare two one-shot runs that differ only in where their training states come from. The always-on-policy run draws fresh rollouts from the current student at every step. The always-off-policy run reuses the $64$ trajectories of Section 4.2 , sampled once from the initial student, for every update, so its training states stay fixed while the student changes.

#### Holding the states fixed leaves the run just as long.

[element]
Figure 10 : Validation accuracy and alignment metrics for the on-policy and off-policy one-shot runs.
[/element]

Figure 10 shows the off-policy run gains accuracy steadily for about $200$ steps and then stops, so its gain builds up over hundreds of steps rather than over the first few updates. The alignment metrics move just as slowly: the off-policy run’s top- $16$ token overlap rises and its overlap-token advantage approaches zero only over hundreds of steps. Removing that supply therefore does not shorten the run.
This suggests that a fixed set of states is enough on its own to keep an OPD run going for hundreds of steps, so the length of a run is not explained by how long fresh states keep arriving.

## 6One-Shot Extends to Multi-Teacher OPD

Section 4.2 showed that a small, semantically diverse query set can match full-data OPD within one domain. Modern post-training pipelines often use multi-teacher on-policy distillation (MOPD) [[Xiao et al., 2026,Xu et al., 2026]] , where one student is trained on several domains in a single run and each query is routed to its domain teacher. This section asks whether the same query-diversity result carries over: can a small, diverse query set for each domain match full-data MOPD?

Takeaways

16 semantically diverse queries within each domain are sufficient to match full-data MOPD, which reaches nearly the same average validation accuracy as separate full-data OPD.

[element]
Figure 11 : Validation accuracy of one-shot, 16-shot, and full-data MOPD.
[/element]

#### Setup.

We train one R1-Distill-1.5B student with MOPD on mathematical reasoning, code generation, and instruction following. The domain teachers from Section 3.1 are JustRL-1.5B for mathematics, Nemotron-1.5B for code, and UltraData-IF-1.5B for instruction following. We exclude agentic tool use because it uses a different student, Qwen-Coder-1.5B. The three settings are one-shot, 16-shot, and full-data MOPD. As in Section 4.2 , the 16-shot setting selects one representative from each of 16 BGE-M3 [[Chen et al., 2024a]] semantic clusters fitted separately within each domain. All settings run for $300$ steps with the hyperparameters in Table 8 .

#### Full-data MOPD matches separate full-data OPD.

At step $300$ , full-data MOPD raises the average validation accuracy from $43.5$ to $52.8$ , recovering $79\%$ of the teacher–student gap. The three corresponding full-data OPD runs reach $53.8$ . This $1.0$ -point difference shows that MOPD remains effective when all three domains are trained in a single run and establishes full-data MOPD as the reference for the query-diversity comparison below.

#### 16 diverse queries per domain match full-data MOPD.

Increasing the query count from 1 to 16 per domain raises the average validation accuracy from $50.1$ to $52.9$ at step $300$ . Full-data MOPD reaches $52.8$ , so 16-shot MOPD recovers $101\%$ of full-data MOPD’s gain. The result is not driven by one domain: calculated separately, the corresponding proportions are $93\%$ for mathematics, $136\%$ for code generation, and $109\%$ for instruction following. Because 16-shot and full-data MOPD use the same rollout and optimization budgets, expanding each domain’s query set beyond 16 diverse queries produces no additional average gain in this setting.

Together, these results are consistent with a state-space account for MOPD: semantically diverse queries can induce complementary states within each domain, allowing 16 queries per domain to match full-data MOPD.

## 7Discussion

The mechanism above shifts attention from prompts to the regions of reasoning states they induce. An input is useful when the student’s rollouts reach regions where the teacher can provide useful supervision. Stating a problem is only one way to reach them. This view raises two broader questions: how much task content must the input itself contain, and why does the same one-shot setting behave differently under OPD and RLVR? We discuss these questions through content-light and off-domain inputs, followed by a controlled comparison with one-shot RLVR.

### 7.1Inputs as State Generators

The state-generator view suggests that task content may be unnecessary if an input still induces trajectories that the teacher can supervise. We stress-test this prediction with an empty user turn ending in <think> , the same template with a simple domain system prompt, and general-domain queries from WildChat [[Zhao et al., 2024]] (Figure 12 ). The WildChat set contains $192{,}824$ predominantly English queries from general conversation, creative writing, role-playing, and rewriting. Conservative heuristics label only $0.17\%$ as mathematics-related and $2.63\%$ as code-related.

[element]
(a) Base template <|begin_of_sentence|> <|User|> <think> \n (b) System template — Code <|begin_of_sentence|> Solve competitive programming problems. Read from stdin, write to stdout. <|User|> <think> \n Figure 12 : The two content-light templates used as OPD training inputs.
[/element]

[element]
Figure 13 : Validation accuracy and actor entropy for mathematics OPD trained on the real training set, on a content-light template, or on general-domain WildChat queries.
[/element]

#### Content-light and off-domain inputs can still induce useful states.

All three conditions track the real-query baseline, which lifts the three-benchmark average from $59.1$ to $69.8$ , and all three reach that level on a third to a half of its rollout tokens (Figure 13 ). Code generation gives the same picture (Appendix E.1 ).
However, the requirement is not empty: a scaffold that closes the block at once, <think>\n</think>\n , collapses into short meta-level replies we cannot train on.
These results do not imply that task content is generally dispensable. They show that content is not the only source of useful signal in OPD. An input also acts as a state generator, and even content-light or off-domain inputs can sometimes lead the student to states that support useful teacher supervision. Explicit domain content changes final performance by about one point in these experiments. This provides a complementary view of OPD data efficiency. Data quality depends not only on the problems being collected, but also on the state regions they induce and the teacher targets available there.

### 7.2Relation to One-Shot RLVR

One-shot OPD and one-shot RLVR [[Wang et al., 2026a]] share the same surface phenomenon that a single query supports continued improvement over hundreds of updates in OPD and over a thousand updates in RLVR. The source of learning is different. OPD receives a teacher target at every visited state, while RLVR receives a verifiable outcome for each trajectory. We compare them on the same medium query to isolate this difference (Figure 14 ).

[element]
Figure 14 : Validation accuracy and training dynamics for one-shot OPD and one-shot RLVR on the same query. Both take a batch of $64$ queries per step. RLVR draws $8$ rollouts per query and scores each by outcome under GRPO. OPD draws $1$ and scores every sampled token against the teacher. Training accuracy is the fraction of rollouts that solve the training query.
[/element]

#### Useful inputs differ across objectives.

RLVR updates from trajectory outcomes. Its training input must therefore define a task whose sampled outcomes can be verified. The outcomes must also vary enough to produce a useful advantage. OPD instead updates from the teacher–student gap at each visited state. It can receive useful signal whenever a prompt leads to trajectories that expose such gaps. The value of an OPD input is therefore not tied directly to verifiability or difficulty. Our results further show that even weak domain alignment can work in the settings we study. Coverage of the induced state regions still matters (Section 4 ), but obtaining such coverage is less restrictive than obtaining tasks with verifiable outcomes.

#### OPD extracts more signal from the same query.

Both algorithms obtain fresh rollouts from the training query at every step. Over $1000$ steps, OPD closes $72\%$ of its gap to the teacher. Its gain in validation accuracy is more than twice that of RLVR. This advantage remains when the two methods are matched by rollout tokens rather than training steps (Figure 14 ).

The RLVR signal weakens as the model learns to solve the training query. Its groups soon become nearly unanimous, which leaves no advantage under GRPO. OPD does not depend on variation in trajectory outcomes. It can continue learning from local teacher–student gaps after the query is solved, although these gaps gradually shrink. One-shot RLVR is therefore limited by the outcome variation available from its query. One-shot OPD is instead limited by how quickly it absorbs dense teacher supervision, as shown in Sections 4 and 5 . The objectives also imply different ceilings. RLVR is not tied to a teacher’s distribution, while OPD is trained to match one.

## 8Related Work

#### On-policy distillation.

On-policy distillation (OPD) trains the student on its own rollouts with teacher supervision [[Song and Zheng, 2026]] . MiniLLM [[Gu et al., 2024]] optimized this under reverse KL by policy gradient, and GKD [[Agarwal et al., 2024]] generalized it across divergences and on- and off-policy mixtures. [Yang et al. [2026]] cast OPD as dense KL-constrained RL whose per-token log-ratio acts as an implicit reward. OPD now appears across frontier post-training stacks [[Yang et al., 2025,Zeng et al., 2026,Xiao et al., 2026,Xu et al., 2026,Blakeman et al., 2026]] , several of which adopt the multi-teacher form we study in Section 6 , routing each query to a domain specialist. A growing line explains why OPD is effective, and does so entirely on the algorithm side. One thread studies its objective and update geometry: the dense per-token target replaces the single sequence-level scalar of outcome-reward RL, which in parameter space steers an early-locked, low-rank update subspace [[Cai et al., 2026,Shen et al., 2026]] and can be recast as token-level policy gradient [[Ko et al., 2026]] , connecting to analyses that separate the dense updates of SFT from the localized subnetworks of RL [[Mukherjee et al., 2026,Zhu et al., 2025]] . A second thread charts when OPD succeeds or fails: a thinking pattern shared between teacher and student and the limits of an over-strong teacher [[Li et al., 2026]] , which tokens are actually learnable [[Armandpour et al., 2026,Wang et al., 2026b]] , and recurring failure modes with simple fixes [[Zhu et al., 2026,Fu et al., 2026]] . These accounts take the training set as given. We instead ask how many training inputs OPD needs, and find that the binding resource is not the data but the rate at which the student absorbs it.

#### Data-efficient reasoning post-training.

A parallel line studies how little data elicits strong reasoning. Under sparse outcome-reward RL, [Wang et al. [2026a]] show that a single example keeps improving a model for over a thousand steps, the precedent we adapt to the dense-signal setting of OPD. LIMR [[Li et al., 2025]] prunes the RLVR training set several-fold by ranking each example’s learning impact. With supervised fine-tuning, LIMA [[Zhou et al., 2023]] , LIMO [[Ye et al., 2025]] , and s1 [[Muennighoff et al., 2025]] elicit competitive alignment or reasoning from on the order of a thousand curated examples. Within OPD, scheduling supervision over reasoning prefixes cuts training cost severalfold [[Zhang et al., 2026]] . Broader post-training work selects high-value subsets by influence or quality [[Xia et al., 2024,Chen et al., 2024b,He et al., 2025a,He et al., 2025c]] . These works share our data-efficiency theme, but their supervision still depends on curated problems, delivered as a sparse outcome scalar in the RL case or as reference solutions in the SFT case. OPD instead supplies a dense per-token target at each visited prefix and remains effective on queries the student never solves (Section 3 ), where the teacher signal stays informative even on incorrect rollouts [[Armandpour et al., 2026]] , so the limiting factor moves from whether a useful signal is available, the standing difficulty under sparse reward, to how fast an on-policy student absorbs an already-abundant one (Section 5.2 ).

#### Synthetic and unsupervised post-training data.

A growing body of work reduces post-training’s dependence on curated, human-written data. One direction synthesizes the inputs: Self-Instruct [[Wang et al., 2023]] and Evol-Instruct [[Xu et al., 2024]] bootstrap instructions from a language model, and Magpie [[Xu et al., 2025]] elicits full instruction–response pairs from nothing but a chat-template prefix. A second direction removes ground-truth labels from RL [[He et al., 2026]] , replacing verifiable rewards with self-generated signals such as model confidence or entropy [[Zhao et al., 2026b,Agarwal et al., 2026]] and majority agreement across rollouts [[Zuo et al., 2026]] , or drawing reward from unlabeled corpora [[Dong et al., 2025]] . A third removes external problems altogether, letting the model propose its own tasks [[Zhao et al., 2026a,Zweiger et al., 2026]] . These lines relax supervision on the reward or the query source. Our template- and WildChat-OPD results (Section 7.1 ) push the query side further: because the teacher supplies dense supervision at each visited prefix, the query’s role reduces to placing the student in a useful reasoning state, so an off-domain chat log or even a content-free template drives OPD nearly as well as a curated problem.

## 9Conclusion, Limitations, and Future Work

#### Conclusion.

Motivated by one-shot RLVR [[Wang et al., 2026a]] , we asked whether a single query is also enough for OPD, and it is: one-shot OPD keeps the student improving for hundreds of steps and recovers most of full-data OPD’s gain, across model families and task domains. The data and the algorithm sides give one answer: OPD is data-overfed but algorithm-starved. A query acts on the student through the states its rollouts reach, and one query already covers most of the state space full-data OPD visits, while 16 semantically diverse queries match full-data training in single-domain OPD and in MOPD alike. What limits a run is the absorption rate: the proportion of the remaining teacher–student gap one update absorbs keeps falling, at the same pace on one query as on all 17k, so the optimizer rather than the training set sets how long a run takes. An input need only start the student reasoning, which turns data design from collecting problems into choosing teachers.

#### Limitations.

State coverage is a semantic-level proxy measured against a reference space built from full-data rollouts, so it reports how much of that space a query set reaches rather than what it covers on its own, and it weights every cluster equally, however often the cluster is visited and however much teacher signal it still carries. What sets the absorption rate is still open. Our MOPD runs use three domains with one teacher each, leaving open how far the diversity result extends as teachers are added.

#### Future work.

Three directions follow.

•

Selecting data by state coverage: the question is no longer how many queries to collect but which states they induce, so state coverage becomes a selection criterion once it can be estimated from the queries themselves, without a full-data run to measure it against.

•

Making training more step-efficient, by reusing a batch for several epochs under a trust region on the per-token gap, or weighting tokens by how much teacher signal they still carry.

•

Growing the setting, to MOPD with more teachers and domains, and to larger students, agentic tool use, and long contexts, where training data is most expensive to collect.

## References

Agarwal 
et al.
 (2024)

R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos Garea, M. Geist, and O. Bachem

On-policy distillation of language models: learning from self-generated mistakes
.

In 
International Conference on Learning Representations
,

Vol. 
2024
, 
pp. 21246–21263
.

Cited by: 
§1
,

§2.2
,

§8
.

Agarwal 
et al.
 (2026)

S. Agarwal, Z. Zhang, L. Yuan, J. Han, and H. Peng

The unreasonable effectiveness of entropy minimization in llm reasoning
.

Advances in Neural Information Processing Systems
 
38
, 
pp. 107150–107180
.

Cited by: 
§8
.

Armandpour 
et al.
 (2026)

M. Armandpour, F. Ilhan, D. Harrison, A. Jaiswal, D. N. Hoang, F. Faghri, Y. Zhang, M. Cho, and M. Farajtabar

Unmasking on-policy distillation: where it helps, where it hurts, and why
.

arXiv preprint arXiv:2605.10889
.

Cited by: 
§8
,

§8
.

Balunovic 
et al.
 (2025)

M. Balunovic, J. Dekoninck, I. Petrov, N. Jovanović, and M. Vechev

Matharena: evaluating llms on uncontaminated math competitions
.

Advances in Neural Information Processing Systems
 
38
.

Cited by: 
§3.1
.

Blakeman 
et al.
 (2026)

A. Blakeman, A. Thomas, A. Jhunjhunwala, A. Gupta, A. Khattar, A. Rajfer, A. Renduchintala, A. Asif, A. Vavre, A. F. Miranda, 
et al.

Nemotron 3 ultra: open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning
.

arXiv preprint arXiv:2606.15007
.

Cited by: 
§8
.

Cai 
et al.
 (2026)

Y. Cai, D. Cao, L. Lin, C. Luo, X. Xu, K. Yang, W. Liu, S. Yang, T. Zhao, G. Sun, 
et al.

Learning to foresee: unveiling the unlocking efficiency of on-policy distillation
.

arXiv preprint arXiv:2605.11739
.

Cited by: 
§1
,

§8
.

Chen 
et al.
 (2024a)

J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu

Bge m3-embedding: multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation
.

arXiv preprint arXiv:2402.03216
 
4
 (
5
).

Cited by: 
§4.2
,

§6
.

Chen 
et al.
 (2024b)

L. Chen, S. Li, J. Yan, H. Wang, K. Gunaratna, V. Yadav, Z. Tang, V. Srinivasan, T. Zhou, H. Huang, 
et al.

Alpagasus: training a better alpaca with fewer data
.

In 
International Conference on Learning Representations
,

Vol. 
2024
, 
pp. 34767–34797
.

Cited by: 
§8
.

Dong 
et al.
 (2025)

Q. Dong, L. Dong, Y. Tang, T. Ye, Y. Sun, Z. Sui, and F. Wei

Reinforcement pre-training
.

arXiv preprint arXiv:2506.08007
.

Cited by: 
§8
.

Fu 
et al.
 (2026)

Y. Fu, H. Huang, K. Jiang, J. Liu, Z. Jiang, Y. Zhu, and D. Zhao

Revisiting on-policy distillation: empirical failure modes and simple fixes
.

arXiv preprint arXiv:2603.25562
.

Cited by: 
§1
,

§8
.

Grattafiori 
et al.
 (2024)

A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, 
et al.

The llama 3 herd of models
.

arXiv preprint arXiv:2407.21783
.

Cited by: 
§3.1
.

Gu 
et al.
 (2024)

Y. Gu, L. Dong, F. Wei, and M. Huang

Minillm: knowledge distillation of large language models
.

In 
International Conference on Learning Representations
,

Vol. 
2024
, 
pp. 32694–32717
.

Cited by: 
§1
,

§8
.

Guo 
et al.
 (2025)

D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, 
et al.

Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning
.

arXiv preprint arXiv:2501.12948
.

Cited by: 
§3.1
.

He 
et al.
 (2025a)

B. He, N. Ding, C. Qian, J. Deng, G. Cui, L. Yuan, H. Hong, H. Gao, L. Huang, H. Xue, 
et al.

The right time matters: data arrangement affects zero-shot generalization in instruction tuning
.

In 
Findings of the Association for Computational Linguistics: ACL 2025
,

pp. 222–243
.

Cited by: 
§8
.

He 
et al.
 (2025b)

B. He, Z. Qu, Z. Liu, Y. Chen, Y. Zuo, C. Qian, K. Zhang, W. Chen, C. Xiao, G. Cui, 
et al.

Justrl: scaling a 1.5 b llm with a simple rl recipe
.

arXiv preprint arXiv:2512.16649
.

Cited by: 
§3.1
.

He 
et al.
 (2025c)

B. He, W. Zhang, J. Song, C. Qian, Z. Fu, B. Sun, N. Ding, H. Hong, L. Huang, H. Xue, 
et al.

Air: a systematic analysis of annotations, instructions, and response pairs in preference dataset
.

arXiv preprint arXiv:2504.03612
.

Cited by: 
§8
.

He 
et al.
 (2026)

B. He, Y. Zuo, Z. Liu, S. Zhao, Z. Fu, J. Yang, C. Qian, K. Zhang, Y. Fan, G. Cui, 
et al.

How far can unsupervised rlvr scale llm training?
.

In 
International Conference on Learning Representations
,

Vol. 
2026
, 
pp. 14823–14865
.

Cited by: 
§8
.

He 
et al.
 (2024)

Y. He, D. Jin, C. Wang, C. Bi, K. Mandyam, H. Zhang, C. Zhu, N. Li, T. Xu, H. Lv, 
et al.

Multi-if: benchmarking llms on multi-turn and multilingual instructions following
.

arXiv preprint arXiv:2410.15553
.

Cited by: 
§3.1
.

Hendrycks 
et al.
 (2021)

D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt

Measuring mathematical problem solving with the math dataset
.

NeurIPS
.

Cited by: 
§3.1
.

Hui 
et al.
 (2024)

B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu, J. Zhang, B. Yu, K. Lu, 
et al.

Qwen2. 5-coder technical report
.

arXiv preprint arXiv:2409.12186
.

Cited by: 
§3.1
.

Jain 
et al.
 (2025)

N. Jain, K. Han, A. Gu, W. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica

Livecodebench: holistic and contamination free evaluation of large language models for code
.

In 
International Conference on Learning Representations
,

Vol. 
2025
, 
pp. 58791–58831
.

Cited by: 
§3.1
.

Ko 
et al.
 (2026)

J. Ko, S. Abdali, Y. J. Kim, T. Chen, and P. Cameron

Scaling reasoning efficiently via relaxed on-policy distillation
.

arXiv preprint arXiv:2603.11137
.

Cited by: 
§8
.

Li 
et al.
 (2024)

J. Li, E. Beeching, L. Tunstall, B. Lipkin, R. Soletskyi, S. Huang, K. Rasul, L. Yu, A. Q. Jiang, Z. Shen, 
et al.

Numinamath: the largest public dataset in ai4maths with 860k pairs of competition math problems and solutions
.

Hugging Face repository
 
13
 (
9
), 
pp. 9
.

Cited by: 
§3.1
.

Li 
et al.
 (2023)

R. Li, J. Fu, B. Zhang, T. Huang, Z. Sun, C. Lyu, G. Liu, Z. Jin, and G. Li

TACO: topics in algorithmic code generation dataset
.

arXiv preprint arXiv:2312.14852
.

Cited by: 
§E.1
.

Li 
et al.
 (2025)

X. Li, H. Zou, and P. Liu

Limr: less is more for rl scaling
.

arXiv preprint arXiv:2502.11886
.

Cited by: 
§8
.

Li 
et al.
 (2026)

Y. Li, Y. Zuo, B. He, J. Zhang, C. Xiao, C. Qian, T. Yu, H. Gao, W. Yang, Z. Liu, 
et al.

Rethinking on-policy distillation of large language models: phenomenology, mechanism, and recipe
.

arXiv preprint arXiv:2604.13016
.

Cited by: 
§1
,

§2.3
,

§8
.

Lin 
et al.
 (2024)

Q. Lin, M. Wen, Q. Peng, G. Nie, J. Liao, J. Wang, X. Mo, J. Zhou, C. Cheng, Y. Zhao, 
et al.

Hammer: robust function-calling for on-device language models via function masking
.

arXiv preprint arXiv:2410.04587
.

Cited by: 
§3.1
.

Liu 
et al.
 (2024)

Z. Liu, T. Hoang, J. Zhang, M. Zhu, T. Lan, S. Kokane, J. Tan, W. Yao, Z. Liu, Y. Feng, 
et al.

Apigen: automated pipeline for generating verifiable and diverse function-calling datasets
.

Advances in Neural Information Processing Systems
 
37
, 
pp. 54463–54482
.

Cited by: 
§3.1
.

Luo 
et al.
 (2025)

M. Luo, S. Tan, R. Huang, A. Patel, A. Ariyak, Q. Wu, X. Shi, R. Xin, C. Cai, M. Weber, 
et al.

Deepcoder: a fully open-source 14b coder at o3-mini level
.

Notion Blog
 
1
.

Cited by: 
§E.1
.

Muennighoff 
et al.
 (2025)

N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. Candès, and T. B. Hashimoto

S1: simple test-time scaling
.

In 
Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing
,

pp. 20286–20332
.

Cited by: 
§8
.

Mukherjee 
et al.
 (2026)

S. Mukherjee, L. Yuan, D. Hakkani-Tur, and H. Peng

Reinforcement learning finetunes small subnetworks in large language models
.

Advances in Neural Information Processing Systems
 
38
, 
pp. 132119–132138
.

Cited by: 
§8
.

Olmo 
et al.
 (2025)

T. Olmo, A. Ettinger, A. Bertsch, B. Kuehl, D. Graham, D. Heineman, D. Groeneveld, F. Brahman, F. Timbers, H. Ivison, 
et al.

Olmo 3
.

arXiv preprint arXiv:2512.13961
.

Cited by: 
§3.1
.

OpenBMB (2026)

OpenBMB

UltraData-sft-2605
.

 
Hugging Face
.

External Links: 
Link

Cited by: 
§A.1
,

§3.1
.

Patil 
et al.
 (2025)

S. G. Patil, H. Mao, F. Yan, C. C. Ji, V. Suresh, I. Stoica, and J. E. Gonzalez

The berkeley function calling leaderboard (bfcl): from tool use to agentic evaluation of large language models
.

In 
Forty-second International Conference on Machine Learning
,

Cited by: 
§3.1
.

Penedo 
et al.
 (2025)

G. Penedo, A. Lozhkov, H. Kydlíček, L. B. Allal, E. Beeching, A. P. Lajarín, Q. Gallouédec, N. Habib, L. Tunstall, and L. von Werra

CodeForces
.

 
Hugging Face
.

Note: 
https://huggingface.co/datasets/open-r1/codeforces

Cited by: 
§3.1
.

Pyatkin 
et al.
 (2026)

V. Pyatkin, S. Malik, V. Graf, H. Ivison, S. Huang, P. Dasigi, N. Lambert, and H. Hajishirzi

Generalizing verifiable instruction following
.

Advances in Neural Information Processing Systems
 
38
.

Cited by: 
§A.1
.

Shen 
et al.
 (2026)

Z. Shen, Y. Li, Q. Yin, C. T. Leong, Z. Wang, Y. Chen, R. Han, S. Lee, and Y. R. Fung

On the geometry of on-policy distillation
.

arXiv preprint arXiv:2606.07082
.

Cited by: 
§8
.

Sheng 
et al.
 (2025)

G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu

Hybridflow: a flexible and efficient rlhf framework
.

In 
Proceedings of the Twentieth European Conference on Computer Systems
,

pp. 1279–1297
.

Cited by: 
§3.1
.

Song and Zheng (2026)

M. Song and M. Zheng

A survey of on-policy distillation for large language models
.

arXiv preprint arXiv:2604.00626
.

Cited by: 
§8
.

Team 
et al.
 (2026)

K. Team, T. Bai, Y. Bai, Y. Bao, J. Cai, X. Cai, P. Cao, Y. Cao, Z. Chai, Y. Charles, 
et al.

Kimi k3: open frontier intelligence
.

arXiv preprint arXiv:2607.24653
.

Cited by: 
§1
.

Wang 
et al.
 (2026a)

Y. Wang, Q. Yang, Z. Zeng, L. Ren, L. Liu, B. Peng, H. Cheng, X. He, K. Wang, J. Gao, 
et al.

Reinforcement learning for reasoning in large language models with one training example
.

Advances in Neural Information Processing Systems
 
38
, 
pp. 122721–122764
.

Cited by: 
§1
,

§7.2
,

§8
,

§9
.

Wang 
et al.
 (2023)

Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi

Self-instruct: aligning language models with self-generated instructions
.

In 
Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers)
,

pp. 13484–13508
.

Cited by: 
§8
.

Wang 
et al.
 (2026b)

Y. Wang, S. Lu, Y. Gu, P. Wang, Y. Yang, Z. Yan, C. Xie, J. Wu, and H. Yang

Not all disagreement is learnable: token teachability in on-policy distillation
.

arXiv preprint arXiv:2605.26844
.

Cited by: 
§8
.

Xia 
et al.
 (2024)

M. Xia, S. Malladi, S. Gururangan, S. Arora, and D. Chen

Less: selecting influential data for targeted instruction tuning
.

arXiv preprint arXiv:2402.04333
.

Cited by: 
§8
.

Xiao 
et al.
 (2026)

B. Xiao, B. Xia, B. Yang, B. Gao, B. Shen, C. Zhang, C. He, C. Lou, F. Luo, G. Wang, 
et al.

Mimo-v2-flash technical report
.

arXiv preprint arXiv:2601.02780
.

Cited by: 
§1
,

§1
,

§6
,

§8
.

Xu 
et al.
 (2026)

A. Xu, B. Lin, B. Xue, B. Wang, B. Xu, B. Wu, B. Zhang, C. Lin, C. Dong, C. Ling, 
et al.

DeepSeek-v4: towards highly efficient million-token context intelligence
.

arXiv preprint arXiv:2606.19348
.

Cited by: 
§1
,

§1
,

§6
,

§8
.

Xu 
et al.
 (2024)

C. Xu, Q. Sun, K. Zheng, X. Geng, P. Zhao, J. Feng, C. Tao, Q. Lin, and D. Jiang

WizardLM: empowering large pre-trained language models to follow complex instructions
.

In 
International Conference on Learning Representations
,

Vol. 
2024
, 
pp. 30745–30766
.

Cited by: 
§8
.

Xu 
et al.
 (2025)

Z. Xu, F. Jiang, L. Niu, Y. Deng, R. Poovendran, Y. Choi, and B. Y. Lin

Magpie: alignment data synthesis from scratch by prompting aligned llms with nothing
.

In 
International Conference on Learning Representations
,

Vol. 
2025
, 
pp. 76346–76382
.

Cited by: 
§8
.

Yang 
et al.
 (2025)

A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, 
et al.

Qwen3 technical report
.

arXiv preprint arXiv:2505.09388
.

Cited by: 
§1
,

§8
.

Yang 
et al.
 (2026)

W. Yang, W. Liu, R. Xie, K. Yang, S. Yang, and Y. Lin

Learning beyond teacher: generalized on-policy distillation with reward extrapolation
.

arXiv preprint arXiv:2602.12125
.

Cited by: 
§2.2
,

§8
.

Ye 
et al.
 (2025)

Y. Ye, Z. Huang, Y. Xiao, E. Chern, S. Xia, and P. Liu

Limo: less is more for reasoning
.

arXiv preprint arXiv:2502.03387
.

Cited by: 
§8
.

Yu 
et al.
 (2026)

Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, 
et al.

Dapo: an open-source llm reinforcement learning system at scale
.

Advances in Neural Information Processing Systems
 
38
, 
pp. 113222–113244
.

Cited by: 
§3.1
.

Zeng 
et al.
 (2026)

A. Zeng, X. Lv, Z. Hou, Z. Du, Q. Zheng, B. Chen, D. Yin, C. Ge, C. Huang, C. Xie, 
et al.

Glm-5: from vibe coding to agentic engineering
.

arXiv preprint arXiv:2602.15763
.

Cited by: 
§1
,

§8
.

Zeng 
et al.
 (2025)

Z. Zeng, H. Ivison, Y. Wang, L. Yuan, S. S. Li, Z. Ye, S. Li, J. He, R. Zhou, T. Chen, 
et al.

Rlve: scaling up reinforcement learning for language models with adaptive verifiable environments
.

arXiv preprint arXiv:2511.07317
.

Cited by: 
§3.1
.

Zhang 
et al.
 (2026)

D. Zhang, Z. Yang, S. Janghorbani, J. Han, A. Ressler II, Q. Qian, G. D. Lyng, S. S. Batra, and R. E. Tillman

Fast and effective on-policy distillation from reasoning prefixes
.

In 
Findings of the Association for Computational Linguistics: ACL 2026
,

pp. 25553–25569
.

Cited by: 
§8
.

Zhang 
et al.
 (2025)

Z. Zhang, J. Zhu, X. Ge, Z. Zhao, Z. Zhou, X. Li, X. Feng, J. Yao, and B. Han

Co-reward: self-supervised reinforcement learning for large language model reasoning via contrastive agreement
.

arXiv e-prints
, 
pp. arXiv–2508
.

Cited by: 
§3.1
.

Zhao 
et al.
 (2026a)

A. Zhao, Y. Wu, T. Wu, Q. Xu, Y. Yue, M. Lin, S. Wang, Q. Wu, Z. Zheng, and G. Huang

Absolute zero: reinforced self-play reasoning with zero data
.

Advances in Neural Information Processing Systems
 
38
, 
pp. 105816–105879
.

Cited by: 
§8
.

Zhao 
et al.
 (2024)

W. Zhao, X. Ren, J. Hessel, C. Cardie, Y. Choi, and Y. Deng

Wildchat: 1m chatgpt interaction logs in the wild
.

In 
International Conference on Learning Representations
,

Vol. 
2024
, 
pp. 34590–34605
.

Cited by: 
§7.1
.

Zhao 
et al.
 (2026b)

X. Zhao, Z. Kang, A. Feng, S. Levine, and D. Song

Learning to reason without external rewards
.

In 
International Conference on Learning Representations
,

Vol. 
2026
, 
pp. 2548–2581
.

Cited by: 
§8
.

Zhou 
et al.
 (2023)

C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma, A. Efrat, P. Yu, L. Yu, 
et al.

Lima: less is more for alignment
.

Advances in Neural Information Processing Systems
 
36
, 
pp. 55006–55021
.

Cited by: 
§8
.

Zhu 
et al.
 (2025)

H. Zhu, Z. Zhang, H. Huang, D. Su, Z. Liu, J. Zhao, I. Fedorov, H. Pirsiavash, Z. Sha, J. Lee, 
et al.

The path not taken: rlvr provably learns off the principals
.

arXiv preprint arXiv:2511.08567
.

Cited by: 
§8
.

Zhu 
et al.
 (2026)

S. Zhu, X. Ye, H. Lu, W. Shi, and G. Liu

The many faces of on-policy distillation: pitfalls, mechanisms, and fixes
.

arXiv preprint arXiv:2605.11182
.

Cited by: 
§1
,

§8
.

Zuo 
et al.
 (2026)

Y. Zuo, K. Zhang, L. Sheng, S. Qu, G. Cui, X. Zhu, H. Li, X. Long, E. Hua, B. Qi, 
et al.

Ttrl: test-time reinforcement learning
.

Advances in Neural Information Processing Systems
 
38
, 
pp. 131459–131483
.

Cited by: 
§8
.

Zweiger 
et al.
 (2026)

A. Zweiger, J. Pari, H. Guo, Y. Kim, and P. Agrawal

Self-adapting language models
.

Advances in Neural Information Processing Systems
 
38
, 
pp. 74084–74115
.

Cited by: 
§8
.

## Appendix ADetails for Section3

### A.1Experimental Setup and Training Queries

#### Instruction-following setup.

Unlike the other primary-domain teachers, the instruction-following teacher is trained specifically for our experiments. Starting from DeepSeek-R1-Distill-Qwen-1.5B, we post-train the model on synthetic instruction-following tasks with programmatically checkable constraints. We use GRPO with rule-based outcome rewards that indicate whether the response satisfies the target-turn constraints. Training samples four completions per prompt at temperature $0.9$ , uses a maximum prompt length of $4{,}096$ tokens and a maximum response length of $16{,}384$ tokens, and optimizes with AdamW at learning rate $1\times 10^{-6}$ , train batch size $16$ , and clip ratio $0.2$ . The step-600 checkpoint is used as the instruction-following teacher, improving IFBench [[Pyatkin et al., 2026]] from $14.33$ to $20.65$ and Multi-IF final-turn accuracy from $20.84$ to $28.58$ .

For this domain, “full-data” denotes the subset sampled from UltraData-SFT-2605 [[OpenBMB, 2026]] rather than the complete dataset, and the one-shot query is drawn at random from that subset.

#### Mathematical queries for one-shot OPD (Table2, Figure15).

We sample 8 rollouts from the untrained student for each of the first 64 queries of DAPO-Math-17K and assign difficulty by the resulting pass rate, which selects t59 ( $8/8$ ), t22 ( $4/8$ ), and t56 ( $0/8$ ) as the easy, medium, and hard queries. How often each is solved during training follows that assignment: the easy query stays near $1.0$ throughout, the medium one rises from $0.36$ to about $0.89$ , and the hard one is never solved in $300$ steps, yet all three produce comparable benchmark gains in Figure 6 .

[element]
Difficulty ID Query Easy t59 In triangle $ABC$ , let $D$ be the foot of the altitude from $A$ . Suppose that $AD=4$ , $BD=3$ , $CD=2$ , and $AB$ is extended past $B$ to a point $E$ such that $BE=5$ . Determine the value of $CE^{2}$ . Medium t22 Let $S$ be the set of triples $(a,b,c)$ of non-negative integers such that $a+b+c$ is even. Determine the value of $\sum_{(a,b,c)\in S}\frac{1}{2^{a}3^{b}5^{c}}$ .
This sum can be expressed as $\frac{m}{n}$ , where $m$ and $n$ are relatively prime positive integers. Compute $m+n$ . Hard t56 Byan is playing a game called “raven, raven, falcon” with his three friends. His friends sit in a circle, and Byan walks clockwise around them, tapping each friend he passes on the head and saying either “raven” or “falcon,” each with probability $\frac{1}{2}$ . The game ends when Byan has said “falcon” twice. The probability that one of his friends will be called a “falcon” twice can be expressed as $\frac{m}{n}$ , where $m$ and $n$ are relatively prime positive integers. Find $100m+n$ . Table 2: Mathematical queries used in the query-difficulty experiment, labelled by the student’s pass rate before OPD training.
[/element]

[element]
Figure 15 : Fraction of the 64 rollouts per step that solve the training query itself, for the three queries of Table 2 .
[/element]

#### Default hyperparameters for OPD (Table3).

Unless otherwise noted, all experiments use these defaults. The top- $K$ row covers both objectives: the mathematical one-shot runs of Section 3.2 use $K=16$ , and all other experiments use the sampled-token objective.

[element]
Table 3 : Default hyperparameters for OPD. Item Value Rollout batch size 64 Mini batch size 64 Responses per prompt 1 KL coefficient 0.0 LogProb top- $K$ 0(default) / 16 Top- $K$ strategy Student Top- $K$ Loss aggregation token-mean Training temperature 1.0 Top- $p$ 1.0 Optimizer AdamW ( $\beta_{1}{=}0.9$ , $\beta_{2}{=}0.999$ , wd $=0.01$ ) Learning rate 1e-6 Gradient clip norm 1.0 Max prompt length 1024 (math); 4096 (code, IF, agentic) Max response length 7168 (math, code, IF); 2048 (agentic)
[/element]

## Appendix BDetails for Section4

### B.1Training-State Coverage

#### Construction.

Rollouts are collected by optimization step rather than at random, so that every stage of training is equally represented and the budget is comparable across settings, which run for the same number of steps: at each step from $1$ to $500$ of the full-data run we take $5$ rollouts, three building the clusters and two forming full data (held-out), for $12{,}000$ reference and $8{,}000$ held-out states. PCA reduces these to $50$ dimensions, retaining $71.9\%$ of the variance, before $K$ -means. Every evaluated setting is then measured on that same schedule, drawing $2$ of the $5$ rollouts at each step and accumulating over the run, so the step- $300$ budget of $600$ rollouts matches the held-out split. We repeat the draw $30$ times; bands and error bars are one standard deviation over these repetitions, which reflects rollout subsampling rather than variation across trained models.

#### The ordering is stable across$K$, reference construction, and state positions (Tables4and5, Figure16).

Coverage depends on how finely the reference states are clustered and on which states define the clusters, so we vary both. The first table repeats the analysis at $K\in\{50,100,200,500\}$ : finer clusters lower absolute coverage, but coverage still increases from 1- to 4- to 16-shot with 16-shot close to full data. The second rebuilds the clusters from the union of $2{,}000$ rollouts per setting instead of from full-data states alone, and the ordering survives; because states from all settings help define these pooled clusters, pooled coverage is not a recovery measure, and its rows are to be read only against each other. The figure varies the positions read from each rollout, comparing the final non-padding token and fixed absolute positions $2{,}000$ and $4{,}000$ against the 8 relative positions of the main analysis. All four preserve the ordering. The fixed-position variants land close to the main construction, whereas the final token gives markedly lower coverage when there are few queries, which is what one expects of a position governed by response length and termination; we therefore read 8 relative positions, which sample throughout a response of any length.

[element]
Table 4: Full-data-anchored training-state coverage at step $300$ under different cluster counts $K$ . $K$ 1-shot 4-shot 16-shot Full data (held-out) 50 0.755 0.829 1.000 1.000 100 0.727 0.803 1.000 1.000 200 0.715 0.798 0.989 1.000 500 0.564 0.695 0.904 0.956
[/element]

[element]
Table 5: Training-state coverage under alternative cluster constructions, at $K=200$ . Reference construction 1-shot 4-shot 16-shot Full data (held-out) Full-data anchored 0.663 0.773 0.953 0.998 Equal-size pooled 0.750 0.845 0.958 0.966
[/element]

[element]
Figure 16 : Training-state coverage under alternative state-extraction positions, at $K=200$ . Error bars show one standard deviation over rollout subsamples.
[/element]

#### One-shot has a heavier tail of distant states, but is not separable from full data (Table6).

Nearest-centroid assignment places a state in some cluster however far it lies from every full-data state, so coverage alone cannot tell a setting that fills the reference space thinly from one that leaves it. We therefore also measure, for each state, the mean cosine distance to its ten nearest reference states in the unreduced $1536$ -dimensional space, and call a state off-reference when that distance exceeds the $95$ th percentile of the held-out full-data states. One-shot OPD exceeds the threshold more often than the other settings, so its distances have a heavier upper tail. But an AUROC of $0.552$ against held-out full-data states is close to chance, so one-shot states are not separable from full-data states as a whole.

[element]
Table 6: Reference-distance diagnostics across query-set sizes. AUROC compares each setting with the held-out full-data states. The last row is the complete full-data pool, including the rollouts that built the clusters, so its exceedance rate need not equal the $5\%$ the threshold is set at. Setting Exceedance (%) Mean distance AUROC 1-shot 17.2 0.283 0.552 4-shot 4.4 0.218 0.477 16-shot 3.8 0.231 0.512 Full data (all) 4.0 0.215 0.476
[/element]

### B.2Controls on the Training Data

#### Diversity in 16-shot OPD (Figure17).

Every query on the ladder of Section 4.2 represents a different semantic cluster, so it raises diversity and count together. Holding the count at 16 and drawing the second set from a single cluster separates them: the 16 cluster representatives reach $70.9$ validation accuracy at step $300$ against $69.9$ for the single-cluster set, which is about what 4-shot OPD reaches with a quarter as many queries, and their state coverage separates the same way and does so early, $98.9\%$ against a single-cluster set that flattens near $76.8\%$ within $100$ steps. Queries from one cluster keep returning the student to the part of the state space it has already seen.

[element]
Figure 17 : Two 16-shot query sets that differ only in diversity, against full-data OPD.
[/element]

#### Query scheduling in 2-shot OPD (Figure18).

Holding a two-query set fixed and varying only when each query is shown, the sequential schedule trains on the hard query for the first $300$ steps of a $600$ -step budget and on the medium query for the rest, while the mixed schedule puts $32$ copies of each in every batch. At step $600$ they reach $68.8$ and $69.1$ , both above the $67.4$ of the one-shot baseline they share, so ordering a fixed query set barely matters next to what the second query itself buys. These runs train longer than those of Section 4.2 and take the hard query as their baseline, so neither their horizon nor their baseline is interchangeable with Figure 8 .

[element]
Figure 18 : Effect of query scheduling in 2-shot OPD. The vertical dotted line marks the query switch in the sequential schedule.
[/element]

## Appendix CDetails for Section5

### C.1Estimating the Absorption Rate and the Measurement Window

Section 5.2 measures under three conventions. Why each one:

•

$\boldsymbol{[t/1.4,\,1.4t]}$ . A difference between adjacent steps is mostly noise once $d$ moves slowly, so $v_{t}$ is read as the slope of $\log d$ across the endpoints of this window, using the median $d$ near each. Only centers whose window fits inside the measured range are plotted, which is why the main text reads the rate at steps $50$ and $200$ rather than at step $300$ .

•

$\boldsymbol{d_{t}/d_{30}}$ . Each run measures its distance on the states it visits and the four sit at different levels by step $30$ , so plotting each against its own value there lets the curves be compared by shape rather than by level.

•

Start at step $30$ . Gradient clipping binds until then, and a clipped update is not the update the objective asked for: $v_{t}$ inside that window measures what one fixed-size step buys rather than what one step buys, which is why the measured rate rises there rather than falls.

Table 7 collects the readings Section 5.2 quotes from this range: the proportion of the step- $30$ distance still left at step $300$ , and the factor by which the absorption rate falls between steps $50$ and $200$ .

[element]
Table 7: Alignment dynamics over steps $30$ – $300$ for on-policy OPD, at the default learning rate. Setting (queries) 1 4 16 Full data $d_{300}/d_{30}$ 0.16 0.17 0.18 0.22 $v_{50}/v_{200}$ 5.6 5.2 4.2 6.4
[/element]

### C.2The Learning Rate Rescales the Step Axis

We repeat on-policy one-shot OPD at half, the default, and twice the default learning rate, holding every other setting fixed, and read all three over the same $300$ steps as Section 5.2 .

[element]
Figure 19 : Validation accuracy and teacher–student distance for one-shot OPD at half, one, and twice the default learning rate. Distance is on a logarithmic vertical axis against linear time.
[/element]

#### The learning rate rescales the step axis and leaves the slowdown untouched.

A larger learning rate reaches any given distance sooner, the three curves separated by roughly the ratio of the learning rates (Figure 19 ). Changing the horizontal axis from the step $t$ to $u=(\mathrm{lr}/10^{-6})\,t$ brings them onto one curve: the spread between the three runs narrows from about $2.2\times$ to about $1.1\times$ . The learning rate therefore sets only how fast a run moves along that one curve, not the shape of the curve itself. It bends flatter as it goes, just as the four query settings of Section 5.2 do against the step count. A proportionally larger learning rate therefore shortens the run proportionally, over the fourfold range we tested, without reaching whatever makes alignment slow in the first place.

## Appendix DDetails for Section6

Table 8 shows the default hyperparameters for MOPD.

[element]
Table 8 : Default hyperparameters for MOPD (multi-teacher on-policy distillation). Item Value Number of teachers 3 (math / code / IF) LogProb top- $k$ 0 KL coefficient 0.0 Loss aggregation seq-mean-token-mean Rollout batch size 64 Mini batch size 64 Responses per prompt 1 Training temperature 1.0 Top- $p$ 1.0 Optimizer AdamW ( $\beta_{1}{=}0.9$ , $\beta_{2}{=}0.999$ , wd $=0.01$ ) Learning rate 1e-6 Gradient clipping 1.0 Max prompt length 4096 Max response length 7680
[/element]

## Appendix EDetails for Section7

### E.1Template and WildChat OPD in Code Generation

Section 7.1 reports the four input conditions in mathematics. Here we report the same four in code generation.

#### Setup.

The student is R1-Distill-1.5B, the teacher DeepCoder-1.5B-Preview [[Luo et al., 2025]] , and the baseline full-data OPD on TACO [[Li et al., 2023]] . The base template, the system template, and the WildChat queries are exactly those of Section 7.1 ; the system template’s one-line instruction is the competitive-programming instruction of Figure 12 b. We evaluate on LiveCodeBench v6 under avg@3.

[element]
Figure 20 : Validation accuracy and actor entropy for code-generation OPD.
[/element]

#### Results.

Figure 20 shows the four conditions ending within $1.7$ points of one another, and the token accounting separating them much further than the step accounting does: over the same $500$ steps the baseline spends $277$ M rollout tokens against $18$ M for the base template, with the other two in between. Actor entropy orders the conditions exactly as it does in mathematics, with WildChat highest and the system template lowest. The one difference is that no condition falls behind here, whereas WildChat costs about a point in mathematics.


