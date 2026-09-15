# 2608.31046

ID: 2608.31046
Source: https://arxiv.org/html/2608.31046v1
Authors: Yi Ding Ruqi Zhang Affiliation: Department of Computer Science,
Purdue University,
USA Email: {ding432,ruqiz}@purdue.edu



# Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement

Yi Ding

  

Ruqi Zhang

Affiliation: 
Department of Computer Science,
Purdue University,
USA

Email: 
{ding432,ruqiz}@purdue.edu

###### Abstract

On-policy distillation (OPD) offers dense token-level supervision as an alternative to the sparse outcome-level advantages of reinforcement learning with verifiable rewards (RLVR).
However, the teacher scores student-generated trajectories that are inherently off-policy for it, so the reliability of its supervision, and hence the source of the student’s improvement, remains unclear.
We quantitatively analyze teacher supervision during OPD training and find substantial noise whose prevalence increases with teacher scale. Surprisingly, the student policy is insensitive to such noise, converging to comparable performance regardless of whether noisy supervision is retained or removed.
Does OPD distill at all? By analyzing what drives its gains, we find that learning concentrates on low log-probability tokens, and using a single fixed negative advantage matches the performance of teacher-provided ones. This suggests that OPD works largely by suppressing low log-probability tokens, which requires no teacher. These findings motivate On-Policy Self-Adaptation (OPSA), a supervision-free method using entropy-adaptive negative advantages.
It assigns stronger learning signals to high-entropy positions, suppressing tail tokens, and evenly redistributing probability mass among head tokens.
Compared with the base Qwen3-1.7B , OPSA improves Avg@32 by 35.41 points on AIME24, corresponding to a 263% relative gain, and more than doubles Pass@32 across all three benchmarks. It also outperforms OPD by 16.77 points in Avg@32 on AIME24.
Extensive experiments and analyses across model families and tasks further demonstrate its effectiveness and generalizability.

[element]

[/element]

Hugging Face Github

[element]
Figure 1: Left: Overview of different on-policy reinforcement learning algorithms. Unlike existing methods that derive advantages from external supervision, OPSA enables self-improvement with no external supervision by assigning entropy-adaptive negative advantages to low-probability tokens. This suppresses tail tokens and redistributes their mass among head tokens, sharpening low-entropy positions while preserving diversity at high-entropy forks. Right: Training dynamics and performance of OPSA. OPSA eliminates the teacher supervision used in OPD, and further outperforms OPD by 12% in Avg@4. On Qwen3-1.7B , OPSA improves Avg@32 by 263-307% across mathematical reasoning benchmarks and more than doubles Pass@32.
[/element]

## 1Introduction

Reinforcement learning (RL) has become a dominant post-training paradigm to improve reasoning capabilities of large language models (LLMs) [(Guo et al., 2025;Yang et al., 2025;GLM-5-Team et al., 2026)] , offering stronger performance and generalization than offline methods such as supervised fine-tuning (SFT) [(Chu et al., 2025;Chen et al., 2025)] . Reinforcement learning with verifiable rewards (RLVR), including GRPO [(Shao et al., 2024)] and DAPO [(Yu et al., 2026)] , samples multiple responses per question and assigns advantages by normalizing verifiable rewards within each group (Fig. 1 (a)). However, response-level rewards provide coarse and sparse supervision for long-horizon reasoning [(Yue et al., 2025)] . Moreover, when responses within a group share the same correctness outcome, their normalized advantages vanish [(Wang et al., 2026b;Ding et al., 2026)] , weakening the learning signal and making training unstable or prone to collapse.

On-Policy Distillation (OPD) [(Agarwal et al., 2024;Gu et al., 2024)] addresses this limitation by using a strong teacher to provide token-level advantages through reverse Kullback-Leibler (KL) optimization on student-sampled trajectories (Fig. 1 (b)) [(Burnham & Anderson, 2001;Lu & Lab, 2025)] . However, it requires shared vocabularies and white-box access to teacher logits, restricting teacher selection. On-Policy Self-Distillation (OPSD) [(Hübotter et al., 2026;Zhao et al., 2026;Shenfeld et al., 2026)] retains the same OPD paradigm but replaces the external teacher with the policy itself conditioned on hints, such as reference answers (Fig. 1 (c)). Constructing such hints still requires additional sampling or annotation and may risk information leakage [(Yang et al., 2026)] .

More fundamentally, OPD and OPSD share the same advantage-assignment paradigm, which asks the teacher to score student-sampled prefixes that are inherently off-policy for it. This raises the question of whether the teacher can still provide reliable supervision in such off-policy contexts [(Xie et al., 2026;Fu et al., 2026;Hou et al., 2026)] .
Our analysis reveals a substantial fraction of noisy teacher supervision, where noisy means a negative advantage on a correct answer or a positive advantage on an incorrect one, and this fraction grows with teacher scale. Surprisingly, training exclusively on these noisy trajectories achieves performance comparable to both standard OPD and OPD trained with the noisy trajectories removed, suggesting that OPD’s gains may not arise from the behavior matching it is designed to achieve. This raises a fundamental question: what actually drives student improvement in OPD?

To answer this question, we analyze OPD’s gains from the perspectives of token selection and learning signals. We find that (i) improvement is driven primarily by low log-probability tokens sampled by the student, and (ii) negative advantages are critical, as replacing all OPD advantages with a single fixed negative value yields comparable performance. These observations suggest that OPD’s gains come from suppressing low-probability tokens sampled by the student, requiring no teacher.
Fixed negative advantages reproduce much of OPD’s gains, but they discard its fine-grained token-level signals by treating all tokens equally. We therefore study how much negative signal each token should receive, and find that token entropy determines it, with stronger signal at high-entropy positions giving better performance.

Altogether, we propose On-Policy Self-Adaptation (OPSA), a supervision-free token-level RL method that assigns entropy-adaptive negative advantages to low-probability tokens (Fig. 1 (d)).
We summarize our contributions as follows:

•

We find that teacher supervision in OPD is highly noisy and uncover a surprising insensitivity of the student policy to such noise. Training exclusively with noisy supervision, excluding noisy supervision, and standard OPD all converge to comparable accuracy after similar numbers of training steps.

•

We identify the key drivers of OPD improvement: effective learning is concentrated on low-logp student-sampled tokens, and fixed negative advantages on these tokens can match standard OPD. This suggests that OPD’s gains stem less from teacher distillation than from suppressing low-probability tokens, questioning the necessity of teacher supervision.

•

We propose On-Policy Self-Adaptation, a supervision-free token-level RL method. It reshapes the policy distribution by suppressing tail-token probabilities and redistributing the probability mass among head tokens. This sharpens the distribution at low-entropy positions while preserving diversity at high-entropy fork tokens to support effective exploration.

•

We show through extensive experiments that OPSA generalizes across model families and tasks. On Qwen3-1.7B , OPSA improves Avg@32 by 263%–307% relative to the base model across the three benchmarks and more than doubles Pass@32 on each benchmark.

## 2Teacher Supervision in OPD Is Highly Noisy, but Students Improve Regardless

### 2.1Preliminary

On-Policy Distillation (OPD) [(Agarwal et al., 2024;Gu et al., 2024)] optimizes the reverse Kullback–Leibler (KL) divergence between the student and teacher distributions over the same prefixes, which is computed using the K1 estimator [(Lu & Lab, 2025)] :

[element]
$\mathrm{KL}(\pi_{s}\mid\mid\pi_{t})=\mathbb{E}_{y\sim\pi_{s}(\cdot\mid x)}\sum_{i=1}^{|y|}\left[\log\pi_{s}(y_{i}\mid x;y_{<i})-\log\pi_{t}(y_{i}\mid x;y_{<i})\right],$ (1)
[/element]

where $y_{<i}=[y_{1},\cdots,y_{i-1}]$ denotes prefixes of the response $y$ sampled from student policy $\pi_{s}$ . In practice, OPD provides token-level advantage signals $A_{i}$ for the student policy during the Reinforcement Learning (RL) training, and the loss function can be written as:

[element]
$\mathcal{L}_{\text{OPD}}=-\mathbb{E}\left[\frac{1}{|y|}\sum_{i=1}^{|y|}A_{i}\log\pi_{s}(y_{i}|x;y_{<i})\right],\hskip 10.0ptA_{i}=\log\frac{\pi_{t}(y_{i}\mid x;y_{<i})}{\pi_{s}(y_{i}\mid x;y_{<i})}.$ (2)
[/element]

The loss function optimizes the student policy by assigning positive or negative credits to the student-sampled tokens based on teachers’ preference, pushing it closer to teachers’ behavior in each state.

### 2.2Noisy Teacher Signals in OPD

Since the KL divergence is computed conditioned on prefixes sampled from the student distribution, the teacher must score trajectories it would not generate. This raises a key concern: can the teacher provide reliable supervision in such off-policy settings [(Fu et al., 2026;Xie et al., 2026;Wang et al., 2026a)] ?

##### Definition of Noisy Signals.

Previous work [(Hou et al., 2026)] often measures supervision noise at the trajectory level. However, the quality of intermediate reasoning steps is difficult to assess quantitatively, as a per-token reference for what the supervision should be is generally unavailable. Therefore,
we focus on tokens corresponding to verifiable final answers enclosed in \boxed{} , whose correctness is determined by the verifier. We call the supervision noisy when the sign of the teacher-provided advantage on these tokens disagrees with the verifiable reward, i.e., when incorrect answers receive positive advantages or correct answers receive negative ones. Such supervision provides the student with misleading learning signals during RL training.

##### Setup.

We use Qwen3-1.7B as the student policy $\pi_{s}$ with thinking mode disabled, and use Qwen3-4B/30B-A3B/235B-A22B-Instruct as teacher $\pi_{t}$ , respectively.
For the following analysis, we randomly sample one correct and one incorrect response from $\pi_{s}$ for each of 500 questions in DAPO-17k [(Yu et al., 2026)] , with correctness verified based on ground-truth answers.

[element]
Figure 2: Performance of Qwen3-1.7B as the student model under supervision from teachers of different scales. (a): The proportion of teacher-provided advantage signals whose directions disagree with verifiable rewards on student-sampled trajectories. Red and blue bars denote noise rates in correct and incorrect trajectories, respectively. (b): Avg@4 performance on AIME24 during OPD training with different variants of teacher-provided advantage signals.
[/element]

##### Teacher Supervision is Highly Noisy, and Noisier at Scale.

As shown in Fig. 2 (a), supervision from teacher models of all sizes contains substantial noise. For the 4B teacher, 20.4% of correct trajectories receive negative advantages on their answer tokens, while 40.8% of incorrect trajectories receive positive advantages, yielding an overall noise rate of 30.6%. The noise rate increases with teacher size, from 30.6% for the 4B teacher to 34.7% for the 30B-A3B teacher and 50.6% for the 235B-A22B teacher.
In addition, the largest teacher consistently tends to assign negative advantages: 97.8% of answer tokens enclosed in \boxed{} receive negative advantages even when the answer is correct, while 96.6% also receive negative advantages when the answer is incorrect. This leads to an overall noise rate of approximately 50%. These results suggest that, as the teacher becomes more capable, its supervision on student-generated trajectories becomes overwhelmingly negative and increasingly insensitive to answer correctness.
We attribute this to the growing distributional mismatch between the student and teacher policies, which makes student-generated trajectories increasingly off-policy from the teacher’s perspective.

### 2.3Students Are Insensitive to Noise in Teacher Supervision

Analysis in Section 2.2 reveals that OPD training involves a substantial amount of noisy token-level supervision. To isolate its effect on learning, we conduct a controlled filtering experiment by partitioning trajectories according to whether they contain noisy signals defined in Section 2.2 . We compare standard OPD trained on all trajectories against two variants trained exclusively on trajectories with or without noisy signals.
As shown in Fig. 2 (b), all three settings converge to comparable performance after a similar number of gradient steps. Remarkably, even when training is restricted to trajectories containing noisy advantages, the student improves at a rate comparable to standard OPD.
This finding reveals a surprising insensitivity of OPD to supervision noise in the teacher-provided advantages.
In particular, when the teacher supervision is substantially noisy, the student’s improvement may not come from matching the teacher’s behavior. This raises a more fundamental question:

Research Question

OPD improves student performance even when teacher supervision is highly noisy.
This suggests that the gains may not come from knowledge transfer, the mechanism OPD is built on. What, then, drives student improvement in OPD?

## 3Where Does Student Improvement Come From?

In this section, we progressively disentangle where the performance gains of OPD come from. Specifically, Sections 3.1 and 3.2 investigate which tokens affect improvement and what kinds of learning signals are effective, respectively. We summarize our findings as follows:

Key Findings

•

Which Tokens?
 Most high-log-probability tokens sampled by the student provide negligible gradients and little effective learning signal. OPD’s performance gains instead arise primarily from the small fraction of low-log-probability tokens. (§
3.1
)

•

Which Signals?
 Negative advantages, which constitute the majority of OPD signals, are the ones that matter. Notably, replacing all OPD advantages, both positive and negative, with a fixed negative value yields performance comparable to standard OPD. (§
3.2
)

### 3.1Which Tokens Contribute to OPD Training?

Findings in Section 2.3 suggest that not all tokens generated by the student policy contribute effective learning signals for OPD. Revisiting the loss in Eq. 2 , let $z_{t}^{v}$ denote the logit for token $v$ conditioned on context $y_{<t}$ . We have the logit-level gradients for each token [(Zhu et al., 2026;Jia et al., 2026)] :

[element]
$-\frac{\partial\mathcal{L}_{\text{OPD}}}{\partial z_{t}^{v}}\propto\begin{cases}A_{t}\left(1-\pi_{s}(v\mid x;y_{<t})\right),&\text{if }v=y_{t}\text{ (sampled token)}\\
-A_{t}(\pi_{s}(v\mid x;y_{<t})),&\text{if }v\neq y_{t}\text{ (unsampled token)}\end{cases}.$ (3)
[/element]

We observe that the gradient vanishes in two regimes: when $|A_{t}|$ is small, and when $\pi_{s}(y_{t}\mid x,y_{<t})$ approaches one for the sampled token. Tokens in either regime have negligible impact on training.

##### Token Mass Concentrates at Near-Zero Advantages.

We visualize the distribution of token-level advantages in Fig. 3 (a). A substantial fraction of the tokens is concentrated near zero: $29.2\%$ of tokens have exactly zero advantage, and $51.7\%$ have advantage magnitude below $10^{-4}$ .

[element]
Figure 3: Qwen3-1.7B student with 4B-Instruct teacher. (a): Token-level advantage distribution during OPD, with most tokens receiving near-zero advantages. (b): Fraction of near-zero advantages among tokens within different top-percentile ranges of student logp. (c): Training only on top-logp tokens yields limited improvement, suggesting that these tokens provide weak learning signals.
[/element]

##### Near-Zero Advantages Coincide with the Student’s High-Logp Tokens.

Fig. 3 (b) shows that tokens with near-zero advantages ( $|A|<10^{-4}$ ) are concentrated among those assigned high log probability by the student. When the student is extremely confident in a token, the teacher, conditioned on the same student-generated prefix, often assigns similarly high probability to that token, resulting in only a negligible difference between their logp and, consequently, an advantage close to zero.

##### High-Logp Tokens Contribute Nothing to On-Policy Training.

To test whether high-logp tokens provide any effective learning signal, we conduct controlled on-policy training experiments. Specifically, we train on varying proportions of the student’s top-logp tokens using either the original OPD advantages (calculated A) or random advantages (random A) sampled evenly from $[-1,1]$ . As shown in Fig. 3 (c), restricting training to these tokens yields no noticeable improvement in the student’s AIME24 performance. Interestingly, the performance remains essentially unchanged even when the original advantages are replaced with random values. These results suggest that high-logp tokens provide no effective learning signal during on-policy training, largely regardless of the advantages assigned to them.

### 3.2Which Learning Signals Drive Policy Improvement?

Revisiting the formulation of the OPD advantage in Eq. 2 , the reverse KL is computed on student-sampled trajectories, which are inherently off-policy for the teacher. The teacher therefore assigns lower probabilities to many sampled tokens, resulting in predominantly negative advantages, consistent with the results in Fig. 3 (a).
A related observation appears in the RL setting as well, where [Zhu et al. (2026)] show that Negative Sample Reinforcement (NSR), which learns only from negative signals, can improve policy performance. This motivates us to ask whether the gains achieved by OPD are driven primarily by its large proportion of negative advantages, rather than by the specific supervision provided by the teacher.

We use Qwen3-1.7B as the student model and compare three advantage assignment schemes. The first follows standard OPD, using Qwen3-4B-Instruct as the teacher model to compute token-level advantages. The other two are teacher-free variants that assign a fixed advantage of $-0.5$ or $+0.2$ to each selected token. Since high-logp tokens provide little effective training gradient (Section 3.1 ), we restrict training to the 20% of tokens with the lowest student logp.

[element]
Figure 4: Training dynamics of standard OPD and on-policy training with fixed advantages.
[/element]

##### A Fixed Negative Advantage Alone Improves Students.

We report the training dynamics of the different advantage-assignment schemes in Fig. 4 . First, OPD restricted to the 20% of tokens with the lowest student log-probabilities achieves performance comparable to that of full-token OPD, further supporting the findings in Section 3.1 . More notably, training with a fixed negative advantage yields a steady improvement in the student policy’s avg@4 performance on AIME24. Similar to standard OPD, the response length also increases gradually throughout training and eventually stabilizes at approximately 12K tokens. In contrast, training with a fixed positive advantage leads to policy collapse: during the first 40 training steps, the response length progressively decreases to nearly zero while the gradient norm explodes. After the collapse, the policy produces degenerate outputs consisting largely of seemingly random or garbled tokens.

Takeaway

Our analysis suggests that OPD gains may arise not from distilling teacher knowledge, but from suppressing low-probability tokens sampled by the student itself, an operation that requires no teacher at all. This observation motivates the method introduced in Section 
4
.

## 4Methodology

Section 3 showed that OPD’s improvements survive the removal of the teacher, of positive advantages, and of all but the lowest-logp tokens. The remaining question is: given that negative signal on low-logp tokens is what drives improvement, how much negative signal should each such token receive? Section 4.1 shows that token entropy determines the amount of signal; Section 4.2 turns the findings into On-Policy Self-Adaptation (OPSA), a supervision-free training algorithm, and Section 4.3 analyzes why it works.

### 4.1Entropy Determines the Amount of Negative Signals

Section 3.1 suggested that low-logp tokens are the ones that matter, but logp alone does not distinguish two very different situations. A token can have low log probability because the policy is uncertain, with probability mass spread across many plausible tokens, so that even head tokens have relatively low logp. Or the student distribution is sharply concentrated on only a few tokens, while the sampled token happens to fall in the tail of a confident distribution. [Wang et al. (2026c);He et al. (2026)] have shown that high-entropy tokens play an important role in reinforcement learning and may benefit from stronger learning signals. Motivated by these findings, we compare two entropy-aware schemes that dynamically adjust the magnitude of the negative advantage:

[element]
$A_{i}^{\text{dyn}}=A_{i}^{\text{fix}}-\frac{1}{4}\delta\cdot r_{i},\hskip 10.0ptr_{i}=2\frac{H_{i}-H_{\text{min}}}{H_{\text{max}}-H_{\text{min}}}-1,$ (4)
[/element]

where $H_{\min}$ and $H_{\max}$ are the minimum and maximum entropy over the lowest- $20\%$ -logp positions within each response, so that $r_{i}\in[-1,1]$ measures a token’s entropy relative to its own rollout. The parameter $\delta$ controls the relationship between the advantage magnitude $|A_{i}|$ and the token entropy $H_{i}$ : $\delta=1$ assigns larger-magnitude negative advantages to higher-entropy tokens, $\delta=-1$ reverses this, and $\delta=0$ recovers the fixed-negative-advantage setting of Section 3.2 .

[element]
Figure 5: Training dynamics of on-policy training with dynamic negative advantages.
[/element]

##### Adaptive Advantages with$\delta=1$Outperform OPD.

The results in Fig. 5 show that, surprisingly, when the magnitude of the negative advantage is positively correlated with token entropy ( $\delta=1$ ), the policy exhibits steady improvement in avg@4 on AIME24, ultimately reaching 50.0% and substantially outperforming standard OPD at 35.13%. Comparing the positively correlated ( $\delta=1$ ) and negatively correlated ( $\delta=-1$ ) variants further demonstrates that on-policy training benefits substantially from assigning larger-magnitude negative advantages to high-entropy tokens. In contrast, the negatively correlated variant becomes unstable between steps 350 and 450, maintains consistently higher gradient norms throughout training, and ultimately performs slightly worse than the fixed-negative-advantage baseline. These results highlight the importance of concentrating stronger learning signals on high-entropy tokens for effective and stable on-policy training.

### 4.2On-Policy Self-Adaptation

These results give a complete on-policy training recipe, which we call O n- P olicy S elf- A daptation ( OPSA ). OPSA updates only the lowest-logp tokens, assigns them negative advantages, and scales the magnitude with entropy. It requires no external supervision of any kind, including teacher signals, ground-truth rewards, or reference answers.
Setting $A_{i}^{\text{fix}}=-\frac{3}{4}$ and $\delta=1$ in Eq. ( 4 ) and restricting the update to $\mathcal{S}_{\text{lowest}20}$ , the training objective of OPSA is defined as follows:

[element]
$\mathcal{L}_{\text{OPSA}}=-\mathbb{E}\left[\frac{1}{{\color[rgb]{0.7773,0.1563,0.1563}|\mathcal{S}_{\text{lowest}20}|}}\sum_{{\color[rgb]{0.7773,0.1563,0.1563}i\in{\mathcal{S}}_{\text{lowest}20}}}{\color[rgb]{0.7773,0.1563,0.1563}A_{i}^{\text{dyn}}}\log\pi_{\theta}(y_{i}|x;y_{<i})\right],\hskip 4.0pt{\color[rgb]{0.7773,0.1563,0.1563}A_{i}^{\text{dyn}}=-\frac{1}{2}-\frac{H_{i}-H_{\text{min}}}{2(H_{\text{max}}-H_{\text{min}})}}$ (5)
[/element]

Since OPSA eliminates the need for a teacher model during training, it avoids additional forward passes through a teacher model to compute advantages. Instead, advantages are derived directly from the student policy’s token-level entropy, introducing negligible computational overhead and substantially reducing training time. Detailed efficiency results are reported in the Appendix B.3 .

### 4.3Why Does OPSA Work?

To further understand the mechanism of OPSA, we conduct a detailed analysis by categorizing tokens into four cases based on token entropy and sampled-token probability: (a) a tail token at a high-entropy position, (b) a head token at a high-entropy position, (c) a tail token at a low-entropy position, and (d) a head token at a low-entropy position. Based on the logit-level gradients in Eq. 3 , Fig. 6 illustrates how the negative advantages in OPSA reshape the policy distribution in each case.

[element]
Figure 6: Examples of OPSA updates. (a) At high-entropy positions, OPSA suppresses sampled tail tokens and reallocates probability to head tokens. (b) At high-entropy positions, OPSA redistributes probability among head tokens, with little effect on the tail. (c) OPSA avoids sampling tail tokens at low-entropy positions. (d) OPSA preserves high-confidence predictions.
[/element]

##### Tail Tokens: OPSA Suppresses Low-Probability Reasoning Branches.

During on-policy training, stochastic sampling is typically used instead of greedy decoding to maintain rollout diversity. At relatively high sampling temperatures, however, extremely low-probability tail tokens may occasionally be sampled, causing the reasoning process to enter unlikely and potentially erroneous branches. Because OPSA adaptively assigns negative advantages to the 20% of policy-generated tokens with the lowest logp, its policy update suppresses these sampled tail tokens and reallocates their probability mass to higher-probability head tokens, as illustrated in Fig. 6 (a)(c). In this way, OPSA steers subsequent reasoning away from low-probability branches and toward more confident alternatives.

##### Head Tokens: OPSA Evenly Redistributes Mass Without Collapsing Diversity.

Suppressing tail tokens makes sampling increasingly head-dominated, which raises the concern of diversity collapse. Fig. 6 (b) shows why this does not occur. At a high-entropy position, the negative advantage assigned to a sampled head token redistributes probability mass among competing head tokens rather than further sharpening the distribution onto a single prediction, while leaving the tail-token probabilities largely unaffected. This behavior preserves output diversity and the policy’s ability to explore alternative reasoning paths. Moreover, because OPSA updates only the 20% of sampled tokens with the lowest student log-probabilities, high-confidence tokens at low-entropy positions are typically excluded from training. As shown in Fig. 6 (d), OPSA therefore preserves confident predictions at low-entropy positions, where high precision is particularly important [(Zhang et al., 2026b)] .

##### Reshaped Distribution Enables Reflective Reasoning.

Consistent with prior observations [(Ding et al., 2026;Zhao et al., 2025a)] , we find that high token-level entropy is closely associated with self-reflective reasoning behavior. At such high-entropy positions, the model is more likely to generate reflective tokens such as “wait” and “but”, which often signal reflection or self-correction and lead to higher-quality reasoning. The example in Fig. 6 illustrates how OPSA increases the total probability mass of the head-token set at high-entropy positions while redistributing it more evenly within the set. This makes diverse reflective branches more likely to be explored. We provide a quantitative analysis of this phenomenon in Section 5.3.2 .

## 5Experiments

### 5.1Experimental Setup

##### Models and Datasets.

Our experiments cover models from the Qwen3 [(Yang et al., 2025)] and Qwen3.5 [(Team, 2026)] families, including Qwen3-1.7B , Qwen3-4B , and Qwen3.5-9B . All models are trained and evaluated in non-thinking mode, unless otherwise specified. All models are trained on DAPO-17k [(Yu et al., 2026)] using only the questions, without access to labels and ground-truth answers. We evaluate the models on three in-domain mathematical reasoning benchmarks, AIME24, AIME25, and HMMT25, as well as the out-of-domain benchmark MBPP+ [(Liu et al., 2023)] (Code), and GPQA-Diamond [(Rein et al., 2023)] (Q&A).

[element]
Methods Dense Signal No Verifiable Reward No External Teacher No Hint RLVR [(Shao et al., 2024)] ✗ ✗ ✓ ✓ TTRL [(Zuo et al., 2026)] ✗ ✓ ✓ ✓ OPD [(Lu & Lab, 2025)] ✓ ✓ ✗ ✓ OPSD [(Zhao et al., 2026)] ✓ ✓ ✓ ✗ OPSA (Ours) ✓ ✓ ✓ ✓ Table 1: Comparison of supervision signals in different on-policy training methods.
[/element]

##### Baselines.

We compare OPSA with representative on-policy training methods, including reinforcement learning with verifiable rewards (RLVR), such as GRPO [(Shao et al., 2024)] ; Test-Time Reinforcement Learning (TTRL) [(Zuo et al., 2026)] , which does not require external supervision; standard On-Policy Distillation (OPD) [(Lu & Lab, 2025)] ; and On-Policy Self-Distillation (OPSD) [(Zhao et al., 2026)] , which replaces an external teacher with hints corresponding to questions. Table 1 summarizes the key properties of these methods. OPSA combines their desirable characteristics: it provides dense token-level learning signals while requiring neither verifiable rewards, external teacher models, nor auxiliary hints.

##### Implementation Details.

We implement OPSA using the slime [(Zhu et al., 2025)] framework and optimize the objective in Eq. 5 . The loss is computed only over the 20% of policy-sampled tokens with the lowest log-probabilities. All experiments are conducted on 8 NVIDIA H100 or H200 GPUs.

### 5.2Main Results

[element]
Methods Math (I.D.) O.O.D. AIME24 AIME25 HMMT25 MBPP+ GPQA D avg@32 pass@32 avg@32 pass@32 avg@32 pass@32 avg@32 Qwen3-1.7B 13.44 40.00 9.69 30.00 5.73 23.33 58.24 27.92 w/ OPSA 48.85 80.00 35.31 66.67 23.33 50.00 59.44 32.40 $\Delta$ +35.41 +40.00 +25.62 +36.67 +17.60 +26.67 +1.20 +4.48 $\uparrow$ 263.5% $\uparrow$ 100.0% $\uparrow$ 264.4% $\uparrow$ 122.2% $\uparrow$ 307.2% $\uparrow$ 114.3% $\uparrow$ 2.1% $\uparrow$ 16.0% Qwen3-4B 23.33 56.67 20.52 56.67 13.13 33.33 66.93 38.46 w/ OPSA 62.08 83.33 58.44 83.33 37.40 60.00 68.35 41.29 $\Delta$ +38.75 +26.66 +37.92 +26.66 +24.27 +26.67 +1.42 +2.83 $\uparrow$ 166.1% $\uparrow$ 47.0% $\uparrow$ 184.8% $\uparrow$ 47.0% $\uparrow$ 184.8% $\uparrow$ 80.0% $\uparrow$ 2.1% $\uparrow$ 7.4% Qwen3.5-9B 76.35 93.33 56.04 93.33 44.48 86.67 77.33 70.53 w/ OPSA 87.81 96.67 76.98 96.67 67.40 93.33 79.27 73.70 $\Delta$ +11.46 +3.34 +20.94 +3.34 +22.92 +6.66 +1.94 +3.17 $\uparrow$ 15.0% $\uparrow$ 3.6% $\uparrow$ 37.4% $\uparrow$ 3.6% $\uparrow$ 51.5% $\uparrow$ 7.7% $\uparrow$ 2.5% $\uparrow$ 4.5% Table 2: Performance of OPSA across different models on in-domain mathematical and out-of-domain code generation and general Q&A tasks. All models are evaluated in non-thinking mode.
[/element]

##### OPSA Consistently Improves Different Models.

Table 2 demonstrates that OPSA substantially improves performance across model families and scales. For the Qwen3 series, OPSA more than doubles Avg@32 on every mathematical reasoning benchmark. In particular, for Qwen3-1.7B , it yields relative improvements of 263% on both AIME24 and AIME25, and 307% on HMMT25, indicating a substantial enhancement in mathematical reasoning capability.
To verify that these gains are not limited to models with relatively limited post-training, we further evaluate OPSA on the Qwen3.5 series. Despite the already strong mathematical reasoning performance of Qwen3.5-9B , which achieves Avg@32 scores of 76.35 on AIME24 and 44.48 on HMMT25, OPSA delivers consistent additional improvements of 11.46 and 22.92 points, respectively.
Moreover, OPSA consistently improves Pass@32. Across the Qwen3 series, it yields relative gains ranging from 50% to 122%. For Qwen3.5-9B , despite the base model already achieving near-saturated Pass@32, OPSA further improves them by 3.34 and 6.66 points on AIME and HMMT25, respectively.

[element]
Methods AIME24 AIME25 HMMT25 Average avg@32 pass@32 avg@32 pass@32 avg@32 pass@32 avg@32 pass@32 Qwen3-1.7B 13.44 40.00 9.69 30.00 5.73 23.33 9.62 31.11 + GRPO 33.96 70.00 25.31 50.00 15.10 43.33 24.79 54.44 + TTRL 19.90 30.00 9.79 30.00 5.73 23.33 11.81 27.78 + OPD 32.08 73.33 20.52 50.00 13.85 40.00 22.15 54.44 + OPSD 33.33 73.33 22.50 53.33 14.90 43.33 23.58 56.67 + OPSA 48.85 80.00 35.31 66.67 23.33 50.00 35.83 65.56 $\Delta$ =OPSA-RL best +14.89 +6.67 +10.00 +13.34 +8.23 +6.67 +11.04 +8.89 Qwen3-1.7B ${}_{\text{Thinking}}$ 46.56 80.00 36.25 73.33 22.92 60.00 35.24 71.11 + OPSA 52.50 83.33 42.79 73.33 26.85 60.00 40.71 72.22 Table 3: Comparison of different on-policy RL methods on Qwen3-1.7B .
[/element]

##### OPSA Outperforms Baselines Using Zero External Supervision.

We compare OPSA with several representative on-policy RL baselines. For clarity, TTRL performs test-time training directly on AIME24, and we evaluate its best checkpoint; OPD uses Qwen3-4B-Instruct as the teacher model.
As shown in Table 3 , OPSA consistently outperforms all baselines in both Avg@32 and Pass@32 while requiring no external supervision. Averaged across the three benchmarks, OPSA surpasses the best baseline by 11.04 points in Avg@32 and 8.89 points in Pass@32.
Although TTRL is also free from external supervision, its self-consistency-based training tends to sharpen the policy distribution around a local optimum, substantially degrading Pass@ $k$ . OPSD removes the external teacher by conditioning the student policy on additional hints to construct a teacher distribution. However, we find that it yields meaningful gains only when thinking mode is disabled for the student but enabled for the teacher, thereby creating a substantial distributional mismatch at the first token position. OPD and GRPO, two widely adopted post-training methods, achieve comparable overall performance. As shown in Table 6 in Appendix B.3 , OPD, OPSD, and OPSA require fewer rollout generations during training and are therefore substantially more efficient than GRPO.

##### Generalization of OPSA.

To evaluate its out-of-domain generalization, we further test the models on the code-generation benchmark MBPP+ and general Q&A benchmark GPQA-Diamond. As shown in Table 2 , OPSA consistently improves both tasks across different models, demonstrating its ability to generalize beyond the training domain.
Additionally, we disable thinking mode ( enable_thinking=false ) during all on-policy rollouts used for OPSA training. Table 3 also reports performance when thinking mode is enabled at inference time. Interestingly, with thinking mode disabled, the OPSA model achieves performance comparable to that of the base model with thinking mode enabled, suggesting that OPSA shifts the policy toward longer-form reasoning behaviors. Enabling thinking mode for the OPSA model yields further improvements, demonstrating that the gains from OPSA remain complementary to the model’s native thinking capability.

### 5.3More Analysis and Ablation

#### 5.3.1OPSA Elicits Reflective Long-Form Reasoning

[element]
Figure 7: OPSA elicits long-form reasoning by generating more reflective tokens compared to base models. Moreover, AIME24 Avg@4 performance increases positively with response length.
[/element]

Section 4.3 shows that, at high-entropy positions, OPSA redistributes probability mass among head tokens rather than concentrating it on a single prediction. Such positions often correspond to “fork” tokens in the chain of thought [(Wang et al., 2026c;Zhang et al., 2026b)] , where different tokens may initiate distinct reasoning branches. Fig. 7 (a) shows that the response length increases steadily throughout OPSA training.
To better understand this change, we compare the frequency of reflective tokens in reasoning trajectories before and after training. As shown in Fig. 7 (b), the OPSA-trained model produces substantially more reflective expressions, indicating more frequent reflection and self-correction during reasoning. We further observe a clear positive correlation between response length and answer accuracy. Together, these findings suggest that OPSA promotes exploration at high-entropy forks, leading the model toward longer and potentially more productive reasoning branches characterized by increased reflection and self-correction.

##### Performance Drops When Fork Tokens Are Masked.

[element]
Figure 8: Training dynamics of OPSA when masking fork tokens.
[/element]

To validate whether OPSA’s gains primarily arise from probability redistribution at fork tokens in the chain of thought, we conduct an ablation study by masking fork positions whose head-token sets contain reflective words, thereby excluding these positions from training.
As shown in Fig. 8 , masking these fork tokens largely eliminates the increases in both response length and accuracy observed under OPSA. Moreover, the response length collapses at approximately 300 training steps.
These results demonstrate that applying OPSA at fork positions suppresses low-probability tail tokens while redistributing probability mass among competing head tokens. This encourages the policy to explore longer and more reflective reasoning branches, which appears to be a major source of its performance improvement. More details are given in Appendix B.3 .

#### 5.3.2OPSA Doesn’t Hurt Models’ Diversity

[element]
Figure 9: Diversity analysis of OPSA. (a) As reasoning length increases, more opportunities for branching arise. At each fork, OPSA distributes probability more evenly across alternative tokens, allowing multiple sampling to explore different reasoning paths and thereby increasing response diversity. (b–c) Jaccard-distance comparisons on AIME24 before and after OPSA training, using 32 sampled responses per problem.
[/element]

Although the analysis in Section 4.3 shows that OPSA sharpens the overall token distribution, particularly at low-entropy positions, it redistributes probability mass among competing head tokens at high-entropy fork positions. This mechanism allows OPSA to preserve response diversity in long-form reasoning. We quantify diversity using Jaccard distance (JD) [(Broder, 1997;Wang & Wan, 2018)] , where lower values indicate greater similarity between responses. Fig. 9 reports JD for the base and OPSA-trained models across different response lengths. As the number of generated tokens increases, the JD gap between the two models gradually narrows and eventually approaches zero, indicating that OPSA preserves a level of long-form response diversity comparable to that of the base model. This result suggests that OPSA continues to explore alternative reasoning branches at successive fork positions, producing the branching, tree-like reasoning patterns illustrated in Fig. 9 (a) without causing diversity collapse. The Pass@32 results in Table 2 provide further evidence that OPSA-induced distribution sharpening neither restricts the policy’s exploration space nor degrades its pass@k performance.

[element]
Figure 10: Ablation study of OPSA with different training token ratios.
[/element]

#### 5.3.3Ablation Study of The Fraction of Trained Tokens

We have examined the advantage-assignment scheme of OPSA in Section 3.2 , where negative advantages positively correlated with token entropy yield the strongest improvement. Here, we ablate the proportion of low-log-probability tokens used for training. Fig. 10 reports the Avg@4 training curves when OPSA is applied to the lowest 10%, 20% (ours), 30%, and 40% of tokens ranked by their on-policy log-probabilities.

Training on only the bottom 10% performs substantially worse than the other settings. We find that these tokens consist almost entirely of low-probability tail tokens outside the top-1 prediction. Applying negative advantages exclusively to them over-sharpens the policy distribution and causes a pronounced entropy decrease, thereby limiting further improvement. In contrast, the 20%, 30%, and 40% settings all increase Avg@4 to above 45. These results indicate that OPSA is not highly sensitive to the exact token-selection ratio and that training on only the bottom 20% of sampled tokens is sufficient to achieve substantial gains.

## 6Related Works

##### Supervision Signals in On-Policy Distillation.

OPD optimizes a student policy on its self-generated trajectories. GKD [(Agarwal et al., 2024)] combines student-generated and supervised trajectories under flexible divergence objectives, while MiniLLM [(Gu et al., 2024)] adopts reverse-KL distillation to reduce exposure bias. More recently, [Lu & Lab (2025)] improve OPD efficiency for reasoning tasks using the K1 estimator. Another line of work [(Zhao et al., 2026;Shenfeld et al., 2026;Hübotter et al., 2026)] replaces the external teacher with the same policy conditioned on additional hints, yielding on-policy self-distillation (OPSD). However, OPSD retains the OPD training paradigm, leaving unresolved whether the constructed teacher can provide reliable supervision on student-generated trajectories. Recent studies aim to avoid potential training noise in OPD through token selection [(Xu et al., 2026;Fu et al., 2026)] or fine-grained credit assignment [(Xing et al., 2026;Xie et al., 2026;Hou et al., 2026)] , but still rely on external supervision. In contrast, we investigate the source of OPD’s gains and propose OPSA, which enables self-improvement without external supervision.

##### Label-Free Policy Improvement.

Label-free training enables a policy to improve using signals derived from the model itself, without external supervision. Self-rewarding methods [(Yuan et al., 2024;Ding & Zhang, 2026)] use an LLM-as-a-Judge [(Zheng et al., 2023)] to score the model’s own responses and construct reward signals. TTRL [(Zuo et al., 2026)] and EMPO [(Zhang et al., 2026a)] instead infer pseudo-gold answers from GRPO rollouts through majority voting or clustering, and then use them for reward assignment. Intuitor [(Zhao et al., 2025b)] adopts self-certainty, using trajectory-level entropy as a reward to encourage the policy toward more confident outputs.
However, these methods depend heavily on the policy’s initial capability. If the correct answer does not lie near a mode of the policy distribution, coarse trajectory-level rewards may reinforce an incorrect mode, causing the policy to become increasingly overconfident and ultimately leading to a collapse in Pass@ $k$ . In contrast, OPSA does not force the policy to converge toward any self-generated reward signal. Instead, it provides finer-grained token-level learning signals through entropy-adaptive negative advantages, simultaneously improving precision at low-entropy positions and exploration at high-entropy fork tokens, thereby enhancing the policy’s overall reasoning capability.

## 7Conclusion

The central contribution of this work is a systematic investigation into the source of performance gains in OPD. We find that much of OPD’s improvement can be reproduced without teacher supervision by assigning fixed negative advantages to low-probability tokens, while scaling these advantages according to token-level entropy further enhances performance. Building on these findings, we introduce O n- P olicy S elf- A daptation ( OPSA ), an external-supervision-free framework that enables policy self-improvement through entropy-adaptive negative advantages.

Our analysis shows that OPSA suppresses low-probability tail tokens while redistributing probability mass among competing head tokens. This adaptively sharpens the overall token distribution, improving prediction confidence at low-entropy positions while preserving exploration at high-entropy reasoning forks. As a result, OPSA encourages longer and more reflective reasoning trajectories, leading to substantial improvements in both Avg@32 and Pass@32.

Notably, OPSA outperforms OPD without teacher supervision while exhibiting similar training dynamics, including response-length growth. This observation calls for a reexamination of the mechanism underlying OPD with the K1 estimator: its gains may arise not only from making the student imitate a stronger teacher, but also from reshaping the student policy’s own probability distribution. More broadly, our results show that the policy’s internal token-level uncertainty can itself provide an effective fine-grained learning signal for on-policy self-improvement, without rewards, reference answers, or teacher-provided supervision.

## References

Agarwal et al. (2024)

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem.

On-policy distillation of language models: Learning from self-generated mistakes.

In 
International Conference on Learning Representations
, volume 2024, pp.  21246–21263, 2024.

Broder (1997)

Andrei Z Broder.

On the resemblance and containment of documents.

In 
Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No. 97TB100171)
, pp.  21–29. IEEE, 1997.

Burnham & Anderson (2001)

Kenneth P Burnham and David R Anderson.

Kullback-leibler information as a basis for strong inference in ecological studies.

Wildlife research
, 28(2):111–119, 2001.

Chen et al. (2025)

Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie.

Sft or rl? an early investigation into training r1-like reasoning large vision-language models.

arXiv preprint arXiv:2504.11468
, 2025.

Chu et al. (2025)

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma.

Sft memorizes, rl generalizes: A comparative study of foundation model post-training.

arXiv preprint arXiv:2501.17161
, 2025.

Ding & Zhang (2026)

Yi Ding and Ruqi Zhang.

Sherlock: Self-correcting reasoning in vision-language models.

Advances in Neural Information Processing Systems
, 38:101638–101672, 2026.

Ding et al. (2026)

Yi Ding, Ziliang Qiu, Bolian Li, and Ruqi Zhang.

Learning self-correction in vision–language models via rollout augmentation.

In 
Forty-third International Conference on Machine Learning
, 2026.

Fu et al. (2026)

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao.

Revisiting on-policy distillation: Empirical failure modes and simple fixes.

arXiv preprint arXiv:2603.25562
, 2026.

GLM-5-Team et al. (2026)

GLM-5-Team, :, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, Chenzheng Zhu, Congfeng Yin, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, Jiajie Zhang, Jian Jiao, Jiaqi Guo, Jingsen Wang, Jingzhao Du, Jinzhu Wu, Kedong Wang, Lei Li, Lin Fan, Lucen Zhong, Mingdao Liu, Mingming Zhao, Pengfan Du, Qian Dong, Rui Lu, Shuang-Li, Shulin Cao, Song Liu, Ting Jiang, Xiaodong Chen, Xiaohan Zhang, Xuancheng Huang, Xuezhen Dong, Yabo Xu, Yao Wei, Yifan An, Yilin Niu, Yitong Zhu, Yuanhao Wen, Yukuo Cen, Yushi Bai, Zhongpei Qiao, Zihan Wang, Zikang Wang, Zilin Zhu, Ziqiang Liu, Zixuan Li, Bojie Wang, Bosi Wen, Can Huang, Changpeng Cai, Chao Yu, Chen Li, Chengwei Hu, Chenhui Zhang, Dan Zhang, Daoyan Lin, Dayong Yang, Di Wang, Ding Ai, Erle Zhu, Fangzhou Yi, Feiyu Chen, Guohong Wen, Hailong Sun, Haisha Zhao, Haiyi Hu, Hanchen Zhang, Hanrui Liu, Hanyu Zhang, Hao Peng, Hao Tai, Haobo Zhang, He Liu, Hongwei Wang, Hongxi Yan,
Hongyu Ge, Huan Liu, Huanpeng Chu, Jia’ni Zhao, Jiachen Wang, Jiajing Zhao, Jiamin Ren, Jiapeng Wang, Jiaxin Zhang, Jiayi Gui, Jiayue Zhao, Jijie Li, Jing An, Jing Li, Jingwei Yuan, Jinhua Du, Jinxin Liu, Junkai Zhi, Junwen Duan, Kaiyue Zhou, Kangjian Wei, Ke Wang, Keyun Luo, Laiqiang Zhang, Leigang Sha, Liang Xu, Lindong Wu, Lintao Ding, Lu Chen, Minghao Li, Nianyi Lin, Pan Ta, Qiang Zou, Rongjun Song, Ruiqi Yang, Shangqing Tu, Shangtong Yang, Shaoxiang Wu, Shengyan Zhang, Shijie Li, Shuang Li, Shuyi Fan, Wei Qin, Wei Tian, Weining Zhang, Wenbo Yu, Wenjie Liang, Xiang Kuang, Xiangmeng Cheng, Xiangyang Li, Xiaoquan Yan, Xiaowei Hu, Xiaoying Ling, Xing Fan, Xingye Xia, Xinyuan Zhang, Xinze Zhang, Xirui Pan, Xu Zou, Xunkai Zhang, Yadi Liu, Yandong Wu, Yanfu Li, Yidong Wang, Yifan Zhu, Yijun Tan, Yilin Zhou, Yiming Pan, Ying Zhang, Yinpei Su, Yipeng Geng, Yong Yan, Yonglin Tan, Yuean Bi, Yuhan Shen, Yuhao Yang, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yurong Wu, Yutao Zhang, Yuxi Duan, Yuxuan Zhang,
Zezhen Liu, Zhengtao Jiang, Zhenhe Yan, Zheyu Zhang, Zhixiang Wei, Zhuo Chen, Zhuoer Feng, Zijun Yao, Ziwei Chai, Ziyuan Wang, Zuzhou Zhang, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang.

Glm-5: from vibe coding to agentic engineering, 2026.

URL 
https://arxiv.org/abs/2602.15763
.

Gu et al. (2024)

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang.

Minillm: Knowledge distillation of large language models.

In 
International Conference on Learning Representations
, volume 2024, pp.  32694–32717, 2024.

Guo et al. (2025)

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al.

Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.

arXiv preprint arXiv:2501.12948
, 2025.

He et al. (2026)

Yuhang He, Haodong Wu, Siyi Liu, Hongyu Ge, Hange Zhou, Keyi Wu, Zhuo Zheng, Qihong Lin, Zixin Zhong, and Yongqi Zhang.

Where hindsight credit can reside: A signed-capacity view of token updates in rlvr.

arXiv preprint arXiv:2604.11056
, 2026.

Hou et al. (2026)

Wenjin Hou, Shangpin Peng, Weinong Wang, Zheng Ruan, Yue Zhang, Zhenglin Zhou, Mingqi Gao, Yifei Chen, Kaiqi Wang, Hongming Yang, et al.

Uni-opd: Unifying on-policy distillation with a dual-perspective recipe.

arXiv preprint arXiv:2605.03677
, 2026.

Hübotter et al. (2026)

Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al.

Reinforcement learning via self-distillation.

arXiv preprint arXiv:2601.20802
, 2026.

Jia et al. (2026)

Nan Jia, Haojin Yang, Xing Ma, Jiesong Lian, Shuailiang Zhang, Weipeng Zhang, Ke Zeng, Xunliang Cai, and Zequn Sun.

Asymmetric on-policy distillation: Bridging exploitation and imitation at the token level.

arXiv preprint arXiv:2605.06387
, 2026.

Liu et al. (2023)

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang.

Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation.

Advances in neural information processing systems
, 36:21558–21572, 2023.

Lu & Lab (2025)

Kevin Lu and Thinking Machines Lab.

On-policy distillation.

Thinking Machines Lab: Connectionism
, 2025.

doi: 
10.64434/tml.20251026
.

https://thinkingmachines.ai/blog/on-policy-distillation.

Rein et al. (2023)

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman.

Gpqa: A graduate-level google-proof q&a benchmark.

arXiv preprint arXiv:2311.12022
, 2023.

Shao et al. (2024)

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al.

Deepseekmath: Pushing the limits of mathematical reasoning in open language models.

arXiv preprint arXiv:2402.03300
, 2024.

Shenfeld et al. (2026)

Idan Shenfeld, Mehul Damani, Jonas Hübotter, and Pulkit Agrawal.

Self-distillation enables continual learning.

arXiv preprint arXiv:2601.19897
, 2026.

Team (2026)

Qwen Team.

Qwen3.5: Accelerating productivity with native multimodal agents, February 2026.

URL 
https://qwen.ai/blog?id=qwen3.5
.

Wang et al. (2026a)

Bing Wang, Shaotian Yan, Chen Shen, Sinan Fan, Ximing Li, Rui Miao, Xiaosong Yuan, Zhanming Shen, Jieping Ye, et al.

Backtracking when it strays: Mitigating dual exposure biases in llm reasoning distillation.

arXiv preprint arXiv:2605.19433
, 2026a.

Wang et al. (2026b)

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen.

Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning.

Advances in Neural Information Processing Systems
, 38:30865–30891, 2026b.

Wang & Wan (2018)

Ke Wang and Xiaojun Wan.

Sentigan: Generating sentimental texts via mixture adversarial networks.

In 
IJCAI
, pp.  4446–4452, 2018.

Wang et al. (2026c)

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang, et al.

Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning.

Advances in Neural Information Processing Systems
, 38:115452–115486, 2026c.

Xie et al. (2026)

Yan Xie, Sijie Zhu, Tiansheng Wen, Bo Chen, and Yifei Wang.

On the position bias of on-policy distillation.

arXiv preprint arXiv:2606.22600
, 2026.

Xing et al. (2026)

Xingrun Xing, Haoqing Wang, Boyan Gao, Ziheng Li, and Yehui Tang.

Trust region on-policy distillation.

arXiv preprint arXiv:2606.01249
, 2026.

Xu et al. (2026)

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard.

Tip: Token importance in on-policy distillation.

arXiv preprint arXiv:2604.14084
, 2026.

Yang et al. (2025)

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al.

Qwen3 technical report.

arXiv preprint arXiv:2505.09388
, 2025.

Yang et al. (2026)

Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan.

Self-distilled rlvr.

arXiv preprint arXiv:2604.03128
, 2026.

Yu et al. (2026)

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al.

Dapo: An open-source llm reinforcement learning system at scale.

Advances in Neural Information Processing Systems
, 38:113222–113244, 2026.

Yuan et al. (2024)

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston.

Self-rewarding language models.

arXiv preprint arXiv:2401.10020
, 2024.

Yue et al. (2025)

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al.

Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks.

arXiv preprint arXiv:2504.05118
, 2025.

Zhang et al. (2026a)

Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, and Yatao Bian.

Right question is already half the answer: Fully unsupervised llm reasoning incentivization.

Advances in neural information processing systems
, 38:67345–67372, 2026a.

Zhang et al. (2026b)

Ruixiang Zhang, Richard He Bai, Huangjie Zheng, Navdeep Jaitly, Ronan Collobert, and Yizhe Zhang.

Embarrassingly simple self-distillation improves code generation.

arXiv preprint arXiv:2604.01193
, 2026b.

Zhao et al. (2025a)

Jiachen Zhao, Yiyou Sun, Weiyan Shi, and Dawn Song.

Can aha moments be fake? towards quantifying decorative and true thinking in chain-of-thought.

arXiv preprint arXiv:2510.24941
, 2025a.

Zhao et al. (2026)

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover.

Self-distilled reasoner: On-policy self-distillation for large language models.

arXiv preprint arXiv:2601.18734
, 2026.

Zhao et al. (2025b)

Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song.

Learning to reason without external rewards.

arXiv preprint arXiv:2505.19590
, 2025b.

Zheng et al. (2023)

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al.

Judging llm-as-a-judge with mt-bench and chatbot arena.

Advances in neural information processing systems
, 36:46595–46623, 2023.

Zhu et al. (2026)

Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng.

The surprising effectiveness of negative reinforcement in llm reasoning.

Advances in Neural Information Processing Systems
, 38:126546–126573, 2026.

Zhu et al. (2025)

Zilin Zhu, Chengxing Xie, Xin Lv, and slime Contributors.

slime: An llm post-training framework for rl scaling.

https://github.com/THUDM/slime
, 2025.

GitHub repository. Corresponding author: Xin Lv.

Zuo et al. (2026)

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Xinwei Long, Ermo Hua, Biqing Qi, et al.

Ttrl: Test-time reinforcement learning.

Advances in Neural Information Processing Systems
, 38:131459–131483, 2026.

## Appendix ALimitations and Future Directions

Due to limited computational resources, our experiments focus on relatively small models of up to 9B parameters. It therefore remains unclear whether OPSA scales effectively to larger models or mixture-of-experts architectures. Moreover, our analysis suggests that OPSA primarily operates by redistributing the policy’s existing probability mass. Its benefits may therefore be limited for heavily post-trained models whose output distributions are already overly sharp and exhibit very low entropy. Although OPSA requires no external annotations and can serve as a plug-and-play post-training method, it may not substantially expand the policy’s underlying exploration frontier, as reflected by the relatively modest improvements in thinking-mode Pass@ $k$ . A promising direction is to combine OPSA’s fine-grained negative-advantage assignment with other reinforcement learning methods to further improve policy exploration.
In addition, OPSA outperforms OPD without teacher supervision while exhibiting similar training dynamics, including response-length growth. This finding calls for a reexamination of the mechanism underlying OPD with the K1 estimator and motivates further theoretical analysis to identify the true source of OPD’s performance gains.

## Appendix BExperimental Details

### B.1Training Configuration

[element]
Hyperparameter Value Framework slime ( 0.2.4 ) Training engine Megatron ( 0.16.0rc0 ) Rollout engine SGLang ( 0.5.14 ) GPUs 8xH100 GPUs Learning rate 1 $\times$ 10 -6 Rollout batch size 64 n samples per prompt 1 Enable Thinking False Global batch size 64 Training decoding Temperature 1.0, top-k -1, top-p 1.0, max response length 12000 Evaluation decoding Temperature 0.7, top-k 20, top-p 0.8, max response length 32768 Save & Evaluation frequency Save every 20 steps, validate every 20 steps Checkpoint selection metric Avg@4 on validation set Table 4: Training hyperparameters for OPSA.
[/element]

We provide the detailed OPSA training configuration in Table 4 . For rollout during training, we adopt the default decoding parameters from the slime repository. For evaluation, we use the decoding parameters recommended in the Qwen3 model card. Following prior work [(Zhao et al., 2026;Hou et al., 2026)] , we select the best checkpoint for each method based on Avg@4 performance.

### B.2Evaluation Configuration

[element]
Hyperparameter Value Rollout engine SGLang ( 0.5.14 ) Temperature 0.7 Top-k 20 Top-p 0.8 Max new tokens 32768 Enable thinking False Samples per prompt 32 Metric Avg@32 & Pass@32 Table 5: Evaluation hyperparameters for OPSA.
[/element]

We provide the detailed evaluation configuration in Table 5 . We use SGLang for inference, generate 32 responses per prompt with thinking mode disabled, and report both Avg@32 and Pass@32.

### B.3More Details

##### Training and Inference Overhead of OPSA.

[element]
Training Inference Methods Step Time (s) Token Counts Time (s) Avg@32 Qwen3-1.7B - 4457 1.78 13.44 + GRPO 186.2 19108 5.27 33.96 + OPD 61.2 15286 5.73 32.08 + OPSA 46.3 23205 6.58 48.85 Qwen3-4B - 8015 3.31 23.33 + OPSA 68.6 20972 6.97 62.08 Qwen3.5-9B - 6847 5.29 76.35 + OPSA 214.7 9695 6.31 87.81 Table 6: Training and inference overhead of OPSA compared to baselines.
[/element]

We compare the training and inference overhead of different methods in Table 6 . OPSA and OPD train faster than GRPO because they do not require extensive rollout sampling to construct response groups. OPSA is even more efficient than OPD, as it neither deploys an additional teacher model nor performs forward passes through a larger and slower teacher. This allows more GPUs to be allocated to policy rollout and optimization. At inference time, OPSA steers the policy toward more reflective reasoning branches, producing longer responses while achieving higher accuracy.

##### Reflective Word Set for Experiments in Fig.8.

We list the detailed reflective word set used in experiments in Fig. 8 as follows:

Reflective Word Set

"wait", "however", "but", "alternatively", "hmm", "perhaps", "check", "might", "actually"

Following the standard OPSA training procedure, we first identify the 20% of token positions with the lowest student log-probabilities. At each selected position, we inspect the student’s top-5 candidate tokens. If any candidate belongs to the predefined reflective word set, we classify the position as a reflective fork and exclude it from the training objective.

##### Jaccard Distance in Section5.3.2.

We measure response diversity using the pairwise Jaccard distance between
sets of token-level 4-grams. For each response, we retain its first $L$ tokens, where $L\in\{512,1024,2048,4096,\mathrm{Full}\}$ , and denote
the resulting set of unique 4-grams by $\mathcal{G}_{4}(r^{(\leq L)})$ . The distance between two responses is
defined as

[element]
$d_{\mathrm{J}}(r_{i},r_{j};L)=1-\frac{\left|\mathcal{G}_{4}(r_{i}^{(\leq L)})\cap\mathcal{G}_{4}(r_{j}^{(\leq L)})\right|}{\left|\mathcal{G}_{4}(r_{i}^{(\leq L)})\cup\mathcal{G}_{4}(r_{j}^{(\leq L)})\right|}.$ (6)
[/element]

For each AIME24 problem, we average the distance over all response pairs
among the $R=32$ sampled responses, and then report the macro-average
over all $P=30$ problems:

[element]
$D_{\mathrm{J}}(L)=\frac{1}{P}\sum_{p=1}^{P}\frac{2}{R(R-1)}\sum_{1\leq i<j\leq R}d_{\mathrm{J}}(r_{p,i},r_{p,j};L).$ (7)
[/element]

A larger $D_{\mathrm{J}}(L)$ indicates greater response diversity. We
additionally report the difference between OPSA and the base model as

[element]
$\Delta D_{\mathrm{J}}(L)=D_{\mathrm{J}}^{\mathrm{OPSA}}(L)-D_{\mathrm{J}}^{\mathrm{Base}}(L).$ (8)
[/element]

## Appendix CAdditional Results

### C.1Comparison to Negative Sample Reinforcement

[element]
Methods AIME24 AIME25 HMMT25 Average avg@32 pass@32 avg@32 pass@32 avg@32 pass@32 avg@32 pass@32 Qwen3-1.7B 13.44 40.00 9.69 30.00 5.73 23.33 9.62 31.11 + GRPO 33.96 70.00 25.31 50.00 15.10 43.33 24.79 54.44 + NSR 32.08 73.33 24.17 60.00 16.15 43.33 24.13 58.89 + OPSA 48.85 80.00 35.31 66.67 23.33 50.00 35.83 65.56 Table 7: Comparison of different on-policy RL methods on Qwen3-1.7B .
[/element]

NSR [(Zhu et al., 2026)] decomposes RLVR trajectories into correct and incorrect samples and performs trajectory-level reinforcement learning with negative advantages. Specifically, it assigns a fixed negative advantage to every token in an incorrect trajectory, thereby improving exploration while maintaining stable policy entropy during training. The key distinction between OPSA and NSR is that OPSA requires no supervisory signal, including verifiable rewards. Instead, it assigns token-level negative advantages to low-log-probability tokens across all trajectories and dynamically modulates these advantages according to token entropy, enabling substantially finer-grained optimization.
Combined with our analysis in Section 4.3 , these results suggest that OPSA suppresses low-probability tail tokens and redistributes their probability mass more evenly among high-probability head tokens. This mechanism preserves accuracy at low-entropy positions while enhancing exploration at high-entropy fork tokens. As shown in Table 7 , OPSA consistently outperforms both GRPO and NSR in single-sample accuracy and pass@ $k$ , validating the effectiveness of our approach. The results further demonstrate that applying negative advantages to low-probability tokens even within correct trajectories is important for improving model performance.

### C.2Low Entropy Can Still Support Effective Exploration

[element]
Figure 11: Training dynamics of different methods on Qwen3-1.7B and pass@32 performance.
[/element]

As shown in Figure 11 , OPSA exhibits substantially lower training entropy than GRPO and NSR while consistently achieving higher pass@32 on AIME24, AIME25, and HMMT25. This result demonstrates that aggregate entropy alone is not a reliable indicator of a model’s exploration ability. Effective exploration depends more critically on how uncertainty is allocated across token positions. At high-entropy fork tokens, OPSA maintains a relatively balanced probability distribution among plausible head tokens, enabling the model to explore multiple promising reasoning branches. At low-entropy positions, it further concentrates probability mass on high-confidence tokens, improving sampling accuracy and preventing trajectories from entering implausible, low-confidence branches. Consequently, OPSA achieves lower overall entropy without sacrificing diversity and meaningful exploration at critical decision points, consistent with our analysis in Section 5.3.2 .

### C.3Performance Comparison Under Similar Token Budgets

[element]
Methods Token Counts Avg@32 Qwen3-1.7B 4457 13.44 + GRPO 19108 33.96 + GRPO (“wait”) 23261 32.81 + OPD 15286 32.08 + OPD (“wait”) 23472 31.67 + OPSA 23205 48.85 Table 8: AIME24 performance of different RL strategy on similar inference token budgets.
[/element]

To verify that OPSA’s gains are not merely a consequence of generating longer responses, we construct a token-budget-matched inference control for the GRPO and OPD baselines. Specifically, we first generate a baseline response, remove its terminal \boxed{} answer, append a single “wait” token to the remaining reasoning trace, and then resume decoding. A minimum generation-length constraint is applied so that the resulting responses approximately match the average response length of the OPSA-trained model. We evaluate the first complete \boxed{} answer produced after reaching the target token budget while keeping all other decoding settings unchanged. As shown in Table 8 , allocating additional inference-time tokens does not improve the baselines, and OPSA continues to substantially outperform both GRPO and OPD under comparable response lengths. These results indicate that OPSA’s improvements cannot be explained by response length alone. Instead, they are consistent with OPSA reshaping the policy distribution to improve precision at low-entropy positions and encourage exploration at high-entropy positions, thereby steering generation toward more effective reflective reasoning branches.

### C.4OPSA as a Cold-Starting for GRPO Training

[element]
Figure 12: Training dynamics of GRPO training started from the OPSA cold-starting 4B model.
[/element]

OPSA demonstrates strong effectiveness during post-training, improving both accuracy and Pass@ $k$ while preserving sampling diversity. In this section, we investigate whether OPSA, as an external-supervision-free method, can serve as a “free-lunch” cold start before further RL training and enable additional performance gains. We conduct experiments on Qwen3-4B , using an OPSA checkpoint as the initialization and further training it with GRPO on DAPO-17k. Figure 12 reports the resulting training dynamics. On the validation set, Avg@4 continues to improve steadily, increasing by approximately 9 points after 40 training steps, while the training curve remains smooth without signs of collapse. These results suggest that OPSA can provide an effective cold start for subsequent RL training, further extending the model’s capabilities.


