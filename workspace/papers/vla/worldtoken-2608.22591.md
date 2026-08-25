# 2608.22591 (from arXiv HTML; MinerU fallback)



# WorldToken: Time-First Sequence Modeling for Robotic Imitation Learning

Chunkai Yang

  
Andong Yang

  

  
Chao Gao

###### Abstract

Can a robot read the physical world as a language model reads text? In language, the basic sequence unit is given through the text token. In the physical world, there is an equally natural organizing axis: physical time. Yet each policy timestep carries heterogeneous inputs, such as multiple camera views, proprioception, and language conditioning, so we keep heterogeneity within each policy timestep and let physical time define the top-level sequence. In this work, we design WorldToken to instantiate this time-first principle. Its defining operation is to resolve observation heterogeneity within each policy timestep, producing a single world token before temporal modeling. The resulting instantiation consists of three components: a multimodal encoder that constructs the world token within each policy timestep, a causal temporal Transformer that contextualizes the world-token sequence, and a diffusion action head that generates action chunks from the current history representation. We evaluate the complete WorldToken instantiation through three questions: whether it can learn effective policies, how its performance changes with data and model capacity under a fixed sequence organization, and how trained WorldToken policies use visible temporal context. On 23 RoboCasa tasks, an 85M-parameter policy trained from scratch on 2,900 generated demonstrations per task reaches a task-averaged closed-loop success rate of approximately 60%, entering the performance range reported by billion-scale pretrained systems. A complete $5\times 5\times 2$ scaling study shows consistent gains from additional data, with diminishing returns beyond moderate model size. To examine temporal-context use, we directly vary the visible history of trained policies: truncating RoboCasa history to one or two visible policy timesteps lowers the closed-loop success rate of all 50 policies, and on RMBench Blocks Ranking, reducing visible history from 146 to 8 seconds lowers success from 95% to 28%, while the same policy sustains the demonstrated behavior for over 850 seconds in an extended rollout. Together, these results establish the empirical feasibility of the complete WorldToken instantiation and characterize its data-scaling and temporal-context behavior under the tested recipes. They do not establish superiority over alternative sequence organizations or isolate which components of the complete implementation drive the observed performance. Instead, they establish time-first organization as a concrete, physically interpretable interface for studying embodied sequence modeling.

11footnotetext:

ychunkai@126.com,
School of Remote Sensing and Information Engineering,
Wuhan University, Wuhan, China.
22footnotetext:

andongyang@mail.tsinghua.edu.cn,
Department of Electronic Engineering,
Tsinghua University, Beijing, China.
33footnotetext:

chao.gao@cantab.net,
Institute for AI Industry Research,
Tsinghua University, Beijing, China.
11footnotetext: Corresponding author: Chao Gao.

## 1 Introduction

### 1.1 Can a robot organize physical interaction as a causal sequence?

Robot policies are moving from short-horizon reactive control toward large-scale vision–language–action sequence modeling. Yet these policies do not share a common answer to a basic modeling question: what should each token in the policy sequence represent? Inputs from a single policy timestep may be expanded into multiple modality-specific tokens; history may instead be handled by a recurrent module, an explicit memory, or a retrieval mechanism. All of these designs can use temporal information, but they organize time through different sequence units.

What each token represents shapes the interfaces between within-timestep multimodal encoding, temporal context modeling, and action generation. Different sequence organizations define top-level positions through different combinations of temporal, modal, spatial, and generation structure. WorldToken instead uses the policy timestep itself as the unit of the top-level temporal sequence, performing multimodal fusion within each timestep before temporal modeling.

In this work, we introduce WorldToken, a time-first policy instantiation. We treat each policy timestep as one observation–decision–replanning event: the robot receives multimodal observations, produces a short action sequence, acts on the environment, and then observes again. This closed-loop interaction forms the basic unit of temporal organization. Within each policy timestep, WorldToken fuses the multimodal observation into a single observation-derived world token, so that each timestep contributes one token to the top-level causal sequence. Action generation then operates on the resulting history-conditioned representation. We refer to this modeling view as time-first organization: physical time, represented by ordered policy timesteps, defines the primary top-level sequence axis, while multimodal fusion occurs within each timestep.

![Refer to caption](drafts/images/worldtoken-2608.22591/fig1.png)

*Figure 1: (a) A language model causally contextualizes homogeneous text tokens and directly appends its generated token. (b) WorldToken treats an embodied episode as a causal interaction with the physical world: each policy timestep contributes one world token to the temporal context, while the policy produces an action chunk that acts on the environment. The resulting observation provides the next world token, so the high-level sequence advances through successive policy timesteps.*

We evaluate this design through three empirical questions: whether it learns effective multitask control and scales with data and model capacity (Section 3); whether its decisions depend on the recent history it receives (Section 4); and whether extended visible context shapes sustained ordered behavior over long horizons (Section 5).

On 23 RoboCasa household-manipulation tasks, an 85.3M-parameter WorldToken policy reaches a 59.45% task-averaged closed-loop success rate at the largest tested dataset size of 2,900 demonstrations per task. All trainable policy modules are initialized from scratch; only the frozen CLIP text encoder is pretrained. Although target data, pretraining, and reporting conventions differ, published RoboCasa systems with billion-scale pretrained backbones report roughly 50–79% success rates, with most results between 58% and 70% (Appendix Table 14). At a matched dataset size of 300 demonstrations per task, WorldToken exceeds the from-scratch BC-Transformer baseline by 12–16 percentage points. Across a complete $5\times 5\times 2$ sweep over dataset size, model size, and training seed, every increase in data improves performance, while moderate model capacity is sufficient under the tested recipes.

We further find that all 50 RoboCasa policies show lower closed-loop success rates when their visible history is truncated at inference without retraining, while separately trained short-context policies recover most of this loss. This shows that the policies systematically use their temporal input, but that much of the observed dependence reflects adaptation to the training context rather than an irreducible task requirement.

Finally, an RMBench Blocks Ranking case study tests whether extended visible context affects sustained ordered behavior. With the checkpoint and 100 initial conditions fixed, reducing visible history from 608 to 32 world tokens lowers success from 95% to 28%, with failures concentrated in episodes requiring several ordered swaps rather than a single local manipulation. In exploratory extended rollouts, one trajectory completes 31 correctly ordered swaps, including 26 after the context begins to slide, despite training demonstrations ending at first success and containing at most five swaps. This shows that the learned behavior can continue well beyond both the demonstrated horizon and a single visible context span.

Together, these results establish that the complete WorldToken instantiation can learn effective multitask control. Within this fixed policy family, performance improves systematically with target-domain data, and trained policies make consequential use of temporal context.

Our contributions are three-fold:

1. 1.

We propose WorldToken, a time-first policy instantiation that organizes interaction history as one observation-derived world token per policy timestep. By resolving multimodal heterogeneity within each policy timestep, the temporal sequence has a fixed physical granularity: its length equals the number of policy decisions, independently of the number of perceptual or modality-specific tokens used within each step. This factorization separates within-timestep multimodal encoding, temporal context modeling, and action generation, while providing a context axis that is directly interpretable in policy timesteps.
2. 2.

We provide a systematic characterization of how the complete WorldToken instantiation scales with target-domain data and model capacity in large-scale multitask robotic imitation learning. Across 23 RoboCasa tasks, an 85M-parameter WorldToken policy reaches approximately 60% mean closed-loop success across 23 tasks, while a complete $5\times 5\times 2$ sweep over dataset size, model size, and training seed shows consistent gains from additional data and diminishing returns from model capacity under the tested regimes.
3. 3.

We provide controlled evidence for how temporal context is used and when extended context matters. Same-checkpoint history truncation across all 50 RoboCasa sweep policies, together with separately trained short-context policies, separates learned history utilization from task-intrinsic context requirements. On RMBench Blocks Ranking, longer visible context substantially improves sustained ordered behavior, while successful rollouts continue far beyond a single context-window span.

## 2 WorldToken policy architecture

Figure 2 summarizes the WorldToken architecture. We first define the policy decomposition, followed by the within-timestep multimodal encoder, causal temporal backbone, and generative action decoder.

![Refer to caption](drafts/images/worldtoken-2608.22591/fig2.png)

*Figure 2: Detailed WorldToken architecture. (a) A shared within-timestep encoder independently processes multiview images, proprioception, and task conditioning at every policy timestep. Learned readout tokens gather the observation and are projected into the world token $z_{t}$. (b) A causal temporal Transformer processes one world token per policy timestep and selects the current history representation $h_{t}$; the default temporal sequence contains no historical action tokens. (c) A DiT action denoiser receives adaLN-Zero conditions from $h_{t}$ and diffusion step $k$, generates an $H$-step action chunk, executes the first $H_{\mathrm{exec}}$ actions, and then replans from a new observation.*

### 2.1 A causal sequence of policy timesteps

Consider a temporally ordered robot record of length $T$,

|  | $$ \tau=(\tilde{o}_{1},\tilde{a}_{1},\tilde{o}_{2},\tilde{a}_{2},\ldots,\tilde{o}_{T},\tilde{a}_{T}). $$ |  | (1) |
|---|---|---|---|

The policy operates at observation–decision–replanning events indexed by $t$, which may subsample this lower-level record; throughout, a policy timestep means one such event rather than one simulator control step. At policy timestep $t$ the policy receives

|  | $$ o_{t}=\{I_{t}^{1:K},p_{t},c\}, $$ |  | (2) |
|---|---|---|---|

where $I_{t}^{1:K}$ are images from $K$ views, $p_{t}$ is proprioception, and $c$ is the task condition (a continuous language embedding or a task identifier, usually fixed within an episode). It then predicts an $H$-step action chunk $A_{t}=(a_{t,0},\ldots,a_{t,H-1})$.

WorldToken decomposes the policy into three sequential modules:

|  | $$ z_{t}=E_{\theta}(o_{t}),\qquad h_{t}=T_{\phi}(z_{\leq t}),\qquad A_{t}\sim\mathcal{D}_{\psi}(\cdot\mid h_{t}). $$ |  | (3) |
|---|---|---|---|

The within-timestep encoder $E_{\theta}$ compresses the current observation into $z_{t}\in\mathbb{R}^{d}$; the causal temporal backbone $T_{\phi}$ processes $z_{1},\ldots,z_{t}$ in temporal order and produces the history-conditioned representation $h_{t}$; the action decoder $\mathcal{D}_{\psi}$ generates the chunk. With $\Theta=(\theta,\phi,\psi)$, the complete policy is $\pi_{\Theta}(A_{t}\mid o_{\leq t}):=\mathcal{D}_{\psi}\bigl(A_{t}\mid T_{\phi}(E_{\theta}(o_{1}),\ldots,E_{\theta}(o_{t}))\bigr)$.

“One timestep, one token” constrains only the temporal representation. The within-timestep encoder may contain many camera views, visual patches, modality-specific features, and readout states. After fusion, the input at temporal position $t$ is always a single observation-derived representation of policy timestep $t$. Extending history means including earlier world tokens in the causal attention window, without a separate history encoder or retrieval rule.

The environment separates a generated action from the next sequence input:

|  | $$ \hat{A}_{t}\sim\pi_{\Theta}(\cdot\mid o_{\leq t}),\qquad\hat{A}^{\mathrm{exec}}_{t}=\hat{A}_{t}[0:H_{\mathrm{exec}}], $$ |  | (4) |
|---|---|---|---|

|  | $$ o_{t+1}\leftarrow\mathrm{EnvStep}(\hat{A}^{\mathrm{exec}}_{t}),\qquad z_{t+1}=E_{\theta}(o_{t+1}). $$ |  | (5) |
|---|---|---|---|

The resulting observation, rather than the generated action itself, supplies the next world token. Executed actions are not inserted into the sequence. Instead, their effects reach the context through subsequent images and proprioception. We do not assume that observations losslessly recover action history; adding executed actions explicitly is a meaningful extension, but not part of the default model.

Only action generation is supervised in WorldToken. The time-first sequence could, in principle, also serve as the conditioning structure for generative modeling of future observations, for example through $p_{\omega}(o_{t+1}\mid o_{\leq t},A_{t})$. This paper studies its action-only imitation-learning realization (Section 7).

### 2.2 Within-timestep multimodal encoding

A single robot observation is high bandwidth: multiview images carry spatial information, proprioception specifies robot configuration, and the task condition specifies the behavioral objective. The within-timestep encoder compresses these heterogeneous inputs into a fixed-width temporal representation $z_{t}$.

For each camera, a visual stem maps $I_{t}^{i}$ to visual patch tokens. Although different benchmarks may use different visual front ends, all visual features are projected to the model width and augmented with embeddings that identify spatial position, camera view, and modality before multimodal fusion. If camera $i$ produces $M_{i}$ tokens,

|  | $$ V_{t}^{i}=\bigl(v_{t,1}^{i},\ldots,v_{t,M_{i}}^{i}\bigr). $$ |  | (6) |
|---|---|---|---|

Proprioception and task conditioning are projected into $u_{t}^{\mathrm{prop}}$ and $u_{t}^{\mathrm{task}}$. The within-timestep sequence is

|  | $$ X_{t}=\bigl[V_{t}^{1};\ldots;V_{t}^{K};u_{t}^{\mathrm{prop}};u_{t}^{\mathrm{task}}\bigr]. $$ |  | (7) |
|---|---|---|---|

We append learned readout tokens $Q=(q^{1},\ldots,q^{R})$, with $R=4$ in every reported model. As shown in Figure 2(a), an observation-token query may attend to every observation-token key but not to a readout-token key; a readout-token query may attend to all observation and readout keys. Thus, the learned readout tokens aggregate the evolving multimodal observation without influencing the observation-token representations.

Let $r_{t}^{1},\ldots,r_{t}^{R}$ denote the final outputs at the readout positions:

|  | $$ (r_{t}^{1},\ldots,r_{t}^{R})=F_{\mathrm{enc}}([X_{t};Q])_{\mathrm{readout}}. $$ |  | (8) |
|---|---|---|---|

Concatenation, RMSNorm, and a linear projection produce the only token passed to the temporal backbone:

|  | $$ z_{t}=W_{z}\,\mathrm{RMSNorm}([r_{t}^{1};\ldots;r_{t}^{R}]). $$ |  | (9) |
|---|---|---|---|

### 2.3 Temporal backbone with causal self-attention

We define the world-token trajectory as

|  | $$ Z_{1:T}=(z_{1},z_{2},\ldots,z_{T}). $$ |  | (10) |
|---|---|---|---|

We instantiate the causal temporal backbone $T_{\phi}$ as a Transformer based on the Qwen2 decoder architecture (53), initialized from scratch, to map the world-token trajectory to history-conditioned states:

|  | $$ (h_{1},\ldots,h_{T})=T_{\phi}(z_{1},\ldots,z_{T}),\qquad h_{t}=T_{\phi}(z_{\leq t})_{t}. $$ |  | (11) |
|---|---|---|---|

One-dimensional RoPE (47) is applied along the world-token sequence to encode policy time. Causal self-attention ensures that $h_{t}$ depends only on $z_{\leq t}$. During training, sampled windows or episode prefixes are processed in parallel with the same causal mask used to prevent access to future observations.

The action decoder reads only the last valid representation $h_{t}$. Earlier observations must first affect $h_{t}$ through the causal backbone before they can influence the current action distribution. We treat an episode as a continuous session, much as a language model accumulates a conversation prefix within one session: each policy timestep contributes a new world token, and existing tokens form the context for subsequent decisions.

Let $C$ be the maximum visible-history length. At policy timestep $t$ the policy reads

|  | $$ z_{\max(1,t-C+1):t}. $$ |  | (12) |
|---|---|---|---|

If $C$ covers the episode so far, the policy receives full-episode context; once the episode exceeds $C$, the earliest tokens are evicted.

When the context window slides, the retained world tokens are reindexed contiguously before recomputation, with no absolute position embeddings added. Because standard RoPE attention depends on relative position differences, subtracting a common offset from all retained indices leaves their pairwise temporal geometry unchanged. Context-length ablations therefore change only which history tokens are visible.

### 2.4 Generative action decoding

At policy timestep $t$, WorldToken generates an $H$-step action chunk

|  | $$ A_{t}=(a_{t,0},a_{t,1},\ldots,a_{t,H-1})\in\mathbb{R}^{H\times d_{a}}. $$ |  | (13) |
|---|---|---|---|

Each action dimension is normalized to $[-1,1]$ using statistics from the training demonstrations and unnormalized before execution.

We model the conditional action distribution with diffusion. For a normalized ground-truth chunk $A_{t}$, training samples diffusion step $k$ and Gaussian noise $\epsilon$:

|  | $$ A_{t}^{(k)}=\sqrt{\bar{\alpha}_{k}}\,A_{t}+\sqrt{1-\bar{\alpha}_{k}}\,\epsilon,\qquad\epsilon\sim\mathcal{N}(0,I). $$ |  | (14) |
|---|---|---|---|

A DiT denoiser (40) predicts the injected noise conditioned on the current history representation $h_{t}$:

|  | $$ \hat{\epsilon}=\epsilon_{\psi}\bigl(A_{t}^{(k)},k,h_{t}\bigr),\qquad L_{\mathrm{act}}=\mathbb{E}_{t,k,\epsilon}\left[\left\|\epsilon-\epsilon_{\psi}\bigl(A_{t}^{(k)},k,h_{t}\bigr)\right\|_{2}^{2}\right]. $$ |  | (15) |
|---|---|---|---|---|---|

The diffusion timestep and $h_{t}$ condition each DiT block through adaLN-Zero modulation. At inference, action chunks are generated from Gaussian noise using stochastic DDPM sampling (21).

Control follows a receding-horizon scheme. Each query generates $H$ actions but executes only the first $H_{\mathrm{exec}}$:

|  | $$ \hat{A}_{t}\sim\mathcal{D}_{\psi}(\cdot\mid h_{t}),\qquad\hat{A}^{\mathrm{exec}}_{t}=(\hat{a}_{t,0},\ldots,\hat{a}_{t,H_{\mathrm{exec}}-1}). $$ |  | (16) |
|---|---|---|---|

After executing this prefix, the policy observes the environment again, appends the next world token, and replans.

## 3 Can WorldToken achieve effective multitask control and scale systematically?

The first empirical question is whether a concrete WorldToken instantiation can learn effective multitask control and respond systematically to additional data and model capacity. At the largest dataset size, with 2,900 target-domain demonstrations per task, the 85.3M-parameter policy reaches a mean closed-loop success rate of 59.45% across 23 household-manipulation tasks, averaged over training seeds; its best individual evaluation run reaches 60.1%. We examine the consistency and source of these gains through a complete $5\times 5\times 2$ sweep over dataset size, model size, and training seed.

### 3.1 Evaluation design

We evaluate five training-set sizes,

|  | $$ D\in\{50,100,300,1000,2900\}, $$ |  |
|---|---|---|

and five model sizes from 44.3M to 1.49B parameters, with two training seeds per condition, yielding 50 trained policies. All conditions use the same data-construction procedure and the scheduled final checkpoint, without rollout-based checkpoint selection. Each policy is evaluated three times under the official seen/unseen-scene protocol, with 50 episodes for each of 23 tasks per run, for a total of 172,500 closed-loop episodes.

We report mean closed-loop success rate (SR) together with holdout stochastic RMSE, an offline measure of how closely the policy reproduces expert actions when given held-out expert observation histories. Lower RMSE indicates closer action matching on expert trajectories, whereas closed-loop SR evaluates behavior under the policy’s own rollouts. Full protocol details are provided in Appendix A.

### 3.2 Effective multitask control from target-domain data

With 300 generated demonstrations per task, the 85.3M-parameter WorldToken policy reaches a mean SR of 46.83% across training seeds, compared with 31.28% for our official-code BC-Transformer reproduction under the same task set, data scale, and evaluation registry.

With 2,900 demonstrations per task, mean SR rises to 59.45% for the 85.3M-parameter policy and 60.26% for the 218.8M-parameter policy, with the best individual run reaching 62.0% (Appendix Table 12). Apart from the frozen CLIP text encoder used for task conditioning, the policy is trained from scratch without robot-policy, robot-trajectory, or policy-video pretraining. For scale reference, published RoboCasa systems with large-scale pretraining report approximately 50–79% SR under differing data, task, initialization, checkpoint-selection, and aggregation protocols (Appendix Table 14).

### 3.3 Scaling with target-domain data

Additional target-domain data improves WorldToken consistently across the complete scaling grid. Across two training seeds, five model sizes, and four adjacent increases in dataset size, holdout RMSE decreases in all 40 paired comparisons. Closed-loop SR also improves in all 40 comparisons. Increasing $D$ from 50 to 2,900 reduces RMSE by 47.0–56.8% across model sizes and increases closed-loop SR by 32.7–39.3 percentage points.

![Refer to caption](drafts/images/worldtoken-2608.22591/fig3.png)

*Figure 3: Complete $5\times 5$ RoboCasa sweep over dataset size and model size. Top: mean closed-loop SR over three complete evaluation runs for each training seed. Bottom: holdout stochastic RMSE for the corresponding policies. Every cell displays its numerical value.*

The reduction in RMSE is also highly regular across dataset sizes. After averaging the two training seeds at each model size, the five dataset-size points are well described by the descriptive power law

|  | $$ \mathrm{RMSE}_{P}(D)\approx a_{P}D^{-\alpha_{P}}, $$ |  | (17) |
|---|---|---|---|

where $P$ indexes model size. The fitted exponents $\alpha_{P}$ range from 0.151 to 0.211, with $R^{2}$ between 0.990 and 0.999 (Figure 4, left). At each dataset size, policies are trained for approximately 100 passes through the training set, and holdout RMSE approaches a plateau before the end of training across the scaling grid. We therefore interpret these results primarily as data scaling under sufficiently trained conditions, while noting that larger datasets naturally involve more optimization steps.

Closed-loop SR gains become smaller at the largest dataset sizes. Across the ten matched comparisons between $D=1000$ and $D=2900$, covering all training seeds and model sizes, RMSE decreases by 17.10% on average, with a sample standard deviation of 1.76 percentage points. In contrast, closed-loop SR improves by only 3.39 percentage points on average, with a sample standard deviation of 2.12 percentage points. Thus, additional data continues to reduce action-prediction error on held-out demonstrations after closed-loop performance has begun to approach a plateau.

*Figure 4: Scaling of holdout stochastic RMSE on logarithmic axes. Left: data scaling with individual-seed results, two-seed means, and descriptive power-law fits. Right: model-size slices at each dataset size.*

### 3.4 Scaling with model capacity

Increasing model size from 44.3M to 218.8M parameters reduces holdout RMSE at every dataset size and for both training seeds. Increasing model size beyond 218.8M parameters does not yield a consistent further reduction: neither the 648.9M- nor the 1.49B-parameter policy consistently outperforms the 218.8M-parameter policy. The reliable RMSE benefit of increased capacity is therefore concentrated between 44.3M and 218.8M parameters.

Closed-loop SR shows a weaker capacity trend. SR generally improves as capacity increases, but the higher-capacity policies do not exhibit a stable ordering.

### 3.5 Interaction between data and model capacity

As shown in Figure 4, right, the RMSE benefit of capacity changes systematically with dataset size. The reduction from 44.3M to 218.8M parameters grows from roughly 5% at small dataset sizes to roughly 23% at large dataset sizes. The 85.3M- and 218.8M-parameter policies are close at small dataset sizes but separate as the dataset grows, while the 648.9M-parameter policy progressively approaches the 218.8M-parameter policy. At $D=2900$, the 218.8M- and 648.9M-parameter policies have nearly identical RMSE. As more data become available, the reliable benefit of capacity is therefore increasingly concentrated between 44.3M and 218.8M parameters.

Closed-loop SR does not show the same interaction as clearly. At the largest dataset size, policies with 85.3M parameters and above achieve similar SR, leaving little consistent separation on the higher-capacity side.

The exponent describing the relationship between RMSE and dataset size increases from 0.151 for the 44.3M-parameter policy to approximately 0.21 for the 218.8M- and 648.9M-parameter policies, consistent with their steeper RMSE improvements as data increase. Within the tested range, this pattern resembles the coordinated scaling of data and model capacity observed in language modeling (24; 22). Section 6 examines the relationship between holdout RMSE and closed-loop SR directly.

## 4 Does WorldToken use recent history?

The second empirical question is whether the decisions of this WorldToken instantiation depend on the recent history it receives. We study this question through two complementary interventions on RoboCasa. First, we truncate the visible history of every policy in the main scaling sweep while keeping its weights and evaluation conditions fixed. Second, we train policies with different context lengths to measure how they adapt to the history available during training. All 50 policies trained with $C_{\mathrm{train}}=10$ lose SR when evaluated with one or two visible policy timesteps, whereas short-context policies recover most of this loss. WorldToken therefore uses recent history consistently across dataset sizes and model sizes, and its use of that history adapts to the training context.

### 4.1 Evaluation design

All policies in the main $5\times 5\times 2$ sweep are trained with $C_{\mathrm{train}}=10$. For every dataset size, model size, and training seed, we hold the checkpoint, 1,150 initial conditions, environment seeds, rollout seed, and execution protocol fixed while setting

|  | $$ C_{\mathrm{test}}\in\{1,2,5,10\}. $$ |  |
|---|---|---|

We call this intervention same-checkpoint history truncation. We evaluate each policy once with visible-history lengths of one, two, and five policy timesteps. The $C_{\mathrm{test}}=10$ result is the mean of the three complete evaluation runs reported in Section 3. The episode identities and protocol are matched across context lengths, although stochastic diffusion and simulation do not produce bitwise identical executions.

We then train the 218.8M-parameter policy with $D=300$ and matched training and evaluation context lengths,

|  | $$ C_{\mathrm{train}}=C_{\mathrm{test}}\in\{1,2,5,10\}. $$ |  |
|---|---|---|

All conditions use 30,000 optimizer steps and the same total number of supervised action-chunk targets. We use two training seeds for each context length, and every final checkpoint undergoes three complete evaluation runs. Together, the two interventions measure how fixed policies use their learned history and how policies adapt when the available training history changes.

### 4.2 Fixed policies consistently use recent history

![Refer to caption](drafts/images/worldtoken-2608.22591/fig5.png)

*Figure 5: History truncation of the same trained policy. Rows are training seeds; columns show the $5\times 5$ SR difference for one, two, and five visible policy timesteps relative to $C_{\mathrm{test}}=10$.*

Reducing the visible history to one or two policy timesteps lowers SR for all 50 policies in the scaling sweep. The smallest declines are 3.5 percentage points with one visible policy timestep and 1.4 percentage points with two. This pattern shows that recent observations contribute to the decisions of all trained policies rather than only under a particular experimental condition or evaluation run.

Most of the observed benefit is already present at five visible policy timesteps. At this context length, 47 of the 50 policies are within two percentage points of their $C_{\mathrm{test}}=10$ result. Figure 5 reports the change for every policy. The full sweep therefore shows consistent recent-history use, with about five visible policy timesteps capturing most of its closed-loop benefit on RoboCasa.

### 4.3 Short-context policies adapt to their training context

*Table 1: Holdout stochastic RMSE and closed-loop SR when training and evaluation context lengths match. SR dispersion is the sample standard deviation of three fixed-seed evaluation repeats.*

| Context $C$ | RMSE, seed 0 | RMSE, seed 1 | SR, seed 0 | SR, seed 1 |
|---|---|---|---|---|
| 1 | 0.150260 | 0.147767 | 46.75% $\pm$ 1.02 pp | 47.68% $\pm$ 1.50 pp |
| 2 | 0.149383 | 0.147528 | 46.87% $\pm$ 0.84 pp | 48.52% $\pm$ 1.36 pp |
| 5 | 0.137165 | 0.139603 | 51.48% $\pm$ 0.26 pp | 50.00% $\pm$ 0.92 pp |
| 10 | 0.131710 | 0.133098 | 48.12% $\pm$ 1.70 pp | 48.38% $\pm$ 1.09 pp |

*Figure 6: Holdout RMSE during training and final closed-loop performance under different context lengths. Left: holdout stochastic RMSE across optimizer steps for two training seeds. Right: final closed-loop SR; bars show the sample standard deviation over three evaluation repeats.*

Table 1 and Figure 6 show that policies trained with short contexts recover most of the SR lost through history truncation. Evaluating the $D=300$ policies trained with $C_{\mathrm{train}}=10$ using $C_{\mathrm{test}}=1$ yields SRs of 27.91% and 27.83% for the two training seeds. Policies trained and evaluated with $C=1$ reach 46.75% and 47.68% under the same evaluation protocol. Policies trained and evaluated with $C=2$ show the same adaptation, approaching the performance of those trained and evaluated with $C=10$.

The two metrics respond differently to context length. Holdout RMSE decreases as context increases from one to ten visible policy timesteps for both training seeds, showing that longer demonstration histories reduce action-prediction error on held-out trajectories. Closed-loop SR is non-monotonic: policies trained and evaluated with $C=5$ perform best for both seeds, while those trained and evaluated with $C\in\{1,2\}$ remain close to those with $C=10$.

### 4.4 History use reflects adaptation to the training context

The two interventions provide a consistent account of recent-history use. A policy trained with $C_{\mathrm{train}}=10$ incorporates temporal cues during training and loses performance when those cues are removed at evaluation. When the policy is trained from the start with a shorter context, it adapts its decisions to the available history and recovers much of the closed-loop performance. The comparison between five and ten visible policy timesteps further shows that longer history can reduce action-prediction error on held-out trajectories without producing a corresponding increase in task success. Across the RoboCasa scaling sweep, WorldToken systematically uses its temporal input; at the $D=300$, 218.8M-parameter anchor, $C=5$ yields the highest closed-loop SR under the tested training recipe.

## 5 Does extended context improve sustained ordered behavior?

Section 4 showed that roughly five visible policy timesteps capture most of
the observed history benefit on RoboCasa. We therefore turn to RMBench Blocks
Ranking, where ordered behavior spans many more policy timesteps and the same
block arrangement can occur at different positions in a demonstrated swap
sequence. The third empirical question is whether extended visible context
improves sustained ordered behavior over these longer horizons.

With the checkpoint and 100 initial conditions fixed, increasing visible
history from 32 to 608 world tokens raises evaluator success from 28% to
95% and sharply improves multi-swap execution under the behavioral audit
defined below. An exploratory stress test further shows that the learned
periodic sequence can continue long after the context window begins to slide.

### 5.1 Evaluation design

Blocks Ranking is a single RMBench task in which three colored blocks occupy
left, middle, and right slots $(L,M,R)$ and must be rearranged into the target
order $(1,2,3)$. Official demonstrations press the button once, then follow the
fixed reference swap sequence

|  | $$ \underbrace{\mathrm{swap}(M,R)\rightarrow\mathrm{swap}(L,R)\rightarrow\mathrm{swap}(L,M)}_{\text{one period}}\rightarrow\mathrm{swap}(M,R)\rightarrow\mathrm{swap}(L,R), $$ |  | (18) |
|---|---|---|---|

pressing the button after each swap and stopping at the target. The five initial arrangements require one to five reference swaps.

The next reference swap is not determined by the current block arrangement
alone. For example, $(1,3,2)$ is solved by $\mathrm{swap}(M,R)$ when it is the
initial state, but the same arrangement occurs after two swaps in a five-swap
episode, where the reference sequence continues with $\mathrm{swap}(L,M)$.
Recent history can therefore disambiguate progress through the demonstrated
sequence. The official evaluator, however, checks only the final block
positions, an open right gripper, and the button press, not adherence to the
reference swap sequence.

We start from a 54.3M-parameter checkpoint trained for 5,000 steps on nine
RMBench tasks (50 demonstrations per task, $C=288$) and continue training for
500 steps on 45 Blocks Ranking demonstrations with $C=608$.
Appendix D gives the full continuation recipe,
task-specific loss, and ablation.

We evaluate this fixed checkpoint on the official seed-0 registry of 100
episodes with a 210-second horizon, varying only the maximum visible-history
length:

|  | $$ C\in\{608,288,128,64,32\}. $$ |  | (19) |
|---|---|---|---|

At 0.24 seconds per world token, these settings span 7.68–145.92 seconds of
visible history. We additionally report strict behavioral success,
which requires evaluator success, adherence to the reference swap sequence,
and every block fully inside its intended slot. Evaluator successes that
violate either behavioral criterion are counted as evaluator-only
successes.

### 5.2 Extended context improves sustained multi-swap execution

Evaluator success decreases monotonically as visible history is shortened:
95%, 92%, 59%, 38%, and 28% for
$C=608,288,128,64,32$, respectively.
One correctly executed reference swap followed by a button press spans about
116 world tokens, or 28 seconds, so the tested contexts range from less than
one such cycle to roughly five.

The effect of context length is concentrated in episodes that require sustained
multi-swap execution. Table 2 stratifies the same 100
episodes by the number of swaps $S$ in the reference solution and reports
evaluator success, strict behavioral success, and evaluator-only success.

*Table 2: Blocks Ranking outcomes stratified by the number of swaps $S$ in the
reference solution. Each entry gives evaluator successes / strict behavioral
successes / evaluator-only successes.*

| $S$ (episodes) | $C=608$ | $C=288$ | $C=128$ | $C=64$ | $C=32$ |
|---|---|---|---|---|---|
| 1 (24) | 24/24/0 | 24/24/0 | 24/24/0 | 24/24/0 | 24/24/0 |
| 2 (14) | 14/14/0 | 14/14/0 | 14/14/0 | 3/3/0 | 3/0/3 |
| 3 (28) | 25/25/0 | 23/23/0 | 13/3/10 | 2/0/2 | 0/0/0 |
| 4 (18) | 17/16/1 | 16/16/0 | 5/5/0 | 9/0/9 | 1/0/1 |
| 5 (16) | 15/15/0 | 15/15/0 | 3/3/0 | 0/0/0 | 0/0/0 |
| Total (100) | 95/94/1 | 92/92/0 | 59/49/10 | 38/27/11 | 28/24/4 |

All 24 one-swap episodes are strict behavioral successes at every context
length. The separation appears as the required sequence becomes longer:
$C\in\{608,288\}$ remains reliable through five swaps, whereas
$C\in\{64,32\}$ degrades sharply beyond the first swap.
The intermediate $C=128$ condition can occasionally complete all five swaps,
but does so much less reliably.

![Refer to caption](drafts/images/worldtoken-2608.22591/fig8.png)

*Figure 7: One episode under two visible-history lengths. The checkpoint,
initial state, and evaluation protocol are fixed; only the maximum
visible-history length changes. With $C=608$, the policy completes the
five-swap reference sequence and succeeds at 142.6 seconds. With $C=64$, it
completes the first swap but fails to sustain ordered progress to the target
within the 210-second horizon. Frame timestamps are simulated seconds.*

Behavioral review localizes this degradation primarily to execution stability
rather than sequence-position errors. With short visible histories, the policy
typically retains the ability to complete the first local manipulation, but
later swaps increasingly produce inaccurate placements or stalls. Selecting a
swap associated with the wrong point in the reference sequence occurs in only
a small subset of trajectories and is not the dominant failure mode.

The gap between evaluator and strict behavioral success makes these execution
failures explicit. All 25 evaluator-only successes at $C\leq 128$ leave at
least one block clearly displaced from its intended slot, and 12 also depart
from the reference swap order. At $C=64$, for example, nine of the eighteen
four-swap episodes pass the evaluator, yet none satisfies the strict behavioral
criterion. Extended context therefore contributes primarily to the reliability
of sustained execution across successive swaps, rather than merely preserving
the ability to perform an individual local manipulation.

### 5.3 Periodic behavior can outlast both demonstrations and context

We conduct an exploratory stress test by disabling termination at first success
and allowing each trajectory to continue without resetting the scene or
accumulated history. The environment otherwise evolves normally, and the fixed
$C=608$ context window begins to slide after 145.92 seconds. The nine
environment seeds come from the official evaluation suite and cover all five
non-target initial arrangements. A trajectory ends at its first deviation from
the reference swap sequence or after 1,500 consecutive low-level control steps
without a completed swap.

This intervention removes an important property of the demonstrations. Official
trajectories stop as soon as the target arrangement is first reached, and
therefore contain at most five swaps. The underlying reference behavior,
however, is generated by the repeating three-swap period

|  | $$ \mathrm{swap}(M,R)\rightarrow\mathrm{swap}(L,R)\rightarrow\mathrm{swap}(L,M), $$ |  | (20) |
|---|---|---|---|

of which each demonstration contains only a terminating prefix. Continuing
after first success therefore tests whether the policy can sustain this
periodic rule beyond the demonstrated episode horizon.

*Table 3: Nine exploratory Blocks Ranking stress trajectories continued beyond
first success. $S$ is the number of swaps required to reach the target from the
initial arrangement. “Swaps” counts correctly ordered reference-sequence
swaps; “after success” and “after sliding” count those completed after first
success and after the $C=608$ window begins to slide at 145.92 seconds,
respectively.*

| Env. seed | $S$ | Swaps | After success | After sliding | Last correct | End | Cause |
|---|---|---|---|---|---|---|---|
| 100000 | 2 | 13 | 11 | 8 | 350.52 s | 440.52 s | stall |
| 100001 | 3 | 5 | 2 | 0 | 139.20 s | 229.20 s | stall |
| 100002 | 4 | 17 | 13 | 12 | 470.28 s | 560.28 s | stall |
| 100003 | 2 | 9 | 7 | 4 | 250.14 s | 269.76 s | order deviation |
| 100004 | 3 | 8 | 5 | 3 | 222.12 s | 312.12 s | stall |
| 100007 | 1 | 5 | 4 | 0 | 139.08 s | 229.08 s | stall |
| 100008 | 1 | 31 | 30 | 26 | 856.44 s | 875.88 s | order deviation |
| 100015 | 5 | 5 | 0 | 0 | 139.38 s | 229.38 s | stall |
| 100016 | 1 | 5 | 4 | 0 | 139.32 s | 229.32 s | stall |

Five of the nine trajectories continue the reference sequence after the
$C=608$ context window begins to slide. The longest completes 31 correctly
ordered swaps—ten complete repetitions of the three-swap period plus one
additional swap—with its final correct swap at 856.44 seconds, more than
five times the 145.92-second visible-context span. Thus, despite training
demonstrations containing at most five swaps, the policy sustains the learned
periodic regularity far beyond both the demonstrated horizon and a single
context window.

### 5.4 How extended context supports sustained ordered behavior

The truncation and stress experiments reveal two complementary properties of extended temporal context. First, longer visible histories improve the reliability of repeated manipulation. Contexts shorter than one swap-and-press cycle degrade sharply after the first swap, whereas $C\in\{288,608\}$ supports reliable execution through the full five-swap sequence. The intermediate $C=128$ condition can occasionally complete all five swaps but does so much less reliably.

Second, the stress test places the policy on trajectory prefixes that do not occur in the training demonstrations. Every demonstration terminates after at most five swaps, whereas the stress trajectories continue the same periodic rule beyond those demonstrated prefixes. Nevertheless, five trajectories continue after the $C=608$ context window begins to slide, and the longest maintains the reference order for 31 swaps. This suggests that the policy has learned a repeatable three-swap regularity rather than only reproducing the finite prefixes observed during training.

## 6 Offline action fitting and closed-loop success

Holdout stochastic action root-mean-square error (RMSE) and closed-loop success rate (SR) measure different aspects of policy performance. RMSE evaluates how closely a policy reproduces expert actions on held-out expert trajectories. SR instead evaluates whether the policy can complete the task when its own actions determine the observations and states that follow. The two measures are related, but they are not interchangeable.

Across substantial changes in data scale and model capacity, the two measures generally improve together. Policies that fit held-out expert actions better also tend to achieve higher closed-loop success. This makes holdout RMSE a useful dense indicator of broad improvements during behavior-cloning development. The complete sweep-level analysis is reported in Appendix F.

The agreement becomes less reliable when comparing nearby policies. In the context-length study, for example, using a longer training context improves action fitting on held-out expert trajectories but does not improve closed-loop success. A controlled comparison of where temporal history is processed shows the same qualitative pattern (Appendix E.1). Thus, improved imitation of expert actions does not necessarily translate into better closed-loop performance.

We attribute the gap between RMSE and SR to two main factors. The first is a difference in the states and action targets being evaluated. Offline RMSE is measured on observation histories generated by the expert, whereas a closed-loop policy must act on states produced by its own earlier decisions. Once the policy deviates from an expert trajectory, it may encounter states for which the demonstrations provide little guidance and where recovery becomes important. Moreover, the recorded expert action is not necessarily the only action that can lead to successful task completion. Better agreement with recorded expert actions therefore need not translate directly into better behavior on the states encountered during rollout.

The second source of discrepancy is that action-level accuracy does not directly translate into task-level success. In closed-loop execution, predicted actions interact with the controller and environment dynamics, and the resulting trajectory is evaluated by a task-specific success criterion. Consequently, small execution errors may have little impact on offline action similarity while still causing a rollout to fail. Blocks Ranking illustrates this clearly: a policy trained with the standard imitation objective typically produces the correct block ordering, but may fail the official evaluation because the final button press is not deep enough (Appendix D). Thus, a trajectory can be behaviorally close to the expert demonstration yet receive a different binary success label.

We therefore treat RMSE and SR as complementary measures of policy capability. RMSE provides a dense measure of action fitting on expert trajectories, while SR measures task completion under the policy’s own behavior and the benchmark evaluation procedure. Agreement between them provides stronger evidence of improvement than either measure alone. When they disagree, examining whether the gap arises from unfamiliar rollout states, recovery behavior, execution, or the success criterion is more informative than using one metric as a substitute for the other. Appendix F provides the quantitative analysis, formalization, and repeatability results.

## 7 Discussion: Time-First Modeling as an Interface

We view time-first organization as a stable interface for analyzing embodied sequence models. WorldToken separates within-timestep perceptual processing, cross-timestep temporal computation, and action generation, while keeping the policy timestep as the semantic unit of the temporal sequence. This separation is useful because architectural changes can be made at one stage without simultaneously changing what a position in the history represents. For example, one can vary how aggressively observations are compressed, how temporal information is integrated, or how actions are decoded while preserving the same policy-step-aligned context. The benefit is therefore not causal attribution by construction, but a cleaner basis for controlled attribution: when data, compute, optimization, and other confounders are appropriately controlled, behavioral differences can be associated more directly with the stage being modified.

This interface also motivates aggressive compression of past observations. A robot does not necessarily need to preserve every perceptual detail from every timestep, because perception is repeatedly renewed through interaction with the environment. Unlike many multimodal language-model settings, where an image may be presented only once and its details must therefore remain accessible for later reasoning, an embodied agent can affect what it observes next through its actions and viewpoint. This creates the possibility of recovering task-relevant visual detail when it becomes useful, rather than carrying a high-bandwidth representation of every past observation throughout the entire history. The present WorldToken policy does not perform active sensing, but this perspective suggests that temporal compression and active perception can be designed jointly: a compact temporal representation may be sufficient when the agent can subsequently obtain the perceptual information needed for decisions.

Separately, we view scaling behavior as an important criterion for evaluating sequence architectures. Performance in a limited data regime or at a single operating point may not reflect how an architectural choice behaves as data, model capacity, context length, or task horizon increase. Architectural comparisons are therefore more informative when they examine whether observed differences persist or change with scale.

## 8 Related work

### 8.1 Sequence organization in robot policies

Recasting decision making as sequence modeling has become an important route in robot learning and generalist agents. Decision Transformer organizes returns, states, and actions as a causal sequence, while Gato serializes data across tasks, modalities, and embodiments into a common token stream and uses one Transformer to produce text, actions, and other outputs from context (8; 42). These works establish that decision processes admit sequence formulations, but they leave open a more basic structural choice: when robot experience is written as a sequence, what should constitute the basic unit of temporal context modeling?

Robot Transformers answer this question differently. RT-1 compresses each visual frame into multiple visual tokens with TokenLearner, so one observation timestep occupies several high-level positions; ICRT aggregates vision and proprioception into a state token and interleaves it with action features in a causal sensorimotor sequence; GR-1 lets language, image sequences, robot state, and future-image prediction share one GPT-style sequence (7; 12; 51). Physical time is present in all three, but top-level positions can differ in modality, provenance, or generation role.

A companion design choice is how history modeling and action generation divide labor. Diffusion Policy represents continuous, multimodal action-sequence distributions with conditional diffusion; ACT generates chunked action sequences with a conditional VAE decoder; Octo summarizes context with learned readout tokens and attaches a lightweight diffusion head; and $\pi_{0}$ adds a separately parameterized flow-matching action expert (10; 54; 13; 6). Modularizing expressive action generation away from the backbone is therefore well precedented; the distinguishing question is in what form temporal context is organized and accessed before action generation.

Compact timestep representations also have direct precedents. The BC-Transformer baseline in RoboCasa encodes each observation timestep, applies non-causal self-attention within a fixed ten-observation window, and lets every window position emit an action prediction through a Gaussian mixture head (37; 2). HAMLET introduces moment tokens that compress each timestep of a pretrained VLA and aggregates them with a lightweight memory module, and Chronos represents each control step with one state-representative token propagated through a selective state-space model (31; 56).

Against this background, WorldToken adopts a time-first organization in which each policy timestep contributes one observation-derived token to a causal temporal sequence. We study the resulting policy family through data and model-size scaling and controlled temporal-context interventions.

### 8.2 Temporal context, memory, and partial observability

History-conditioned control predates modern Transformers: recurrent imitation policies such as BC-RNN maintain a hidden state across timesteps, and Hiveformer jointly models language, multiview observations, and full observation/action history for multitask manipulation (36; 14).

It is useful to distinguish history availability, history necessity, and history utilization. Availability asks whether the architecture can receive past observations. Necessity asks whether the task is partially observable without them. Utilization asks whether a trained policy’s behavior actually depends on history. A large input window establishes availability but proves neither necessity nor utilization. Truncating or perturbing observation history with fixed policy weights primarily tests utilization (9; 46; 56).

By how history is carried, existing designs fall into three broad classes. Recurrent state compression folds the past into a continually updated hidden state, as in BC-RNN and Chronos’s selective state-space model. Explicit sequence context keeps past interactions as a directly attendable sequence: Hiveformer is an early full-history Transformer; HALO targets spurious historical correlations with VQA-supervised relevance and sparse top-$K$ attention; PRISM extends visuomotor memory to minute scale with gated attention and hierarchical compression; and controlled Diffusion Policy studies show that context-length gains depend on conditioning, denoising architecture, variable-length training, and data conditions (44; 45; 1). Dedicated memory subsystems specify write, update, or retrieval rules: RMBench proposes Mem-0 as an explicit-memory reference, MemoryVLA maintains a Perceptual-Cognitive Memory Bank, and RoboMME evaluates memory-augmented VLA integration across several dimensions (9; 46; 11). A shared lesson across these lines is that long context does not automatically equal effective memory.

WorldToken is closest to explicit sequence context: it adds no memory bank and no hand-designed retrieval or update rule, and it measures history utilization through controlled context interventions on trained checkpoints.

### 8.3 Latent representations and predictive world models

Compressing observations into latent state and predicting the environment there is a classic world-model strategy. PlaNet learns latent dynamics for planning, Dreamer learns through imagined trajectories, and JEPA-style methods predict abstract representations rather than reconstructing all inputs (17; 16; 18; 32; 3). V-JEPA 2-AC adds an action-conditioned latent world model for robot planning, while GR-1 and ACT-JEPA combine behavior learning with future image or latent prediction (4; 51; 48).

WorldToken shares the intuition of forming a compact representation before decision making, but it does not define its temporal representation through future-state prediction. A world token is a physical-time-aligned policy representation, not a predictive world-model state; attaching predictive objectives to these tokens is a compatible extension (Section 7).

### 8.4 Scaling in robot learning

Data and parameter scaling provide a separate empirical lens. RT-1 varies data size, model size, and diversity; Open X-Embodiment/RT-X studies cross-robot data aggregation; and OpenVLA combines vision–language pretraining with large robot datasets (7; 38; 29). Robot-data studies show that quantity, environment and object diversity, and support composition all matter (35). ScaleDP and broader imitation-learning studies further show that capacity behavior depends on architecture, tokenization, data, and optimization (58; 39).

Existing scaling work thus already indicates that scaling behavior is architecture dependent. We accordingly treat scaling as an architectural probe: a finite factorial sweep over dataset size and model capacity within one fixed, from-scratch time-first family, characterizing that family within the tested range.

## 9 Conclusion and Limitations

WorldToken provides a concrete empirical realization of time-first sequence modeling. The results in this paper therefore concern the complete WorldToken instantiation under the tested recipes, rather than time-first organization in isolation. Within that scope, the study establishes a workable policy family for systematically studying data scale, model capacity, and temporal context through a policy-step-aligned sequence interface.

The broader contribution of time-first organization is the interface it defines rather than an established performance advantage. Keeping the semantic unit of temporal context fixed makes sequence structure itself easier to intervene on and compare, while leaving perceptual compression, temporal computation, and action generation as separable design choices. Whether this organization yields measurable advantages over alternative ways of structuring embodied sequences remains an open comparative question.

The present evidence is limited to simulation and action-only imitation learning. The main scaling study remains within one WorldToken family, the extended-context evidence is concentrated in a single long-horizon case study, and the experiments do not isolate the effects of individual components or compare alternative sequence organizations under fully matched conditions. They also do not test real-robot uncertainty, explicit action history, predictive objectives, or measured system-level efficiency. These boundaries limit the empirical claims to the complete WorldToken instantiation studied here.

## AI use statement

Large language models were used throughout this work under author direction: for literature collection, experimental statistics, code development, and manuscript drafting, with additional assistance in experimental design. All core ideas, the experimental program, and the conclusions originate from the authors. The authors verified the paper’s numerical claims against the underlying configurations, logs, evaluation records, and analysis artifacts, reviewed all AI-assisted text, code, and figures, and take full responsibility for the paper’s content.

## Reproducibility statement

Sections 2–5 specify the model interfaces, experimental comparisons, and context interventions; complete training, rollout, and aggregation details are consolidated in Appendix A and the benchmark-specific appendices. Appendix G maps each result family to its frozen archive records and identifies unavailable cells. The planned public release includes training and evaluation code, materialized run configurations, frozen evaluation registries, figure reconstruction scripts, and the archived-build values used for those cells.

## References

- Agarwal et al. (2026)
A. Agarwal, A. Wei, T. Kargin, M. Zeng, C. Becker, A. K. Dayi, P. Parrilo, A. Ozdaglar, and R. Tedrake

Training and evaluating diffusion policies with long context lengths.

arXiv preprint arXiv:2606.16447.

External Links: [Link](https://arxiv.org/abs/2606.16447)

Cited by: §8.2.
- ARISE Initiative (2024)
ARISE Initiative

RoboCasa BC-Transformer implementation in the robocasa branch of robomimic.

Note: Official software repository

External Links: [Link](https://github.com/ARISE-Initiative/robomimic/tree/robocasa)

Cited by: §8.1.
- Assran et al. (2023)
M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas

Self-supervised learning from images with a joint-embedding predictive architecture.

In IEEE/CVF Conference on Computer Vision and Pattern Recognition,

External Links: [Link](https://arxiv.org/abs/2301.08243)

Cited by: §8.3.
- Assran et al. (2025)
M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, et al.

V-JEPA 2: Self-supervised video models enable understanding, prediction and planning.

arXiv preprint arXiv:2506.09985.

External Links: [Link](https://arxiv.org/abs/2506.09985)

Cited by: §8.3.
- Bjorck et al. (2025)
J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al.

GR00T N1: An open foundation model for generalist humanoid robots.

arXiv preprint arXiv:2503.14734.

External Links: [Link](https://arxiv.org/abs/2503.14734)

Cited by: Table 14.
- Black et al. (2025)
K. Black, N. Brown, D. Driess, A. Esmail, M. R. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, L. Smith, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky

$\pi_{0}$: A Vision-Language-Action Flow Model for General Robot Control.

In Robotics: Science and Systems,

External Links: [Document](https://dx.doi.org/10.15607/RSS.2025.XXI.010),
[Link](https://arxiv.org/abs/2410.24164)

Cited by: Table 14,
§8.1.
- Brohan et al. (2023)
A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. S. Ryoo, G. Salazar, P. R. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. H. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich

RT-1: Robotics transformer for real-world control at scale.

In Robotics: Science and Systems,

External Links: [Document](https://dx.doi.org/10.15607/RSS.2023.XIX.025),
[Link](https://arxiv.org/abs/2212.06817)

Cited by: §8.1,
§8.4.
- Chen et al. (2021)
L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch

Decision transformer: Reinforcement learning via sequence modeling.

In Advances in Neural Information Processing Systems,

Vol. 34.

External Links: [Link](https://arxiv.org/abs/2106.01345)

Cited by: §8.1.
- Chen et al. (2026)
T. Chen, Y. Wang, M. Li, Y. Qin, H. Shi, Z. Li, Y. Hu, Y. Zhang, K. Wang, Y. Chen, et al.

RMBench: Memory-dependent robotic manipulation benchmark with insights into policy design.

arXiv preprint arXiv:2603.01229.

External Links: [Link](https://arxiv.org/abs/2603.01229)

Cited by: §8.2,
§8.2.
- Chi et al. (2023)
C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. C. M. Burchfiel, and S. Song

Diffusion policy: Visuomotor policy learning via action diffusion.

In Robotics: Science and Systems,

External Links: [Document](https://dx.doi.org/10.15607/RSS.2023.XIX.026),
[Link](https://arxiv.org/abs/2303.04137)

Cited by: §8.1.
- Dai et al. (2026)
Y. Dai, H. Fu, J. Lee, Y. Liu, H. Zhang, J. Yang, C. Finn, N. Fazeli, and J. Chai

RoboMME: Benchmarking and understanding memory for robotic generalist policies.

In International Conference on Machine Learning,

External Links: [Link](https://arxiv.org/abs/2603.04639)

Cited by: §8.2.
- Fu et al. (2024)
L. Fu, H. Huang, G. Datta, L. Y. Chen, W. C. Panitch, F. Liu, H. Li, and K. Goldberg

In-context imitation learning via next-token prediction.

arXiv preprint arXiv:2408.15980.

External Links: [Link](https://arxiv.org/abs/2408.15980)

Cited by: §8.1.
- Ghosh et al. (2024)
D. Ghosh, H. R. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, Y. L. Tan, L. Y. Chen, Q. Vuong, T. Xiao, P. R. Sanketi, D. Sadigh, C. Finn, and S. Levine

Octo: An open-source generalist robot policy.

In Robotics: Science and Systems,

External Links: [Document](https://dx.doi.org/10.15607/RSS.2024.XX.090),
[Link](https://arxiv.org/abs/2405.12213)

Cited by: §8.1.
- Guhur et al. (2023)
P. Guhur, S. Chen, R. G. Pinel, M. Tapaswi, I. Laptev, and C. Schmid

Instruction-driven history-aware policies for robotic manipulations.

In Conference on Robot Learning,

Proceedings of Machine Learning Research, Vol. 205, pp. 175–187.

Note: CoRL 2022 proceedings

External Links: [Link](https://arxiv.org/abs/2209.04899)

Cited by: §8.2.
- Guo et al. (2026)
J. Guo, Q. Li, P. Li, Z. Chen, N. Sun, Y. Su, H. Wang, Y. Zhang, X. Li, and H. Liu

Unified 4d world action modeling from video priors with asynchronous denoising.

arXiv preprint arXiv:2604.26694.

External Links: [Link](https://arxiv.org/abs/2604.26694)

Cited by: Table 14,
Table 14.
- Hafner et al. (2020)
D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi

Dream to control: Learning behaviors by latent imagination.

In International Conference on Learning Representations,

External Links: [Link](https://arxiv.org/abs/1912.01603)

Cited by: §8.3.
- Hafner et al. (2019)
D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson

Learning latent dynamics for planning from pixels.

In International Conference on Machine Learning,

Proceedings of Machine Learning Research, Vol. 97.

External Links: [Link](https://arxiv.org/abs/1811.04551)

Cited by: §8.3.
- Hafner et al. (2025)
D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap

Mastering diverse control tasks through world models.

Nature 640, pp. 647–653.

External Links: [Document](https://dx.doi.org/10.1038/s41586-025-08744-2),
[Link](https://arxiv.org/abs/2301.04104)

Cited by: §8.3.
- Han et al. (2024)
B. Han, J. Kim, and J. Jang

A dual process VLA: Efficient robotic manipulation leveraging VLM.

arXiv preprint arXiv:2410.15549.

External Links: [Link](https://arxiv.org/abs/2410.15549)

Cited by: Table 14.
- Han et al. (2026)
J. Han, S. Jeon, J. Jung, R. Zurbrügg, H. An, T. Portela, M. Hutter, M. Pollefeys, S. Kim, and S. Hong

Geometric action model for robot policy learning.

arXiv preprint arXiv:2606.17046.

External Links: [Link](https://arxiv.org/abs/2606.17046)

Cited by: Table 14.
- Ho et al. (2020)
J. Ho, A. N. Jain, and P. Abbeel

Denoising diffusion probabilistic models.

In Advances in Neural Information Processing Systems,

Vol. 33, pp. 6840–6851.

External Links: [Link](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)

Cited by: §2.4.
- Hoffmann et al. (2022)
J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, et al.

Training compute-optimal large language models.

arXiv preprint arXiv:2203.15556.

External Links: [Link](https://arxiv.org/abs/2203.15556)

Cited by: §3.5.
- Jang et al. (2025)
J. Jang, S. Ye, Z. Lin, J. Xiang, J. Bjorck, Y. Fang, F. Hu, S. Huang, K. Kundalia, Y. Lin, et al.

DreamGen: Unlocking generalization in robot learning through video world models.

arXiv preprint arXiv:2505.12705.

External Links: [Link](https://arxiv.org/abs/2505.12705)

Cited by: Table 14.
- Kaplan et al. (2020)
J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei

Scaling laws for neural language models.

arXiv preprint arXiv:2001.08361.

External Links: [Link](https://arxiv.org/abs/2001.08361)

Cited by: §3.5.
- Kim et al. (2026a)
D. Kim, H. Jang, M. Koo, S. Jang, T. Kim, B. Kim, B. Yoon, C. Jang, D. Choi, D. Han, et al.

RLDX-1 technical report.

arXiv preprint arXiv:2605.03269.

External Links: [Link](https://arxiv.org/abs/2605.03269)

Cited by: Table 14,
Table 14.
- Kim et al. (2026b)
J. Kim, J. Cho, S. Chu, A. Bal, J. Kim, G. Lee, S. Lee, S. H. Kim, B. Han, H. Lee, et al.

Pri4R: Learning world dynamics for vision-language-action models with privileged 4d representation.

arXiv preprint arXiv:2603.01549.

External Links: [Link](https://arxiv.org/abs/2603.01549)

Cited by: Table 14.
- Kim et al. (2026c)
M. J. Kim, Y. Gao, T. Lin, Y. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M. Liu, C. Finn, and J. Gu

Cosmos policy: Fine-tuning video models for visuomotor control and planning.

arXiv preprint arXiv:2601.16163.

External Links: [Link](https://arxiv.org/abs/2601.16163)

Cited by: Table 14.
- Kim and Gu (2026)
M. J. Kim and J. Gu

Cosmos policy: Fine-tuning video models for visuomotor control and planning.

Note: NVIDIA Cosmos CookbookOfficial software recipe and results page; not a separate peer-reviewed paper

External Links: [Link](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2/cosmos_policy/post_training.html)

Cited by: Table 14.
- Kim et al. (2025)
M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn

OpenVLA: An open-source vision-language-action model.

In Proceedings of the 8th Conference on Robot Learning,

Proceedings of Machine Learning Research, Vol. 270, pp. 2679–2713.

External Links: [Link](https://proceedings.mlr.press/v270/kim25c.html)

Cited by: §8.4.
- Kim et al. (2026d)
T. Kim, J. Lee, M. Koo, D. Kim, K. Lee, C. Kim, Y. Seo, and J. Shin

Contrastive representation regularization for vision-language-action models.

In International Conference on Machine Learning,

Note: Initial arXiv submission in 2025; current manuscript revision in 2026

External Links: [Link](https://arxiv.org/abs/2510.01711)

Cited by: Table 14,
Table 14,
Table 14,
Table 14.
- Koo et al. (2026)
M. Koo, D. Choi, T. Kim, K. Lee, C. Kim, Y. Seo, and J. Shin

HAMLET: Switch your vision-language-action model into a history-aware policy.

In International Conference on Learning Representations,

External Links: [Link](https://arxiv.org/abs/2510.00695)

Cited by: Table 14,
Table 14,
§8.1.
- LeCun (2022)
Y. LeCun

A path towards autonomous machine intelligence.

Note: OpenReview position paper

External Links: [Link](https://openreview.net/forum?id=BZ5a1r-kVsf)

Cited by: §8.3.
- Lee et al. (2026)
B. Lee, D. Hwang, D. Kim, H. Lee, M. Park, and J. Choo

See like a robot: Robot-centric pointmaps for vision-language-action models.

arXiv preprint arXiv:2607.11498.

External Links: [Link](https://arxiv.org/abs/2607.11498)

Cited by: Table 14.
- Liang et al. (2025)
J. Liang, P. Tokmakov, R. Liu, S. Sudhakar, P. Shah, R. Ambrus, and C. Vondrick

Video generators are robot policies.

arXiv preprint arXiv:2508.00795.

External Links: [Link](https://arxiv.org/abs/2508.00795)

Cited by: Table 14,
Table 14,
Table 14.
- Lin et al. (2025)
F. Lin, Y. Hu, P. Sheng, C. Wen, J. You, and Y. Gao

Data scaling laws in imitation learning for robotic manipulation.

In International Conference on Learning Representations,

External Links: [Link](https://arxiv.org/abs/2410.18647)

Cited by: §8.4.
- Mandlekar et al. (2022)
A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín

What matters in learning from offline human demonstrations for robot manipulation.

In Conference on Robot Learning,

Proceedings of Machine Learning Research, Vol. 164, pp. 1678–1690.

Note: Conference held in 2021; introduces the robomimic framework

External Links: [Link](https://arxiv.org/abs/2108.03298)

Cited by: §8.2.
- Nasiriany et al. (2024)
S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu

RoboCasa: Large-scale simulation of everyday tasks for generalist robots.

In Robotics: Science and Systems,

External Links: [Link](https://arxiv.org/abs/2406.02523)

Cited by: Table 14,
Table 14,
§F.1,
§8.1.
- Open X-Embodiment Collaboration et al. (2024)
Open X-Embodiment Collaboration et al.

Open x-embodiment: Robotic learning datasets and RT-X models.

In IEEE International Conference on Robotics and Automation,

External Links: [Link](https://arxiv.org/abs/2310.08864)

Cited by: §8.4.
- Pearce et al. (2025)
T. Pearce, T. Rashid, D. Bignell, R. Georgescu, S. Devlin, and K. Hofmann

Scaling laws for pre-training agents and world models.

In International Conference on Machine Learning,

Proceedings of Machine Learning Research, Vol. 267, pp. 48542–48562.

External Links: [Link](https://arxiv.org/abs/2411.04434)

Cited by: §8.4.
- Peebles and Xie (2023)
W. Peebles and S. Xie

Scalable diffusion models with Transformers.

In IEEE/CVF International Conference on Computer Vision,

pp. 4195–4205.

External Links: [Link](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)

Cited by: §2.4.
- Radford et al. (2021)
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever

Learning transferable visual models from natural language supervision.

In International Conference on Machine Learning,

Proceedings of Machine Learning Research, Vol. 139, pp. 8748–8763.

External Links: [Link](https://proceedings.mlr.press/v139/radford21a.html)

Cited by: Appendix A.
- Reed et al. (2022)
S. Reed, K. Zolna, E. Parisotto, S. Gomez Colmenarejo, A. Novikov, G. Barth-Maron, M. Gimenez, Y. Sulsky, J. Kay, J. T. Springenberg, T. Eccles, J. Bruce, A. Razavi, A. Edwards, N. Heess, Y. Chen, R. Hadsell, O. Vinyals, M. Bordbar, and N. de Freitas

A generalist agent.

Transactions on Machine Learning Research.

External Links: [Link](https://arxiv.org/abs/2205.06175)

Cited by: §8.1.
- Ross et al. (2011)
S. Ross, G. Gordon, and D. Bagnell

A reduction of imitation learning and structured prediction to no-regret online learning.

In International Conference on Artificial Intelligence and Statistics,

Proceedings of Machine Learning Research, Vol. 15, pp. 627–635.

External Links: [Link](https://proceedings.mlr.press/v15/ross11a.html)

Cited by: §F.1.
- Shah et al. (2026a)
R. Shah, R. K. Jenamani, X. Zhang, L. Sun, R. Martín-Martín, Y. Zhu, D. Ramanan, and K. Schmeckpeper

Scaling short-term memory of visuomotor policies for long-horizon tasks.

arXiv preprint arXiv:2606.16178.

External Links: [Link](https://arxiv.org/abs/2606.16178)

Cited by: §8.2.
- Shah et al. (2026b)
R. Shah, Y. Li, F. Bello, Y. Zhu, and R. Martín-Martín

Memory retrieval in visuomotor policies for long-horizon robot control.

arXiv preprint arXiv:2606.25136.

External Links: [Link](https://arxiv.org/abs/2606.25136)

Cited by: §8.2.
- Shi et al. (2026)
H. Shi, B. Xie, Y. Liu, L. Sun, F. Liu, T. Wang, E. Zhou, H. Fan, X. Zhang, and G. Huang

MemoryVLA: Perceptual-cognitive memory in vision-language-action models for robotic manipulation.

In International Conference on Learning Representations,

External Links: [Link](https://arxiv.org/abs/2508.19236)

Cited by: §8.2,
§8.2.
- Su et al. (2021)
J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu

RoFormer: Enhanced Transformer with rotary position embedding.

arXiv preprint arXiv:2104.09864.

External Links: [Link](https://arxiv.org/abs/2104.09864)

Cited by: §2.3.
- Vujinovic and Kovacevic (2026)
A. Vujinovic and A. Kovacevic

ACT-JEPA: Novel joint-embedding predictive architecture for efficient policy representation learning.

IEEE Access 14, pp. 78895–78906.

External Links: [Document](https://dx.doi.org/10.1109/ACCESS.2026.3696039),
[Link](https://arxiv.org/abs/2501.14622)

Cited by: §8.3.
- Vuong et al. (2026)
A. D. Vuong, T. V. Vo, A. Sohail, H. Ding, L. Ma, X. Liang, A. Duan, I. Laptev, and I. Reid

World2Act: Latent action post-training from world model dynamics.

arXiv preprint arXiv:2603.10422.

External Links: [Link](https://arxiv.org/abs/2603.10422)

Cited by: Table 14.
- Won et al. (2026)
J. Won, K. Lee, H. Jang, D. Kim, and J. Shin

Dual-stream diffusion for world-model augmented vision-language-action model.

In International Conference on Machine Learning,

Note: Initial arXiv submission in 2025; accepted to ICML 2026

External Links: [Link](https://arxiv.org/abs/2510.27607)

Cited by: Table 14.
- Wu et al. (2024)
H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong

Unleashing large-scale video generative pre-training for visual robot manipulation.

In International Conference on Learning Representations,

External Links: [Link](https://arxiv.org/abs/2312.13139)

Cited by: §8.1,
§8.3.
- Xiaomi Robotics Team et al. (2026)
Xiaomi Robotics Team, J. Guo, P. Jin, J. Li, P. Li, Y. Li, F. Liu, W. Peng, O. Qin, Y. Su, et al.

Xiaomi-Robotics-1: Scaling vision-language-action models with over 100k hours of real-world trajectories.

arXiv preprint arXiv:2607.15330.

External Links: [Link](https://arxiv.org/abs/2607.15330)

Cited by: Table 14.
- Yang et al. (2024)
A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, et al.

Qwen2 technical report.

arXiv preprint arXiv:2407.10671.

External Links: [Link](https://arxiv.org/abs/2407.10671)

Cited by: §2.3.
- Zhao et al. (2023)
T. Z. Zhao, V. Kumar, S. Levine, and C. Finn

Learning fine-grained bimanual manipulation with low-cost hardware.

In Robotics: Science and Systems,

External Links: [Link](https://arxiv.org/abs/2304.13705)

Cited by: §8.1.
- Zheng et al. (2025)
R. Zheng, J. Wang, S. Reed, J. Bjorck, Y. Fang, F. Hu, J. Jang, K. Kundalia, Z. Lin, L. Magne, et al.

FLARE: Robot learning with implicit world modeling.

arXiv preprint arXiv:2505.15659.

External Links: [Link](https://arxiv.org/abs/2505.15659)

Cited by: Table 14,
Table 14,
Table 14.
- Zhou et al. (2026)
Y. Zhou, Y. Wang, N. Wang, S. Xing, S. Tu, X. Li, J. Zhang, N. Jiang, Y. Lin, H. Yang, et al.

Chronos: A physics-informed full-history framework for non-markovian long-horizon manipulation.

arXiv preprint arXiv:2606.30318.

Note: Submitted to IEEE Transactions on Robotics

External Links: [Link](https://arxiv.org/abs/2606.30318)

Cited by: §8.1,
§8.2.
- Zhu et al. (2025)
C. Zhu, R. Yu, S. Feng, B. Burchfiel, P. Shah, and A. Gupta

Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets.

arXiv preprint arXiv:2504.02792.

External Links: [Link](https://arxiv.org/abs/2504.02792)

Cited by: Table 14.
- Zhu et al. (2024)
M. Zhu, Y. Zhu, J. Li, J. Wen, Z. Xu, N. Liu, R. Cheng, C. Shen, Y. Peng, F. Feng, and J. Tang

Scaling diffusion policy in transformer to 1 billion parameters for robotic manipulation.

arXiv preprint arXiv:2409.14411.

External Links: [Link](https://arxiv.org/abs/2409.14411)

Cited by: §8.4.

## Appendix A Implementation and evaluation protocol

The main sweep trains each data scale for approximately 100 loader epochs, with the scheduled optimizer steps in Table 4. One such pass traverses the dataset’s eight sample slots per training demonstration; each slot draws a random ten-token sequence window, so a pass is not an exhaustive enumeration of all valid timestep windows. Across the 49 cells whose complete stochastic-RMSE curves are present in the recovered archive, RMSE continues to decrease between the checkpoints nearest 70% and 100% of scheduled training, but the relative decrease has a median of 1.16%; 40/49 decreases are at most 2%, 48/49 are at most 5%, and the maximum is 5.02%. Every final checkpoint is within 0.86% of the best logged value in its late-training tail. The remaining seed-1, $D=2{,}900$, 1.49B trace is unavailable and excluded from this audit. Preliminary late-checkpoint rollouts on eight seed-0 configurations likewise show modest net SR changes on average, although individual estimates fluctuate. These diagnostics support the uniform use of scheduled final checkpoints; they do not certify exact convergence, compute optimality, or equal optimization completeness across capacities. We therefore do not select checkpoints by rollout. A low-level control step below denotes one environment command rather than one simulator integration step.

#### Task and data registry.

The RoboCasa sweep uses a fixed 23-task subset. We exclude OpenDoubleDoor because its official generated-image pool contains only 1,500 demonstrations and therefore cannot support the common $D=2{,}900$ endpoint. For every retained task, 100 demonstrations are reserved as a common holdout set (2,300 demonstrations total) and are disjoint from every training set. The $D=300$ set is the official 300_demos subset; $D\in\{50,100,1000\}$ are independently fixed, hash-recorded samples from the non-holdout pool. These samples may overlap but are not nested. The $D=2{,}900$ set uses the full non-holdout pool. At a given $D$, the demonstration identities are shared across model capacities and training seeds.

*Table 4: RoboCasa optimizer steps by training demonstrations per task.*

| $D$ | 50 | 100 | 300 | 1000 | 2900 |
|---|---|---|---|---|---|
| Steps | 5k | 10k | 30k | 100k | 280k |

Each camera has its own five-stage CNN stem. The stages use $5\times 5$ convolutions with channels $(96,144,192,384,768)$, followed in each stage by $2\times 2$ max pooling, group normalization, and SiLU; a $128\times 128$ image therefore yields a $4\times 4\times 768$ feature grid. Task text is encoded once by the frozen openai/clip-vit-large-patch14 text tower with projection (41), producing a 768-D vector; this encoder is excluded from the parameter counts. From smallest to largest capacity, the action decoders use (hidden width, layers, attention heads, feed-forward width) $(128,1,4,512)$, $(192,2,6,768)$, $(256,4,8,1024)$, $(384,6,12,1536)$, and $(512,8,16,2048)$. Thus each capacity row jointly scales the within-step encoder, temporal backbone, and action decoder; it is not a backbone-only intervention.

*Table 5: RoboCasa model configurations, parameter counts, and learning rates. Parameter counts include the convolutional visual stems and all policy modules; the frozen CLIP text encoder is excluded. The action-decoder peak learning rate is $3.0\times 10^{-4}$ for every row.*

| Width | Within-step | Temporal | Params | LR |
|---|---|---|---|---|
| $d$ | layers | heads | layers | heads |  | encoder/temporal |
| 512 | 1 | 4 | 2 | 4 | 44.3M | $6.0\times 10^{-4}$ |
| 768 | 2 | 6 | 4 | 6 | 85.3M | $4.25\times 10^{-4}$ |
| 1024 | 4 | 8 | 8 | 8 | 218.8M | $3.0\times 10^{-4}$ |
| 1536 | 6 | 12 | 12 | 12 | 648.9M | $1.5\times 10^{-4}$ |
| 2048 | 8 | 16 | 16 | 16 | 1490.3M | $5.0\times 10^{-5}$ |

*Table 6: RoboCasa training and closed-loop execution recipe. The main sweep uses $C_{\mathrm{train}}=10$; separately trained context variants are identified where they are analyzed.*

| Component | Protocol |
|---|---|
| Inputs | Three $128\times 128$ RGB views, 16-D proprioception, and a frozen 768-D CLIP task embedding |
| Temporal sampling | 20 Hz environment control; one observation, world token, and replanning decision every four control steps (5 Hz) |
| Actions | Same-frame-aligned 12-D commands; $H=10$ predicted, $H_{\mathrm{exec}}=4$ executed before replanning |
| Context and diffusion | $C_{\mathrm{train}}=10$ world tokens; 20-stage cosine schedule and stochastic 20-step DDPM evaluation |
| Optimization | AdamW, $\beta=(0.9,0.95)$, $\epsilon=10^{-8}$, zero weight decay, unit gradient clipping, BF16, and global batch size 192 |
| LR schedule | Encoder/temporal rates in Table 5; action-decoder rate $3.0\times 10^{-4}$; 5% linear warmup, then cosine decay to 10% of each peak rate |

#### Action alignment.

The two benchmarks use different on-disk indexing conventions. In RoboCasa, the observation stored at index $t$ is paired with the same-frame command chunk action[$t$:$t+H$]. In RMBench, the observation is vector[$t$] and the $H$-step absolute joint-position target is vector[$t+1$:$t+H+1$]. In both cases, chunk position zero is the first command applied after conditioning on the current observation; the one-index difference reflects how the two datasets store observations and command targets, not an additional open-loop delay. RoboCasa stores a repeated terminal action in padded chunk slots but masks them, and the objective retains a query only when all $H$ targets are real; RMBench likewise samples only query frames with a complete future chunk. Thus neither benchmark trains on padded terminal targets.

All training and evaluation were conducted on two single-node servers: one with eight NVIDIA A100-80GB GPUs and one with eight NVIDIA H100-80GB GPUs. Each main-sweep $C_{\mathrm{test}}=10$ final checkpoint is executed three times on the same 1,150 episode identities, environment seeds, rollout seed, and protocol. The controlled architecture and single-anchor diagnostics state explicitly when they also use three executions; the $C_{\mathrm{test}}\in\{1,2,5\}$ truncations and direct-action-decoder rows use one execution each. Reported standard deviations are sample standard deviations across the stated executions. Because stochastic diffusion and simulation are not bitwise deterministic, this dispersion is a fixed-seed repeatability measure rather than uncertainty over independently sampled environments.

Holdout RMSE uses eval_seed=0 to select a fixed registry of eight ten-timestep sequence crops from each of the 2,300 held-out demonstrations, rather than exhaustively enumerating the trajectories. Within those crops, one $H=10$ action chunk is generated at every query that has a complete target chunk. RMSE is computed over the unnormalized 12-D command space, and the 23 taskwise RMSE values are macro-averaged. Deterministic and stochastic DDPM evaluations start from the same fixed-seed Gaussian $x_{T}$ for each query. The deterministic sampler follows the DDPM posterior mean at every reverse step and draws no additional reverse-step variance; the stochastic ancestral sampler uses the same initialization and draws the prescribed reverse-step variance.

#### Local BC-Transformer reference.

The local BC-Transformer comparison uses the same 23-task registry, the same $D=300$ demonstration identities, and the same final 1,150-episode evaluation registry, but is not otherwise a recipe-matched control. It uses one training seed (123), batch size 16, 500,000 optimizer steps, the official native recipe with AdamW weight decay 0.01, observation stride one, and native one-action execution with replanning after every action. WorldToken instead uses observation stride four and executes four actions per replanning query. Optimizer, training compute, seed count, observation cadence, and action-execution cadence are therefore not matched.

### A.1 Complete RoboCasa closed-loop success-rate grids

Tables 7 and 8 expose the three fixed-checkpoint $C_{\mathrm{test}}=10$ executions behind every cell of the main scaling sweep. Tables 9–11 give the corresponding absolute SR for the same checkpoints under each inference-time history truncation. The scaling entries are triplets; each truncated-history entry is one execution, as specified in Section 4.2. All entries are percentages over 1,150 episodes, and all cited summaries are complete and have zero crashed episodes.

*Table 7: Complete RoboCasa scaling SR for training seed 0. Each cell is repeat 1 / repeat 2 / repeat 3 in percent; no averaging is applied within the table.*

| $D$ | 44.3M | 85.3M | 218.8M | 648.9M | 1.49B |
|---|---|---|---|---|---|
| 50 | 14.52 / 13.91 / 14.00 | 20.96 / 20.96 / 22.26 | 22.70 / 22.00 / 21.22 | 27.13 / 28.17 / 26.09 | 22.52 / 23.22 / 23.48 |
| 100 | 29.48 / 29.57 / 30.87 | 30.17 / 31.74 / 31.65 | 37.30 / 34.87 / 35.30 | 36.09 / 36.35 / 36.96 | 33.48 / 33.30 / 34.35 |
| 300 | 41.04 / 40.87 / 41.22 | 48.00 / 46.17 / 48.26 | 46.70 / 50.00 / 47.65 | 50.78 / 51.22 / 51.74 | 50.00 / 49.39 / 50.09 |
| 1000 | 48.96 / 49.91 / 48.43 | 55.83 / 56.35 / 54.87 | 59.39 / 58.26 / 57.74 | 56.78 / 58.43 / 57.74 | 56.43 / 56.70 / 55.48 |
| 2900 | 55.04 / 55.48 / 52.70 | 58.96 / 58.17 / 60.09 | 61.04 / 56.96 / 60.00 | 57.65 / 60.52 / 58.26 | 59.48 / 60.43 / 59.04 |

*Table 8: Complete RoboCasa scaling SR for training seed 1, in the same format as Table 7.*

| $D$ | 44.3M | 85.3M | 218.8M | 648.9M | 1.49B |
|---|---|---|---|---|---|
| 50 | 18.17 / 18.96 / 17.30 | 20.35 / 17.65 / 18.78 | 23.39 / 22.70 / 22.09 | 24.17 / 23.74 / 24.17 | 21.48 / 22.00 / 21.48 |
| 100 | 27.30 / 26.87 / 27.91 | 34.43 / 31.83 / 30.87 | 31.91 / 34.00 / 33.83 | 36.43 / 35.74 / 36.70 | 33.13 / 34.35 / 36.35 |
| 300 | 41.13 / 41.74 / 41.22 | 46.52 / 46.35 / 45.65 | 48.87 / 49.13 / 47.13 | 50.52 / 51.91 / 51.74 | 49.83 / 49.83 / 49.91 |
| 1000 | 50.26 / 49.91 / 49.39 | 52.87 / 52.78 / 50.96 | 58.35 / 56.96 / 58.09 | 56.78 / 56.26 / 57.04 | 56.87 / 56.52 / 55.91 |
| 2900 | 54.09 / 56.00 / 53.91 | 59.83 / 59.74 / 59.91 | 61.74 / 62.00 / 59.83 | 58.09 / 57.83 / 57.48 | 58.61 / 60.87 / 58.35 |

Forty-five triplets above are transcribed from recovered complete, zero-crash terminal summaries. Five use exact post-repair counts preserved in the registry or inventory together with the named repair records in Appendix G; none is inferred from the one-decimal labels in Figure 3. Figure labels are the corresponding means rounded to one decimal place.

*Table 9: Absolute RoboCasa SR (%) with one visible history step, using the fixed ten-step-trained checkpoints. Each cell is one complete 1,150-episode execution.*

| Training seed 0 |
|---|
| $D$ | 44.3M | 85.3M | 218.8M | 648.9M | 1.49B |
| 50 | 10.61 | 14.26 | 14.70 | 16.61 | 16.70 |
| 100 | 19.83 | 22.52 | 24.70 | 21.48 | 20.09 |
| 300 | 22.96 | 29.65 | 27.91 | 29.91 | 30.17 |
| 1000 | 34.17 | 37.65 | 38.17 | 38.78 | 38.87 |
| 2900 | 41.39 | 43.74 | 40.43 | 36.61 | 41.04 |
| Training seed 1 |
| $D$ | 44.3M | 85.3M | 218.8M | 648.9M | 1.49B |
| 50 | 12.52 | 10.96 | 14.35 | 16.43 | 14.78 |
| 100 | 21.65 | 21.65 | 21.74 | 17.39 | 19.91 |
| 300 | 30.96 | 29.57 | 27.83 | 29.30 | 28.35 |
| 1000 | 35.83 | 34.43 | 37.57 | 37.65 | 37.57 |
| 2900 | 40.61 | 42.61 | 42.00 | 38.87 | 40.43 |

*Table 10: Absolute RoboCasa SR (%) with two visible history steps, using the fixed ten-step-trained checkpoints. Each cell is one complete 1,150-episode execution.*

| Training seed 0 |
|---|
| $D$ | 44.3M | 85.3M | 218.8M | 648.9M | 1.49B |
| 50 | 10.00 | 16.09 | 17.57 | 18.26 | 17.39 |
| 100 | 26.70 | 24.52 | 25.91 | 25.13 | 25.22 |
| 300 | 30.09 | 35.22 | 34.96 | 39.30 | 37.13 |
| 1000 | 40.52 | 49.22 | 46.78 | 47.57 | 46.78 |
| 2900 | 47.57 | 52.87 | 53.74 | 49.65 | 50.87 |
| Training seed 1 |
| $D$ | 44.3M | 85.3M | 218.8M | 648.9M | 1.49B |
| 50 | 15.57 | 13.57 | 15.13 | 15.04 | 16.09 |
| 100 | 22.52 | 24.96 | 25.91 | 25.04 | 24.43 |
| 300 | 39.91 | 36.43 | 35.74 | 38.17 | 34.78 |
| 1000 | 42.43 | 45.57 | 47.74 | 46.70 | 47.04 |
| 2900 | 49.04 | 53.22 | 54.00 | 52.00 | 48.61 |

*Table 11: Absolute RoboCasa SR (%) with five visible history steps, using the fixed ten-step-trained checkpoints. Each cell is one complete 1,150-episode execution.*

| Training seed 0 |
|---|
| $D$ | 44.3M | 85.3M | 218.8M | 648.9M | 1.49B |
| 50 | 13.22 | 21.65 | 22.26 | 25.74 | 23.57 |
| 100 | 30.35 | 31.13 | 35.48 | 34.61 | 33.30 |
| 300 | 40.52 | 46.35 | 47.04 | 49.91 | 48.35 |
| 1000 | 48.96 | 56.43 | 57.91 | 57.04 | 57.74 |
| 2900 | 53.83 | 60.61 | 59.30 | 57.83 | 58.70 |
| Training seed 1 |
| $D$ | 44.3M | 85.3M | 218.8M | 648.9M | 1.49B |
| 50 | 18.78 | 18.09 | 20.61 | 23.48 | 19.39 |
| 100 | 27.13 | 32.35 | 33.74 | 34.61 | 34.61 |
| 300 | 42.17 | 46.35 | 48.17 | 49.91 | 48.61 |
| 1000 | 49.74 | 52.70 | 57.22 | 57.74 | 57.48 |
| 2900 | 55.22 | 58.70 | 59.13 | 59.74 | 59.39 |

*Table 12: Per-task success of the best individual execution in the main RoboCasa sweep: the 218.8M-parameter model at $D=2900$, training seed 1, repeat 02. Each task has 50 episodes; the total is 713/1,150 = 62.0%.*

| Task | Successes | SR | Task | Successes | SR |
|---|---|---|---|---|---|
| CloseDoubleDoor | 45/50 | 90% | PnPCounterToStove | 16/50 | 32% |
| CloseDrawer | 48/50 | 96% | PnPMicrowaveToCounter | 10/50 | 20% |
| CloseSingleDoor | 45/50 | 90% | PnPSinkToCounter | 25/50 | 50% |
| CoffeePressButton | 49/50 | 98% | PnPStoveToCounter | 22/50 | 44% |
| CoffeeServeMug | 38/50 | 76% | TurnOffMicrowave | 50/50 | 100% |
| CoffeeSetupMug | 23/50 | 46% | TurnOffSinkFaucet | 41/50 | 82% |
| OpenDrawer | 43/50 | 86% | TurnOffStove | 9/50 | 18% |
| OpenSingleDoor | 30/50 | 60% | TurnOnMicrowave | 46/50 | 92% |
| PnPCabToCounter | 9/50 | 18% | TurnOnSinkFaucet | 32/50 | 64% |
| PnPCounterToCab | 23/50 | 46% | TurnOnStove | 36/50 | 72% |
| PnPCounterToMicrowave | 16/50 | 32% | TurnSinkSpout | 39/50 | 78% |
| PnPCounterToSink | 18/50 | 36% |  |  |  |
| All tasks | 713/1,150 | 62.0% |

## Appendix B Hyperparameter selection provenance

The scale-specific recipes of Table 5 were frozen before the main sweep from short calibration runs at $D=300$, in a fixed order: weight decay first, then a discrete learning-rate entry for each capacity. The frozen recipe uses one shared encoder/temporal rate and an action-decoder rate of $3.0\times 10^{-4}$. Calibration runs follow the task set and 50-episode-per-task protocol of Appendix A but enter no scaling curve or main-text statistic, and no scale was retuned after the sweep began. These one-shot calibration rollouts predate the final per-episode seed manifest; they form a separate development evaluation stream and do not reuse the final reporting episode registry. They select nuisance optimization settings only and are not evidence about architecture or capacity.

#### Weight decay.

Before the final scaling family was frozen, a three-way $D=300$ qualification sweep on an earlier 364.0M-parameter predecessor compared weight decay $\{0,0.03,0.1\}$ with all other settings fixed within that predecessor family. The final-checkpoint rollouts scored 44.26%, 41.13%, and 44.09% (509, 473, and 507 of 1,150 episodes). Zero weight decay was carried forward as the common optimizer default. This one-shot predecessor-family diagnostic is neither evidence that zero weight decay is generally optimal nor a comparison among the final architectures. Every WorldToken policy and diagnostic reported here uses zero weight decay; the local BC-Transformer reference retains its native weight decay of 0.01.

#### Learning rate.

The final discrete table was assembled from fixed-protocol candidate rollouts and earlier range-setting calibrations. At the time, we had not yet characterized the substantial execution variability of SR; the later audit in Appendix F.3 shows that small differences of this size can fall within repeatability variation. We therefore report the surviving individual results rather than statistically resolved rankings, and did not retrospectively retune the frozen rates. RMSE is a companion offline diagnostic and did not enter the LR choice.

Table B gives the surviving records underlying each of the five capacity entries. The 85.3M and 648.9M groups are direct three-point shared-rate lookups, and the 1.49B group combines two direct lookup runs with two earlier calibration runs. For 44.3M, the surviving rows are shared-rate legacy candidates; the matching $(3.00/3.00/3.00)\times 10^{-4}$ candidate is unavailable. The 218.8M group contains two component-shifted range-setting runs bracketing the shared-rate frozen reference. Actual encoder/temporal/action-decoder peak rates are shown explicitly. These records document recipe selection only; they are not a cross-capacity LR rule. The table reports deterministic full-chunk task-macro RMSE throughout.

Table 13: Learning-rate calibration and adjudication records ($D=300$, seed 0). LR triples are actual encoder/temporal/action-decoder peak rates in units of $10^{-4}$. SR is one fixed-protocol execution of 1,150 episodes at the archived final checkpoint. RMSE is deterministic holdout full-chunk task-macro RMSE at 30k unless noted. Bold rows record the frozen tuples.

| Capacity | Record | LR (E/T/H) | One-shot SR | Deterministic RMSE |
|---|---|---|---|---|
| 44.3M | lookup (unavailable) | (3.00/3.00/3.00) | — | — |
|  | lookup | (6.00/6.00/3.00) | 42.61% | 0.1478 |
|  | lookup | (12.00/12.00/3.00) | 37.65% | 0.1541 |
| 85.3M | lookup | (3.50/3.50/3.00) | 45.57% | 0.1361 |
|  | lookup | (4.25/4.25/3.00) | 46.61% | 0.1362 |
|  | lookup | (5.00/5.00/3.00) | 43.30% | 0.1358 |
| 218.8M | formula | (3.00/3.00/1.50) | 45.74% | 0.1282 |
|  | frozen ref. | (3.00/3.00/3.00) | 52.35% | 0.1283 |
|  | formula | (6.00/6.00/3.00) | 40.43% | 0.1178 |
| 648.9M | lookup | (1.00/1.00/3.00) | 49.83% | 0.1398 |
|  | lookup | (1.50/1.50/3.00) | 50.17% | 0.1338 |
|  | lookup | (2.25/2.25/3.00) | 49.04% | 0.1270 |
| 1.49B | lookup | (0.375/0.375/3.00) | 48.26% | 0.1459 |
|  | lookup | (0.50/0.50/3.00) | 50.78% | 0.1422 |
|  | earlier run | (0.75/0.75/3.00) | 50.43% | 0.1385 |
|  | earlier run | (1.50/1.50/3.00) | 48.43% | 0.1289 |

—denotes an unavailable matching measurement. For 44.3M, the $(3.00/3.00/3.00)\times 10^{-4}$ numerical artifact is not present in the recovered archive, while the contemporaneous adjudication record selecting $(6.00/6.00/3.00)\times 10^{-4}$ remains available. Recovered 30k logs give deterministic RMSE 0.1478 and 0.1541 for the two surviving rows; earlier manuscript drafts used the last then-available preterminal values 0.1475 and 0.1543.

## Appendix C Selected publicly reported RoboCasa results

Table 14 is a nonexhaustive scale reference containing selected publicly reported rows. Protocols, task sets, target data, external pretraining, checkpoint selection, and aggregation differ, so neither row order nor numerical gaps support a method ranking. Human-50 denotes 50 human demonstrations per task and Gen-$N$ denotes $N$ generated trajectories per task. WorldToken uses 23 tasks; the other entries use 24. Our values are equal-weight means over two training seeds, with each training-seed result averaging three complete executions. The best individual executions are reported in the main text.

*Table 14: Selected public RoboCasa Kitchen performance-scale reference, ordered for lookup by reported success rate. Rows are not protocol matched, and the selection is not exhaustive.*

|  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|
| Method | Target data | External pretraining / initialization | Objective / mechanism | Params | Source / aggregation | Tasks | SR |
| X-WAM (15) | official demos | Wan2.2-5B; 5,800+h / 1.49M robot and simulation trajectories, including RoboCasa | RGB-D 4D modeling; ANS | $\sim$5B | paper / avg. | 24 | 79.2 |
| Xiaomi-Robotics-1 (52) | Gen-300 | Qwen3-VL; 100K+h UMI pretraining; $\sim$10K h cross-embodiment post-training | flow matching; Choice Policies | family: 2.6–10.5B; evaluated size not stated | paper / avg. | 24 | 74.5 |
| GR00T N1.6 + World2Act (49) | 1K expert + $\sim$1K WM synthetic total | GR00T N1.6; Cosmos-Predict2 WM | latent-action post-training | $\sim$3B | paper / avg. | 24 | 72.6 |
| Cosmos Policy P2.5 (28) | Human-50 | Cosmos-Predict2.5 video model | supervised policy post-training | 2B | official recipe / avg. | 24 | 71.1 |
| RLDX-1 (25) | Gen-300 | Qwen3-VL; $\sim$1.5M robot trajectories | multistream action modeling | 6.9B | paper / avg. | 24 | 70.6 |
| FLARE (55) | 1K/task | SigLIP2 image–text initialization; no external robot data | in-domain future-representation alignment | not reported | paper / best of 5 checkpoints | 24 | 70.1 |
| GR00T N1.5 + RS-CL (30) | Gen-300 | GR00T robot pretraining | RS-CL representation regularization | $\sim$2.7B | paper / avg. | 24 | 69.7 |
| GAM (20) | Gen-300 | DA3-Giant / Track4World; T5; 784K robot trajectories | future geometry; depth supervision | 1.4B | paper / avg. | 24 | 69.4 |
| Cosmos Policy P2 (27) | Human-50 | Cosmos-Predict2 video model | supervised policy post-training | 2B | paper / avg. | 24 | 67.1 |
| FLARE (55) | 300/task | SigLIP2; cross-robot action-aware embedding pretraining | future-representation alignment | not reported | paper / best of 5 checkpoints | 24 | 66.4 |
| GR00T N1.5 + HAMLET (31) | Gen-300 | GR00T robot pretraining | history memory | $\sim$2.9B | paper / avg. | 24 | 66.4 |
| GR00T N1.6 (25) | Gen-300 | large-scale robot pretraining | supervised policy post-training | $\sim$3B | official / avg. | 24 | 66.2 |
| Video Policy (34) | Gen-300 | SVD video model | joint video–action diffusion | not reported | paper / mean | 24 | 66.0 |
| GR00T N1.5 (31; 30) | Gen-300 | large-scale robot pretraining | supervised policy post-training | $\sim$2.7B | reprod. / avg. | 24 | 64.1–65.7 |
| $\pi_{0}$-FAST (30) | Gen-300 | PaliGemma; multi-robot data | FAST action tokenizer | $\sim$3.3B | reprod. / avg. | 24 | 63.6 |
| Video Policy (34) | Human-50 | SVD video model | joint video–action diffusion | not reported | paper / mean | 24 | 63.0 |
| $\pi_{0.5}$ + pointmap (33) | Human-50 | $\pi_{0.5}$ multi-source robot/Web pretraining | robot-centric pointmap at train/test | $\sim$3.3B | paper / avg. | 24 | 62.9 |
| $\pi_{0}$ (6; 30) | Gen-300 | PaliGemma; multi-robot data | supervised policy post-training | $\sim$3.3B | reprod. / avg. | 24 | 62.5 |
| DreamZero (15) | official demos | Wan2.2-5B general video model | video-generative policy | $\sim$5B | reprod. / avg. | 24 | 62.4 |
| UWM (FLARE reproduction) (57; 55) | 1K/task | ImageNet/VAE initialization; no external robot trajectories | joint action–future-image latent diffusion | not reported | paper / best of 5 checkpoints | 24 | 60.8 |
| WorldToken (218.8M) | $D{=}2900$ | CLIP text; visual/policy random init. | behavior cloning | 218.8M | ours / 2-seed mean | 23 | 60.3 |
| WorldToken (85.3M) | $D{=}2900$ | CLIP text; visual/policy random init. | behavior cloning | 85.3M | ours / 2-seed mean | 23 | 59.5 |
| GR00T N1.5 + DUST (50) | Gen-300 | pretrained GR00T VLM; random action expert | action–future-latent dual-stream diffusion | $\sim$2.7B | paper / avg. | 24 | 58.5 |
| GR00T N1 + DreamGen (23) | Gen-300 + 10K syn. / task | GR00T robot pretraining | video-generated trajectories | 2.2B | paper / avg. | 24 | 57.6 |
| DP-VLA (19) | Gen-3000 | OpenVLA/OXE pretraining | OpenVLA latent + BC-Transformer | $\sim$7B | paper / avg. | 24 | 57.3 |
| $\pi_{0.5}$ + Pri4R (26) | Human-50 | $\pi_{0.5}$ multi-source pretraining | privileged 3D supervision | $\sim$3.3B | paper / avg. | 24 | 57.0 |
| UVA (34) | Human-50 | pretrained VAE; MAR image generator | joint video–action modeling | not reported | reprod. / mean | 24 | 50.0 |
| GR00T N1 (5) | Gen-300 | VLM; robot / human / synthetic data | supervised policy post-training | 2.2B | paper / avg. | 24 | 49.6 |
| BC-Transformer (37) | Gen-3000 | CLIP text; visual/policy random init. | behavior cloning | not reported | paper / mean | 24 | 47.6 |
| WorldToken (85.3M) | $D{=}300$ | CLIP text; visual/policy random init. | behavior cloning | 85.3M | ours / 2-seed mean | 23 | 46.8 |
| BC-Transformer (37) | Gen-300 | CLIP text; visual/policy random init. | behavior cloning | not reported | paper / mean | 24 | 35.0 |

*Table 14: Selected public RoboCasa Kitchen performance-scale reference (continued).*

## Appendix D Blocks Ranking training objective and archived development diagnostic

#### Training and development protocol.

The reported policy continues the seed-1 nine-task step-5,000 checkpoint for 500 additional optimizer steps on Blocks Ranking, using 45 training demonstrations and five held-out demonstrations. Its materialized recipe uses microbatch size one with eight-step gradient accumulation, AdamW with $\beta=(0.9,0.95)$, $\epsilon=10^{-8}$, zero weight decay, unit gradient clipping, and BF16; encoder/predictor peak learning rates are $4.25\times 10^{-4}$ and the base/action-head peak rate is $3.0\times 10^{-4}$, with 250 warmup steps followed by cosine decay to 10% of peak. The inherited checkpoint carries a per-dimension min–max action normalizer fitted on the nine-task training split, mapping each nondegenerate dimension to $[-1,1]$. The DiT head uses epsilon prediction with 20 squared-cosine DDPM steps, fixed-small variance, and denoised clipping to $[-1,1]$; reported rollouts use stochastic ancestral reverse steps. The step-5,500 checkpoint was selected and reported on the same fixed registry of 100 expert-validated initial conditions. Accordingly, 95/100 is a development-registry result, not an independent held-out test estimate. All history interventions below freeze this checkpoint, episode registry, success predicate, and horizon, so their relative comparison is a same-checkpoint development diagnostic.

The Blocks Ranking continuation uses a task-specific training-only objective for the left-arm descent component of the button press. For each expert target frame $k$ whose measured left-end-effector displacement satisfies $\Delta z_{k}<-0.2$ mm, we form a 14-D raw-action direction $d_{k}$ from the expert increment in the six left-arm joints and set its other eight entries to zero. The measured $\Delta z_{k}$ is used only to select the frame and to scale this local joint-space direction; the loss does not apply forward kinematics to the predicted joint action, and end-effector pose is not a policy observation.

Let $u_{k}$ be the unit direction after action normalization, let $e_{k}=\langle\hat{x}_{0,k}-x_{0,k},u_{k}\rangle$ be the denoised action error along that direction, and let $c_{k}>0$ be the scalar coordinate along $u_{k}$ of the normalized joint-space extrapolation corresponding to 4 mm of additional descent. With $p_{k}=(1.5/4)c_{k}$, the parallel component of the DDPM prediction loss is replaced by

|  | $$ \ell_{\parallel}(e_{k})=3\kappa_{t}\begin{cases}16(p_{k}-e_{k})^{2},&e_{k}<p_{k},\\ 0,&p_{k}\leq e_{k}\leq c_{k},\\ (e_{k}-c_{k})^{2},&e_{k}>c_{k},\end{cases} $$ |  | (21) |
|---|---|---|---|

where $\kappa_{t}=\bar{\alpha}_{t}/(1-\bar{\alpha}_{t})$, with $\bar{\alpha}_{t}$ the cumulative DDPM signal coefficient, maps a denoised-action distance back to the epsilon-prediction space at diffusion step $t$. The orthogonal component retains the ordinary diffusion BC loss, as do all non-descent targets. Thus the factors 3 and 16 apply only to the one-dimensional descent term, and the 1.5–4 mm interval is a zero-loss corridor along the expert-derived local joint-space direction. No button-contact or press-stage label is used.

The archived recipes record the same nine-task step-5,000 starting checkpoint, 45/5 split, 500-step continuation schedule, architecture, 100 expert-validated initial conditions, $C=608$, four executed actions per replanning query, success predicate, and episode horizon. The standard-loss and modified-loss runs were produced under different recorded code revisions, however, so the surviving metadata do not establish that the executions were bitwise identical apart from the objective. We therefore treat this as a controlled development diagnostic under the recorded fields, rather than as an isolated estimate of a universally applicable loss effect.

The original experiment server was subsequently compromised and its primary run store was deleted. The detailed standard-loss aggregate and trajectory diagnostics below are retained in a contemporaneous pre-loss manuscript and evaluation record, but the present recovered archive does not contain enough of that run’s terminal episode output to reaggregate them independently; Appendix G classifies them as archived-only. In contrast, recovered rollout summaries directly reconstruct the modified-loss checkpoint’s 95/100 result and its same-checkpoint history interventions: 95/100, 92/100, 59/100, 38/100, and 28/100 for $C\in\{608,288,128,64,32\}$.

The contemporaneous record reports 13/100 successes for the standard-loss policy and 95/100 for the modified-loss policy. All 13 standard-loss successes also succeed under the modified loss; the modified-loss policy succeeds on 82 additional initial conditions, and both policies fail on the remaining five. Because the rollouts are independent stochastic executions, these matched-seed outcome categories are outcome-level rather than action-level comparisons. The same record reports that the standard-loss policy reaches the correct block order in 99 of 100 episodes—its 13 successes and 86 of its 87 failures—and that, in all 86 order-reaching failures, the deepest button press recorded while the order holds stays short of the $-5$ mm success threshold (deepest $-4.65$ mm, median $-0.54$ mm); the remaining failure never reaches the correct order. For one matched environment seed, the archived diagnostic records about $-1.5$ mm after the target order is ready under standard loss and about $-5.0$ mm under modified loss. Within these development rollouts, the 82-episode outcome gap is therefore consistent with a terminal press-depth difference while block sorting remains largely intact; the surviving evidence does not support a broader causal claim about objectives. Section 6 uses only this bounded example. Every history-length comparison in the main text uses the same recovered modified-loss checkpoint, and the descent objective does not vary with context length; Figure 7 shows the behavioral contrast between history lengths on one evaluation episode.

## Appendix E Single-anchor implementation diagnostics

This section retains the surviving implementation diagnostics for transparency, but does not use them to rank architectures or components. They cover only the $D=300$, $C=10$ operating point: temporal placement and the current-observation bypass use two training seeds, whereas token granularity, visual width, decoder learning rate, and action distribution use one. Most rows have three repeat executions of one checkpoint; the direct action decoders have one. The recorded action-head plan made a second training seed conditional on a clear seed-0 SR gain, and no second seed was added. Replication depth is therefore neither uniform nor sufficient for a structural conclusion; model-structure comparisons would need to be evaluated as scaling behavior over substantially larger data and compute ranges.

### E.1 Temporal placement of history: causal backbone versus denoiser cross-attention

All arms share the same trainable within-timestep encoder ($d=1024$) and training recipe and differ downstream of $z_{t}$. WorldToken is the 218.8M-parameter $D=300$ grid cell: an eight-layer causal temporal backbone and an 11.2M-parameter DiT conditioned on $h_{t}$. DP-like base removes the temporal backbone; the current $z_{t}$ supplies adaLN conditioning, while the nine strictly past, position-encoded $z$ tokens form cross-attention memory that is read at every denoising step (117.5M parameters). The parameter-matched arm widens this denoiser to $d=1024$ with a 3,520-wide feed-forward layer (218.7M total). A fourth six-layer denoiser arm was run for one seed but is omitted because its history-processing cost was not matched.

*Table 15: Temporal-placement diagnostic at $D=300$, $C=10$. RMSE is final-checkpoint holdout task-macro full-chunk error; SR is the mean and sample standard deviation of three fixed-protocol executions.*

| Arm | Params | Seed | RMSE det. | RMSE stoch. | SR |
|---|---|---|---|---|---|
| WorldToken | 218.8M | 0 | 0.12891 | 0.13171 | 48.12% $\pm$ 1.70 pp |
|  |  | 1 | 0.13052 | 0.13310 | 48.38% $\pm$ 1.09 pp |
| DP-like base | 117.5M | 0 | 0.14042 | 0.14230 | 50.78% $\pm$ 0.76 pp |
|  |  | 1 | 0.14033 | 0.14213 | 51.39% $\pm$ 0.78 pp |
| Parameter-matched | 218.7M | 0 | 0.14118 | 0.14297 | 49.48% $\pm$ 0.68 pp |
|  |  | 1 | 0.14087 | 0.14268 | 51.42% $\pm$ 0.61 pp |

At this single anchor, WorldToken has 6.4–8.7% lower expert-action RMSE than the two DP-like arms, while the DP-like arms have 1.4–3.0 percentage points higher mean SR. Thus RMSE and SR order the rows differently, which is the only use of this table in Section 6. Two seeds at one data scale do not establish an architecture ranking or its scaling behavior. As interface arithmetic only, the DP-like design re-reads $C$ history tokens from $H=10$ action-token queries at each of 20 denoising steps, whereas WorldToken performs temporal aggregation outside the denoising loop. No end-to-end latency, memory, or FLOP comparison was measured, including at the $C=608$ Blocks Ranking context.

### E.2 A current-observation bypass around the world token

The default decoder reads only $h_{t}$. The bypass additionally lets every DiT block cross-attend to the 50 post-fusion current-observation tokens, using zero-initialized output projections; it adds 2.63M parameters (+1.2%), and shared parameters start bit-identically to the reference.

*Table 16: Current-observation bypass diagnostic at $D=300$, $C=10$. SR is the mean and sample standard deviation of three fixed-protocol executions.*

| Arm | Seed | RMSE det. | RMSE stoch. | SR |
|---|---|---|---|---|
| WorldToken reference | 0 | 0.12891 | 0.13171 | 48.12% $\pm$ 1.70 pp |
|  | 1 | 0.13052 | 0.13310 | 48.38% $\pm$ 1.09 pp |
| Current-observation bypass | 0 | 0.13547 | 0.14178 | 46.20% $\pm$ 0.20 pp |
|  | 1 | 0.13941 | 0.15257 | 42.00% $\pm$ 0.88 pp |

The bypass has higher RMSE and lower SR for both training seeds at this recipe (SR changes of $-1.9$ and $-6.4$ points), with every bypass execution below every same-seed reference execution. This observation does not establish why performance changes, or that the one-token representation is lossless or generally sufficient.

### E.3 Token granularity and analytic backbone cost

The 85.3M-parameter reference passes one learned world token per timestep. The $K=2$ and $K=4$ variants reshape projections of the four learned readouts into ordered timestep-major token groups. The $K=50$ variant instead passes all 48 post-fusion spatial features plus the proprioception and task tokens; these are fused CNN features, not raw pixels. The decoder reads the final contextualized token of the current timestep in every arm.

For reference, leading full-prefix Transformer matrix-multiplication FLOPs, using two FLOPs per multiply–accumulate, are

|  | $$ F(N)=L(8d^{2}+6df)N+4LdN^{2}. $$ |  | (22) |
|---|---|---|---|

For the $d=768$, $f=2048$, $L=4$ backbone, this is $56{,}623{,}104N+12{,}288N^{2}$, with linear–quadratic crossover $N_{\times}=2d+1.5f=4{,}608$. Table 17 uses $N=10K$. It counts the temporal backbone only under full rectangular attention products; causal kernels may skip masked work, and these values are not measured latency.

*Table 17: Analytic full-prefix temporal-backbone FLOPs at $C=10$, relative to $K=1$. The within-timestep encoder and action decoder are excluded.*

| Tokens per timestep $K$ | Temporal length $N$ | Relative FLOPs |
|---|---|---|
| 1 | 10 | $1.00\times$ |
| 2 | 20 | $2.00\times$ |
| 4 | 40 | $4.03\times$ |
| 50 | 500 | $55.31\times$ |

*Table 18: Single-seed token-granularity diagnostic at the 85.3M-parameter, $D=300$, $C=10$ anchor. RMSE is final-checkpoint stochastic full-chunk task-macro error. SR is the mean and sample standard deviation of three complete fixed-protocol executions. Relative temporal-backbone FLOPs use Table 17.*

| Temporal interface | Params | Rel. FLOPs | RMSE | SR |
|---|---|---|---|---|
| $K=1$, world token | 85.3M | $1.00\times$ | 0.13680 | 47.48% $\pm$ 1.14 pp |
| $K=2$, readout tokens | 87.7M | $2.00\times$ | 0.13661 | 46.78% $\pm$ 1.35 pp |
| $K=4$, readout tokens | 92.4M | $4.03\times$ | 0.13826 | 48.75% $\pm$ 0.80 pp |
| $K=50$, post-fusion tokens | 83.0M | $55.31\times$ | 0.13230 | 49.57% $\pm$ 0.38 pp |

The four one-seed rows do not vary monotonically in either metric. Relative to $K=1$, the $K=50$ row has 3.3% lower RMSE and 2.09 points higher mean SR; taskwise descriptive changes range from +28.7 points on TurnOnSinkFaucet, +12.7 on OpenDrawer and CoffeeSetupMug, to $-13.3$ on CloseSingleDoor and $-8.0$ on OpenSingleDoor. Because parameters and temporal compute are not jointly matched, the table neither identifies an optimal token count nor attributes the differences to information retention, temporal processing, or optimization.

### E.4 Visual width and action-decoder learning rate

At the 218.8M-parameter reference, the visual-width row changes only the final CNN stage from 768 to 1,024 channels (approximately 226.5M total) while retaining $128\times 128$ inputs and a $4\times 4$ feature grid. The two LR rows change only the action-decoder peak rate; encoder and temporal-backbone rates remain $3.0\times 10^{-4}$.

*Table 19: Single-seed visual-width and action-decoder-LR diagnostics at the 218.8M-parameter, $D=300$ anchor. Each row changes one field from the reference. RMSE and SR follow Table 18.*

| Variant | CNN final width | Decoder LR | RMSE | SR |
|---|---|---|---|---|
| 218.8M reference | 768 | $3.0\times 10^{-4}$ | 0.13171 | 48.12% $\pm$ 1.70 pp |
| CNN terminal-width variant | 1,024 | $3.0\times 10^{-4}$ | 0.12917 | 48.55% $\pm$ 1.33 pp |
| Decoder LR $0.5\times$ | 768 | $1.5\times 10^{-4}$ | 0.12801 | 50.84% $\pm$ 1.02 pp |
| Decoder LR $2\times$ | 768 | $6.0\times 10^{-4}$ | 0.13398 | 50.35% $\pm$ 0.40 pp |

Widening the last CNN stage changes mean SR by 0.43 points and RMSE by $-1.9\%$; it does not test higher spatial resolution or additional views. The LR rows change mean SR by 2.72 and 2.23 points while moving RMSE in opposite directions. These one-seed values document recipe sensitivity only and do not select a generally better visual design or learning rate.

### E.5 Action distribution and decoding rule

This diagnostic retains the 85.3M-parameter encoder and temporal backbone while replacing diffusion with a single-forward decoder over the full $H=10$ chunk. One arm predicts a diagonal Gaussian and executes its mean. The other predicts a five-component trajectory mixture and either samples a component or selects the highest-probability component before executing its mean, without within-component noise. Both direct heads minimize exact negative log likelihood over the 120 action scalars; that training loss is not comparable to diffusion noise-prediction loss.

*Table 20: Single-seed action-decoder diagnostic at the 85.3M-parameter, $D=300$ anchor. “Evals” is the number of decoder network evaluations per generated action chunk. Direct-decoder SR uses one complete fixed-protocol execution; the diffusion reference reports three. RMSE matches each row’s rollout decoding rule.*

| Decoder and rule | Params | Evals | RMSE | SR |
|---|---|---|---|---|
| Diffusion, stochastic DDPM | 85.3M | 20 | 0.13680 | 47.48% $\pm$ 1.14 pp |
| Diagonal Gaussian, mean | 85.27M | 1 | 0.18259 | 26.09% |
| Trajectory GMM, sampled-component mean | 85.29M | 1 | 0.15252 | 44.00% |
| Trajectory GMM, argmax-component mean | 85.29M | 1 | 0.14914 | 41.74% |

In the reported seed-0 executions, Gaussian-mean, sampled-mixture, and argmax-mixture decoding are 21.39, 3.48, and 5.74 SR points below the diffusion reference. Because each direct rule has one rollout and one training seed, these values do not establish a decoder ranking, show that diffusion is necessary, or support a component-level attribution.

## Appendix F Supporting analysis for offline action fitting and closed-loop success

This section supplies the sweep-level comparison, metric definition, and repeatability audit used by Section 6. It does not add an independent model-selection claim.

### F.1 Sweep-level offline-to-closed-loop metric analysis

The RoboCasa sweep gives a clear global relation between expert-action fitting and closed-loop performance. All 40 comparable adjacent data increases reduce full-chunk stochastic holdout RMSE and improve mean SR with a ten-step context. This statement uses 49 recovered-exact aggregate RMSE cells and the one archived-only cell identified in Appendix G. All 10 same-data comparisons between the 44.3M- and 218.8M-parameter models have the same direction. Figure 8 shows this descriptive association across all 50 trained policies.

Coarse agreement does not make RMSE a precise ordering for arbitrary nearby checkpoints. The separately trained context-length comparison provides a direct counterexample. From a five-step to a ten-step training context, RMSE decreases for both training seeds and the two-seed mean falls from 0.138384 to 0.132404; SR decreases for both seeds and its mean falls from 50.74% to 48.25%. Across tasks, RMSE improves on 20/23, whereas SR decreases on 15, is unchanged on one, and increases on seven. Longer context therefore improves expert-conditioned action fidelity without inducing a monotonic closed-loop ranking.

The high-data endpoint further exposes unequal conversion. From $D=1000$ to $D=2900$, all 10 matched training-seed/model-size comparisons improve in both metrics. The saved pre-loss aggregate, computed before the endpoint’s full-precision value was lost, reports an average RMSE decrease of 17.10% with a 1.76-point sample standard deviation. Recomputing from the machine-readable ledger, whose sole archived endpoint is retained only as 0.086, gives 17.12% and 1.79 points; this rounding-level difference reflects the unavailable higher-precision endpoint. SR is fully reconstructible and rises by 3.39 points on average with a 2.12-point sample standard deviation. At task level, RMSE improves in all 207 comparisons with recovered terminal records; the remaining 23 belong to the unavailable seed-1, 1.49B endpoint. Two surviving records disagree by one task-level SR comparison: the saved pre-loss manuscript reports 157 increases, 10 ties, and 63 decreases, whereas the later reconstruction record reports 158 increases, 10 ties, and 62 decreases. The repaired taskwise output needed to adjudicate that cell is unavailable, so both triplets are archived-only. Either supports only the same descriptive point that closed-loop conversion is heterogeneous across tasks; no conclusion depends on the one-cell difference. Relative RMSE changes and absolute SR points have different units and aggregation, so the claim is not a numerical comparison of their magnitudes. The single-anchor temporal-placement table provides another descriptive RMSE–SR mismatch, but does not support an architecture ranking (Appendix E.1).

The reported holdout RMSE measures sampled action-chunk similarity on fixed expert observations and histories. Closed-loop SR instead evaluates trajectories induced jointly by the policy, simulator, controller, termination rule, and success predicate. Two gaps help organize this difference without uniquely identifying its causal mechanism.

The first is limited expert-state coverage. After a small policy error, later images, proprioception, object poses, and contacts differ from the expert prefix. A model can become increasingly accurate on recorded trajectories without learning how to return to a successful region after deviation—the classic covariate-shift and recovery-coverage problem of behavior cloning (43). Nor is the recorded action necessarily the only valid target: a state may permit different approaches, contacts, timings, or recovery paths. The RoboCasa generation pipeline uses 50 human demonstrations per task to guide MimicGen in producing roughly 3,000 trajectories (37). More trajectories densify this generated expert distribution but do not automatically provide policy-induced recovery states. Failure-state data, recovery demonstrations, or on-policy data collection would be needed to identify this mechanism.

The second concerns closed-loop execution and the success metric. Continuous sampled actions pass through receding-horizon execution, controllers, contact dynamics, termination, and a discrete predicate before becoming SR. The Blocks Ranking diagnostic in Appendix D gives one bounded example: the archived standard-loss record associates a millimeter-scale difference in final button depression with a different binary outcome while sorting remains largely intact. Surviving RoboCasa records identify TurnOffMicrowave episode 38 (environment seed 116038) as a one-step success, but disagree about its interpretation. The saved manuscript states that reset is not already successful and that the policy action is nonzero; a separate internal analysis record reports that the predicate is true before any policy action and remains true after a zero action. Because the underlying audit artifact is absent, we retain the episode identity and one-step outcome as archived-only but use neither reset/zero-action interpretation (Appendix G). A single such episode can contribute at most $1/1150\approx 0.087$ percentage points to an aggregate and cannot by itself explain multi-point model differences. These examples illustrate only that SR is a joint endpoint rather than a direct scalar readout of model-output quality.

The practical implication is limited to the present target-domain BC setting. RMSE provides a dense diagnostic on fixed expert-trajectory crops and has smaller cross-training-seed relative range than SR in this grid, whereas repeated SR remains necessary for closed-loop capability. Large consistent changes in both are complementary evidence. When lower RMSE stops converting locally into higher SR, attribution to state support, recovery, control, or the evaluator is more informative than another high-resolution scalar ranking.

*Figure 8: Holdout stochastic RMSE versus closed-loop SR for all 50 trained policies. The $x$-axis is task-averaged RMSE over all 12 action dimensions and all ten steps in each sampled action chunk; the $y$-axis is mean SR over three complete evaluations with a ten-step context. Color, marker, and fill encode demonstrations per task $D$, capacity, and training seed. Error bars are fixed-seed repeatability sample standard deviations (Section 3.1). One aggregate RMSE cell is archived-only (Appendix G). Descriptive Spearman correlation is $\rho=-0.980$, summarizing the sweep-level trend rather than ranking nearby checkpoints.*

### F.2 Formal view of offline fitting and closed-loop success

Let the finite held-out expert-trajectory distribution be $p_{E}^{\mathrm{ho}}(\tau)$. The reported holdout RMSE compares fixed-seed sampled ten-step, 12-dimensional action chunks with recorded expert actions on a fixed crop registry, not over exhaustively enumerated trajectories. For task $m$, the registry contains eight ten-timestep sequence crops from each of its 100 held-out demonstrations. Let $\mathcal{Q}_{m}$ be the indexed collection of all pairs $(o_{\leq t},a_{t:t+H-1})$ within those crops for which the complete $H$-step target is valid. Then

|  | $$ \mathrm{RMSE}_{m}=\Biggl[\frac{1}{\lvert\mathcal{Q}_{m}\rvert\,Hd_{a}}\sum_{(o_{\leq t},a)\in\mathcal{Q}_{m}}\left\|\hat{a}(o_{\leq t})-a\right\|_{2}^{2}\Biggr]^{1/2},\qquad\mathrm{RMSE}_{E}=\frac{1}{M}\sum_{m=1}^{M}\mathrm{RMSE}_{m}, $$ |  | (23) |
|---|---|---|---|---|---|

with $M=23$ tasks and $\hat{a}(o_{\leq t})$ one stochastic chunk per query under eval_seed=0. The deterministic variant uses the same initial $x_{T}$ and the posterior-mean reverse transitions defined in Appendix A. The inner factor averages over all scalar action elements of the sampled task registry before the square root, and tasks are then weighted equally. The metric measures action similarity under held-out expert observation and history conditions. WorldToken optimizes diffusion noise-prediction loss rather than action RMSE itself, so RMSE is an offline behavior-cloning diagnostic, not the direct training objective.

Closed-loop evaluation acts on a different object. Under initial-state distribution $\rho_{0}$ and environment/evaluation protocol $\mathcal{M}$, the policy and environment induce

|  | $$ p_{\pi,\mathcal{M}}(\tau\mid\rho_{0}). $$ |  | (24) |
|---|---|---|---|

Define the protocol-specific accepted set

|  | $$ T^{\mathrm{succ}}_{\mathcal{M}}=\{\tau\mid E_{\mathcal{M}}(\tau)=1\}. $$ |  | (25) |
|---|---|---|---|

Then

|  | $$ \mathrm{SR}(\pi;\mathcal{M},\rho_{0})=\Pr_{\tau\sim p_{\pi,\mathcal{M}}(\cdot\mid\rho_{0})}\bigl[\tau\in T^{\mathrm{succ}}_{\mathcal{M}}\bigr]. $$ |  | (26) |
|---|---|---|---|

This set may contain expert paths, other valid solutions, recovery paths, and, under a permissive boundary, very early successes after few environment steps.

At least three gaps separate offline comparison on $p_{E}^{\mathrm{ho}}$ from closed-loop success on $p_{\pi,\mathcal{M}}$. First, offline RMSE uses expert observations and histories, whereas rollout observations are induced by earlier policy actions. Second, finite expert data record only a subset of action targets and behavior modes, while $T^{\mathrm{succ}}_{\mathcal{M}}$ may permit alternatives and recoveries. Third, continuous sampled actions pass through receding-horizon execution, control, contact dynamics, termination, and the evaluator before becoming a binary outcome.

### F.3 Metric stability

To describe dispersion across different units, define relative range

|  | $$ R_{\mathrm{rel}}(x)=\frac{\max_{i}x_{i}-\min_{i}x_{i}}{\bar{x}}. $$ |  | (27) |
|---|---|---|---|

For the three ten-step-context repeats of each of the 50 trained policies, the checkpoint, 1,150 episodes, environment seeds, and rollout seed are fixed. These are execution repeatability checks, not independent rollout-seed samples. All 150 evaluations contain 1,150 valid episodes and zero crashes. SR relative range has median 3.40%, third quartile 5.09%, and maximum 14.24%; corresponding absolute ranges are 1.39, 2.09, and 4.09 points. This scale covers many small differences among adjacent high-performing configurations.

Across training seeds, the same formula is applied to the two results for each paired combination of $D$ and model size. Across 25 pairs, RMSE relative range has median 0.76%, third quartile 1.14%, and maximum 2.26%; SR has 1.74%, 6.35%, and 24.78%. Separately, for 15 completed, non-continuation seed-0 training curves in the historical prefix04 audit, define terminal rebound as $100(\mathrm{RMSE}_{\mathrm{final}}/\min_{s}\mathrm{RMSE}_{s}-1)$, where the minimum is over logged checkpoints. Its median is 0.086% and maximum is 0.581%. This describes terminal-versus-best-logged-checkpoint sensitivity along those training curves; it is not an estimator-convergence test, a fixed-checkpoint repeatability test, or evidence that the RMSE estimator itself is stable, and it is not pooled with the repeat designs. The cross-seed summaries likewise compare trained policies rather than repeated estimates of one checkpoint, so these statistics do not isolate variability by component.

These dispersion summaries explain why the paper treats RMSE and repeated SR as complementary evidence at coarse scale, without using small local differences for fine-grained model selection.

## Appendix G Result provenance and recovery status

The original experiment server was compromised after the experiments were completed, and its primary run store was deleted. The present verification archive was reconstructed from off-server copies, exported summaries, repair logs, the saved pre-loss manuscript, and later reconstruction records. Table 21 maps the paper’s result families to these surviving sources; locations are relative to the recovered experiment archive, and surviving run directories retain their materialized configurations and metadata.

We use three recovery-status labels. Recovered-exact means that a number can be read from, or reaggregated from, a surviving terminal log or complete exported summary. Repair-log means that a post-repair registry or inventory preserves the exact value and a named completion record documents the repair or gap-fill, although a later mirror may have overwritten the terminal summary. Archived-only means that a value or diagnostic is retained in a saved manuscript or reconstruction record but the current archive is insufficient to reproduce it independently. These labels describe record recoverability, not statistical uncertainty.

For the primary RoboCasa sweep, 49 of 50 terminal full-chunk stochastic RMSE cells are recovered-exact. The seed-1, $D=2{,}900$, 1.49B-parameter aggregate RMSE is archived-only: its recovered holdout log ends at 70k of the scheduled 280k steps, so no terminal taskwise RMSE is imputed and task-level RMSE analyses exclude that endpoint. Statistics using this cell inherit its retained 0.001 precision. Of the 50 three-repeat SR cells, 45 are recovered-exact and five are backed by exact post-repair counts and named repair records. The machine-readable ledger in analysis/paper_metrics_full10_20260824 records source, status, and retained precision cell by cell. The task-level SR direction audit has two archived-only variants, 157/10/63 and 158/10/62; the missing repaired taskwise record prevents adjudicating their one-cell difference. The one-step evaluator episode is archived-only, and its conflicting reset/zero-action interpretations are not used. For Blocks Ranking, the modified-loss 95/100 result and the five history-intervention totals 95/92/59/38/28 are recovered-exact, whereas the detailed standard-loss aggregate and press-depth diagnostics in Appendix D are archived-only.

*Table 21: Compact provenance and recovery-status map for the experimental result families.*

|  |  |  |
|---|---|---|
| Result family | Frozen archive record | Scope verified |
| RoboCasa data registry | robocasa_mg23_scaling_keys_h100_seed0.json; manifest_sha256 (registry-body digest) 0ec5b509cf744121ff021e86170087000568bd67f2710d73b5588781cfa6b8d4; file SHA-256 368fb3c6b2038faad8e6b960234a21643c2b5eb1b02434380dc9a6094274081a | Recovered-exact: common 23-task train/holdout selection; independently sampled, potentially overlapping but non-nested $D=50/100/1000$ subsets; official $D=300$ subset; and full $D=2{,}900$ non-holdout pool. |
| Primary RoboCasa sweep | Phase1_5x5_wd0_newfamily/runs; Phase1_5x5_wd0_newfamily/runs_seed1; analysis/paper_metrics_full10_20260824 | RMSE: 49 recovered-exact cells and one archived-only aggregate. SR: 45 recovered-exact cells and five repair-log cells; every cell retains three exact 1,150-episode success counts. |
| Optimizer calibration | E0.1b_WD; E0.2_LR_lite_wd0_newfamily; E0.2_LR_lookup_wd0_newfamily; E0.2_LR_formula_wd0_newfamily; E0.2_LR_localopt_shared_encpred_wd0; E0.2_LR_wd0 | Recovered-exact: the 364.0M predecessor WD records and surviving LR rows. The matching 44.3M $(3/3/3)\times 10^{-4}$ measurement is unavailable and has no reported value; its contemporaneous recipe-adjudication record survives. |
| Local BC-Transformer | E0_baseline/runs/e0_bc_xfmr_mg23_d300_seed123_official500k_cuda4 | One seed (123), 500k-step native official-code recipe at batch 16 and weight decay 0.01; matched 23-task $D=300$ data and final evaluation registry; three complete executions giving 31.28% mean and 32.0% best. |
| RoboCasa context studies | E2_context; same-checkpoint rollout records under the primary-sweep directories | Separately trained $C\in\{1,2,5\}$ controls and fixed-checkpoint visible-history interventions. |
| RMBench Blocks Ranking | E4_rmbench; run tag leftdescentcorridor1p5to4mm_dirw3_shallow16; analysis/ranking_behavior_audit_20260810 | Recovered-exact: modified-loss 95/100 and the five same-checkpoint history interventions (95/92/59/38/28). Archived-only: standard-loss 13/100, matched outcome counts, and press-depth diagnostics. |
| Offline-to-closed-loop audits | Saved pre-loss manuscript/PDF; internal analysis-alignment record; later reconstruction record; surviving primary-sweep summaries | Archived-only: conflicting task-level SR variants 157/10/63 and 158/10/62, and conflicting interpretations of the one-step evaluator episode. Neither conflict is used for a numerical conclusion. Recovered-exact taskwise RMSE analyses exclude the missing seed-1, $D=2{,}900$, 1.49B endpoint. |
| Architecture diagnostics | E3_dp_like; E5_moredetail | Materialized configs, run records, terminal metrics, and three complete executions per reported seed for temporal placement and current-observation bypass. |
| Token-granularity diagnostics | E_multitoken_n2_d300; E_rawtoken_n2_d300 | 85.3M-parameter, $D=300$, seed-0 runs with 2, 4, and 50 temporal tokens per timestep; terminal holdout metrics and three complete executions per checkpoint. |
| Visual and optimization diagnostics | E7; E8 | 218.8M-parameter, $D=300$, seed-0 terminal-CNN-width and action-decoder-learning-rate variants, with terminal holdout metrics and three complete executions. |
| Action-decoder diagnostics | E9_actionhead | 85.3M-parameter, $D=300$, seed-0 diagonal-Gaussian and five-component trajectory-mixture decoders, with terminal holdout metrics and one complete execution per decoding rule. |

