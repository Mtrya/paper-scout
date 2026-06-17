# Hierarchical Advantage Weighting for Online RL Fine-Tuning of VLAs from Sparse Episode Outcomes

Tongyan Fang1,2 Siyuan Huang1† Naiyu Fang1,3 Ganlong Zhao1,3 Zhongjin Luo1,3 Jianbo Liu1 Xiaogang Wang1 Ying Dong2 B Hongsheng Li1,3 B

1ACE Robotics 2Shenzhen International Graduate School, Tsinghua University

3The Chinese University of Hong Kong

†Project leader BCorresponding authors

When pretrained VLA policies are fine-tuned through online RL, each rollout episode produces only a single binary outcome (success or failure), yet the actor update requires per-transition supervision. Existing approaches commonly reduce this sparse outcome to a single scalar reward or advantage signal, which conflates distinct forms of transition-level feedback and provides limited guidance once basic task success becomes achievable. First, a single scalar signal conflates the two objectives of viability and efficiency; once basic success is achieved, the binary label provides no gradient to distinguish efficient completions from slow ones. Second, real-world rollouts mix autonomous and intervention segments; naively assigning episode outcomes across these boundaries introduces incorrect credit assignment. To address these issues, we propose Hierarchical Advantage-Weighted Behavior Cloning (HABC), which trains separate critic heads for these two objectives on different data subsets and combines their outputs with a state-adaptive balance. A state-adaptive gate gt merges their one-step advantages, prioritizing viability when success is uncertain and shifting to efficiency only when viability is high, and converts the result into per-transition weights on the actor loss. Intervention-aware credit assignment further restricts outcome labels to segments executed by the current policy, preventing supervision from leaking across intervention boundaries. In real-robot experiments on three contact-rich bimanual tasks, HABC raises success from supervised fine-tuning (SFT) baselines of 36%, 44%, and 12% to 92%, 88%, and 38%.

Date: Jun 2026

Website: https://acerobotics-vla.github.io/HABC-Website Code: https://github.com/ACERobotics-VLA/HABC

Keywords: Vision-Language-Action Models, Online Reinforcement Learning, Robot Manipulation

## 1 Introduction

Pretrained Vision-Language-Action (VLA) policies [1, 2, 3, 4, 5, 6] have demonstrated remarkable generalization across diverse manipulation tasks, but demonstrations alone are often insufficient for reliable deployment. Supervised fine-tuning (SFT) is bounded by demonstration coverage, and covariate shift causes errors to compound at deployment [7, 8]. To correct the mistakes a policy actually makes, improve robustness beyond the level of teleoperation, and adapt to new deployment conditions, online RL fine-tuning is necessary where the policy must learn from its own experience [9, 10, 11, 12]. In practice, however, each episode produces only a single binary outcome, yet effective actor updates call for per-transition signals.

We observe that this sparse episode label encodes two separable layers of transition-level information: viability— whether the current state can still lead to task completion, and efficiency—given that success is reachable, whether the current transition is advancing toward completion or wasting time. Viability can be supervised from all outcomelabeled windows, while efficiency can only be estimated from successful trajectories. They are also informative at different training stages: viability dominates early when failures are common, while efficiency matters later when success rate is high. Existing methods such as Recap [12] collapse both into a single reward-derived advantage, and critic-filtered behavior cloning hard-selects transitions above a TD-advantage threshold; both approaches lose the structure that would make each signal useful at the right training stage.

![](images/49ba621356c81ec00530179f4cd2ede3dea1121eb5afdc4448f3a0ff6d87d7f7.jpg)  
Figure 1 Overview of Hierarchical Advantage-Weighted Behavior Cloning. HABC fine-tunes a VLA actor using SFT demonstrations and online rollouts with policy execution and human interventions. Given sparse episode outcomes, HABC converts rollouts into transition-level weights with a dual-head critic. The viability head $V _ { v }$ estimates whether a state can still lead to success, while the efficiency head $V _ { e }$ estimates progress toward faster completion. Their one-step advantages, $A _ { v }$ and $A _ { e }$ , are combined by a state-adaptive gate $^ { g _ { t } , }$ , emphasizing viability when success is uncertain and efficiency once viability is high. Intervention-aware credit assignment partitions rollouts by control authority, preventing outcomes from being credited to policy mistakes or human corrections. The resulting per-transition weight wi reweights the flow-matching imitation loss and updates the actor.

Beyond signal decomposition, a second challenge concerns data quality in mixed-control episodes. When a human intervenes mid-execution, the episode outcome reflects both policy execution and human intervention; naively assigning this outcome to all timesteps can upweight the policy mistakes that triggered the intervention or penalize the human’s corrective actions. Prior work acknowledges that not all online rollout data is equally reliable [12], but does not explicitly handle the control-authority boundary within an episode. Restricting outcome labels to policy execution segments provides cleaner supervision while still leveraging the human’s corrective actions as imitation data.

We address these two challenges with Hierarchical Advantage-Weighted Behavior Cloning (HABC, Fig. 1). A dualhead critic separates the two signals: a viability head trained on all outcome-labeled windows and an efficiency head trained on successful trajectories only, whose one-step advantages are combined through a state-adaptive gate into bounded per-transition weights on the actor loss. Intervention-aware credit assignment restricts outcome labels to policy execution segments, preventing supervision from leaking across control-authority boundaries.

The main contributions of this work are as follows:

• Hierarchical signal decomposition and dual-head critic. We show that a single binary episode outcome encodes two separable transition-level signals that require different data and are informative at different training stages. We operationalize this decomposition through a viability head $V _ { v }$ and an efficiency head $V _ { e }$ , whose one-step advantages are combined via a state-adaptive gate into bounded per-transition weights on the actor loss.

• Intervention-aware credit assignment. We restrict outcome labels to policy execution segments according to control authority, preventing credit leakage across intervention boundaries and ensuring clean supervision for both critic heads.

• Real-robot validation with open-source release. On three contact-rich bimanual tasks with deformable objects,

HABC improves success rates from 36%/44%/12% to 92%/88%/38% over SFT baselines. We will release code and data to facilitate reproducibility.

## 2 Related Work

Online RL fine-tuning. HIL-SERL [9] and ConRFT [10] use off-policy RL for robot manipulation, RIPT-VLA [11] applies PPO-style updates directly to VLA policies, iRe-VLA [13] iterates between RL exploration and supervised distillation, VLA-RL [14] uses trajectory-level RL with a process reward model for sparse-reward manipulation, and SimpleVLA-RL [15] scales VLA training via RL with a curriculum. These methods improve task performance through online interaction, but they rely on standard RL machinery and do not focus on how sparse episode outcomes should be converted into transition-level supervision, particularly when policy execution and human intervention are interleaved within episodes.

RL for generative action policies. ReinFlow [16], FPO [17], DPPO [18], and RFS [19] optimize generative policies via policy-gradient estimation through the generative sampling process, while IDQL [20] combines implicit Q-learning with diffusion policy extraction. ARFM [21] adaptively balances RL advantage preservation and flow-loss variance for offline post-training of VLA flow models. Policy-gradient methods require differentiating through the generative process, which can be sample-inefficient for high-dimensional flow-based actors. HABC takes an alternative approach: it trains critics online with TD learning and converts their outputs into per-transition weights on the supervised flowmatching loss, avoiding policy-gradient estimation entirely while still closing the learning loop through online interaction.

Intervention-based imitation learning. DAgger [7] and HG-DAgger [22] aggregate intervention actions as direct supervision. IWR [23] increases the weight of intervention data, Sirius [24] reweights samples using approximated human value judgments, and AIM [25] learns an adaptive criterion for requesting human demonstrations. RaC [26] scales recovery and correction data through human-in-the-loop rollouts for long-horizon tasks, and MILE [27] models the human intervention decision itself to improve data efficiency. These approaches treat intervention windows as data to imitate rather than asking how outcomes should be attributed across a mixed-control episode.

Advantage-weighted and advantage-conditioned actor updates. AWR [28] and AWAC [29] derive actor weights from critic advantage via exp(A/β ˆ ); IQL [30], CQL [31], and Decision Transformer [32] provide alternative policy extraction routes from value estimates or returns; AWM [33] analyzes weighted matching losses from a variancereduction perspective. In the VLA post-training setting, Recap [12] and its scalable system variant SOP [34] convert a reward-derived advantage into a prompt token that conditions the actor, gaining test-time controllability via classifierfree guidance; LWD [35] extends this to fleet-scale offline-to-online RL with distributed robot experience. HABC takes the complementary weighting route: the actor input is unchanged, and two outcome-label-derived heads produce bounded per-transition loss weights consumed only at training time, decomposing the sparse outcome into viability and efficiency signals.

## 3 Method

## 3.1 Problem Setup

At each step, the robot observes $s _ { t } = ( I _ { t } , q _ { t } , \ell )$ where $I _ { t }$ denotes multi-view images, qt the proprioception, and ℓ a language task prompt, and executes an action chunk $a _ { t }$ of horizon H. We call a maximal contiguous interval under one controller (current policy or human) a segment, a fixed-length training sample drawn from a segment a window, and the final policy execution segment after the last intervention a post-intervention policy execution suffix. For notational simplicity, we index each training window by its anchor step t. All methods share a flow-matching VLA actor trained with a weighted flow-matching loss [36, 37, 38, 39]:

$$
\mathcal { L } _ { \pi } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } w _ { i } \left\| v _ { \theta } ( s _ { i } , a _ { i } , \sigma ) - u _ { i } \right\| _ { 2 } ^ { 2 } ,\tag{1}
$$

where $v _ { \theta }$ is the flow-matching velocity, $u _ { i }$ is the ground-truth flow target, and $w _ { i }$ is a scalar weight derived from the viability value $V _ { v }$ and the efficiency value $V _ { e }$ (§3.2). For readability, Eq. (1) shows only the scalar transition weight; route-specific validity masks and intervention action-dimension masks are applied in the standard way (details in Appendix B). Online fine-tuning draws from three data sources: demonstrations $\mathcal { D } _ { \mathrm { S F T } } ;$ autonomous rollouts $\mathcal { D } _ { \mathrm { a u t o } }$ with episode outcome $y \in \{ 0 , 1 \}$ ; and human-intervention data ${ \mathcal { D } } _ { \mathrm { i n t } }$ . The core design question is how to set $w _ { i }$ so that viability and efficiency are extracted from sparse outcomes and routed to the correct data, without leaking credit across control-authority boundaries.

## 3.2 Dual-Head Critic

The scalar weight $w _ { i }$ in Eq. (1) must encode two distinct improvement signals that are informative at different training stages. Early in training, when failures are frequent, the key signal is whether an action keeps the task viable. Later, when success is reliable, the key signal shifts to whether an action advances efficiently. A single critic trained on episode outcomes conflates these two signals, losing the structure that makes each separately actionable at the right training stage. We therefore decompose the sparse outcome into two dedicated heads on the shared backbone $\phi ( s )$

$$
z _ { v } ( s ) = f _ { v } ( \phi ( s ) ) , \quad \hat { V } _ { e } ( s ) = f _ { e } ( \phi ( s ) ) , \quad p _ { v } ( s ) = \mathrm { s i g m o i d } ( z _ { v } ( s ) ) .\tag{2}
$$

$V _ { v } ( s )$ estimates the viability of state $s ,$ defined as $p _ { v } ( s ) = P ( { \mathrm { s u c c e s s } } \mid s )$ , the probability of eventual task success under the current policy. It is trained with binary cross-entropy against the episode outcome $y$ on all labeled policy execution windows. Because the label itself is binary, $V _ { v }$ can be supervised from both successful and failed episodes, making it informative even when the success rate is low.

$V _ { e } ( s )$ estimates the steps to success from state $s ,$ trained only on successful trajectories where this target is well-defined. Non-terminal actions receive a one-step cost of −1; the terminal success action is assigned target 0:

$$
y _ { e } ( s _ { t } ) = \left\{ \begin{array} { l l } { 0 , } & { d _ { t } = 1 , } \\ { - 1 + \mathrm { s g } [ \hat { V } _ { e } ( s _ { t + 1 } ) ] , } & { d _ { t } = 0 , } \end{array} \right.\tag{3}
$$

where $d _ { t } = 1$ denotes the terminal success step and sg[·] denotes stop-gradient. $\hat { V } _ { e }$ is a scalar regression output trained with Huber loss, with target values naturally bounded in $[ - T _ { \mathrm { m a x } } , 0 ]$ where $T _ { \mathrm { m a x } }$ is the episode step limit. As the policy improves and success becomes frequent, $V _ { v }$ saturates near 1 for most states; $V _ { e }$ then becomes the more informative ranking signal, distinguishing fast progress from slow progress.

$V _ { v }$ and $V _ { e }$ are trained jointly. $V _ { v }$ is supervised on all outcome-labeled policy execution windows, including both successful and failed windows whose outcome is attributable to the current policy. $V _ { e }$ is supervised only on successful windows, including successful policy-execution windows and successful intervention windows, because the steps-tosuccess target is not defined for failures. Using $\mathcal { D } _ { \mathrm { a u t o } } ^ { \mathrm { l a b } }$ for the former and $\mathcal { D } _ { \mathrm { s u c c } } = \mathcal { D } _ { \mathrm { a u t o } } ^ { \mathrm { s u c c } } \cup \mathcal { D } _ { \mathrm { i n t } } ^ { \mathrm { s u c c } }$ for the latter, the joint critic loss is

$$
\mathcal { L } _ { \mathrm { c r i t i c } } = \mathbb { E } _ { \mathcal { D } _ { \mathrm { a u t o } } ^ { \mathrm { l a b } } } [ \mathrm { B C E } ( z _ { v } , y ) ] + \mathbb { E } _ { \mathcal { D } _ { \mathrm { s u c c } } } [ \mathrm { H u b e r } ( \hat { V } _ { e } , y _ { e } ) ] .\tag{4}
$$

## 3.3 Advantage-Weighted Actor Update

The dual-head critic separates what should be learned from sparse outcomes, but the actor update in Eq. (1) requires a scalar weight for each training transition. We therefore turn each head into a local improvement signal: transitions that increase viability or make efficient progress are upweighted, while transitions that reduce viability or waste progress are downweighted. We compute one-step advantages for the two heads and combine them through a state-adaptive gate.

The viability advantage measures how much an action improves predicted viability:

$$
A _ { v } = z _ { v } ( s _ { t + 1 } ) - z _ { v } ( s _ { t } ) .\tag{5}
$$

We compute this difference in logit space to preserve resolution near $p _ { v } \approx 1$ . The efficiency advantage measures whether an action shortens the predicted steps to success faster than the one-step baseline:

$$
A _ { e } = - 1 + \hat { V } _ { e } ( s _ { t + 1 } ) - \hat { V } _ { e } ( s _ { t } ) .\tag{6}
$$

Positive $A _ { e }$ indicates the action advances progress beyond expectation; negative $A _ { e }$ indicates slower-than-expected progress. Both are one-step TD residuals: $A _ { v }$ uses zero per-step reward and measures the change in viability logit over one transition, while $A _ { e }$ incorporates the per-step cost $( - 1 )$ consistent with the step-count supervision target in Eq. (3). HABC combines the two advantages through a state-adaptive gate: HABC combines the two advantages through a state-adaptive gate:

$$
g _ { t } = 1 + \operatorname { t a n h } \Bigl ( \bigl ( 1 - p _ { v } ( s _ { t } ) \bigr ) A _ { v } + p _ { v } ( s _ { t } ) A _ { e } \Bigr ) .\tag{7}
$$

When $p _ { v }$ is low, the gate emphasizes $A _ { v } ,$ separating viable states from stuck ones. When $p _ { v }$ is high, the gate emphasizes $A _ { e } ,$ ranking actions by efficiency inside already-viable trajectories. This interpolation happens per state rather than through a global training schedule: within the same batch, a low-viability state is weighted by viability improvement while a high-viability state is weighted by efficiency improvement.

## 3.4 Intervention-Aware Credit Assignment

When policy execution and human intervention appear in the same episode, the source of the final outcome is ambiguous: success may be caused by the policy, by the human correction, or by both. Naively crediting the outcome to all timesteps leaks supervision across the control-authority boundary. Concretely, if the episode succeeds after an intervention, naively crediting the outcome to all timesteps would upweight the pre-intervention policy execution segment that led to the near-failure state (reinforcing the mistakes that triggered the intervention); conversely, if the episode fails after an intervention, it would incorrectly penalize the human’s corrective actions.

Rather than requiring the critic to infer hidden causes from a binary episode label, HABC uses the logged control authority as the attribution boundary. The key observation is that the post-intervention policy execution suffix is the only segment whose outcome is attributable to the current policy: the pre-intervention segment led to a state requiring correction, and the intervention segment reflects human rather than policy decisions. Corrupting $V _ { v }$ with labels from these segments would cause the viability head to upweight the very states that triggered failures, undermining the weighting signal.

HABC therefore partitions each episode by controller. For fully autonomous episodes, the entire trajectory is labeled with $y .$ For episodes with one or more interventions, only the post-intervention policy execution suffix receives the outcome label. Intervention windows are never outcome-labeled; instead they serve a dual role: imitation supervision for the actor and progress targets for $V _ { e } ,$ leveraging the human’s corrective actions as demonstrations without attributing the episode outcome to them (see Appendix C for an illustration).

## 3.5 Training Procedure

We set the pre-normalization weight $\tilde { w } _ { i }$ according to data source: $\tilde { w } _ { i } = g _ { t }$ for successful online rollout samples, $\tilde { w } _ { i } = 0$ for failed online rollout samples, and $\tilde { w } _ { i } = 1$ for SFT and intervention samples; when intervention reweighting (IR) is enabled, $\tilde { w } _ { i } = g _ { t }$ for intervention samples instead. The normalized weight $w _ { i }$ in Eq. (1) is then obtained by unit-mean normalization over valid samples, decoupling the weighting from the effective learning rate:

$$
c = \frac { 1 } { | \mathcal { D } _ { \mathrm { v a l i d } } | } \sum _ { i \in \mathcal { D } _ { \mathrm { v a l i d } } } \tilde { w } _ { i } , \qquad w _ { i } = \tilde { w } _ { i } / \operatorname* { m a x } ( c , \varepsilon ) .\tag{8}
$$

A warmup of $N _ { \mathrm { w u } }$ steps keeps $g _ { t } \ = \ 1$ until the critic heads are minimally calibrated. Intervention reweighting is enabled only after an initial HABC phase, once $V _ { v }$ has been trained from labeled policy execution windows collected during that phase.

## 4 Experiments

## 4.1 Experimental Setup

Tasks. We evaluate on three real-robot dual-arm manipulation tasks (Figure 2): Pencil Pouch, in which the robot inserts a marker and zips a soft pouch closed; Paper Bag, in which the robot opens a flat-folded bag, stands it upright, and inserts a bottle; and Snack Bag, in which the robot sequentially places three items into a pouch and pulls the

penouch

![](images/06f81d11a7004dfbf1750f34336fb65abd8393a1d4fe961875a1016c1d068a7f.jpg)  
Retrieve Pouch

![](images/cfd8a7a5b79e989a6b2e95dfbda51d74c46f337ed37273fc8f3becfee0a27dae.jpg)  
Retrieve Pencil

![](images/6e0318b298c23cf2caf31f20d9684587700219dbc4a73cee6b119f48bfb2edf1.jpg)  
Insert Pencil

![](images/e80db6d43fc664e80065c16ca0d00ede907bf0bcb87c8f8a036b744c68bcf456.jpg)  
Flip Pouch

![](images/d81b3f5ac06528833eb08e4e5cd3c09a78cf02d08289666697c69e04f246b1e1.jpg)  
Flip Pouch

![](images/08c17a548b75cceb2272c871364b69c7fb30004da0afb8879fb437fef1fc44e9.jpg)  
Zip Pouch

![](images/d2542d5731df2930835607b435cdd06d4090914770a90e6160092caf83fb6609.jpg)  
Task Complete  
paperbag

![](images/771b9eb3dcfa6d72ff2b1d65082adc1c402882f0f884d4068503dcc2995179a3.jpg)  
Retrieve Bag

![](images/97e71bcc22d5a2ff43071ff4ec1158a137f0df82bb75802c14eeb62a4c47768a.jpg)  
Open Bag

![](images/16b5c8d977702f4f1180dde7476833fddb143b986af1ec1d4066c8c49d3a3fa6.jpg)

![](images/ca36b155ad33384a79c0d6f51ec3b58a735347d0bdff5c071d0744af90cf3bba.jpg)

![](images/b400a6ed3b367dff9526b27c40777308b0010454af6150353264e1733973006a.jpg)  
Flip Bag

![](images/f0855e12fbf2673a353ec82148f4d0690e3a2c4166da6eec9bdd22914cab3f51.jpg)  
Flip Bag  
Insert Water

![](images/4a01998359386c0ba74687d9bbf42be098e2cb2544e7e7b10960e361a41d4a85.jpg)

![](images/316ee0f2eada6065c63f01ad7c7daf618ccc2e5433c4f1cb4af76396d729dc04.jpg)  
Task Complete  
snackbag  
Place Upright  
Retrieve Bag

![](images/d4267809c79e356edad8b21d4645da37757670759dd6b778bad6e01569bffde8.jpg)

![](images/00630999243f7358eb6fb48eb308fd6767e0b227b137d74d6aa0c8b13223a63c.jpg)  
Place Water

![](images/1fbe2e8e946664ade50d4f349647d1b29887abf777833babb9beccbf71ae6ab3.jpg)  
Place Snack

![](images/48231eba43ee17b3d82d47ba0fd2bc8fb39bd155f0299b0e9b82f5583828594d.jpg)  
Bag Tidy

![](images/e7922e9234e6105a883cfe5de153f082fdf8920892a54b42b65a7ee285e138b6.jpg)  
Close Bag

![](images/30ed0486180179c9eab93d80c6cc41eb9a57da93c2c32705d0d80b95cd21531c.jpg)  
Task Complete

Figure 2 Real-robot bimanual manipulation tasks. We evaluate on three dual-arm tasks involving deformable objects: Pencil Pouch, Paper Bag and Snack Bag.

drawstring closed. All three involve multi-stage bimanual coordination on deformable objects where partial progress and interventions are common.

Implementation. All experiments are conducted on an ARX X5 bimanual robot. Observations consist of three RGB camera streams: a top Intel RealSense D455 and two wrist Intel RealSense D405 cameras. The action space is the robot’s end-effector frame with a chunk size of 50 during training and 25 during inference. We use π0.5 [39] as the base VLA, initialized from its pretrained offline checkpoint, and fine-tune on 8×A800 GPUs. Full hyperparameter details are in Appendix F.

Baselines. We compare five methods: SFT (no online data); Imit-DAgger (50/50 SFT and intervention mix, no rollout data); Imit-Recap (hard-threshold filtering on the critic’s TD residual, adapted from [12]); HABC-V (Vv only, Ve ablated); and HABC (full method). Imit-DAgger follows the intervention-imitation recipe of HG-DAgger [22]; Imit-Recap adopts the hard-threshold filtering mechanism from Recap [12] but omits its advantage-conditioned actor prompt, making it a baseline without intervention-aware credit assignment that uses a single critic’s TD residual. The clean ablation of each factor is captured within the HABC variants: HABC-V isolates the contribution of the efficiency head, while the comparison between Imit-Recap and HABC-V highlights the effect of soft dual-head weighting with intervention-aware credit assignment.

Metrics and training protocol. Each checkpoint is evaluated over 50 trials; we report success rate and mean trajectory length (number of action frames on successful trials only). For the Step 1 comparison, all online methods are initialized from the same SFT checkpoint and trained with an equal online fine-tuning budget; SFT itself receives no online data and serves as the baseline. Step 2 starts from the HABC checkpoint and evaluates continued training with and without intervention reweighting. We additionally report the best checkpoint reached by continuing HABC+IR training for more online rounds, reflecting the full potential of our method given additional interaction data.

## 4.2 Main Results

Figure 3 summarizes Step 1, the equal-budget initial online fine-tuning comparison. HABC achieves the highest success rate on all three tasks (60%, 78%, 22%). A viability-only variant (HABC-V) already surpasses all non-HABC baselines on every task, suggesting that soft viability weighting extracts more useful supervision from sparse outcomes than hard filtering. Imit-DAgger underperforms SFT on Pencil Pouch and Snack Bag because the 50/50 mix biases training toward intervention states.

Trajectory length complements success rate by measuring efficiency only among successful trials; it does not conflate efficiency with failure rate. Upgrading from HABC-V to full HABC consistently reduces trajectory length on every task (−55, −162, −32 frames), suggesting that the efficiency head downweights unproductive motion and favors more direct completions.

![](images/79a3b4ba750237b8fa33998ba4dd78f2b01d5b517f483e0fb585b62fd3a190e5.jpg)  
Figure 3 Main results across three tasks. Top: success rate (%) with Wilson 95% confidence intervals. Bottom: mean trajectory length, measured as number of action frames on successful trials, with standard deviation. Methods are grouped into Step 1, initial online fine-tuning with 5 methods; Step 2, continued training ± intervention reweighting with 2 methods; and the best observed HABC+IR checkpoint. Shorter trajectories indicate faster task completion.

Step 2 starts from the HABC checkpoint and compares continued training with and without intervention reweighting. HABC (cont.) continues without intervention reweighting, while HABC+IR additionally applies $g _ { t }$ to intervention windows. Following the training procedure in Section 3, intervention reweighting is enabled only after the initial HABC phase. With multiple additional rounds under HABC+IR, the best checkpoints reach 92% on Pencil Pouch, 88% on Paper Bag, and 38% on Snack Bag, up from SFT baselines of 36%, 44%, and 12%.

## 4.3 Critic and Weight Analysis

Value head generalization. The viability head $V _ { v }$ provides more than a memorized lookup of training labels (Figure 4). For Pencil Pouch, we evaluate $p _ { v }$ at each observed initial state, parameterized by the pouch-center position on the workspace; each point represents one episode’s initial pouch-center location, and we retain only those where $p _ { v } \ >$ 0.6. As online fine-tuning proceeds, the set of pouch-center positions satisfying $p _ { v } ~ > ~ 0 . 6$ expands progressively (Figure 4, left), indicating that the viability head assigns high viability to an increasingly broad range of observed initial placements. Trajectory-level traces (Figure 4, right) show the same behavior locally: $p _ { v }$ drops sharply when the policy enters an out-of-distribution state and fails to grasp, then recovers steadily during a human intervention that re-establishes a viable grasp. The real-time tracking of viability across both autonomous and intervention segments suggests that $A _ { v }$ provides a useful per-transition viability signal for intervention reweighting.

![](images/dd1283b0a17ae769ed2cacf0a0e9dad0c44138d007eef7216c933c2c1f16a48c.jpg)  
Figure 4 Viability head generalization on Pencil Pouch. Left: each point is one episode’s initial pouch center position; only positions where $p _ { v } ~ > ~ 0 . 6$ are shown. Crosses mark the pouch center. As training progresses, the high-viability region expands, indicating that the viability head assigns high viability to an increasingly broad range of observed initial placements. Right: along a rollout, $p _ { v }$ drops after an OOD grasp failure and recovers during human intervention. This shows that $A _ { v }$ tracks local changes in viability across both autonomous and intervention segments (Section 3.3).

Per-transition weight analysis. Figure 5 illustrates how the two heads produce non-uniform weights along a real rollout. Three highlighted segments demonstrate the division of labor in Eq. (7). At a successful second grasp shown in pink, $p _ { v }$ rises sharply and $A _ { v }$ is large, so the gate upweights the transition through the viability term. During a confused regrasp shown in yellow, $p _ { v }$ stays high and changes little so $A _ { v }$ is uninformative, but $\hat { V } _ { e }$ signals stalled progress, causing $A _ { e }$ to turn negative and the $p _ { v } A _ { \epsilon }$ branch to downweight the inefficient segment. During recovery actions shown in green, both signals climb together and the gate upweights through both terms. This complementary behavior, where $V _ { v }$ reacts to discrete viability-changing events while $V _ { e }$ resolves gradations within high-viability regions, helps explain the empirical improvement of HABC over HABC-V in Figure 3.

![](images/7f7e3070761467ec69717f4c2e5eb6b3a076eac3b263c6bf5d0c485aa86e1eb2.jpg)  
Figure 5 Viability and efficiency signals along a Pencil Pouch rollout. Frames at three highlighted segments (pink: successful grasp; yellow: inefficient regrasp; green: recovery) and the corresponding pv(st) (blue dashed) and normalized $\hat { V } _ { e } ( s _ { t } )$ (black solid) over the episode. At a successful grasp, $p _ { v }$ rises sharply and the transition is upweighted through $A _ { v }$ . During inefficient motion, $p _ { v }$ stays high while $\hat { V _ { e } }$ worsens, so the segment is downweighted through $A _ { e } .$ . During recovery, both signals improve and the transition is upweighted. Efficiency rescaled to [0, 1] for display.

## 4.4 Recovery Behavior Comparison

![](images/ec7f5c0a4d54e2ce4899640b27af975eeda54187c73ed95ac00395c7871dd734.jpg)  
Figure 6 Recovery behavior: HABC vs. SFT baseline. Each row shows a task where the robot encounters a manipulation failure. Top: the SFT baseline fails to recover and the episode terminates unsuccessfully. Bottom: the HABC-trained policy detects the error and executes corrective actions, ultimately completing the task. The dual-head critic’s viability weighting encourages the policy to learn from recovery transitions, enabling autonomous error correction without human intervention.

Figure 6 presents qualitative rollout comparisons between the SFT baseline and the HABC-trained policy on three representative failure-recovery scenarios. In each case, the robot encounters a manipulation error (e.g., a failed grasp, a misaligned insertion, or a dropped object). The $\mathrm { { s F T } }$ policy, lacking exposure to recovery states during training, either repeats the failed action or enters an unrecoverable loop. In contrast, the HABC policy—trained with viabilityweighted transitions that upweight recovery-oriented actions—detects the failure state and executes corrective motions to re-establish task progress. These examples complement the quantitative viability-head analysis in Section 4 by showing that the learned weighting translates into observable recovery behavior at deployment.

## 5 Conclusion and Limitations

We presented HABC, an online RL fine-tuning method for VLAs that converts sparse episode outcomes into pertransition behavior-cloning weights via a dual-head critic and intervention-aware credit assignment, leaving the deployed actor unchanged. The viability head enables learning even when success is rare; the efficiency head reduces trajectory length once success is reliable; and restricting outcome labels to policy execution segments prevents credit leakage across intervention boundaries while preserving human corrections as imitation data. On three contact-rich bimanual tasks, HABC raises success rates from SFT baselines of 36%/44%/12% to 92%/88%/38%, with qualitative evidence of learned recovery behavior in Appendix 4.4.

Intervention-aware credit assignment assumes reliably detected intervention boundaries; noisy labels would corrupt Vv supervision. $V _ { e }$ trains only on successful trajectories, so its signal is weakest precisely when success is rare. HABC is currently evaluated on single-task fine-tuning; extending the dual-head design to multi-task or cross-embodiment settings remains an open direction. In future work, adaptive gating, multi-step advantage estimation, and denser outcome signals for contact-rich recovery are natural next steps to further refine sparse-reward credit assignment.

## References

[1] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[2] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

[3] O. Mees, D. Ghosh, K. Pertsch, K. Black, H. R. Walke, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, et al. Octo: An open-source generalist robot policy. In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024.

[4] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[5] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164, 2024.

[6] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, volume 2025, pages 29982–30009, 2025.

[7] S. Ross, G. Gordon, and D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.

[8] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation. arXiv preprint arXiv:2108.03298, 2021.

[9] J. Luo, C. Xu, J. Wu, and S. Levine. Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning. Science Robotics, 10(105):eads5033, 2025.

[10] Y. Chen, S. Tian, S. Liu, Y. Zhou, H. Li, and D. Zhao. Conrft: A reinforced fine-tuning method for vla models via consistency policy. arXiv preprint arXiv:2502.05450, 2025.

[11] S. Tan, K. Dou, Y. Zhao, and P. Krähenbühl. Interactive post-training for vision-language-action models. arXiv preprint arXiv:2505.17016, 2025.

[12] P. Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo, et al. π0.6: a vla that learns from experience. arXiv preprint arXiv:2511.14759, 2025.

[13] Y. Guo, J. Zhang, X. Chen, X. Ji, Y.-J. Wang, Y. Hu, and J. Chen. Improving vision-language-action model with online reinforcement learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 15665–15672. IEEE, 2025.

[14] G. Lu, W. Guo, C. Zhang, Y. Zhou, H. Jiang, Z. Gao, Y. Tang, and Z. Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. arXiv preprint arXiv:2505.18719, 2025.

[15] H. Li, Y. Zuo, J. Yu, Y. Zhang, Z. Yang, K. Zhang, X. Zhu, Y. Zhang, T. Chen, G. Cui, et al. Simplevla-rl: Scaling vla training via reinforcement learning. arXiv preprint arXiv:2509.09674, 2025.

[16] T. Zhang, C. Yu, S. Su, and Y. Wang. Reinflow: Fine-tuning flow matching policy with online reinforcement learning. Advances in Neural Information Processing Systems, 38:106282–106319, 2026.

[17] D. McAllister, S. Ge, B. Yi, C. M. Kim, E. Weber, H. Choi, H. Feng, and A. Kanazawa. Flow matching policy gradients. arXiv preprint arXiv:2507.21053, 2025.

[18] A. Ren, J. Lidard, L. Ankile, A. Simeonov, P. Agrawal, A. Majumdar, B. Burchfiel, H. Dai, and M. Simchowitz. Diffusion policy policy optimization. In International Conference on Learning Representations, volume 2025, pages 77288–77329, 2025.

[19] E. Su, T. Westenbroek, A. Nagabandi, and A. Gupta. Rfs: Reinforcement learning with residual flow steering for dexterous manipulation. In The Fourteenth International Conference on Learning Representations, 2026.

[20] P. Hansen-Estruch, I. Kostrikov, M. Janner, J. G. Kuba, and S. Levine. Idql: Implicit q-learning as an actor-critic method with diffusion policies. arXiv preprint arXiv:2304.10573, 2023.

[21] H. Zhang, S. Zhang, J. Jin, Q. Zeng, Y. Qiao, H. Lu, and D. Wang. Balancing signal and variance: Adaptive offline rl posttraining for vla flow models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 18755–18763, 2026.

[22] M. Kelly, C. Sidrane, K. Driggs-Campbell, and M. J. Kochenderfer. Hg-dagger: Interactive imitation learning with human experts. In 2019 International Conference on Robotics and Automation (ICRA), pages 8077–8083. IEEE, 2019.

[23] A. Mandlekar, D. Xu, R. Martín-Martín, Y. Zhu, L. Fei-Fei, and S. Savarese. Human-in-the-loop imitation learning using remote teleoperation. arXiv preprint arXiv:2012.06733, 2020.

[24] H. Liu, S. Nasiriany, L. Zhang, Z. Bao, and Y. Zhu. Robot learning on the job: Human-in-the-loop autonomy and learning during deployment. The International Journal of Robotics Research, 44(10-11):1727–1742, 2025.

[25] H. Cai, Z. Peng, and B. Zhou. Robot-gated interactive imitation learning with adaptive intervention mechanism. arXiv preprint arXiv:2506.09176, 2025.

[26] Z. Hu, R. Wu, N. Enock, J. Li, R. Kadakia, Z. Erickson, and A. Kumar. Rac: Robot learning for long-horizon tasks by scaling recovery and correction. arXiv preprint arXiv:2509.07953, 2025.

[27] Y. Korkmaz and E. Bıyık. Mile: Model-based intervention learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 15673–15679. IEEE, 2025.

[28] X. B. Peng, A. Kumar, G. Zhang, and S. Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

[29] N. Ashvin, D. Murtaza, G. Abhishek, and L. Sergey. Accelerating online reinforcement learning with offline datasets. CoRR, vol. abs/2006.09359, 2020.

[30] I. Kostrikov, A. Nair, and S. Levine. Offline reinforcement learning with implicit q-learning. arXiv preprint arXiv:2110.06169, 2021.

[31] A. Kumar, A. Zhou, G. Tucker, and S. Levine. Conservative q-learning for offline reinforcement learning. Advances in neural information processing systems, 33:1179–1191, 2020.

[32] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch. Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34:15084–15097, 2021.

[33] S. Xue, C. Ge, S. Zhang, Y. Li, and Z.-M. Ma. Advantage weighted matching: Aligning rl with pretraining in diffusion models. arXiv preprint arXiv:2509.25050, 2025.

[34] M. Pan, S. Feng, Q. Zhang, X. Li, J. Song, C. Qu, Y. Wang, C. Li, Z. Xiong, Z. Chen, et al. Sop: A scalable online post-training system for vision-language-action models. arXiv preprint arXiv:2601.03044, 2026.

[35] Y. Wang, X. Li, P. Xie, P. Yang, B. Nie, Y. Cai, Q. Zhang, C. Qu, J. Wu, J. Song, et al. Learning while deploying: Fleet-scale reinforcement learning for generalist robot policies. arXiv preprint arXiv:2605.00416, 2026.

[36] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[37] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[38] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

[39] K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. R. Equi, C. Finn, N. Fusai, M. Y. Galliker, et al. π0.5: a vision-language-action model with open-world generalization. In 9th Annual Conference on Robot Learning, 2025.

[40] M. G. Bellemare, W. Dabney, and R. Munos. A distributional perspective on reinforcement learning. In International conference on machine learning, pages 449–458. Pmlr, 2017.

## A Parent Value Head

The pretrained value head $V _ { \psi }$ follows a distributional formulation [40]. It predicts logits over value bins and is trained by cross-entropy against a mixed TD/MC target:

$$
y _ { V } ( s _ { t } ) = ( 1 - c _ { t } ) { \big ( } r _ { t } + \gamma \operatorname { s g } [ V ( s _ { t + 1 } ) ] { \big ) } + c _ { t } R _ { t } ,\tag{9}
$$

Here $\begin{array} { r } { R _ { t } = \sum _ { k > 0 } \gamma ^ { k } r _ { t + k } } \end{array}$ is the Monte-Carlo return. The switch variable $c _ { t } \in \{ 0 , 1 \}$ chooses between the two targets: $c _ { t } { = } 1$ applies the MC target at an episode boundary, while ${ c _ { t } } \mathrm { { = } } 0$ uses the TD target elsewhere. During online finetuning, we continue updating $V _ { \psi }$ on $B _ { S } \cup B _ { O }$ , starting from the pretrained IQL checkpoint. This head is retained for compatibility with the Imit-Recap baseline, but it is not used in HABC’s actor weighting. The base VLA model is π0.5 [39].

## B Compared Update Rules

For completeness, we summarize the update rules used by HABC and the two imitation-style baselines below.

## B.1 HABC

Algorithm 1 One HABC update.   
Require: batches $B _ { S } , B _ { I } , B _ { O }$ (sampled from $\mathcal { D } _ { \mathrm { S F T } } , \mathcal { D } _ { \mathrm { i n t } } , \mathcal { D } _ { \mathrm { a u t o } } ) ;$ actor $\pi _ { \boldsymbol { \theta } } ;$ parent value head $V _ { \psi } ;$ ; critic heads $f _ { v } , f _ { e } ;$   
warmup $N _ { \mathrm { w u } } ;$ intervention-reweight flag IR ▷ $B _ { O } ^ { \mathrm { l a b } } = B _ { O } ^ { \mathrm { s u c c } } \cup B _ { O } ^ { \mathrm { f a i l } }$   
1: Update $V _ { \psi }$ on Eq. (9) over $B _ { S } \cup B _ { O }$ ▷ not used in actor weighting   
2: Update $f _ { v }$ using the minibatch estimate of the first term in Eq. (4) on $B _ { O } ^ { \mathrm { l a b } }$   
3: Update $f _ { e }$ using the minibatch estimate of the second term in Eq. (4) on $B _ { O } ^ { \mathrm { s u c c } } \cup B _ { I } ^ { \mathrm { s u c c } }$   
4: i $\mathbf { f } \mathrm { s t e p } \geq N _ { \mathrm { w u } }$ then   
5: Compute $g _ { t }$ from detached head outputs on $B _ { O } ^ { \mathrm { s u c c } }$ and, if IR, on $B _ { I }$   
6: else   
7: $g _ { t } \gets 1$ for all non-SFT samples   
8: end if   
9: Set $\tilde { w } _ { i } = g _ { t }$ on $B _ { O } ^ { \mathrm { s u c c . } } ;$ set $\tilde { w } _ { i } = 0$ on $B _ { O } ^ { \mathrm { f a i l } }$   
10: If $\mathrm { I R } ,$ set $\tilde { w } _ { i } = g _ { t }$ on $B _ { I } ;$ else set $\tilde { w } _ { i } = 1$ on $B _ { I }$   
11: Unit-mean-normalize non-SFT weights by Eq. (8)   
12: Update $\pi _ { \theta }$ on $B _ { S } \cup B _ { I } \cup B _ { O }$ with Eq. (1)

## B.2 Imit-Recap

Imit-Recap follows the hard-threshold filtering mechanism in Recap [12]. An online transition is included in the actor loss only when the critic’s TD residual exceeds a threshold ϵ, selected on a held-out validation set. This is not a full reproduction of Recap’s advantage-conditioned actor.

Algorithm 2 One Imit-Recap update.   
Require: batches $B _ { S } , B _ { I } , B _ { O }$ (as above); actor $\pi _ { \boldsymbol { \theta } } ;$ value head $V _ { \psi } ;$ ; threshold ϵ   
1: Update $V _ { \psi }$ on Eq. (9) over $B _ { S } \cup B _ { O }$   
2: for each autonomous rollout sample $i \in { \cal B } _ { O }$ do   
3: Compute $\hat { A } ( s _ { i } , a _ { i } ) = r _ { i } + \gamma ( 1 - d _ { i } ) V ( s _ { i } ^ { \prime } ) - V ( s _ { i } )$   
4: Set online weight $w _ { i } = \mathbb { 1 } \big [ \hat { A } ( s _ { i } , a _ { i } ) \ge \epsilon \big ]$   
5: end for   
6: Set SFT weights to 1   
7: Keep intervention weights unchanged under the intervention mask   
8: Update $\pi _ { \theta }$ on $B _ { S } \cup B _ { I } \cup B _ { O }$ with Eq. (1)

## B.3 Imit-DAgger

Imit-DAgger follows a simple intervention-imitation recipe. The actor is trained on a 50/50 mixture of SFT and intervention data, without using rollout transitions or critic-derived reweighting.

```powershell
Algorithm 3 One Imit-DAgger update.
Require: SFT batch $B _ { S } ;$ intervention batch $B _ { I } ;$ actor $\pi _ { \theta }$
1: Sample a 50/50 mixed actor batch from $B _ { S }$ and $B _ { I }$
2: Set SFT weights to 1
3: Set scalar intervention weights to 1; use $M _ { i } ^ { \mathrm { { i n t } } }$ as the per-dimension action mask
4: Update πθ on $B _ { S } \cup B _ { I }$ with Eq. (1)
```

## C Intervention-Aware Credit Assignment Illustration

![](images/36c36e8aceae45886a90bf943109cfa65542a54d256091acb7e2d1a962e124b7.jpg)  
Figure 7 Intervention-aware credit assignment and dual-value training. Supervision routing for the two value heads: Vv uses all outcome-labeled policy-execution windows to predict viability, while $V _ { e }$ uses only successful policy or intervention windows to predict progress/efficiency. Their advantages become transition-level actor weights. In online episodes, the source of the final success/failure label is uncertain, so naive episode-level supervision can assign credit to the wrong controller.

## Windowing Details

For each episode that contains an intervention, we split the trajectory into policy execution segments and intervention segments. If an episode contains an intervention, only the post-intervention policy execution suffix receives the episode outcome label. This suffix is the part executed by the current policy from a corrected state onward. Earlier policy execution segments are kept in the replay buffer but do not receive outcome labels. For fully autonomous episodes, the full trajectory receives the episode outcome label. An intervention window requires at least 10 human-controlled steps within a 50-step window. This intervention-aware split is used for both critic supervision and actor weighting.

## E Data Collection Protocol

Each task starts from 200 SFT demonstration episodes. Online fine-tuning then proceeds in rounds. Each round collects 100 autonomous rollout episodes and trains for 6k gradient steps. Among failed rollouts, approximately half receive human intervention, where the operator takes over and completes the task. The remaining failed rollouts are recorded as unassisted failures. Each round therefore adds 100 autonomous rollout episodes to the replay buffer, consisting of successes, episodes with intervention, and pure failures.

For Pencil Pouch, the initial-phase best checkpoint in Figure 3 is selected after 3 online rounds, corresponding to 300 rollout episodes on top of 200 SFT demonstrations. Continued training with HABC+IR then runs for additional rounds. The final HABC+IR checkpoint reaches 92% success. For Paper Bag, the same schedule applies: 3 initial rounds followed by continued HABC+IR rounds, with a final best checkpoint of 88%. For Snack Bag, we again use 3 initial online rounds followed by continued HABC+IR rounds, with a final best checkpoint of 38%.

## F Hyperparameters

Key constants are $C = 1 0 0$ (episode failure penalty; applied as reward $r ~ = ~ - C$ on failed-episode transitions for the parent value head TD update (Appendix A) and the Imit-Recap advantage computation—HABC’s actor weighting does not use this reward directly), $N _ { \mathrm { w u } } = 5 0 0$ (warmup steps), Huber $\delta = 1 . 0 , \gamma = 0 . 9 9$ , batch size $B = 2 5 6$ , and a stale-rollout cutoff of 10000 model indices. All HABC runs use pure TD supervision for $V _ { e }$ and keep action-expert training disabled. Online fine-tuning is initialized from the pretrained $\pi _ { 0 . 5 } \mathrm { I Q I }$ checkpoint.

## Weight Statistics

To verify that the HABC weighting rule produces meaningful variation rather than near-uniform weights, we report empirical weight statistics from the three tasks.

After warmup, the mean pre-normalization weight $g _ { t }$ on successful autonomous rollout transitions is approximately 0.76, 0.68, and 0.86 for Pencil Pouch, Paper Bag, and Snack Bag respectively. These values indicate that the weighting rule is non-trivial and varies across tasks.

For comparison, Imit-Recap passes approximately 22, 27, and 19 out of every 100 sampled autonomous rollout transitions for Pencil Pouch, Paper Bag, and Snack Bag respectively. The remaining transitions are discarded by the hard threshold. This filtering behavior is more aggressive than HABC’s soft weighting.

When intervention reweighting is enabled, the mean $g _ { t }$ on intervention windows is approximately 1.0, 1.1, and 0.95 for Pencil Pouch, Paper Bag, and Snack Bag respectively. On average, the critic therefore assigns near-uniform weight to intervention windows. Individual intervention transitions still receive non-uniform weights, but the distribution is more concentrated than for autonomous rollout data, consistent with the expectation that interventions are generally productive.

In the main text, mean trajectory length is reported only over successful evaluation trials. We use this metric as a direct readout of trajectory efficiency: once a policy can solve the task, fewer frames indicate less redundant motion and less recovery before completion. The consistent reductions from HABC-V to HABC in Figure 3 therefore support the intended role of $V _ { e }$