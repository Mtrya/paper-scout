# WorldCycle: Self-Verifiable Reinforcement Learning for Long-Horizon Video World Models

Bohai Gu<sup>1,</sup> , Yueyang Yuan<sup>2,</sup> , Taiyi Wu<sup>3</sup>, Dazhao Du<sup>1</sup>, Jian Liu<sup>1</sup>, Xiaoyi Pang<sup>1</sup>, Jie Zhang<sup>1</sup>, Xiaocheng Lu<sup>1</sup>, Haobin Zhong<sup>3</sup>, Xiaotong Zhao<sup>3</sup>, Alan Zhao<sup>3</sup>, Song Guo<sup>1,†</sup>

<sup>1</sup>The Hong Kong University of Science and Technology <sup>2</sup>Wuhan University <sup>3</sup>AI Technology Center, Tencent Video, Tencent

## Abstract

Interactive video world models are essential for long-horizon planning and exploration, yet they sufer from compounding errors. Post-training methods such as reinforcement learning (RL) can improve these models, but they hit a verification bottleneck: for arbitrary action sequences, no ground-truth future state exists to measure long-term drift. Our key insight is that reversible action cycles make this verification possible: a sequence composed with its inverse must analytically return to the initial state, yielding annotation-free supervision on long-horizon correctness. Building on this, we introduce WorldCycle, a self-verifiable RL framework that constructs closed action cycles and their repeated executions from ordinary action sequences, and optimizes two complementary rewards: a spatial closure reward enforcing symmetry between mirrored forward and reverse segments, and a temporal consistency reward aligning states across repeated cycle executions. These rewards force the model to learn actions as consistent state operators rather than memorized temporal patterns, and extend naturally to out-of-distribution composite action cycles that the base model handles poorly. We further release CycleBench, a diagnostic benchmark for state-returning ability under complex action structures. WorldCycle reduces state returning drift by up to 44% and lifts composite-action accuracy nearly 4× over the base model, providing a vital foundation for physically grounded world models.

## Introduction

Interactive video world models (IWMs) generate future visual observations conditioned on an initial observation and a stream of user or agent actions, turning generative video models into interactive simulators (Yang et al. 2024; Wu et al. 2024; Zhu et al. 2025; Xiang et al. 2024). Such models support embodied interaction, robot learning, visual planning, and game-like simulation. Related action-conditioned video world models have also been developed for autonomous driving (Hu et al. 2023; Wang et al. 2024; Gao et al. 2024; Zhang et al. 2025a). These models commonly employ an autoregressive generation paradigm: each newly generated observation becomes the context for predicting the next one. Consequently, small single-step errors can propagate and compound as the rollout extends, gradually eroding the physical and geometric consistency required by downstream longhorizon tasks (Bengio et al. 2015; Janner et al. 2019; Huang et al. 2026; Su et al. 2026).

Post-training methods based on reinforcement learning (RL) have recently emerged as a promising approach for improving video world models (Wu et al. 2025a; Ye et al. 2025). Yet existing approaches predominantly optimize short-horizon visual quality or per-step action alignment. The fundamental obstacle to long-horizon improvement is the absence of efective supervision. For an arbitrary action sequence, no ground-truth future state is available to measure accumulated error. The most recent work, WorldCompass (Wang et al. 2026a), attempts to address this by rewarding clip-level action following, but such objectives primarily improve local action accuracy and remain insuficient for correcting fine-grained spatial drift over long horizons.

We identify a new angle on this problem by focusing on a specific and ubiquitous class of dynamics: reversible trajectories such as camera ego-motion. For a reversible action cycle, physical law dictates that the final state must be identical to the initial state, providing an absolutely precise, annotationfree reference. Yet as shown in Figure 1, both WorldPlay and WorldCompass (Sun et al. 2026; Wang et al. 2026a) fail this minimal check: an inverse action sequence fails to recover the starting view (spatial closure failure), and the same action produces diferent displacements at diferent rollout positions (temporal consistency failure). These failures compound with horizon length and are invisible to any short-horizon reward. Crucially, the same physical prior that exposes the failure also resolves the supervision bottleneck: the cycle-return constraint provides an exact, label-free target for the long-horizon trajectory. Starting from this insight, we identify two remaining challenges: (i) reward sparsity, where an endpoint-only cycle signal provides too little gradient to localise intermediate errors; and (ii) temporal drift, where AR accumulation makes the same action behave differently at diferent rollout depths. The harder second level concerns composite actions such as moving forward while turning, which must also return after their inverse programs. Pre-trained baselines such as WorldPlay exhibit a 5× accuracy collapse on composite-action versus simple-action settings of identical rollout length (Table 1), indicating that composite motions are out-of-domain: pretraining data rarely provides GT video demonstrations of composite trajectories.

We propose WorldCycle, a Self-Verifiable Reinforcement Learning framework specifically designed to tackle statereturning failure in long-horizon video world models under reversible action trajectories. The guiding principle is to turn closed action programs into dense supervision. We exploit the mirror structure of an inverse action program: every partial forward trajectory has a corresponding reverse state that should coincide with it. We construct mirrored frame pairs at every intermediate depth of a cycle and compare them with dense visual evidence, yielding a spatial closure reward that converts each intermediate chunk into an independently verifiable closure check, matching the chunkwise structure of AR post-training and removing the need for a single sparse endpoint signal. We further repeat cycles and compare co-indexed frames across cycles, producing a temporal state consistency reward that penalises drift of identical actions over time, directly targeting the temporal drift obstacle. These two rewards force the model to learn actions as consistent state operators rather than memorized temporal patterns. The same cycle-based objective also applies to composite actions. Although the base model may not reliably execute such action combinations, WorldCycle can directly optimize them through cycle closure rewards, without requiring ground-truth video supervision.

![](images/b45b460118577e453ce500f23ee4958d0721d2c7fb17f2497d5a361caa07d669.jpg)  
Figure 1: State-returning failure in interactive video world models. Even state-of-the-art models fail a minimal sanity check: an inverse action sequence such as forward-then-backward does not recover the starting view (spatial closure failure), and executing the same action from what should be the same state at diferent rollout steps produces inconsistent displacements (temporal consistency failure). Both failures accumulate as autoregressive rollouts extend, yet neither is captured by short horizon rewards. WorldCycle turns closed action cycles into annotation-free supervision on long-horizon correctness.

We further introduce CycleBench, a benchmark suite covering inverse, repeated, and composite cycles, enabling video IWMs to be evaluated as simulators rather than only as shorthorizon generators. Because no existing benchmark diagnoses state-returning consistency, existing evaluations ask whether the model follows actions correctly, not whether its induced state-transition system is cycle-consistent (Ying et al. 2026; Ye et al. 2026). CycleBench fills this gap with four task types and four scenario settings spanning short-, mid-, and long-horizon lengths as well as composite-action conditions. Our contributions are threefold:

❶ A new long-horizon supervision perspective. We reveal state-returning failures in long-horizon video IWMs and introduce reversible action cycles as a self-verifiable supervision signal, requiring no ground-truth trajectories.

❷ WorldCycle: a self-verifiable RL framework for reversible dynamics. We introduce two complementary trajectory-level rewards, spatial cycle closure and temporal state consistency, that overcome autoregressive drift on in-domain actions and generalize to out-ofdistribution composite action cycles.

❸ CycleBench and empirical gains. We establish the first benchmark for state-returning consistency across reversible, repeated, and composite actions. WorldCycle reduces state returning drift by up to 44% and improves composite-action accuracy nearly 4× over the baseline.

## Related Work

Video World Models. Classical world models learn predictive environment dynamics for control and planning, typically through compact internal or latent representations (Ha and Schmidhuber 2018; Hafner et al. 2019; Schrittwieser et al. 2020; Hafner et al. 2025). Recent video world models instead represent predicted environment evolution directly in pixel space, building on advances in video generation and visual world modeling (Ho et al. 2022; Kondratyuk et al. 2024; Brooks et al. 2024; Alonso et al. 2024). Interactive systems such as Genie, GameNGen, Oasis, WorldPlay, and HY-World condition generation on user actions for longer-horizon or real-time interaction (Bruce et al. 2024; Valevski et al. 2024;

Decart and Etched 2024; Sun et al. 2026; HY-World Team 2026; Google DeepMind 2024, 2025; Zhang et al. 2025b; Gao et al. 2026). These models pursue fidelity along a single forward direction: each step should look right and follow the current action. Long-range coherence, when addressed, is treated as an appearance- or geometry-level property (Chen et al. 2025; Wu et al. 2025b; Yan et al. 2023; Po et al. 2025). None of these objectives can ask whether the rollout has reached the state the actions demand, because along an arbitrary open trajectory no reference state exists. Reversible and closed action trajectories are the exception: their composed transformation must be the identity, fixing the correct outcome analytically and without annotation. WorldCycle turns this structure into a trajectory-level training signal.

Reinforcement Learning for World Model Post-training. RL-based post-training has produced strong gains in language models, from RLHF to GRPO (Ouyang et al. 2022; Shao et al. 2024; Guo et al. 2025), and the paradigm has been extended to difusion models via DDPO, AlignProp, DifusionDPO, Flow-GRPO, DanceGRPO, and Difusion-NFT (Black et al. 2024; Prabhudesai et al. 2023; Wallace et al. 2024; Liu et al. 2025; Xue et al. 2025; Zheng et al. 2025). For world models specifically, RLVR-World and RLIR apply RL post-training with verifiable or inversedynamics rewards (Wu et al. 2025a; Ye et al. 2025), and WorldCompass scores each generated clip against an inversepose action-following reward and a visual-quality preference model (Wang et al. 2026a). All existing methods supervise individual clips or short segments against an external scorer, placing no constraint on the trajectory as a whole. The correct long-horizon state is unknown for arbitrary action sequences, so trajectory-level error cannot be penalized directly. World-Cycle resolves this by constructing rewards whose target is fixed by the algebraic identity of reversible cycles, yielding a label-free objective that requires no external annotation.

## Method

## Preliminaries

Interactive Video World Models. We formulate an interactive video world model as a learned state-transition system $\mathcal { M } _ { \theta } = ( \mathcal { S } , \mathcal { A } , f _ { \theta } )$ , where $f _ { \theta } : \mathcal { S } \times \mathcal { A }  \mathcal { S }$ maps a state and an action to the next state, $s _ { t + 1 } = f _ { \theta } ( s _ { t } , a _ { t } )$ . For a reversible action $^ { a , }$ let $T _ { a } : { \mathcal { S } } \to { \mathrm { \hat { \mathcal { S } } } }$ denote its true state transformation and $\hat { T } _ { a }$ the transformation realized by the learned model. Given a trajectory $\gamma = ( a _ { 1 } , \dots , a _ { T } )$ , we define $T _ { \gamma } = T _ { a _ { T } } \circ \cdot \cdot \cdot \circ T _ { a _ { 1 } }$ and define $\hat { T } _ { \gamma }$ analogously. Even a well-trained model realizes each action only up to a residual transformation, $\hat { T } _ { a } = T _ { a } \circ E _ { a }$ with $E _ { a } \approx I$ . Consequently,

$$
\begin{array} { r l } & { \hat { T } _ { \gamma } = \hat { T } _ { a _ { T } } \circ \dotsb \circ \hat { T } _ { a _ { 1 } } } \\ & { \qquad = ( T _ { a _ { T } } \circ E _ { a _ { T } } ) \circ \dotsb \circ ( T _ { a _ { 1 } } \circ E _ { a _ { 1 } } ) \neq T _ { \gamma } . } \end{array}\tag{1}
$$

Because autoregressive predictions become the context for subsequent generation, these residual transformations generally do not cancel and instead compound with the rollout horizon. Directly measuring the resulting deviation would require access to the correct transformation $T _ { \gamma }$ , which is unknown for an arbitrary action sequence at training time. We refer to this as the verification bottleneck: long-horizon transition error exists, but no reference state is available against which it can be directly scored, forcing existing post-training objectives to rely on locally verifiable proxies.

Action-induced Transformation Group. The verification bottleneck admits a structural exception. We consider the subset of reversible interactions whose induced transformations form a group $\mathcal { G } = ( \tau , \circ )$ . Every action a admits an inverse $a ^ { - 1 }$ , while the inverse trajectory $\gamma ^ { - 1 }$ cancels the complete transformation induced by $\gamma \colon$

$$
T _ { a } \circ T _ { a ^ { - 1 } } = I , \qquad T _ { \gamma ^ { - 1 } \circ \gamma } = I .\tag{2}
$$

The identity target is determined entirely by the algebraic structure of the actions and is therefore independent of scene content, the initial state, and intermediate observations. Closed reversible trajectories consequently turn an otherwise unknown long-horizon target into an analytically known one, providing a content-independent and annotation-free verifier for accumulated transition error.

## Self-Verifiable Reinforcement Learning

WorldCycle converts this analytical identity constraint into a trajectory-level training signal. For a closed trajectory $\gamma ,$ recurrently executing the learned transition function, denoted by $f _ { \theta } ( s , \bar { \gamma ) }$ , should recover the state from which the trajectory started. We define state returning through the condition

$$
T _ { \gamma } = I \quad \Longrightarrow \quad f _ { \theta } ( s , \gamma ) = s , \qquad \forall s \in \mathcal { S } .\tag{3}
$$

This condition is stronger than local action following. A model may respond correctly to every individual action while its composed transformation $\hat { T } _ { \gamma }$ still deviates from identity because of accumulated residuals. We therefore construct two complementary objectives. The spatial objective densely verifies mirrored states within a reversible cycle, whereas the temporal objective verifies whether corresponding cycle phases remain stable across repeated executions.

## State Discrepancy Measurement.

Because the underlying physical state s is inaccessible, we leverage dense point tracking within the image space as an observable proxy to measure state consistency. Given two frames $o _ { i }$ and $o _ { j } .$ , we track N valid correspondence points and denote their image coordinates by $p _ { n } ^ { i } , p _ { n } ^ { j }$ . The frame-pair discrepancy is the size-normalized mean displacement

$$
d ( o _ { i } , o _ { j } ) = \frac { 1 } { N ( H + W ) } \sum _ { n = 1 } ^ { N } \left. p _ { n } ^ { i } - p _ { n } ^ { j } \right. _ { 2 } ,\tag{4}
$$

where $H , W$ are the image dimensions. Invalid or lowconfidence matches are excluded.

Spatial Closure Reward. We construct a symmetric cycle consisting of a forward action sequence followed by its exact inverse, $\gamma _ { c } = ( a _ { 1 } , \dots , a _ { m } , a _ { m } ^ { - 1 } , \dots , a _ { 1 } ^ { - 1 } )$ , for which $T _ { \gamma _ { c } } = I$ . The state reached after the first i forward actions should match the state reached at the mirrored phase $2 m - i$ of the reverse trajectory. Rather than supervising only the final endpoint, we use every mirrored pair to provide dense evidence throughout the rollout:

$$
R _ { \mathrm { s p a t i a l } } = { \frac { 1 } { m + 1 } } \sum _ { i = 0 } ^ { m } \exp [ - \alpha d ( o _ { i } , o _ { 2 m - i } ) ] .\tag{5}
$$

![](images/7cb9a6b07f7d2eede8e367d7865e017ae199ba28a2d04eda217a6c6cdbac0db9.jpg)  
Figure 2: Overview of WorldCycle. From ordinary action sequences we construct closed reversible cycles and their repeated executions. The spatial closure reward compares every mirrored forward-reverse frame pair within a single cycle, providing dense supervision on where the trajectory begins to drift. The temporal consistency reward compares phase-aligned frames of the same action across repeated cycles, penalizing drift of identical actions over time. Both rewards are trajectory-level, annotation-free, and jointly optimized with a DifusionNFT-style objective.

Here $o _ { i }$ denotes the generated observation at phase $i ,$ and $d ( \cdot , \cdot )$ measures the discrepancy between two observations that should correspond to the same underlying state. This mirrored supervision localizes where a reversible trajectory begins to drift, avoiding the sparse credit assignment produced by an endpoint-only reward.

Temporal State Consistency Reward. Spatial closure verifies state agreement within a single cycle, but does not guarantee that the recovered state behaves consistently when the cycle is executed again. A valid state-transition system should satisfy $s _ { t } = s _ { t ^ { \prime } } \Rightarrow f _ { \theta } ( s _ { t } , a ) = f _ { \theta } ( s _ { t ^ { \prime } } , a )$ . Identical states under identical actions must induce identical transitions. To enforce this, we repeat the cyclic action program for K cycles. Let $o _ { k } ^ { i }$ denote the observation at step i of cycle k (where each cycle has length L). We anchor all subsequent cycles to the first cycle and define the reward as:

$$
R _ { \mathrm { t e m p o r a l } } = { \frac { 1 } { ( K - 1 ) ( L + 1 ) } } \sum _ { k = 2 } ^ { K } \sum _ { i = 0 } ^ { L } \exp \left[ - \alpha d ( o _ { k } ^ { i } , o _ { 1 } ^ { i } ) \right] .\tag{6}
$$

Using the first cycle as a shared anchor avoids allowing the reference state to drift through adjacent-cycle comparisons. The resulting reward verifies not only whether each cycle closes, but also whether repeated returns produce consistent subsequent evolution over an extended horizon.

## Cycle-Consistent RL Training

Trajectory-level cycle rewards require long rollouts, which are computationally expensive and challenging for standard RL algorithms to optimize. To make training practical, we introduce a structured curriculum to prevent learning shortcuts, alongside an eficient difusion RL objective.

Warm-up-and-Combine Schedule. Temporal consistency is unreliable if individual cycles do not close: phase-aligned comparisons across cycles are uninterpretable when the cycle has not been grounded. We therefore begin with a spatial-only warm-up $( t \bar { < } t _ { \mathrm { w } } )$ before activating the temporal reward.

$$
R _ { \mathrm { c y c l e } } ^ { ( i ) } ( t ) = \lambda _ { s } R _ { \mathrm { s p a t i a l } } ^ { ( i ) } + { \bf 1 } [ t \geq t _ { \mathrm { w } } ] \lambda _ { t } R _ { \mathrm { t e m p o r a l } } ^ { ( i ) } .\tag{7}
$$

After $t _ { \mathrm { w } } ,$ both rewards remain jointly active. A sequential strategy that switches to temporal-only risks catastrophic forgetting of spatial closure (Kirkpatrick et al. 2017; Kaplanis, Shanahan, and Clopath 2019; Yu et al. 2020). An alternating strategy avoids forgetting but introduces gradient conflict: each objective switch partially reverses the previous optimization direction. Joint optimization resolves both issues simultaneously: spatial closure provides the geometric grounding on which temporal consistency depends, and the two gradients reinforce rather than compete with each other. Multi-Scale Cycle Sampling. A fixed cycle length may introduce a temporal-position shortcut, where the model memorizes when to reverse rather than learning the inverse relation between actions. We therefore enforce cycle closure at multiple temporal scales. Given a rollout horizon $H = n { \cdot } 2 ^ { Q }$ where n is the residual factor and $2 ^ { Q }$ denotes the largest power-of-two factor of H, we construct hierarchical cycles by partitioning the rollout into $2 ^ { k } ( k \in \{ 0 , \ldots , Q - 1 \dot { \} } ) \mathrm { c y } -$ cles of length $H / 2 ^ { k }$ , each consisting of a forward segment followed by its inverse. We randomly sample one hierarchy level k during training, thus preventing reward hacking.

Reward Composition. The cycle reward is the trajectorylevel backbone of our objective, but on its own it underconstrains two ingredients that a video world model must still get right: local action fidelity and per-frame visual quality. To ensure the cycle reward is not satisfied through visually degenerate shortcuts, we compose the final training reward from three complementary signals:

$$
R ^ { ( i ) } ( t ) = R _ { \mathrm { c y c l e } } ^ { ( i ) } ( t ) + \lambda _ { a } R _ { \mathrm { a c t } } ^ { ( i ) } + \lambda _ { v } R _ { \mathrm { H P S } } ^ { ( i ) } ,\tag{8}
$$

![](images/701c27170e0c183c5e2e175f38a54f09a63d38ed31db2bae3b00ba41fc30a811.jpg)  
Figure 3: Overview of CycleBench. Four complementary tasks probe diferent aspects of accumulated transition error: T1 measures where error appears along a reversible path; T2 measures the residual after a non-symmetric closed trajectory; T3 tracks error growth across repeated cycles; T4 measures error propagation across structurally diferent cycles. All tasks are further evaluated under three horizon settings (short/mid/long) and one out-of-domain composite-action setting.

where $R _ { \mathrm { a c t } }$ is an action-following score (Wang et al. 2026a) and $R _ { \mathrm { H P S } }$ is a visual-quality score (Ma et al. 2025).

Trajectory-Level Optimization with Eficient Rollouts. Cycle-consistency is a global property: rewards are computed over the complete rollout before optimization. The reward design is optimizer-agnostic and can be integrated into any difusion RL framework supporting autoregressive rollout. However, policy-gradient methods such as DanceGRPO (Xue et al. 2025) require timestep-wise logprobability tracking and often reduce rollout diversity under long-horizon optimization (Wang et al. 2026a). We therefore adopt a DifusionNFT-style negative-aware objective (Zheng et al. 2025), which performs supervised regression toward reward-weighted positive and negative velocity directions:

$$
\mathcal { L } _ { \mathrm { R L } } ^ { ( i ) } = r ^ { ( i ) } \Vert v _ { \theta } ^ { + } - v ^ { ( i ) } \Vert _ { 2 } ^ { 2 } + ( 1 - r ^ { ( i ) } ) \Vert v _ { \theta } ^ { - } - v ^ { ( i ) } \Vert _ { 2 } ^ { 2 } ,\tag{9}
$$

where $r ^ { ( i ) } = \mathrm { N o r m } ( R ^ { ( i ) } )$ is the normalized cycle reward. Following existing work (Wang et al. 2026a), we backpropagate only through the trailing chunk while keeping previous chunks as frozen context to reduce memory cost for long rollouts, preserving trajectory-level supervision with a per-step cost comparable to single-clip training.

Analysis on Composite Action Generalization. Composite actions lie outside the base model’s pretraining distribution. WorldCycle addresses this directly: a composite action sequence composed with its inverse still forms a closed cycle, so the cycle-closure reward provides annotation-free supervision on composite trajectories without requiring any ground-truth video. The mechanism that makes this efective is transition residual reduction. Consider a composite trajectory $\gamma _ { \mathrm { c o m p } } = ( c _ { 1 } , \dots , c _ { T } )$ , where each $c _ { t }$ may be a primitive action or a combination of multiple motion components. Its learned rollout can be written as

$$
\hat { T } _ { \gamma _ { \mathrm { c o m p } } } = T _ { \gamma _ { \mathrm { c o m p } } } \circ \mathcal { E } _ { \gamma _ { \mathrm { c o m p } } } ,\tag{10}
$$

where $\mathcal { E } _ { \gamma _ { \mathrm { c o m p } } }$ denotes the accumulated transition residual along the trajectory. For a nominally closed composite trajectory, $T _ { \gamma _ { \mathrm { c o m p } } } = { \dot { I } }$ , and the final drift is therefore determined by the residual accumulation. By optimizing cycle closure, WorldCycle reduces local transition errors and their propagation across action sequences, thereby encouraging the model to recover the underlying action composition algebra rather than memorizing specific trajectories. Consequently, the learned dynamics generalize to unseen combinations of translations, rotations, and their compositions.

## CycleBench

CycleBench measures how transition errors accumulate over extended autoregressive rollouts. For an arbitrary open trajectory, the correct future state is unavailable, so long-horizon error cannot be scored directly. CycleBench sidesteps this by constructing reversible and closed action trajectories whose composed transformation is analytically the identity: any discrepancy between rollout states that should coincide is an observable estimate of accumulated error. The benchmark consists of 47 action trajectories evaluated from 380 initial frames, organized as four tasks × four settings, and provides correspondence-based metrics that localize where errors emerge and how they propagate.

Benchmark Tasks. Each task uses analytically known frame correspondences within closed trajectories to probe a distinct aspect of error accumulation. T1 (Reversible-Cycle) executes a forward action sequence followed by its exact inverse and evaluates whether mirrored forward-reverse frames coincide, providing a dense path-wise probe. T2 (Closed-Cycle) executes a non-retracing closed path (e.g., a rectangle) whose net transformation is still identity, testing endpoint closure without symmetric replay. T3 (Repeated-Cycle) repeats the same closed cycle K times and tracks both endpoint-error growth and phase-aligned drift across repetitions. T4 (Cascaded-Cycle) composes two structurally different closed cycles $C ^ { ( \mathrm { i } ) } \circ C ^ { ( 2 ) }$ and asks whether residuals from the first cycle contaminate the second.

Scenario Settings. Every task is evaluated under four settings along two axes—rollout horizon and action complexity: short-term (125 frames, basic in-domain case), midterm (253 frames, tests error amplification under longer generation), long-term (381 frames, primary stress-test for compounding error), and composite-action (125 frames with multi-component controls, out-of-domain w.r.t. pretraining). Matched trajectory structure across the three horizons attributes diferences to autoregressive accumulation rather than task semantics.

Metrics. To decouple evaluation from the CoTracker-based reward used during post-training (Karaev et al. 2024), CycleBench adopts RoMa (Edstedt et al. 2024) as an independent dense correspondence estimator. For any two frames that should coincide, the mean 2D displacement of valid RoMa pixel matches, measured in pixels, defines the framepair distance $d _ { \mathrm { R o M a } } ( \hat { I } _ { i } , \hat { I } _ { j } )$ ; pairs with too few matches are assigned the image diagonal. Three complementary metrics are reported: ESC↓ (Endpoint State Closure), the distance between the generated endpoint and the initial frame; RPS↓ (Reverse-Path Symmetry), the average distance between mirrored forward-reverse pairs; and RCS↓ (Repeated-Cycle Stability), the average phase-aligned distance across cycles.

Table 1: CycleBench results (weighted average over four benchmark tasks). Model size and venue are indicated after each baseline. WorldCompass and WorldCycle are both post-trained from WorldPlay; reported gains are relative to WorldCompass.
<table><tr><td></td><td colspan="5">Short-term (125 frames)</td><td colspan="5">Mid-term (253 frames)</td></tr><tr><td>Method</td><td>ESC↓</td><td>RPS↓</td><td>RCS↓</td><td>Acc↑</td><td>Qual↑</td><td>ESC↓</td><td>RPS↓</td><td>RCS↓</td><td>Acc↑</td><td>Qual↑</td></tr><tr><td>Lingbot World v2 (14B)</td><td>0.198</td><td>0.143</td><td>0.085</td><td>0.464</td><td>12.11</td><td>0.227</td><td>0.233</td><td>0.062</td><td>0.480</td><td>10.58</td></tr><tr><td>WorldPlay (8B, ICML&#x27;26)</td><td>0.076</td><td>0.057</td><td>0.066</td><td>0.635</td><td>9.66</td><td>0.092</td><td>0.065</td><td>0.053</td><td>0.695</td><td>9.00</td></tr><tr><td>WorldCompass (ICML&#x27;26)</td><td>0.038</td><td>0.034</td><td>0.026</td><td>0.829</td><td>11.06</td><td>0.048</td><td>0.039</td><td>0.025</td><td>0.868</td><td>10.67</td></tr><tr><td>WorldCycle</td><td>0.026 ↓32% 0.019 ↓44%</td><td></td><td>0.020↓23%</td><td>0.833+0.5%</td><td>11.24+2%</td><td>0.038 ↓21%</td><td>0.028↓28%</td><td>0.018↓28%</td><td>0.878+1%</td><td>10.81+1%</td></tr></table>

<table><tr><td></td><td colspan="5">Composite-action (125 frames)</td><td colspan="5">Long-term (381 frames)</td></tr><tr><td>Method</td><td>ESC↓</td><td>RPS↓</td><td>RCS↓</td><td>Acc↑</td><td>Qual↑</td><td>ESC↓</td><td>RPS↓</td><td>RCS↓</td><td>Acc↑</td><td>Qual↑</td></tr><tr><td>Lingbot World v2 (14B)</td><td>0.204</td><td>0.143</td><td>0.102</td><td>0.414</td><td>10.81</td><td>0.221</td><td>0.186</td><td>0.123</td><td>0.033</td><td>11.73</td></tr><tr><td>WorldPlay (8B, ICML&#x27;26)</td><td>0.119</td><td>0.088</td><td>0.057</td><td>0.136</td><td>8.98</td><td>0.302</td><td>0.202</td><td>0.173</td><td>0.073</td><td>4.71</td></tr><tr><td>WorldCompass (ICML&#x27;26)</td><td>0.073</td><td>0.059</td><td>0.040</td><td>0.499</td><td>10.12</td><td>0.187</td><td>0.104</td><td>0.074</td><td>0.084</td><td>9.81</td></tr><tr><td>WorldCycle</td><td>0.057 ↓22% 0.042 ↓29%</td><td></td><td>0.026↓35%</td><td>0.553+11%</td><td>10.42 +3%</td><td>0.163 ↓13% 0.087 ↓16% 0.049 ↓34% 0.095 +13%</td><td></td><td></td><td></td><td>10.31 +5%</td></tr></table>

## Experiments

## Implementation Details

Baselines. We compare WorldCycle against three publicly available interactive video world models. Lingbot World v2 (Gao et al. 2026) is a 14B-parameter open-source foundation model with the largest capacity among the compared systems. WorldPlay (8B, ICML 2026) (Sun et al. 2026) and WorldCompass (ICML 2026) (Wang et al. 2026a) are the strongest smaller-scale baselines for action-conditioned generation; WorldCompass is the closest prior work, applying reinforcement-learning post-training to WorldPlay using perclip inverse-pose and visual-quality rewards. Both World-Compass and our WorldCycle are initialized from WorldPlay, isolating the efect of the post-training objective.

Evaluation. All methods are evaluated on CycleBench (380 videos) under four task types (T1–T4) and four scenario settings: short-term (125 frames), mid-term (253 frames), longterm (381 frames), and composite-action (125 frames). Primary metrics are ESC↓, RPS↓, RCS↓, and Action Accuracy↑; HPSv3↑ quality is reported to verify that cycle training does not degrade visual fidelity.

Training Data and Setup. We post-train the 8B WorldPlay AR checkpoint on approximately 4,000 image–caption pairs from real-world scenes. Each training instance samples a base or combined action sequence and constructs closed reversible cycles and repeated executions on the fly, so no cycle-specific supervision is annotated. Rollouts are chunked into 16-frame video clips as in the base model. We use rollout group size G = 16, up to K = 8 repeated cycles per hierarchical level, and sample one hierarchy level per training step to keep the per-step rollout cost fixed. To optimize the model for longhorizon generation, we set the maximum generation length to 16 clips. The spatial warm-up runs for $t _ { \mathrm { w } } = 3 0 0$ steps, after which the temporal reward is activated. The reward weights are set to $\lambda _ { s } = 1 . 0 , \lambda _ { t } = 0 . 5 , \lambda _ { v } = 1 . 0$ and $\lambda _ { a } = 2 . 0 .$ Optimization uses AdamW with learning rate $1 \times 1 0 ^ { - 5 }$ . The training process runs 3 days on 8 H200 GPUs.

## Qualitative and Quantitative Results

Table 1 reports results across all CycleBench tasks and settings. We break down the key findings into three distinct aspects of model performance:

Scaling Alone Does Not Solve State Consistency: Lingbot World v2 (14B) achieves the highest HPSv3 quality due to its larger capacity, but lags every RL post-trained baseline on cycle-consistency metrics (ESC, RPS, RCS are 1.4–8.3× worse than WorldCycle), confirming that state-returning consistency is not solved by scaling alone.

Superiority Across Extended Horizons: WorldCycle consistently outperforms both WorldPlay and WorldCompass. On short-term simple actions it reduces ESC by 32% and RPS by 44% over WorldCompass while maintaining comparable action accuracy (0.833 vs. 0.829), confirming that cycle-consistency gains do not come at the cost of local interaction following. The gains persist as the horizon extends: at 253 frames ESC drops by 21% and RPS by 28%; at 381 frames RCS falls by 34%, the largest relative gain across horizon settings, reflecting the direct contribution of the temporal consistency reward to long-range stability.

Strong Generalization to Out-of-Domain Composite Actions: On composite actions, where the base WorldPlay’s accuracy collapses to 0.136, WorldCycle achieves an accuracy of 0.553 (a 4× improvement over the base and +11% over WorldCompass), confirming that cycle supervision generalizes efectively to unseen action compositions.

Figure 4 provides qualitative confirmation. While baselines generate locally plausible frames, they accumulate visible drift upon returning to the initial state: colors shift, geometric layouts distort, and structural details diverge. This degradation is particularly severe in the out-of-domain compositeaction setting (left). WorldCycle recovers states substantially

![](images/996274ea93cdeaeaf358ddf62b1abe8a7992cae7ae3fd73a193c3db7b6bc5d09.jpg)  
Figure 4: Qualitative comparison on CycleBench (253 frames). Left: composite-action setting (forward+turn then inverse). Right: simple-action setting (forward then backward). Each method row contains eight sampled frames in total, with four keyframes from each setting: two from the outward phase and two from the return phase. The bottom strip shows a cropped zoom of the return frame alongside the first-frame reference for each method.

Table 2: Ablation on CycleBench. Left: basic actions across three horizons (short / mid / long). Right: OOD composite actions reported with the full metric set. Reward isolates each cycle signal; Schedule isolates the warm-up-and-combine training order.
<table><tr><td></td><td colspan="2">Short-term</td><td colspan="2">Mid-term</td><td colspan="2">Long-term</td><td colspan="5">Composite-action (OOD)</td></tr><tr><td>Variant</td><td>ESC↓</td><td>Acc↑</td><td>ESC↓</td><td>Acc↑</td><td>ESC↓</td><td>Acc↑</td><td>ESC↓</td><td>RPS↓</td><td>RCS↓</td><td>Acc↑</td><td>Qual↑</td></tr><tr><td>Base model</td><td>0.076</td><td>0.635</td><td>0.092</td><td>0.695</td><td>0.302</td><td>0.073</td><td>0.119</td><td>0.088</td><td>0.057</td><td>0.136</td><td>8.98</td></tr><tr><td>Reward ablation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spatial only</td><td>0.033</td><td>0.712</td><td>0.050</td><td>0.744</td><td>0.212</td><td>0.081</td><td>0.071</td><td>0.054</td><td>0.038</td><td>0.473</td><td>9.77</td></tr><tr><td>Temporal only</td><td>0.059</td><td>0.651</td><td>0.078</td><td>0.712</td><td>0.268</td><td>0.076</td><td>0.098</td><td>0.076</td><td>0.048</td><td>0.378</td><td>9.20</td></tr><tr><td>Schedule ablation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sequential (S→T)</td><td>0.028</td><td>0.797</td><td>0.041</td><td>0.845</td><td>0.176</td><td>0.088</td><td>0.061</td><td>0.045</td><td>0.030</td><td>0.529</td><td>10.21</td></tr><tr><td>Joint from scratch</td><td>0.030</td><td>0.769</td><td>0.044</td><td>0.821</td><td>0.180</td><td>0.085</td><td>0.065</td><td>0.052</td><td>0.036</td><td>0.510</td><td>10.04</td></tr><tr><td>Full (Ours)</td><td>0.026</td><td>0.833</td><td>0.038</td><td>0.878</td><td>0.163</td><td>0.095</td><td>0.057</td><td>0.042</td><td>0.026</td><td>0.553</td><td>10.42</td></tr></table>

closer to the starting view in both settings, as the bottom crop strips make this visually unambiguous.

## Ablation study

We analyze component contributions in Table 2.

Each reward addresses a distinct failure mode. Removing the temporal signal primarily degrades long-range stability and composite-action generalization (e.g., long-term ESC increases from 0.163 to 0.212). Conversely, removing the spatial signal causes broader failure across all settings, because temporal supervision relies on individual cycles approximately closing; without the spatial anchor, phase-aligned comparisons become meaningless.

Warm-up-and-combine is essential. The schedule ablation confirms that the ordering of the two rewards is non-trivial. Joint-from-scratch training consistently underperforms our warm-up schedule: the temporal reward has no meaningful reference before individual cycles are grounded, so its gradient adds noise rather than signal. Sequential training (spatialonly then temporal-only) recovers some performance but ultimately drifts because it discards the spatial constraint during the temporal phase. Our warm-up-and-combine strategy retains the spatial anchor throughout optimization, yielding the best results across all settings.

Training does not harm visual quality. Table 2 shows that the full WorldCycle model achieves the highest HPSv3 score (10.42) among all variants, a 16% gain over the base model (8.98), ruling out the hypothesis that cycle-consistency improvements are purchased at the cost of visual fidelity.

## Conclusion

We presented WorldCycle, a self-verifiable RL framework that turns reversible action cycles into dense, annotation-free supervision for long-horizon video world models. Spatial closure and temporal state consistency rewards jointly drive the model to learn actions as consistent state operators, overcoming reward sparsity and temporal drift while generalizing to composite action cycles outside the base model’s pretraining distribution. We further introduced CycleBench, the first benchmark evaluating video world models as state-transition simulators. Extending the cycle-return principle to approximate reversibility and other action modalities such as robotic manipulation are natural directions for future work.

## References

Alonso, E.; Jelley, A.; Micheli, V.; Kanervisto, A.; Storkey, A.; Pearce, T.; and Fleuret, F. 2024. Difusion for World Modeling: Visual Details Matter in Atari. In Advances in Neural Information Processing Systems.

Bengio, S.; Vinyals, O.; Jaitly, N.; and Shazeer, N. 2015. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. In Advances in Neural Information Processing Systems, volume 28.

Black, K.; Janner, M.; Du, Y.; Kostrikov, I.; and Levine, S. 2024. Training Difusion Models with Reinforcement Learning. In International Conference on Learning Representations.

Brooks, T.; Peebles, B.; Holmes, C.; DePue, W.; Guo, Y.; Jing, L.; Schnurr, D.; Taylor, J.; Luhman, T.; Luhman, E.; Ng, C.; Wang, R.; and Ramesh, A. 2024. Video Generation Models as World Simulators. OpenAI technical report.

Bruce, J.; Dennis, M.; Edwards, A.; Parker-Holder, J.; Shi, Y.; Hughes, E.; Lai, M.; Mavalankar, A.; Steigerwald, R.; Apps, C.; Aytar, Y.; Bechtle, S.; Behbahani, F.; Chan, S. C. Y.; Heess, N.; Gonzalez, L.; Osindero, S.; Ozair, S.; Reed, S.; Zhang, J.; Zolna, K.; Clune, J.; de Freitas, N.; Singh, S.; and Rocktäschel, T. 2024. Genie: Generative Interactive Environments. In Proceedings ofthe 41st International Conference on Machine Learning.

Chen, T.; Hu, X.; Ding, Z.; and Jin, C. 2025. Learning World Models for Interactive Video Generation. arXiv preprint arXiv:2505.21996.

Decart; and Etched. 2024. Oasis: A Universe in a Transformer. Project website.

Edstedt, J.; Sun, Q.; Bökman, G.; Wadenbäck, M.; and Felsberg, M. 2024. RoMa: Robust Dense Feature Matching. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19790–19800.

Gao, S.; Yang, J.; Chen, L.; Chitta, K.; Qiu, Y.; Geiger, A.; Zhang, J.; and Li, H. 2024. Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability. arXiv preprint arXiv:2405.17398.

Gao, Z.; Wang, Q.; Zhu, J.; Chen, J.; Liu, Z.; Bai, Q.; Wang,J.; Yuan, Y.; Wang, H.; Lu, Y.; Cheng, K. L.; Zhang, H.; Gao,J.; Feng, T.; Liu, Y.; Yao, Y.; Xu, Y.; Zhu, X.; Shen, Y.; andOuyang, H. 2026. Infinite Worlds with Versatile Interactions.arXiv preprint arXiv:2607.07534.

Google DeepMind. 2024. Genie 2: A Large-Scale Foundation World Model. Google DeepMind Blog.

Google DeepMind. 2025. Genie 3: A New Frontier for World Models. Google DeepMind Blog.

Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948.

Ha, D.; and Schmidhuber, J. 2018. World models. arXiv preprint arXiv:1803.10122.

Hafner, D.; Lillicrap, T.; Fischer, I.; Villegas, R.; Ha, D.; Lee, H.; and Davidson, J. 2019. Learning Latent Dynamics for Planning from Pixels. In Proceedings of the 36th

International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, 2555–2565. PMLR.

Hafner, D.; Pasukonis, J.; Ba, J.; and Lillicrap, T. 2025. Mastering diverse control tasks through world models. Nature, 640(8059): 647–653.

Ho, J.; Salimans, T.; Gritsenko, A. A.; Chan, W.; Norouzi, M.; and Fleet, D. J. 2022. Video Difusion Models. In Advances in Neural Information Processing Systems, volume 35.

Hu, A.; Russell, L.; Yeo, H.; Murez, Z.; Fedoseev, G.; Kendall, A.; Shotton, J.; and Corrado, G. 2023. GAIA-1: A Generative World Model for Autonomous Driving. arXiv preprint arXiv:2309.17080.

Huang, J.; Ye, Z.; Hu, X.; He, T.; Zhang, G.; Shi, S.; Bian, J.; and Jiang, L. 2026. LIVE: Long-Horizon Interactive Video World Modeling. In Proceedings of the 43rd International Conference on Machine Learning. ArXiv preprint arXiv:2602.03747.

Huang, Z.; He, Y.; Yu, J.; Zhang, F.; Si, C.; Jiang, Y.; Zhang, Y.; Wu, T.; Jin, Q.; Chanpaisit, N.; et al. 2024. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21807–21818.

HY-World Team. 2026. HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds. arXiv preprint arXiv:2604.14268.

Janner, M.; Fu, J.; Zhang, M.; and Levine, S. 2019. When to Trust Your Model: Model-Based Policy Optimization. In Advances in Neural Information Processing Systems, volume 32.

Kaplanis, C.; Shanahan, M.; and Clopath, C. 2019. Policy Consolidation for Continual Reinforcement Learning. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, 3242–3251. PMLR.

Karaev, N.; Rocco, I.; Graham, B.; Neverova, N.; Vedaldi, A.; and Rupprecht, C. 2024. CoTracker: It Is Better to Track Together. In European Conference on Computer Vision, volume 15120 of Lecture Notes in Computer Science, 18–35. Springer.

Kirkpatrick, J.; Pascanu, R.; Rabinowitz, N.; Veness, J.; Desjardins, G.; Rusu, A. A.; Milan, K.; Quan, J.; Ramalho, T.; Grabska-Barwinska, A.; Hassabis, D.; Clopath, C.; Kumaran, D.; and Hadsell, R. 2017. Overcoming Catastrophic Forgetting in Neural Networks. Proceedings ofthe National Academy ofSciences, 114(13): 3521–3526.

Kondratyuk, D.; Yu, L.; Gu, X.; Lezama, J.; Huang, J.; Schindler, G.; Hornung, R.; Birodkar, V.; Yan, J.; Chiu, M.- C.; Somandepalli, K.; Akbari, H.; Alon, Y.; Cheng, Y.; Dillon, J.; Gupta, A.; Hahn, M.; Hauth, A.; Hendon, D.; Martinez, A.; Minnen, D.; Sirotenko, M.; Sohn, K.; Yang, X.; Adam, H.; Yang, M.-H.; Essa, I.; Wang, H.; Ross, D. A.; Seybold, B.; and Jiang, L. 2024. VideoPoet: A Large Language Model for Zero-Shot Video Generation. In Proceedings of the 41st International Conference on Machine Learning.

Liu, J.; Liu, G.; Liang, J.; Li, Y.; Liu, J.; Wang, X.; Wan, P.; Zhang, D.; and Ouyang, W. 2025. Flow-GRPO: Training Flow Matching Models via Online RL. arXiv preprint arXiv:2505.05470.

Ma, Y.; Shui, Y.; Wu, X.; Sun, K.; and Li, H. 2025. HPSv3: Towards Wide-Spectrum Human Preference Score. arXiv:2508.03789.

Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C. L.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; Schulman, J.; Hilton, J.; Kelton, F.; Miller, L.; Simens, M.; Askell, A.; Welinder, P.; Christiano, P.; Leike, J.; and Lowe, R. 2022. Training Language Models to Follow Instructions with Human Feedback. In Advances in Neural Information Processing Systems.

Po, R.; Nitzan, Y.; Zhang, R.; Chen, B.; Dao, T.; Shechtman, E.; Wetzstein, G.; and Huang, X. 2025. Long-Context State-Space Video World Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 8733–8744.

Prabhudesai, M.; Goyal, A.; Pathak, D.; and Fragkiadaki, K. 2023. Aligning Text-to-Image Difusion Models with Reward Backpropagation. arXiv preprint arXiv:2310.03739.

Schrittwieser, J.; Antonoglou, I.; Hubert, T.; Simonyan, K.; Sifre, L.; Schmitt, S.; Guez, A.; Lockhart, E.; Hassabis, D.; Graepel, T.; Lillicrap, T.; and Silver, D. 2020. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. Nature, 588: 604–609.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300.

Su, Z.; Hu, T.; Zhang, J.; Wang, R.; Yi, R.; Ma, L.; and Tao, D. 2026. Cycle-World: Mitigating Error Accumulation in Long-Term Video World Models via Reverse-Prediction Cycle Consistency. arXiv preprint arXiv:2607.11836.

Sun, W.; Zhang, H.; Wang, H.; Wu, J.; Wang, Z.; Wang, Z.; Wang, Y.; Zhang, J.; Wang, T.; and Guo, C. 2026. World-Play: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling. In Proceedings of the 43rd International Conference on Machine Learning. ArXiv preprint arXiv:2512.14614.

Valevski, D.; Leviathan, Y.; Arar, M.; and Fruchter, S. 2024. Difusion Models Are Real-Time Game Engines. arXiv preprint arXiv:2408.14837.

Wallace, B.; Dang, M.; Rafailov, R.; Zhou, L.; Lou, A.; Purushwalkam, S.; Ermon, S.; Xiong, C.; Joty, S.; and Naik, N. 2024. Difusion Model Alignment Using Direct Preference Optimization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Wang, Y.; He, J.; Fan, L.; Li, H.; Chen, Y.; and Zhang, Z. 2024. Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14749–14759.

Wang, Z.; Wang, T.; Zhang, H.; Zuo, X.; Wu, J.; Wang, H.; Sun, W.; Wang, Z.; Cao, C.; Zhao, H.; Guo, C.; and

Zhao, Z. 2026a. WorldCompass: Reinforcement Learning for Long-Horizon World Models. In Proceedings ofthe 43rd International Conference on Machine Learning.

Wang, Z.; Zhang, W.; Zhang, W.; Zhang, F.; Tan, X.; Qin, Y.; and Li, G. 2026b. World Models as Group Actions. arXiv preprint arXiv:2605.24578.

Wu, J.; Yin, S.; Feng, N.; He, X.; Li, D.; Hao, J.; and Long, M. 2024. iVideoGPT: Interactive VideoGPTs Are Scalable World Models. In Advances in Neural Information Processing Systems, volume 37.

Wu, J.; Yin, S.; Feng, N.; and Long, M. 2025a. RLVR-World: Training World Models with Reinforcement Learning. In Advances in Neural Information Processing Systems.

Wu, T.; Yang, S.; Po, R.; Xu, Y.; Liu, Z.; Lin, D.; and Wetzstein, G. 2025b. Video World Models with Long-Term Spatial Memory. arXiv preprint arXiv:2506.05284.

Xiang, J.; Liu, G.; Gu, Y.; Gao, Q.; Ning, Y.; Zha, Y.; Feng, Z.; Tao, T.; Hao, S.; Shi, Y.; Liu, Z.; Xing, E. P.; and Hu, Z. 2024. Pandora: Towards General World Model with Natural Language Actions and Video States. arXiv preprint arXiv:2406.09455.

Xue, Z.; Wu, J.; Gao, Y.; Kong, F.; Zhu, L.; Chen, M.; Liu, Z.; Liu, W.; Guo, Q.; Huang, W.; et al. 2025. DanceGRPO: Unleashing GRPO on Visual Generation. arXiv preprint arXiv:2505.07818.

Yan, W.; Hafner, D.; James, S.; and Abbeel, P. 2023. Temporally Consistent Transformers for Video Generation. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, 39062–39098. PMLR.

Yang, S.; Du, Y.; Ghasemipour, K.; Tompson, J.; Kaelbling, L.; Schuurmans, D.; and Abbeel, P. 2024. Learning Interactive Real-World Simulators. In International Conference on Learning Representations.

Ye, Y.; He, T.; Yang, S.; and Bian, J. 2025. Reinforcement Learning with Inverse Rewards for World Model Post-Training. arXiv preprint arXiv:2509.23958.

Ye, Y.; Lu, X.; Jiang, Y.; Gu, Y.; Zhao, R.; Liang, Q.; Pan, J.; Zhang, F.; Wu, W.; and Wang, A. J. 2026. MIND: Benchmarking Memory Consistency and Action Control in World Models. arXiv:2602.08025.

Ying, K.; Hu, H.; Ren, S.; Li, J.; Chen, F.; Wang, Z.; Cao, X.; Cai, X.; and Ding, H. 2026. WBench: A Comprehensive Multi-Turn Benchmark for Interactive Video World Model Evaluation. arXiv preprint arXiv:2605.25874.

Yu, T.; Kumar, S.; Gupta, A.; Levine, S.; Hausman, K.; and Finn, C. 2020. Gradient Surgery for Multi-Task Learning. In Advances in Neural Information Processing Systems, volume 33, 21326–21337. Curran Associates, Inc.

Zhang, K.; Tang, Z.; Hu, X.; Pan, X.; Guo, X.; Liu, Y.; Huang, J.; Yuan, L.; Zhang, Q.; Long, X.-X.; Cao, X.; and Yin, W. 2025a. Epona: Autoregressive Difusion World Model for Autonomous Driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 27220–27230.

Zhang, Y.; Peng, C.; Wang, B.; Wang, P.; Zhu, Q.; Kang,F.; Jiang, B.; Gao, Z.; Li, E.; Liu, Y.; and Zhou, Y. 2025b.

Matrix-Game: Interactive World Foundation Model. arXiv preprint arXiv:2506.18701.

Zheng, K.; Chen, H.; Ye, H.; Wang, H.; Zhang, Q.; Jiang, K.; Su, H.; Ermon, S.; Zhu, J.; and Liu, M.-Y. 2025. DiffusionNFT: Online Difusion Reinforcement with Forward Process. arXiv preprint arXiv:2509.16117.

Zhu, F.; Wu, H.; Guo, S.; Liu, Y.; Cheang, C.; and Kong, T. 2025. IRASim: A Fine-Grained World Model for Robot Manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 9834–9844.

Zhu, Z.; Huang, K.-R.; Xu, Z.; Li, R.; Wu, B.; Bai, R.; Wu, M.; Paul, S.; and Tu, Z. 2026. 4KLSDB: A Large-Scale Dataset for 4K Image Restoration and Text-to-Image Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 3052–3061.

## Benchmark Details

## Benchmark Overview

CycleBench measures how transition errors accumulate over extended autoregressive rollouts. For an arbitrary open action trajectory, the correct future state is unavailable, so long-horizon error cannot be scored directly. CycleBench sidesteps this verification bottleneck by constructing reversible and closed action trajectories whose composed transformation is analytically the identity. Any discrepancy between rollout states that should coincide therefore provides an observable estimate of accumulated transition error.

The benchmark consists of 47 action trajectories evaluated from 380 initial frames, organized as four benchmark tasks and four scenario settings. The four tasks probe where transition error emerges, how much residual remains at closure, how it grows across repeated executions, and whether it propagates from one closed cycle to the next. The four settings vary rollout horizon and action complexity, covering shortterm, mid-term, long-term, and composite-action conditions. Initial-Frame Collection. CycleBench contains 380 initial frames from two complementary sources. We select 236 images from 4KLSDB (Zhu et al. 2026), which predominantly cover realistic, photographic scenes. To increase the diversity of scene content and visual styles, we additionally synthesize 144 images using GPT Image 2. Together, the collected and generated images provide a balanced set of realistic and synthetic environments for evaluating world-model rollouts.

## Benchmark Tasks

Each task uses analytically known frame correspondences within closed trajectories to probe a distinct aspect of accumulated transition error.

T1: Reversible-Cycle. T1 executes a forward action sequence $H = ( a _ { 1 } , \ldots , a _ { K } )$ , followed by its exact inverse $H ^ { - 1 } = ( a _ { K } ^ { - 1 } , \ldots , a _ { 1 } ^ { - 1 } )$ . Under ideal dynamics, the complete forward–inverse trajectory induces the identity transformation. Moreover, each state reached during the forward segment should coincide with its mirrored state during the inverse segment. T1 therefore evaluates whether mirrored forward–reverse frames coincide, providing a dense pathwise probe that localizes where transition error first appears and how it develops along the rollout.

T2: Closed-Cycle. T2 executes a non-retracing closed action sequence C, such as a rectangular or polygonal path. Although the trajectory does not symmetrically replay its earlier path, its ideal net transformation is still the identity, so the generated endpoint should coincide with the initial observation. T2 therefore tests endpoint closure after a structurally closed but non-symmetric trajectory, preventing success through simple reverse-path replay.

T3: Repeated-Cycle. T3 repeats the same closed cycle C for K executions. If one cycle has length L, the endpoints at times $L , 2 L , \ldots , K L$ should all coincide with the initial observation, while phase-aligned states across executions should remain consistent. T3 tracks both endpoint-error growth and phase-aligned drift, directly measuring whether small residuals are amplified as the rollout horizon increases.

T4: Cascaded-Cycle. T4 sequentially executes two structurally diferent closed cycles, $\dot { C } ^ { ( 1 ) }$ and $C ^ { ( 2 ) }$ , where the second begins from the generated endpoint of the first:

$$
I _ { 0 } \xrightarrow { C ^ { ( 1 ) } } \hat { I } _ { T _ { 1 } } \xrightarrow { C ^ { ( 2 ) } } \hat { I } _ { T _ { 1 } + T _ { 2 } } .
$$

Because both cycles ideally induce identity transformations, the intermediate and final endpoints should each coincide with $I _ { 0 }$ . T4 asks whether the residual produced by the first cycle contaminates the second, thereby probing cross-cycle error propagation under diferent action compositions.

## Scenario Settings

Every task is evaluated under four settings along two complementary axes: rollout horizon and action complexity. The short-, mid-, and long-term settings preserve matched trajectory structures while progressively extending the rollout, so diferences can be attributed primarily to autoregressive error accumulation rather than to diferent task semantics. The composite-action setting instead tests more complex, multi-component controls.

Short-term. The short-term setting contains 125-frame rollouts and serves as the basic in-domain case for measuring where transition error first emerges and how much residual remains after a relatively short reversible or closed trajectory.

Mid-term. The mid-term setting extends the rollout to 253 frames while preserving the corresponding trajectory structure. It tests whether residuals observed at short horizons are amplified under longer autoregressive generation.

Long-term. The long-term setting further extends the rollout to 381 frames and serves as the primary stress test for compounding error, including endpoint residuals, reversepath discrepancies, and repeated-cycle drift.

Composite-action. The composite-action setting uses 125-frame rollouts with controls that combine multiple motion components, such as simultaneous translation and rotation. These controls are out of domain with respect to the base model’s pretraining distribution and test whether cycle consistency generalizes to unseen action compositions rather than only to primitive controls.

## RoMa Correspondence Distance

To decouple evaluation from the CoTracker-based correspondence reward used during post-training, CycleBench adopts RoMa as an independent dense correspondence estimator. Given two generated frames $\hat { I } _ { i }$ and $\hat { I } _ { j }$ , RoMa produces correspondences $\{ ( \mathbf { p } _ { k } ^ { ( i ) } , \mathbf { p } _ { k } ^ { ( j ) } , c _ { k } ) \} _ { k = 1 } ^ { N _ { i j } }$ , where $\mathbf { p } _ { k } ^ { ( i ) } , \mathbf { p } _ { k } ^ { ( j ) } \in \mathbb { R } ^ { 2 }$ are matched pixel coordinates and $c _ { k }$ is the associated confidence. We retain the valid matches $\mathcal { V } _ { i j } = \{ k \mid c _ { k } \geq \tau _ { \mathrm { m a t c h } } \}$ and define the frame-pair distance as

$$
d _ { \mathrm { R o M a } } ( { \hat { I } } _ { i } , { \hat { I } } _ { j } ) = \frac { 1 } { | \mathcal { V } _ { i j } | } \sum _ { k \in \mathcal { V } _ { i j } } \left\| \mathbf { p } _ { k } ^ { ( i ) } - \mathbf { p } _ { k } ^ { ( j ) } \right\| _ { 2 } .\tag{11}
$$

For frame pairs that should correspond to the same underlying state, this distance provides an observable estimate of the accumulated transition residual. A smaller value indicates better alignment between the matched visual structures.

Algorithm 1: WorldCycle Training Process   
Input: World model π , EMA reference $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } .$ , training data D, level weights ω, group size G, repeat count K, warm-up threshold   
$t _ { \mathrm { w } } ,$ weights $\lambda _ { s } , \lambda _ { t } , \lambda _ { a } , \lambda _ { v }$   
Output: Optimized world model π   
for training step $t = 1 , 2 , \dots$ . do   
/<sub>\*</sub> Multi-scale cycle sampling: one hierarchy level per step $\star /$   
k ∼ ω   
for $( o _ { 0 } , a _ { 1 : T } ) \in \mathcal { D }$ do   
/<sub>\*</sub> Construct closed cycle at hierarchy level $k \mathrm { ~ \bf ~ \nabla ~ } \star /$   
$\gamma _ { c }  ( B _ { k } \| B _ { k } ^ { - 1 } ) ; \quad \Gamma _ { K }  ( \gamma _ { c } , \cdot . . , \gamma _ { c } )$ ; // K repetitions   
/ Group rollout from EMA reference (G independent samples) /   
$\{ \tau ^ { ( i ) } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( o _ { 0 } , \Gamma _ { K } )$   
/ Compute per-rollout rewards /   
for $i = 1 , \dots , G$ do   
$R _ { s } ^ { ( i ) } \gets \mathrm { S p a t i a l C l o s u r e } ( \tau ^ { ( i ) } )$ $/ /$ Dense mirrored-pair verification   
$R _ { t } ^ { ( i ) } \gets \mathbf { 1 } [ t \geq t _ { \mathrm { w } } ]$ · TemporalStationarity $( \tau ^ { ( i ) } )$ ; // Activated after spatial warm-up   
$R ^ { ( i ) } \gets \lambda _ { s } R _ { s } ^ { ( i ) } + \lambda _ { t } R _ { t } ^ { ( i ) } + \lambda _ { a } R _ { \mathrm { a c t } } ^ { ( i ) } + \lambda _ { v } R _ { \mathrm { H P S } } ^ { ( i ) }$   
end   
/<sub>\*</sub> Group-normalize rewards and optimize (DiffusionNFT) $\star /$   
$r ^ { ( i ) } \gets \mathrm { N o r m } \big ( \{ R ^ { ( i ) } \} _ { i = 1 } ^ { G } \big )$ // $( R ^ { ( i ) } - \mu ) / \sigma$ within group   
$\mathcal { L } _ { \mathrm { R L } }  \frac { 1 } { G } \sum _ { i = 1 } ^ { G } [ r ^ { ( i ) }  v _ { \theta } ^ { + } - v ^ { ( i ) }  _ { 2 } ^ { 2 } + ( 1 - r ^ { ( i ) } )  v _ { \theta } ^ { - } - v ^ { ( i ) }  _ { 2 } ^ { 2 } ]$   
$\theta \gets \mathrm { A d a m W } ( \theta , \nabla _ { \theta } \mathcal { L } _ { \mathrm { R L } } ) ; \quad \theta _ { \mathrm { o l d } } \gets \mathrm { E M A } ( \theta _ { \mathrm { o l d } } , \theta )$ // Update model and reference   
end   
end

Because all evaluated videos share the same resolution, $\mathrm { C y - }$ cleBench reports the displacement directly in pixels. To prevent failed matching from being interpreted as low error, frame pairs with fewer than $\mathrm { \bar { \Phi } } _ { N _ { \mathrm { m i n } } }$ valid correspondences are assigned the maximum image-space distance $D _ { \mathrm { i m g } } = \sqrt { H ^ { 2 } + W ^ { 2 } }$ , where H and W denote the frame height and width.

## Evaluation Metrics

CycleBench reports three complementary correspondencebased metrics, ordered as in the main results table: ESC measures the final closure residual, RPS measures discrepancies along mirrored forward–reverse paths, and RCS measures phase-aligned drift across cycle executions. Lower values indicate better state-returning consistency.

Endpoint State Closure (ESC). ESC is the distance between the generated endpoint and the initial frame:

$$
\begin{array} { r } { \mathcal { L } _ { E S C } ^ { ( \xi ) } = d _ { \mathrm { R o M a } } ( I _ { 0 } , \hat { I } _ { T } ) . } \end{array}\tag{12}
$$

It summarizes the total residual remaining after the complete closed trajectory as a single global closure error.

Reverse-Path Symmetry (RPS). RPS is the average distance between mirrored forward–reverse frame pairs. For a standard T1 trajectory ξ of length $T = 2 K$ , let

$$
\mathcal { P } _ { \xi } ^ { \mathrm { s y m } } = \{ ( t , T - t ) \ | \ 1 \le t < K \}
$$

denote the set of non-trivial mirrored pairs. We define

$$
\mathcal { L } _ { R P S } ^ { ( \xi ) } = \frac { 1 } { | \mathcal { P } _ { \xi } ^ { \mathrm { s y m } } | } \sum _ { ( i , j ) \in \mathcal { P } _ { \xi } ^ { \mathrm { s y m } } } d _ { \mathrm { R o M a } } ( \hat { I } _ { i } , \hat { I } _ { j } ) .\tag{13}
$$

Unlike an endpoint-only measure, RPS provides a dense path-wise diagnostic of where residual error emerges during reversible generation.

Repeated-Cycle Stability (RCS). RCS is the average distance between phase-aligned states across cycle executions. For T3, we anchor all later executions to the first cycle. If each cycle has length L, the correspondence set is

$$
\mathcal { P } _ { \xi } ^ { \mathrm { c y c } } = \{ ( \ell , ( r - 1 ) L + \ell ) \ | \ r = 2 , \ldots , K , \ \ell = 0 , \ldots , L \} .
$$

The metric is

$$
\mathcal { L } _ { R C S } ^ { ( \xi ) } = \frac { 1 } { | \mathcal { P } _ { \xi } ^ { \mathrm { c y c } } | } \sum _ { ( i , j ) \in \mathcal { P } _ { \xi } ^ { \mathrm { c y c } } } d _ { \mathrm { R o M a } } \big ( \hat { I } _ { i } , \hat { I } _ { j } \big ) .\tag{14}
$$

For T4, the same principle is applied to analytically coincident closure states at the cycle handof and after the second cycle. A low RCS indicates that repeated or cascaded cycle execution does not progressively amplify transition residuals.

Auxiliary Metrics. In addition to the three stateconsistency metrics, CycleBench reports Action Accuracy, using the action-following evaluator adopted by WorldCompass, and HPSv3 visual quality. These auxiliary metrics verify that improvements in cycle consistency do not come at the cost of local action following or visual fidelity.

Table 3: VBench video quality comparison across four settings. All six dimensions are higher-is-better (↑). Results are weighted averages over T1–T4 tasks within each setting (380 videos total). WorldCycle consistently matches or exceeds WorldCompass and improves over WorldPlay across all settings.
<table><tr><td></td><td colspan="6">Short-term (125 frames)</td><td colspan="6">Mid-term (253 frames)</td></tr><tr><td>Method</td><td>Aesth.↑</td><td>S.Cons↑</td><td>B.Cons↑</td><td>Img↑</td><td>T.Flick↑</td><td>Mot↑</td><td>Aesth.↑</td><td>S.Cons↑</td><td>B.Cons↑</td><td>Img↑</td><td>T.Flick↑</td><td>Mot↑</td></tr><tr><td>WorldPlay (8B, ICML&#x27;26)</td><td>0.653</td><td>0.950</td><td>0.948</td><td>0.744</td><td>0.956</td><td>0.985</td><td>0.631</td><td>0.930</td><td>0.939</td><td>0.723</td><td>0.955</td><td>0.983</td></tr><tr><td>WorldCompass (ICML&#x27;26)</td><td>0.676</td><td>0.963</td><td>0.953</td><td>0.770</td><td>0.946</td><td>0.983</td><td>0.665</td><td>0.954</td><td>0.948</td><td>0.772</td><td>0.944</td><td>0.983</td></tr><tr><td>WorldCycle</td><td>0.676</td><td>0.968</td><td>0.955</td><td>0.773</td><td>0.945</td><td>0.985</td><td>0.665</td><td>0.961</td><td>0.948</td><td>0.771</td><td>0.943</td><td>0.984</td></tr></table>

<table><tr><td colspan="6">Composite-action (125 frames)</td><td colspan="6">Long-term (381 frames)</td></tr><tr><td>Method</td><td>Aesth.↑</td><td>S.Cons↑</td><td>B.Cons↑</td><td>Img↑</td><td>T.Flick↑</td><td>Mot↑</td><td>Aesth.↑</td><td>S.Cons↑</td><td>B.Cons↑</td><td>Img↑ T.Flick↑</td><td>Mot↑</td></tr><tr><td>WorldPlay (8B, ICML&#x27;26)</td><td>0.637</td><td>0.948</td><td>0.943</td><td>0.733</td><td>0.960</td><td>0.985</td><td>0.579</td><td>0.818</td><td>0.891 0.572</td><td>0.980</td><td>0.989</td></tr><tr><td>WorldCompass (ICML’26)</td><td>0.666</td><td>0.960</td><td>0.948</td><td>0.762</td><td>0.943 0.981</td><td>0.665</td><td>0.871</td><td>0.921</td><td>0.707</td><td>0.958</td><td>0.984</td></tr><tr><td>WorldCycle</td><td>0.672</td><td>0.966</td><td>0.948</td><td>0.768</td><td>0.944</td><td>0.984 0.697</td><td>0.907</td><td>0.925</td><td>0.729</td><td>0.961</td><td>0.988</td></tr></table>

![](images/2a9f8c1afcd1ca9c716ae3ecca8ebed1497e0b8d08ef726158b87d9874a58213.jpg)

![](images/0a519c76d3c23293dd1ace26141eb451fe0ff6e62b1db23be8a56ee93fac7d26.jpg)  
Figure 5: Evolution of cycle consistency and visual quality metrics during the RL training of WorldCycle. These metrics are evaluated on a fixed CycleBench validation subset at every training step.

## WorldCycle Training Algorithm

Algorithm 1 summarizes the complete training procedure of WorldCycle. At each training step, we sample a hierarchy level and construct a closed action cycle by concatenating an action block with its inverse, which is then repeated to form a long-horizon trajectory. The EMA reference model generates a group of independent rollouts, from which we compute the spatial closure, temporal consistency, actionfollowing, and visual-quality rewards. The temporal reward is activated only after the spatial warm-up stage, while both cycle rewards remain jointly active thereafter. The groupnormalized rewards are finally used in a DifusionNFT-style objective to update the world model, followed by an EMA update of the reference model.

## Additional Analysis Details

Training Dynamics. Figure 5 further examines the optimization process when WorldCycle is initialized from WorldPlay. All three cycle-consistency metrics decrease steadily during post-training and converge below their initial values. Meanwhile, HPSv3 increases and remains above the WorldPlay initialization after convergence. These trajectories indicate that the combined objective provides stable optimization of long-horizon consistency without sacrificing visual quality.

Long-Horizon Qualitative Stability. Figure 6 provides a qualitative analysis under the 381-frame long-term setting, in which the model executes five repeated cycles. The baselines exhibit distinct forms of accumulated visual error. LingBot World v2 undergoes pronounced color drift, with the sky becoming progressively oversaturated and unnaturally blue. WorldPlay and WorldCompass instead develop pronounced visual artifacts in the sky region, including whitening and unnatural linear patterns in the final return frame. In contrast, WorldCycle maintains stable colors, textures, and scene content throughout the rollout, producing a final return frame that remains closely aligned with the initial observation. These results demonstrate that the cycle-consistency rewards effectively suppress the accumulation of both color drift and visual artifacts over repeated long-horizon transitions.

## VBench Video Quality Evaluation

To verify that WorldCycle’s training does not degrade perceptual video quality, we evaluate all methods on six VBench dimensions (Huang et al. 2024): Aesthetic Quality, Subject Consistency, Background Consistency, Imaging Quality, Temporal Flickering, and Motion Smoothness. These dimensions are computed independently of the CycleBench reward, providing a complementary quality signal. Lingbot-World-2 VBench scores are omitted here due to limited sample size (12 videos); overall VBench scores for Lingbot-World-2 are: Aesthetic 0.743, Subj. Cons. 0.882, Bg. Cons. 0.923, Imaging 0.746, Temp. Flick. 0.961, Motion 0.979.

WorldCycle matches or exceeds WorldCompass on nearly all dimensions across all four settings, and consistently outperforms the base WorldPlay model. Subject Consistency and Imaging Quality show the largest gains over World-Play across all settings, consistent with the observation that reducing transition residuals also stabilizes frame-level appearance. The improvements are sustained in the compositeaction setting, confirming that cycle training on out-ofdistribution action sequences does not introduce quality degradation.

![](images/71441afd704a9dd2697921eb35ef5f6a0bd130a03633979040fe7da4c43857ee.jpg)  
Figure 6: Long-horizon qualitative comparison over five repeated cycles (381 frames). Each row shows sampled frames from the same long-term rollout. The bottom row enlarges the red-boxed region in the final return frame and compares it with the corresponding region in the first frame. LingBot World v2 exhibits severe color drift, most visibly through the increasingly saturated blue sky, whereas WorldPlay and WorldCompass develop pronounced visual artifacts, including whitening in the sky and unnatural linear patterns around the rooftop structure. In contrast, WorldCycle preserves stable colors, textures, and scene content, yielding a final return frame that remains close to the initial observation after all five cycles.

## Discussion and Limitations

WorldCycle does not treat reversible action cycles merely as a specialized state-returning task. Instead, it uses them as a controlled mechanism for exposing and correcting long-horizon transition errors that would otherwise be difficult to observe. Because the composed transformation of a closed trajectory is analytically known, mirrored and repeated states provide dense verification without requiring ground-truth video supervision. The consistent improvements across diferent rollout horizons and action structures indicate that reducing local transition residuals improves the overall trajectory-level dynamics. Moreover, the gains on outof-distribution composite actions suggest that the learned dynamics increasingly capture the underlying compositional structure of actions rather than simply memorizing specific action programs. Closed cycles should therefore be understood as a self-verifiable supervision mechanism, rather than a restriction on the model’s applicable trajectory space.

The main limitation of WorldCycle is that its exact supervision signals rely on action structures with explicit inverse operations or analytically known closure relations. For irreversible or partially irreversible processes, such as object deformation, contact-rich interaction, and persistent state changes, an action sequence cannot generally recover its initial state by simply applying inverse controls. The current cycle construction therefore cannot be directly applied to these dynamics. Extending self-verifiable supervision to irreversible processes through conservation laws, equivalent terminal states, or other analytically verifiable constraints represents an important direction for future work.

## Relation to Concurrent Cycle-Based Methods

Two concurrent works explore cyclic constraints for improving video world models from complementary perspectives. Cycle-World (Su et al. 2026) introduces reverse-prediction cycle consistency for long-horizon video generation: an auxiliary reverse model reconstructs the preceding latent chunk from the current generated chunk and is further reused for inference-time latent correction. World Models as Group Actions (Wang et al. 2026b) instead formalizes actionconditioned dynamics through group actions and enforces identity, inverse, and composition consistency using synthesized latent-space supervision.

Despite these high-level similarities, the central distinction lies in the role of cyclic structure in the optimization pipeline. Both concurrent approaches formulate cycle consistency as a diferentiable loss for model training or fine-tuning. In contrast, WorldCycle focuses specifically on reinforcement-learning post-training of a pretrained interactive world model. Our action cycles are not introduced merely as another structural regularizer. Their analytically known closure relations instead serve as self-verifiable reward functions, converting sampled long-horizon rollouts into annotation-free scalar feedback. This directly addresses the verification bottleneck in world-model post-training, where ground-truth future videos are generally unavailable. The three methods also difer in how their cycles are instantiated. Cycle-World forms a temporal prediction loop between adjacent latent chunks, whereas World Models as Group Actions regularizes the latent dynamics to respect predefined algebraic relations. WorldCycle constructs closed action programs in the environment and evaluates their complete generated trajectories using spatial closure and temporal consistency rewards. Thus, the cyclic structure serves three distinct purposes: reverse-prediction regularization, group-action regularization, and self-verifiable reward construction for RL post-training, respectively.