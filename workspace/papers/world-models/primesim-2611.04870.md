<!-- page 1 of 42 -->

arXiv:2611.04870v1 [cs.RO] 3 Nov 2026

+PrimeBOT启元机器人

# PrimeSim: A Generalist World Model Simulator from Large-Scale Cross-Embodiment Demonstrations

Travis Davies<sup>1</sup>, Wentao Ma<sup>1,2</sup>, Lihao Guo<sup>1,3</sup>, Tengfei Ma<sup>1,4</sup>, Xiaowen Chang<sup>1,5</sup>, Jiafeng Xu<sup>7</sup>, Shilong Wang<sup>1,3,8</sup>, Yiqi Deng<sup>1,6</sup>, Shaowen He<sup>1,7</sup>, Xiaoqi Li<sup>1,7</sup>, Qian Zhu<sup>1</sup>, Ran Cheng<sup>1</sup>, Hao Dong<sup>1,7</sup>

<sup>1</sup>PrimeBot Research,<sup>2</sup>Southeast University,<sup>3</sup>Shanghai Jiao Tong University,<sup>4</sup>The Chinese University of Hong Kong,<sup>5</sup>University of Chinese Academy of Sciences,<sup>6</sup>Beihang University,<sup>7</sup>Peking University, <sup>8</sup>Shanghai Innovation Institute

## Abstract

A robot world model that serves as a simulator must do more than generate plausible videos: it must faithfully follow commanded actions, maintain consistency across camera views, and sustain coherent long-horizon rollouts. We present **PrimeSim**, an action-conditioned, multiview world model simulator that addresses these challenges through progressive midtraining and posttraining. Starting from a pretrained video generator, we midtrain on 30,110 hours of robot demonstrations spanning 22 platforms. To improve action following, we introduce a unified cross-embodiment action representation and **frame-aligned dual-path conditioning** to strengthen action–frame correspondence. To maintain multiview consistency, we develop **view-aware interleaved attention** with explicit camera identities, enabling cross-view information exchange while preserving view-specific representations. We further introduce a **fast-video, slow-memory posttraining** framework that transforms the bidirectional video generator into an efficient autoregressive simulator. By combining fast causal generation, self-rollout training, and hierarchical temporal memory, PrimeSim mitigates error accumulation from self-generated histories and supports coherent long-horizon simulation. PrimeSim runs at 10.1 FPS, remains stable over long-horizon rollouts on rigid and deformable manipulation tasks, and transfers to held-out platforms within 100 to 200 gradient steps where the pretrained backbone does not. Against five open-source action-conditioned world models it leads four of the five WorldSimProbe suites and posts the best overall EWMBench score, while fine-grained local calibration still trails the field; its rollouts are faithful enough to support GRPO-based policy optimization. We release the model weights, inference code, and training code.

**Project Page:** [https://primebottech.github.io/PrimeSim-website](https://primebottech.github.io/PrimeSim-website)**Correspondence:** [travisdavies@qiyuan-robot.com](mailto:travisdavies@qiyuan-robot.com)

## 1 Introduction

A world model that renders the consequences of a policy’s actions can serve as a simulator: rollouts become parallel, repeatable, and free of the risk of damaging hardware. This sidesteps both the danger and the sample cost of online training on real hardware [43, 61, 64, 65], and avoids the sim-to-real gap [19, 34] and the per-environment engineering that hand-built physics simulators demand. The idea runs from model-based

1

<!-- page 2 of 42 -->

![Image block](doc:8ea4708/tier:basic/page:2/block:1)

Figure 1 PrimeSim. (a) The midtraining corpus: roughly 30.1K hours of curated robot data spanning humanoid, dual-arm, single-arm and handheld UMI [3, 14] platforms, mixing real and simulated footage. (b) The model is conditioned on a language prompt, a unified action representation over head, torso, waist, end-effector, gripper and chassis, and three camera views, and predicts future frames; post-training makes generation autoregressive with a selected KV-cache memory at four denoising steps. (c) Capabilities: fidelity against prior systems, action following measured against simulation ground truth, few-shot adaptation to an unseen embodiment, and use as a simulator: success rate in PrimeSim predicts success rate in the LIBERO [51] physics simulator with r = 0.99, so policies can be trained in the model and the result carried back to the robot.

reinforcement learning, where agents learn from imagined rollouts alone [30–32, 79], through interactive video generation for games and open-ended worlds [4, 7, 20, 23, 55, 66, 73], to action-conditioned robot world models [25, 26, 29, 50]. What such a model must do is narrower than generating plausible video: it must adhere to the actions a policy issues, keep several camera views consistent, and stay coherent over rollouts far longer than the window it was trained on. We present PrimeSim, a world model built as a robot simulator that meets all three requirements.

To enable faithful action following and consistent multiview generation, we adapt a pretrained video model to robot dynamics through **midtraining** on 30,110 hours of demonstrations spanning 22 robot platforms. For action following, we unify heterogeneous robot controls into a shared action representation and introduce **frame-aligned dual-path action conditioning**, combining temporally aligned attention with direct action injection to establish precise correspondence between robot controls and generated motion. For multiview consistency, we introduce **view-aware interleaved attention**, enabling cross-view information exchange while preserving camera-specific representations through explicit camera identities. Through joint conditional flow-matching training, these designs allow a single model to learn action-conditioned, multiview robot dynamics across diverse embodiments, establishing the foundation for an interactive world model simulator.

While midtraining establishes action-conditioned multiview generation within a fixed temporal window, turning the model into a practical robot simulator introduces a fundamental challenge: autoregressive simulation requires continuously conditioning on the model’s own imperfect predictions, whereas conventional video generators are trained on fixed-length, ground-truth contexts. This mismatch leads to error accumulation over extended rollouts and is compounded by the computational cost of iterative video generation. To address these challenges, we develop a **fast-video, slow-memory posttraining** framework that progressively

2

<!-- page 3 of 42 -->

transforms the bidirectional video generator into an efficient, long-horizon autoregressive simulator. First, we enable fast generation through chunk-causal adaptation and four-step consistency distillation, making sequential simulation and subsequent self-rollout training computationally tractable. Second, we improve self-conditioned stability through self-rollout training with asymmetric distribution matching against the frozen bidirectional teacher, allowing the model to learn from its own generated histories rather than relying solely on ground-truth context. Third, we introduce **hierarchical temporal memory** to support long-horizon coherence, preserving the initial scene, sparse long-term history, and dense recent motion within a bounded KV cache. Together, these components enable efficient autoregressive simulation that remains robust to self-generated context and scales beyond the original training window.

Together, these designs enable PrimeSim to generate temporally coherent rollouts up to 105 seconds at 10.1 FPS, supporting diverse robotic manipulation tasks involving both rigid and deformable objects. Compared with five open-source action-conditioned world models, PrimeSim achieves state-of-the-art performance on WorldSimProbe [16], leading four of its five suites: global trajectory coverage by 10.0%, action-source behavior preservation by 41.8%, and interaction dynamics by 48.4%, each measured against the strongest competing model on that axis, with interaction grounding also led at 85.7. On EWMBench, PrimeSim reaches the best overall score of 6.09, ahead of both GE-Sim variants (5.82 and 5.79) and every instruction-conditioned model, and is best on five of the eight dimensions, including perfect scores on high-level semantics, trajectory alignment and logical consistency, with action semantics (BLEU) 72.7% above the strongest GE-Sim variant. Beyond benchmark fidelity, we demonstrate the practical utility of PrimeSim as a learned simulator for GRPO-based policy optimization [27], enabling policy improvement through model-generated interactions. We release the model weights, inference code, and training code to support further research.

## Our main contributions are summarized as follows:

• We present **PrimeSim**, an action-conditioned, multiview world model simulator that transforms a pretrained video generator into an interactive robot world model through a progressive midtraining and posttraining framework.

• We develop a large-scale **midtraining framework** using 30,110 hours of demonstrations across 22 robot platforms. With unified cross-embodiment action representations, frame-aligned dual-path conditioning, and view-aware interleaved attention, PrimeSim learns action-faithful and multiviewconsistent robot dynamics across diverse embodiments.

• We introduce a **fast-video, slow-memory posttraining framework** that combines efficient causal generation, self-rollout training with asymmetric distribution matching, and hierarchical temporal memory. This enables fast, self-conditioned autoregressive simulation while mitigating long-horizon error accumulation beyond the original training window.

• We demonstrate **strong performance** across diverse rigid and deformable object manipulation tasks: 10.1 FPS at four sampling steps, coherent rollouts of up to 105 seconds, **6.09 overall on EWMBench** (first), and **first on four of five** WorldSimProbe suites against five open-source action-conditioned world models, retaining the lead on two of the three action-realisation axes and interaction dynamics. We further validate PrimeSim as a learned simulator for GRPO-based policy optimization.

## 2 Data Engine

Our goal for training data was to learn from a massive and diverse pool of settings while keeping our representations simple. To this end, we assembled 30,110 hours of robot demonstration data from a mix of open-source and proprietary sources. Dataset statistics are shown in Fig. 3, and an overview of the datasets used is provided in Table 12. In general, we include any demonstration data that can be converted into our unified format with reasonable effort, whether humanoid, dual-arm, single-arm, UMI [14], or simulated. We describe the conversion pipeline, quality filtering, and dataset composition below and in the appendix.

**Robot Data Pooling.** Motivated by the low cost of data collection, recent state-of-the-art methods pretrain on large egocentric human videos with a smaller ratio of robot data [25, 45, 83]. However, aligning robot actions to latent actions or predicted hand-to-end-effector poses adds additional complexity. Since pooled

3

<!-- page 4 of 42 -->

![Image block](doc:8ea4708/tier:basic/page:4/block:1)

Figure 2 Overview of the data pipeline. Raw demonstrations from teleoperation, UMI, and simulation are converted through four stages: video quality filtering, action quality filtering, action preprocessing, and data alignment. Each stage lists the operations applied to the incoming data. The pipeline produces a unified corpus of 30,110 hours across 20 dataset families and 22 robot platforms (Table 12).

robot datasets, open- and closed-source together, now largely match the scale of the egocentric corpora those models relied on, we skip the alignment problem entirely: we train directly on robot demonstrations from both sources and simply unify their action and state spaces using the general action representation described in Sec. 3.2.1.

**Dataset Staging.** Each training stage serves a different purpose, so each should be trained on a different data distribution. For midtraining, we require vast and highly diverse data sources, so we use the entire dataset. For posttraining, the model has already seen the full corpus and the stage mainly aims to improve long-horizon and real-time responsiveness rather than to broaden coverage. We therefore train it on small datasets collected for specific downstream tasks, which are long enough to contain complete episodes and representative of what the deployed simulator is asked to continue, rather than on a subsample of the pretraining mixture.

## 2.1 Data Pipeline

## 2.1.1 Stage 1: Video Quality Filtering

**Spatiotemporal validity.** We discard clips whose motion or temporal characteristics fall outside the corpus distribution. Motion is assessed via dense optical flow [22]: given the flow field $\mathbf { u } _ { t } ( p )$ between adjacent frames, we compute

$$
M \big ( V \big ) = \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \frac { 1 } { \big | \Omega \big | } \sum _ { \boldsymbol { p } \in \Omega } \big \| \mathbf { u } _ { t } \big ( \boldsymbol { p } \big ) \big \| _ { 2 } ,\tag{1}
$$

where Ω is the image domain. The motion score is not used to retain only high-motion samples; instead, it supports distribution control. Near-static clips are downsampled so the corpus is not dominated by a small number of platforms that remain stationary. Motion scores are read within a platform rather than across the corpus, because the flow a camera records at rest varies markedly between fixed, handheld and wrist-mounted views. The retained data therefore spans subtle in-hand manipulation, camera movement, and larger end-effector displacement without collapsing toward either motion extreme.

**Artifact detection.** We remove frames and clips containing JPEG encoding artifacts, and discard any whose state or action values are numerically corrupted.

**Episode length filtering.** Clips shorter than 1 second are excluded, as they do not provide enough temporal context for autoregressive training.

## 2.1.2 Stage 2: Action Quality Filtering

**Embodiment-type filtering.** Datasets with dexterous hands or humanoid legs represent a small fraction of the pooled corpus. To keep the action representation simple, we filter these datasets out. This removes legged locomotion, not humanoids as a class: humanoid platforms on a wheeled chassis are kept, because their arm, torso and chassis commands all fit the slots the interface already provides, and they supply the largest single share of the corpus (Fig. 3).

4

<!-- page 5 of 42 -->

![Chart block](doc:8ea4708/tier:basic/page:5/block:1)

![Image block](doc:8ea4708/tier:basic/page:5/block:2)

Figure 3 Overview of the PrimeSim training corpus: 30,110 hours, 2.38B frames and 1.52M episodes from 1,773 repositories over 22 robot platforms, at an 84:16 real-to-simulated ratio. Left: the task-category and embodiment distributions. Right: example frames from each of the five embodiment classes.

**Anomaly detection.** We discard sequences whose action signals contain discontinuities inconsistent with continuous robot motion. We detect these as outliers in the first difference of each action channel: an isolated step change far outside the distribution of adjacent steps indicates a sensor or logging fault rather than a commanded acceleration. Datasets containing corrupted video frames, or action values outside their admissible range, are discarded as well.

**Missing action detection.** Some datasets leave certain action features blank, such as end-effector pose or chassis velocity. If these values cannot be recovered through forward kinematics, the affected data is dropped.

## 2.1.3 Stage 3: Data Alignment

**Frequency alignment.** We preserve each dataset’s native sampling rate at collection time, but resample to a fixed 10 Hz during training. If a dataset cannot be resampled to exactly 10 Hz within a 0.03 s timestamp tolerance, we use the closest achievable frequency.

**Camera alignment.** The number of cameras varies across embodiments. To make batches collatable, we fix the model’s camera count to three. Datasets with more than three cameras have their uncommon cameras dropped; datasets with fewer have the primary camera repeated until three slots are filled.

**Prompt alignment.** Datasets differ in whether they provide natural-language task descriptions. For datasets that include prompts, we keep them as is. For datasets that do not, we generate a prompt from the task metadata with a template-based generator, so that every training sample is paired with a task description.

**Format conversion.** All datasets are converted to LeRobot [8] format. We chose this format for three reasons: (i) its storage is space-efficient, (ii) many open-source datasets already use it, and (iii) recent updates have made it reliable for large-scale training at the tens-of-thousands-of-hours scale used in this project.

## 2.1.4 Stage 4: Action Alignment

**End-effector unification.** Datasets store end-effector pose inconsistently, and we unify it in three steps. For datasets that provide joint-space commands without end-effector poses, we look up the robot’s open-sourced URDF and compute them using the forward kinematics implementation in LeRobot [8]. For datasets with

5

<!-- page 6 of 42 -->

![Image block](doc:8ea4708/tier:basic/page:6/block:1)

Figure 4 The PrimeSim model architecture. Left: the whole system, including the input data modalities, the VAE, the text encoder and the PrimeSim decoder. Right: the action conditioning mechanisms, a framewise attention mask and a frame-action adaptor.

poorly defined action labels, we discard the labels and use proprioceptive data instead, taking the pose one timestep ahead as the current timestep’s action. Rotations vary across datasets in both parametrization and quaternion component ordering, so we convert every rotation to a matrix and then to a quaternion in $( x , y , z , w )$ order, with the real value last. Datasets lacking both end-effector poses and usable joint-space commands are excluded.

**PrimeSim action unification.** To unify actions across heterogeneous embodiment platforms, we convert every action into a shared representation consisting of bimanual end-effector poses with quaternion rotations, full-body joint angles, gripper commands, and chassis velocity in $\left( v _ { x } , v _ { y } , \omega \right)$ . Gripper commands are binary open/close signals and are kept in absolute form. We refer the reader to Sec. 3.2.1 for the full specification.

**Relative action chunk.** We convert absolute actions into relative action chunks by expressing every action in the coordinate frame of the proprioceptive state at frame 0 of the current window. For body joints, we take the difference between the current joint configuration and the reference configuration,

$$
\tilde { q } _ { t } = q _ { t } - q _ { 0 } .\tag{2}
$$

For end-effector pose, we express the current translation and rotation in the reference frame. Let $p _ { t } \in \mathbb { R } ^ { 3 }$ and $R_{t} \in SO(3)$ denote the end-effector translation and rotation at time t, and let $p _ { 0 } , R _ { 0 }$ denote their values at the reference frame. The relative pose is

$$
\tilde { p } _ { t } = R _ { 0 } ^ { \top } \big ( p _ { t } - p _ { 0 } \big ) ,\tag{3}
$$

$$
\tilde { R } _ { t } = R _ { 0 } ^ { \top } R _ { t } .\tag{4}
$$

Equivalently, in homogeneous form, $\tilde { T } _ { t }   =   T _ { 0 } ^ { - 1 } T _ { t }$ where $T _ { t }   =   \left[ R _ { t } \; \middle | \; p _ { t } \right]$ . Gripper commands and chassis velocity are left unchanged, as both are already expressed in a frame-independent form.

## 3 PrimeSim

PrimeSim aims to serve as a world model simulator for long-horizon robot manipulation, generating future camera observations conditioned on robot actions. Unlike conventional video generation, robot simulation requires the model to faithfully execute prescribed actions, maintain consistency across camera views, and preserve coherent scene dynamics over extended trajectories. Moreover, the simulator must support low-latency autoregressive generation to enable efficient interaction with robot policies.

6

<!-- page 7 of 42 -->

To address these requirements, we adopt a progressive training framework that transforms a pretrained video generator into an action-conditioned, multiview, autoregressive simulator. We first initialize PrimeSim from a pretrained video generation backbone that provides general visual and motion priors. During <u>midtraining</u>, we introduce structured action conditioning and multiview adaptation to learn robot dynamics from largescale demonstrations. During <u>posttraining</u>, we progressively adapt the model for causal generation, few-step sampling, and long-horizon self-conditioned rollouts with persistent temporal memory.

## 3.1 Pretrain

We initialize from the Wan2.2-TI2V-5B pretrained weights [75], which provides a strong generative prior over natural video: object permanence, lighting and material statistics, and short-horizon motion patterns that would be prohibitively expensive to relearn from robot data alone. This stage establishes the general world priors, spatiotemporal coherence and high-fidelity texture that the later stages build on. We adopt this checkpoint as given: all of our own training happens in the midtraining and posttraining stages that follow.

## 3.2 Midtrain

Midtraining adapts the pretrained video generator to action-conditioned robot dynamics. While the pre-trained backbone provides general visual and motion priors, it lacks explicit control over robot actions and consistent generation across multiple camera views. To address these limitations, we introduce three complementary components: a unified cross-embodiment action representation, frame-aligned dual-path conditioning, and view-aware interleaved attention. We jointly train these components with the video generation backbone on 30,110 hours of robot demonstrations spanning 22 platforms (Sec. 2).

**Problem Formulation.** Our objective during midtraining is to learn the visual consequences of prescribed robot actions across multiple camera views. At robot timestep $t ,$ let $o _ { t }   =   \left( \mathbf { I } _ { t } , q _ { t } \right)$ denote the observation, where $\mathbf { I } _ { t } = \left( I _ { t } ^ { 1 } , \cdots , I _ { t } ^ { n } \right)$ contains images from n cameras and $q _ { t }$ represents the robot’s proprioceptive state.

Given an observation context $\mathcal { O } _ { t } = \left( o _ { t - K + 1 } , \cdots , o _ { t } \right)$ , an H-step action trajectory $A_{t} = \left( a_{t + 1}, \cdots, a_{t + H} \right)$ , and an optional language instruction $\ell ,$ we formulate multiview video prediction as

$$
\mathbf { I } _ { t + 1 : t + H } \sim \mathcal { W } _ { \mathrm { B D } } \big ( \cdot \big | \mathcal { O } _ { t } , A _ { t } , \ell \big ) ,\tag{5}
$$

where $\mathcal { W } _ { \mathrm { B D } }$ denotes the bidirectional world model learned during midtraining. Unlike autoregressive simulation, the model jointly predicts the future video within a fixed temporal window. This formulation provides the foundation for the action and multiview conditioning mechanisms described below.

## 3.2.1 Action Representation

We define actions as changes in robot poses. Given consecutive video tokens, an action token $a _ { t }$ denotes the pose change that produces the transition into latent frame t. Each embodiment source is transformed into a fixed 31-dimensional representation that captures a shared underlying geometric structure across different action domains, with the slots an embodiment lacks set to zero.

**General Action Representation.** At a high level, actions can include up to four components: effector poses for the agent’s effectors, grasp states, pose of the head, waist, and torso, and the chassis velocity. To cover manipulation poses across almost all robot platforms’ pose representation, we represent arm states as end-effector poses, grasp states as binary flags, head/torso/waist as joint angles, and the chassis as a planar velocity $\left( v _ { x } , v _ { y } , \omega \right)$ . As each embodiment’s degrees of freedom varies, we leave redundancy in each joint slot to account for this (see Fig. 5).

PrimeSim Action Representation

![Image block](doc:8ea4708/tier:basic/page:7/block:14)

Figure 5 General action representation in PrimeSim, covering single-arm, bimanual, UMI, and wheeled humanoid embodiments.

7

<!-- page 8 of 42 -->

**Relative Action Chunk.** All actions are defined relative to the initial frame of each window, i.e., $\tilde { a } _ { t + i | t } =$ $a _ { t + i } \ominus a _ { t }$ . This relative action representation for conditioning resolves several core issues in cross-embodiment learning. First, embodiments with missing features can simply be represented as 0 (no movement). Second, the absolute origin offset of each embodiment is cancelled out by the relative transformation. Third, it strengthens the per-frame action conditioning signal by grounding all changes to the conditioning first frame, as well as concentrating the action distribution. Relative actions have been explored in prior models, but with deltas taken from the previous frame [25]. We argue that anchoring to the first conditional frame of the window instead provides a stronger conditioning signal for world models. An ablation for relative action chunk is provided in Table 9.

## 3.2.2 Action Conditioning

Effectively controlling a world model to generate future frames that faithfully follow commanded actions remains a fundamental challenge. Existing models often hallucinate robot motions inconsistent with the conditioned actions [13, 67]. We attribute this limitation in part to the lack of explicit temporal alignment between actions and generated frames. Under standard global attention, each frame can aggregate action information from multiple timesteps, making it difficult to identify which action should govern its evolution. To address this, we introduce a <u>frame-aligned dual-path conditioning</u> mechanism, which strengthens action–frame correspondence by explicitly binding actions to their corresponding frames through two complementary pathways. while the second directly injects action information into frame representations, ensuring that action conditioning does not rely solely on attention routing. Together with our first-frame-anchored relative action representation, these designs provide temporally precise and explicit control over generated robot motions. We evaluate their individual contributions in Table 10.

**Action Tokenizer.** To enable frame-aligned conditioning, we first establish a temporal correspondence between robot actions and video latents. Specifically, we encode each relative action chunk into a compact sequence of latent action tokens using a two-layer MLP, producing one action token for each latent video frame. This representation provides a shared temporal structure for the two conditioning pathways described below, allowing each frame to be associated with the action responsible for its evolution.

**Framewise Action-Conditioning.** Given the aligned action tokens, our first conditioning pathway explicitly constrains how actions interact with video representations. Rather than allowing every frame to attend to actions from arbitrary timesteps, we introduce a shifted-diagonal attention mask that restricts frame-to-action attention to the corresponding temporal transition. Specifically, since action a<sub>t</sub> drives the transition into frame t + 1, frame t attends only to $a _ { t - 1 } ,$ while the initial conditioning frame receives no action conditioning. This prevents interference from temporally unrelated actions and establishes an explicit action–frame correspondence at the attention level. While prior work has explored frame-level action conditioning [29], our approach enforces this correspondence directly through the attention mask and combines it with first-frame-anchored relative action chunks, which specify the intended motion relative to a consistent reference state.

**Action-Frame Adapter.** Although framewise attention establishes which action each frame should attend $\mathbf { t o } ,$ it does not guarantee that the corresponding action signal sufficiently influences the frame representation. In particular, action information may still be underweighted when competing with rich visual features during attention. We therefore introduce a complementary conditioning pathway that directly injects the aligned action information into each frame token. Specifically, a lightweight action–frame adapter computes a residual feature shift from the corresponding action token and integrates it into the frame representation (Figure 4). Unlike attention-based conditioning, this pathway provides a direct route for action information, independent of the learned attention weights. The two pathways thus serve distinct but complementary roles: framewise attention determines <u>which</u> action governs each frame, while the adapter ensures that the selected action directly modulates its representation. Their combination strengthens action–frame binding and encourages generated motions to remain consistent with the specified action trajectory.

8

<!-- page 9 of 42 -->

## 3.2.3 Multiview Adaption

A fundamental challenge in action-conditioned multiview generation is ensuring that different camera views remain consistent with the same evolving physical scene. This is particularly challenging in robotics, where wrist-mounted cameras move with the robot and may observe one another, creating strong dependencies across views. Treating each view independently can lead to inconsistent robot motions and object interactions, while unrestricted global attention may obscure view-specific information. To address this, we introduce <u>view-aware interleaved attention</u> with explicit camera identities, which enables cross-view information exchange while preserving view-specific representations.

Specifically, we organize the 30 DiT blocks to alternate between intra-view and global attention: two out of every three blocks restrict attention to tokens within each camera view, while every third block enables information exchange across all three views. This design allows the model to capture view-specific spatial and temporal relationships locally, while periodically integrating information across cameras to maintain a coherent representation of the shared physical scene.

However, effective cross-view interaction also requires the model to distinguish the source of each visual token, particularly when features from different cameras are mixed through global attention. We therefore prepend a dedicated camera token to each view, explicitly encoding its view identity throughout the generation process. The two designs serve complementary roles: interleaved attention controls <u>how</u> information is exchanged across views, while camera tokens preserve <u>which</u> view each representation belongs to. Together, they promote consistent multiview generation without sacrificing view-specific information (Figure 4). We validate the contribution of both components in Table 11.

## 3.2.4 Training Objective

Having established the unified action representation, frame-aligned action conditioning, and view-aware multiview attention, we jointly optimize these components under a conditional latent flow-matching objective. Rather than introducing separate objectives for action following or multiview consistency, both capabilities are learned through the conditional generation of future multiview observations.

Let $x _ { 0 } = \mathcal { E } \big ( \mathbf { I } _ { t + 1 : t + H } \big )$ denote the clean latent representation of the target multiview video, where E is the video encoder. We collect the conditioning inputs as $c = \left( \mathcal { O } _ { t } , A _ { t } , \ell \right)$ , comprising the observation context, action trajectory, and optional language instruction.

For a sampled flow timestep $\tau \in [ 0 , 1 ]$ and Gaussian noise $\boldsymbol { \epsilon } \sim \mathcal { N } ( 0 , \mathbf { I } )$ , we construct the noisy latent and its target velocity:

$$
x _ { \tau } = \big ( 1 - \tau \big ) x _ { 0 } + \tau \epsilon , \qquad v _ { \tau } = \epsilon - x _ { 0 } .\tag{6}
$$

The conditional velocity network $u _ { \theta _ { \mathrm { B D } } }$ is trained to predict the target velocity:

$$
\mathcal { L } _ { \mathrm { f l o w } } \big ( \boldsymbol { \theta } _ { \mathrm { B D } } \big ) = \mathbb { E } _ { \big ( \boldsymbol { x } _ { 0 } , \boldsymbol { c } \big ) , \boldsymbol { \epsilon } , \boldsymbol { \tau } } \left[ \left\| \boldsymbol { u } _ { \boldsymbol { \theta } _ { \mathrm { B D } } } \big ( \boldsymbol { x } _ { \tau } , \boldsymbol { \tau } , \boldsymbol { c } \big ) - \boldsymbol { v } _ { \tau } \right\| _ { 2 } ^ { 2 } \right] ,\tag{7}
$$

where $\theta _ { \mathrm { B D } }$ denotes the parameters of the bidirectional world model, and τ indexes the generative flow process rather than robot time.

Importantly, all camera views are generated jointly, and robot actions are incorporated directly into the velocity prediction network through the proposed conditioning mechanisms. This allows the model to learn action-conditioned visual dynamics and cross-view dependencies under a unified generative objective.

The resulting bidirectional world model $\mathcal { W } _ { \mathrm { B D } }$ provides the foundation for subsequent autoregressive adaptation. During posttraining, we retain flow matching for initial causal adaptation, while introducing consistency distillation and distribution matching to enable efficient, self-conditioned, long-horizon simulation.

9

<!-- page 10 of 42 -->

![Image block](doc:8ea4708/tier:basic/page:10/block:1)

Figure 6 Overview of the post-training pipeline, from the midtrained base model through the four adaptation stages to the deployed simulator.

## 3.3 Posttrain

Midtraining equips PrimeSim with the ability to predict action-conditioned, multiview robot videos within a fixed temporal window. However, accurate short-horizon generation does not by itself yield a practical world model simulator. The midtrained model generates clips with bidirectional attention and requires many denoising steps, whereas policy interaction calls for continuously extending a rollout from the model’s own predictions. This introduces three additional challenges: autoregressive generation must respect temporal causality, sampling must be fast enough for repeated rollouts, and errors and missing scene context must not accumulate uncontrollably over long horizons.

We therefore posttrain PrimeSim to transform <u>fixed-window video prediction</u> into <u>fast, self-conditioned, long-horizon simulation</u>. Our guiding principle is <u>fast video, slow memory</u>: generate each new chunk with few-step sampling while retaining informative history across much longer timescales. We realize this through four progressive stages: (1) causal adaptation enables chunk-wise autoregressive generation; (2) consistency distillation produces a four-step sampler; (3) clip-level self-rollout with asymmetric distribution matching improves generation conditioned on the model’s own outputs; and (4) episode-level adaptation combines self-rollout training with bounded, multiscale temporal memory. The stages are sequential: causal generation enables rollout, fast sampling makes rollout training tractable, and self-conditioned training prepares the model for long-horizon continuation.

## 3.3.1 Stage 1: Autoregressive Modelling

To enable continuous simulation, we first adapt the bidirectional model $\mathcal { W } _ { \mathrm { B D } }$ obtained from midtraining into a chunk-causal model $\mathcal { W } _ { C }$ . Unlike the midtrained model, which jointly generates all frames within a fixed temporal window, the causal model generates video chunks sequentially, conditioned on previously generated observations. We achieve this through <u>chunk-causal attention</u>, which preserves bidirectional attention within each chunk while restricting cross-chunk attention to preceding chunks. This allows the model to retain its learned action-conditioned multiview generation capabilities while supporting autoregressive prediction.

Specifically, we adopt teacher forcing [9, 18, 36, 85, 101] during causal adaptation, where each chunk is conditioned on clean ground-truth history. Let $B _ { i , \mathrm { c l e a n } }$ and $B _ { i , \mathrm { n o i s y } }$ denote the clean and noisy representations of chunk i, respectively. The attention context for each noisy chunk is defined as

$$
\mathcal { S } \big ( B _ { i , \mathrm { n o i s y } } \big ) = \Big \{ B _ { j , \mathrm { c l e a n } } : j < i \Big \} \cup \big \{ B _ { i , \mathrm { n o i s y } } \big \}   ,\tag{8}
$$

where $S ( \cdot )$ denotes the accessible attention context. This design prevents information leakage from future chunks while allowing all chunks to be trained in parallel. The same chunk partition also determines the noise granularity of the diffusion-forcing objective. Stage 1 therefore keeps the flow-matching loss of Eq. 7; what changes is the attention mask the loss is evaluated under, which is what we mean by diffusion forcing rather than a new objective.

Through this adaptation, PrimeSim acquires the ability to generate future observations autoregressively

10

<!-- page 11 of 42 -->

rather than within a fixed bidirectional window. However, teacher forcing introduces an exposure bias: the model is trained with clean historical observations but must condition on its own imperfect predictions at inference. Increasing the chunk size from $k = 1 \; \mathrm { t o } \; k = 3$ reduces the number of sequential generation chunks from nine to three in our training window, mitigating but not eliminating error accumulation. To enable subsequent training on self-generated histories, we first reduce the computational cost of autoregressive generation through few-step distillation in Stage 2.

## 3.3.2 Stage 2: Causal Consistency Distillation

Although Stage 1 enables autoregressive generation, the resulting causal model $\mathcal { W } _ { \mathrm { C } }$ still requires 50 denoising steps per chunk, making continuous simulation computationally expensive. More importantly, the high sampling cost makes it impractical to train on self-generated rollouts, where gradients must propagate through multiple generation chunks. To address this, we adopt Causal Consistency Distillation (Causal CD) [97] to obtain an efficient four-step generator. Specifically, we use $\mathcal { W } _ { C }$ as a frozen <u>causal teacher</u> and initialize a <u>causal student</u> $\mathcal { W } _ { \theta }$ from its weights. Both models share the same chunk-causal architecture, allowing the student to preserve the autoregressive structure learned in Stage 1 while substantially reducing the number of sampling steps.

Rather than supervising the student with complete teacher-generated trajectories, Causal CD performs online distillation by matching local transitions of the teacher’s flow trajectory. Let $x _ { t }$ denote a noisy video latent at flow time $t ,   \hat { x } _ { 0 , C }$ the teacher’s clean estimate, and $x _ { t - \Delta t } ^ { C }$ the target obtained from a local teacher ODE step. The distillation objective is

$$
\mathcal { L } _ { \mathrm { C D } } \big ( \theta \big ) = \mathbb { E } _ { \boldsymbol { x } _ { t } , t , \Delta t } \left[ w \big ( t \big )   d \Big ( G _ { \theta } \big ( \boldsymbol { x } _ { t } , \boldsymbol { \hat { x } } _ { 0 , C } , t \big ) , \boldsymbol { x } _ { t - \Delta t } ^ { \mathbb { C } } \Big ) \right] ,\tag{9}
$$

where $G _ { \theta }$ denotes the student’s local flow-map update and $d ( \cdot , \cdot )$ measures the discrepancy between student and teacher transitions. By matching nearby flow states, the student learns to approximate the teacher’s multi-step generation process with substantially fewer sampling steps. Moreover, the online objective requires only a local teacher ODE step per update, avoiding the expensive precomputation and storage of complete teacher trajectories.

To further reduce the memory overhead of maintaining teacher and student models simultaneously, we optimize LoRA [33] adapters while keeping the midtrained backbone weights frozen. Together, these designs produce a four-step causal student that substantially accelerates autoregressive generation and, crucially, makes self-rollout training computationally tractable. Nevertheless, distillation alone does not resolve the exposure bias inherited from teacher-forced training, nor does it guarantee visual fidelity under repeated self-conditioned generation. We therefore further optimize the distilled student on its own generated rollouts in Stage 3.

## 3.3.3 Stage 3: Clip-Level Self-Rollout with Asymmetric DMD

Although Stage 2 substantially accelerates autoregressive generation, the distilled student $\mathcal { W } _ { \theta }$ still faces two limitations. First, training with clean historical observations introduces exposure bias, as the model must condition on its own imperfect predictions during inference. Second, consistency distillation has a forward-KL-like, mode-covering effect, potentially producing averaged predictions with reduced perceptual detail. To address both limitations, we introduce <u>clip-level self-rollout training with asymmetric distribution matching</u>. Specifically, the student autoregressively generates video chunks within each training clip, with subsequent chunks conditioned on its own previously generated latents through the causal KV cache, rather than on ground-truth history. This exposes the model to the conditioning distribution encountered during inference.

To preserve generation quality under self-conditioned rollouts, we further optimize the student using an asymmetric Distribution Matching Distillation (DMD) objective [89, 90]. Unlike Stage 2, which distills a causal teacher into a causal student, we now use the frozen bidirectional model $\mathcal { W } _ { \mathrm { B D } }$ from midtraining as the reference teacher. This design exploits the stronger fixed-window generation capability of the bidirectional

11

<!-- page 12 of 42 -->

model to supervise the autoregressive student, without imposing the teacher’s bidirectional attention structure on the student. We adopt the mode-seeking reverse-KL objective

$$
D _ { \mathrm { K L } } \left( p _ { \theta } \big ( \cdot \mid c \big ) \left\| p _ { \mathrm { B D } } \big ( \cdot \mid c \big ) \right) , \right.\tag{10}
$$

where $p _ { \theta }$ and $p _ { \mathrm { B D } }$ denote the conditional distributions of the causal student and bidirectional reference model, respectively, and c represents the corresponding conditioning information, including visual context, robot actions, and camera-view identities.

To optimize this objective, we introduce a trainable <u>fake-score network</u> $v _ { \phi }$ that estimates the distribution of student-generated samples. Together with the frozen bidirectional reference, it provides the score difference used to update the student. Let ${ \tilde { x } } _ { 0 }$ denote the clean video latent generated by the student, and let $\hat { x } _ { 0 , \phi }$ and $\hat { x } _ { 0 , \mathrm { B D } }$ denote the clean-sample estimates of the fake-score network and bidirectional teacher, respectively, evaluated on the same noisy latent. We construct the normalized score-difference direction as

$$
\nabla _ { \mathrm { K L } } = \frac { \hat { x } _ { 0 , \phi } - \hat { x } _ { 0 , \mathrm { B D } } } { \operatorname* { m a x } \bigl ( \mathbb { E } \left| \tilde { x } _ { 0 } - \hat { x } _ { 0 , \mathrm { B D } } \right| , \delta \bigr ) } , \qquad \delta = 1 0 ^ { - 8 } ,\tag{11}
$$

where the normalization stabilizes the update when the student’s generated distribution deviates substantially from the reference. The student is optimized using the surrogate loss

$$
\mathcal { L } _ { \mathrm { g e n } } ( \theta ) = \frac { 1 } { 2 } \left\| \tilde { x } _ { 0 } - \mathrm { s g } \left( \tilde { x } _ { 0 } - \nabla _ { \mathrm { K L } } \right) \right\| _ { 2 } ^ { 2 } ,\tag{12}
$$

where sg denotes the stop-gradient operator. Meanwhile, the fake-score network is trained on studentgenerated samples using the standard flow matching objective,

$$
\mathcal { L } _ { \mathrm { c r i t i c } } \big ( \phi \big ) = \mathbb { E } _ { \tilde { x } _ { 0 } , \epsilon , \tau } \left[ \big \| v _ { \phi } \big ( x _ { \tau } , \tau , c \big ) - \big ( \epsilon - \tilde { x } _ { 0 } \big ) \big \| _ { 2 } ^ { 2 } \right] ,\tag{13}
$$

where $x _ { \tau } = ( 1 - \tau ) \tilde { x } _ { 0 } + \tau \epsilon$ is the noisy interpolation of the generated latent, and $\epsilon \sim \mathcal { N } ( 0 , I )$

Through clip-level self-rollout and asymmetric distribution matching, Stage 3 enables the four-step causal student to learn from its own generated history while receiving high-quality supervision from the bidirectional reference. However, training remains confined to fixed-length clips, whereas practical robot simulation requires coherent generation over complete episodes. This remaining short-training–long-inference gap motivates the episode-level self-rollout and persistent temporal memory introduced in Stage 4.

## 3.3.4 Stage 4: Episode-Level Long-Horizon Adaptation

While Stage 3 enables self-conditioned generation within short clips, reliable robot simulation requires maintaining coherent predictions over trajectories far longer than the training window. At this scale, two challenges become particularly critical: prediction errors accumulate as generated chunks repeatedly serve as future conditioning, and important scene information may be lost from recent context due to occlusion or camera motion. To address both challenges, we extend self-rollout training from fixed-length clips to complete robot episodes and introduce a <u>persistent hierarchical temporal memory</u>. The former exposes the model to long-horizon error accumulation during training, while the latter preserves scene information across multiple temporal scales. Together, they enable the student to generate extended trajectories using its own predictions rather than relying on clean historical observations.

Stage 4 inherits the asymmetric DMD objective of Stage 3 unchanged. What changes is what the rollouts span: the student is now supervised on episodes rather than fixed clips, so the distribution it is matched against includes the errors it accumulates over a long horizon rather than only over a few chunks.

**Episode-level self-rollout.** We first extend self-conditioned generation to complete robot episodes, allowing the student to experience the long-horizon conditioning distribution encountered during inference. Specifi cally, the student generates successive video chunks autoregressively, with each continuation conditioned on its own previously generated history through the causal KV cache. Generator gradients are accumulated

12

<!-- page 13 of 42 -->

across the episode, followed by a single optimization update at the episode boundary. Meanwhile, the fake-score network is updated after each generated window to track the evolving student distribution. The KV cache persists throughout the episode and is reset only when a new episode begins. Unlike clip-level training, this procedure directly exposes the student to errors that accumulate across successive generations, encouraging more stable long-horizon predictions.

Memory in world models is usually conditioned on encoded ground-truth observations [29, 50, 63, 67] or on a cache of clean historical frames [81, 86, 92]. Both supply, at inference, data the model does not itself produce, which opens a train-inference gap. PrimeSim instead trains its memory on its own rollouts, so the distribution it conditions on during training is the one it will have at inference.

**Hierarchical temporal memory.** Although episode-level self-rollout exposes the model to longer trajectories, it does not by itself prevent the loss of historical information. Retaining every generated frame is computationally expensive, while keeping only recent frames can discard important scene states. We therefore introduce a <u>hierarchical temporal memory</u> that organizes the KV cache into three complementary components: <u>global sink</u>, <u>slow memory</u>, and <u>local memory</u>.

The global sink retains the initial observed latent state as a persistent reference throughout the episode. The local memory stores recent consecutive generated latents, preserving fine-grained motion information. The slow memory, in contrast, maintains a sparsely sampled representation of longer-term history. After each generation window, the previous local-memory entries are merged with the existing slow-memory bank and temporally downsampled to a fixed capacity, while the newly generated latents populate the next local-memory bank. Temporal positions are preserved during downsampling, and synchronized frame indices are used across all camera views to maintain aligned multiview histories.

This hierarchical design allows each generated chunk to access the initial scene, sparse long-term context, and dense recent motion within a bounded memory budget. Importantly, except for the initial observed state, all memory entries originate from the student’s own generated history. The same memory update mechanism is used during training and inference, avoiding the distribution mismatch introduced by teacher-forced historical context.

**Latent-space continuation.** To further improve the efficiency and consistency of long-horizon generation, we maintain the student’s continuation entirely in latent space. Rather than decoding each generated segment and re-encoding it as a new conditioning observation, the student carries its history forward through the persistent KV cache. Continuation windows receive neither a new first-frame latent nor a clean ground-truth prefix, ensuring that the autoregressive process remains conditioned on the model’s own predictions. This avoids repeated video decoding and encoding along the generation path and allows the student to extend trajectories without repeatedly reinitializing visual context.

**Amortized supervision.** A remaining challenge is supervising episode-length student rollouts using the bidirectional reference and fake-score networks, which operate within fixed temporal windows. We decouple long-horizon generation from local distribution supervision. For each continuation, we decode the preceding segment once and re-encode only its final frame as a conditioning observation for the score networks, as required by the causal VAE. This operation is performed outside the student’s gradient path, and the re-encoded observation is never fed back into the student’s autoregressive generation. Consequently, the student can accumulate history across complete episodes, while the score networks evaluate local transitions within their original temporal windows. This enables long-horizon self-rollout training without extending the temporal context of the supervising models.

**Action re-anchoring.** Finally, to preserve consistent action conditioning across long rollouts, we express actions relative to the reference pose recorded at the beginning of each episode, rather than resetting the reference at every generation window. This maintains a consistent action coordinate frame across continuation boundaries and prevents changes in the reference frame from introducing ambiguity into the conditioned motion trajectory.

Together, the four posttraining stages transform PrimeSim from a fixed-window, action-conditioned video generator into a <u>fast, self-conditioned, long-horizon world model simulator</u>. Few-step distillation enables

13

<!-- page 14 of 42 -->

efficient autoregressive generation, while episode-level self-rollout and hierarchical temporal memory support coherent simulation across extended robot trajectories. At inference, PrimeSim requires only the four-step causal student and its bounded KV cache, generating future multiview observations conditioned on robot actions without access to ground-truth future frames.

## 4 Experiments

Six sets of experiments follow, in the order the paper’s argument develops. We first place PrimeSim against existing world models on benchmarks built for embodied prediction, together with the inference cost that decides whether a model can serve as a simulator at all. We then ask what midtraining actually acquires, by adapting to platforms and to scenes and camera placements the model has never seen. We examine what post-training changes over long horizons, whether the simulator is faithful enough to evaluate and to improve a policy, and finally which of the design choices carry the result.

## 4.1 Comparison to the State of the Art

We compare on three axes, chosen because a simulator has to be right about different things than a video generator. <u>Trajectory fidelity</u> asks whether the commanded motion is actually realised; <u>scene and semantic consistency</u> asks whether the world still looks and means the same while that motion happens; and <u>inference</u> cost asks whether the model can be run inside a control loop at all. WorldSimProbe separates the first two into its own axes, EWMBench aggregates them into a single score, and neither measures the third, so we report throughput separately.

## 4.1.1 WorldSimProbe Evaluation

WorldSimProbe [16] isolates a property that reconstruction metrics do not capture: whether a world model behaves as a simulator. It is built on the Observable Simulator Contract, which requires that supplied actions induce corresponding agent motion, and that the environment respond to that realized motion rather than to the action that was merely commanded. This targets a failure mode invisible to PSNR, in which the robot does not move as instructed while the scene reacts as though it had.

The benchmark decomposes the observable simulator contract into five diagnostic suites. T1 (Local Action Calibration) tests whether the model reproduces the simulator’s response to small and large action perturbations. T2 (Global Trajectory Coverage) evaluates whether it follows globally different but physically executable trajectories transferred across tasks and scenes. T3 (Action-Source Behavior Preservation) examines whether behavioral differences among expert, human-teleoperated, and policy-generated actions, such as speed, smoothness, and corrective motion, are retained. T4 (Interaction Grounding) uses validated no-contact counterfactuals to test whether the model avoids hallucinating interactions unsupported by the realized robot motion. T5 (Interaction Dynamics) assesses whether both robot motion and object response match intended interaction primitives such as pushing, pulling, rotating, shaking, and dropping.

We evaluate PrimeSim with a sink token that provides lightweight historical context for bidirectional architectures. To improve physical understanding, we additionally construct training data from three sources: multiple trajectories sampled from VLA task executions, perturbed variants of these sampled trajectories, and trajectories obtained after randomly replacing objects in the scene. Random visual noise is also applied during training to reduce the distribution shift caused by the mismatch between training and evaluation resolutions. PrimeSim generates 192 × 256 images per view and predicts 33 frames at 10 FPS. Because the benchmark does not explicitly score appearance, task success, or coarse rollout-level responsiveness, the resulting comparison is primarily behavioral, and high scores indicate simulator fidelity rather than visual quality or downstream utility.

Against five open-source action-conditioned world models, PrimeSim ranks first on four of the five suites, leading two of the three action realization axes (T2 at 67.9 and T3 at 72.3) and interaction dynamics (T5 at 37.1). The one axis we do not win is informative rather than incidental. LingBot-VA leads local action calibration (77.0 against 70.2), indicating that our fine-grained control precision is not yet the best in the field even where coarse trajectory coverage is. Interaction grounding (T4) is led by us at 85.7 against Ctrl-World’s

14

<!-- page 15 of 42 -->

Table 1 WorldSimProbe results on LIBERO [51], against five open-source action-conditioned world models. T1–T3 measure action realization (local action calibration, global trajectory coverage, action-source behavior preservation); T4–T5 measure interaction response (interaction grounding, interaction dynamics). Bold marks the best value per suite; higher is better. PrimeSim leads four of the five, all but local action calibration.

| Model | T1 ↑ | T2 ↑ | T3 ↑ | T4 ↑ | T5 ↑ | Mean ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| LingBot-VA | 77.0 | 60.2 | 51.0 | 57.8 | 22.4 | 53.7 |
| DreamDojo | 69.5 | 41.5 | 50.3 | 65.3 | 23.5 | 50.0 |
| Cosmos-3-Nano | 67.3 | 55.9 | 47.4 | 61.1 | 25.0 | 51.3 |
| IRASim | 54.0 | 54.7 | 47.9 | 68.2 | 22.4 | 49.4 |
| Ctrl-World | 62.7 | 61.7 | 50.4 | 77.5 | 25.0 | 55.5 |
| PrimeSim | 70.2 | 67.9 | 72.3 | 85.7 | 37.1 | 66.6 |

77.5, and it follows from the same design decision read the other way: PrimeSim trains memory on its own rollouts rather than on encoded ground-truth observations, so the model learns the environment’s response to motion it actually produced, which is what this suite measures.

## 4.1.2 EWMBench Evaluation

EWMBench [93] evaluates embodied world models on eight dimensions covering scene consistency (SceneC), motion accuracy (HSD, Dyn and nDTW) and semantic alignment with the task instruction (BLEU, CLIP, Logics), plus a Diversity term across candidates. The overall score is the sum of the eight dimensions, each in [0, 1]. We evaluate on the standard 21-episode community split across 7 AgiBot World [2] tasks (as used in [24]). For each episode we generate multiple rollouts under a fixed action-conditioned protocol, exploring a small family of sampler settings so that both fidelity and cross-sample diversity are represented; scores reflect EWMBench metric suite on these generations.

These episodes are in-domain: AgiBot World contributes to the training corpus, and the models compared here are fine-tuned from the midtrained checkpoint on these datasets before scoring. The EWMBench numbers therefore measure how well the architecture absorbs an in-domain benchmark rather than zero-shot transfer to it; Section 4.2 is where held-out platforms are evaluated.

PrimeSim rolls out open-loop at 10 FPS, conditioned on the first frame, the task instruction and the groundtruth action trajectory, in 33-frame windows chained over the episode. Videos are generated at 192 × 256, scored at the EWMBench frame size of 480 × 640 on the head-camera stream against the 10 FPS ground-truth lane. These quality results use the multi-step sampling recipe (50 denoising steps, CFG 1.5), while the wall-clock latency numbers in Sec. 4.1.3 use the four-step distilled Stage-3 sampler.

As shown in Table 2, PrimeSim reaches an overall score of 6.09, exceeding the action-conditioned GE-Sim (LTX, 5.82) reported under the same metric accounting; the action-conditioned Thinker-WM variant stands at 5.03. The instruction- and text-conditioned field sits further below: EnerVerse-FT reaches 4.70 in the EWMBench paper’s own evaluation, the strongest recent embodied world models on the same 21-episode split, Thinker-WM (4.66) and Qwen-RobotWorld (4.60), cluster just under it, and the strongest general video models remain below 3.9. We emphasize that the second block of the table is not iso-protocol with ours, as those models condition on instructions or text rather than actions.

## 4.1.3 Inference Speed Comparison

We benchmark the wall-clock inference throughput of the four-step distilled PrimeSim against recent actionconditioned world models, with CausalWM included as a related text-conditioned generative reference.

**Protocol.** All timings are measured on a single NVIDIA A100-80GB in bfloat16 at batch size one. The unit of work is one autoregressive chunk of future frames under each model’s released inference recipe, timed by wall clock with CUDA synchronization, including GPU-to-CPU materialization when it is part of the public API. Checkpoint loading, model construction and warm-up chunks are excluded, and we report

15

<!-- page 16 of 42 -->

Table 2 EWMBench scores. Higher is better; Overall is the sum of the eight dimensions. The first block is actionconditioned like ours; the second block is instruction- or text-conditioned and therefore not iso-protocol with ours (e.g. Thinker-WM generates at 5 FPS upsampled to 30 FPS for scoring). GE-Sim rows are computed from the published Genie Envisioner simulation evaluation [50], aligned to the standard EWMBench column names. Thinker-WM, Qwen-RobotWorld, LVP, Sora2 and Kling26 rows are as reported on the same 21-episode split in [24], compiled there from [95]. <sup>†</sup>EnerVerse-FT is from the EWMBench paper’s own evaluation [93]. ‡Thinker-WM (action-cond.) conditions on the ground-truth action sequence; only its motion dimensions and Overall are reported [24], averaged over the evaluated checkpoints.

<table><tr><td>Method</td><td>SceneC ↑</td><td>HSD↑</td><td>Dyn↑</td><td>nDTW↑</td><td>Div ↑</td><td>BLEU↑</td><td>CLIP↑</td><td>Logics ↑</td><td>Overall ↑</td></tr><tr><td>Action-conditioned</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GE-Sim (LTX) [50]</td><td>0.900</td><td>0.940</td><td>0.780</td><td>0.980</td><td>0.011</td><td>0.330</td><td>0.908</td><td>0.970</td><td>5.82</td></tr><tr><td>GE-Sim (COSMOS2) [50]</td><td>0.910</td><td>0.870</td><td>0.850</td><td>0.970</td><td>0.010</td><td>0.310</td><td>0.902</td><td>0.970</td><td>5.79</td></tr><tr><td>Thinker-WM (action-cond.) [24]‡</td><td></td><td>0.737</td><td>0.364</td><td>0.874</td><td></td><td></td><td></td><td></td><td>5.03</td></tr><tr><td>PrimeSim (ours)</td><td>0.887</td><td>1.000</td><td>0.666</td><td>1.000</td><td>0.010</td><td>0.570</td><td>0.954</td><td>1.000</td><td>6.09</td></tr><tr><td colspan="10">Instruction- or text-conditioned (different protocol)</td></tr><tr><td>EnerVerse-FT [35]†</td><td>0.943</td><td>0.536</td><td>0.536</td><td>0.596</td><td>0.069</td><td>0.180</td><td>0.864</td><td>0.978</td><td>4.70</td></tr><tr><td>Thinker-WM [24]</td><td>0.888</td><td>0.564</td><td>0.363</td><td>0.710</td><td>0.012</td><td>0.269</td><td>0.902</td><td>0.952</td><td>4.66</td></tr><tr><td>Qwen-RobotWorld [95]</td><td>0.914</td><td>0.566</td><td>0.343</td><td>0.671</td><td>0.011</td><td>0.208</td><td>0.883</td><td>1.000</td><td>4.60</td></tr><tr><td>LVP [10]</td><td>0.880</td><td>0.425</td><td>0.043</td><td>0.623</td><td>0.009</td><td>0.218</td><td>0.900</td><td>0.952</td><td>4.05</td></tr><tr><td>Sora2 [59]</td><td>0.853</td><td>0.281</td><td>0.349</td><td>0.275</td><td>0.031</td><td>0.247</td><td>0.910</td><td>0.947</td><td>3.89</td></tr><tr><td>Kling26 [42]</td><td>0.821</td><td>0.327</td><td>0.182</td><td>0.342</td><td>0.017</td><td>0.259</td><td>0.901</td><td>1.000</td><td>3.85</td></tr></table>

steady-state means; per-model warm-up schedules and timing scopes are given in Appendix B. Because multi-view generators emit several RGB images per time step, we report two rates. <u>Temporal FPS</u> (TempFPS) counts rollout time steps produced per second. <u>View-frame FPS</u> (ViewFPS) counts each camera view as one frame and is our primary throughput metric: comparing only TempFPS would penalize a multi-view model, since each time step already renders V images. For single-view models the two rates coincide; PrimeSim’s ViewFPS is three times its TempFPS.

**Results.** Table 3 ranks the measured systems by ViewFPS. PrimeSim is fastest: 30.35 ViewFPS (10.12 TempFPS), rendering a 33-frame three-view window at 288 × 384 in 3.16 s.

Among external systems the closest comparison is GE-Sim 2.0, which is also four-step distilled and three-view. It reaches 6.73 ViewFPS, 1.67× the resolution- and horizon-matched BWM setting at 4.04 ViewFPS. Two published numbers are not comparable on this bench and we do not use them: GE-Sim 2.0 reports 10.9 Hz [63] and DreamDojo 10.81 FPS [25], both measured on H100 rather than A100, and DreamDojo’s distilled student weights were not public when we ran. BWM’s own configuration (672 × 896, 48 frames) falls to 0.78 FPS, the clearest illustration of how strongly throughput depends on resolution and horizon.

**Caveats.** The rows are not matched in resolution, horizon or denoising budget: PrimeSim and GE-Sim 2.0 use four-step distilled samplers, while the public BWM and Ctrl-World checkpoints run 50-step sampling. Few-step distillation is part of the released systems, so the comparison is between released inference recipes rather than matched compute. Ctrl-World generates at the lower native 192 × 320 per view, and CausalWM is text- and first-frame-conditioned rather than action-conditioned. PrimeSim’s throughput also depends on rollout length: longer continuous-KV episodes measure 7.63 TempFPS as the KV cache grows, and we quote the short-window steady state. Finally, PrimeSim is timed at 288 × 384 per view under the RoboTwin 2.0 [11] Stage-3 evaluation contract, whereas the quality results above use 192 × 256 generations; the latency geometry is stated explicitly and is not shared with those tables. This benchmark covers world-model chunk generation only, not an end-to-end control loop with a policy.

16

<!-- page 17 of 42 -->

Table 3 Inference throughput on a single NVIDIA A100-80GB (bfloat16), measured as steady-state wall-clock throughput per autoregressive chunk under each method’s released inference recipe. TempFPS counts rollout time steps per second; ViewFPS counts each camera view as one frame and coincides with TempFPS for single-view models. Rows are not matched in resolution, horizon or denoising steps. PrimeSim is measured at $2 8 8 \times \bar { 3 } 8 4$ per view under the RoboTwin 2.0 Stage-3 contract (33-frame windows, continuous KV, 32 usable frames per window), a different geometry from the 192 × 256 setting of the quality evaluations above. <sup>†</sup>CausalWM is text- and first-frame-conditioned, not actionconditioned.

| Model | Views | Res./ view | Future T | Steps T | empFPS ↑ | ViewFPS ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Ours (PrimeSim Stage-3) | 3 | 288 × 384 | 32 (33) | 4 | 10.12 | 30.35 |
| GE-Sim 2.0 [63] | 3 | 384 × 512 | 25 | 4 | 2.24 | 6.73 |
| BWM [67] (fair-aligned) | 1 | 384 × 512 | 25 | 50 | 4.04 | 4.04 |
| Ctrl-World [29] | 3 | 192 × 320 | 5 | 50 | 0.87 | 2.60 |
| BWM [67] (official) | 1 | 672 × 896 | 48 | 50 | 0.75 | 0.75 |
| CausalWM [83]<sup>†</sup> | 1 | 480 × 640 | 25 | 4+4+4 | 0.66 | 0.66 |

## 4.2 What Midtraining Learns

Midtraining is where the model acquires its action interface: action conditioning and multiview coupling are introduced and trained over 20 dataset families and 22 robot platforms (Sec. 3.2). The question we ask here is what else that stage leaves behind. If cross-embodiment training yields transferable features rather than a fit to the training distribution, those features should survive transfer along three axes at once: to platforms the model has never driven, to scenes it has never seen, and to camera placements it has never been mounted at. All three survive, and the first does so at a cost low enough to change what adding a new robot means in practice: an action-following rollout within 100 to 200 gradient steps, where the pretrained backbone does not reach one at ten to twenty times the budget. At a learning rate of $3 \times 1 0 ^ { - 5 } ,$ , two hundred steps move the parameters by roughly $1 0 ^ { - 2 } ,$ so this is a small amount of optimisation rather than a finetune to convergence. The comparison therefore tests what midtraining already provides, not how much can be recovered by fitting each new platform. The protocol is in Table 4. Posttraining is analysed separately in Sec. 4.3, because it changes how the model is run rather than what it has learned.

Table 4 Few-shot adaptation to held-out embodiments. Both initialisations are finetuned under an identical recipe with single-trajectory batches at a batch size of 1 and a learning rate of $3 \times 1 0 ^ { - 5 } ,$ , and neither receives an embodiment identifier or any embodiment-specific parameters. Entries are gradient steps to an action-following rollout, meaning one in which the arm tracks the commanded trajectory and the scene responds to the motion actually realized. <sup>†</sup>No action-following rollout was reached within this budget.

| Platform | Morphology | Midtrained | Pretrained |
| --- | --- | --- | --- |
| xArm 7 | 7-DoF single arm | 200 | 2000<sup>†</sup> |
| ViperX 300 | bimanual | 100 | 2000<sup>†</sup> |
| UMI gripper | handheld | 200 | 2000<sup>†</sup> |

**Embodiment Generalisation.** Midtraining and posttraining are designed to broaden the model’s coverage of embodiments and dynamics. We probe whether that coverage has in fact yielded transferable knowledge by asking what a new platform costs. We hold out three platforms from the 20 dataset families of our training corpus (Table 12): a single 7-DoF arm (<u>xArm 7</u>) drawn from Open X-Embodiment [17], a bimanual platform (<u>ViperX 300</u>, as used by ALOHA [98]) and the original Stanford UMI gripper [14]. Similar embodiments do appear in the corpus, through Franka in DROID [40] and RoboMIND [78], ALOHA-style simulation in RoboTwin 2.0, and the two UMI corpora, and that similarity is what gives cross-embodiment training something to transfer. It does not make adaptation automatic, however: the kinematics, camera mounting and data distribution all differ, so whatever transfers must be re-expressed for the new body, which is the gap a practitioner bridges when deploying to a new robot. UMI is the sharpest case, since other handheld grippers appear in pre-training while this device does not.

17

<!-- page 18 of 42 -->

![Image block](doc:8ea4708/tier:basic/page:18/block:1)

(a) xArm 7, midtrained, 200 gradient steps.

![Image block](doc:8ea4708/tier:basic/page:18/block:3)

(b) xArm 7, pretrained backbone, 2000 gradient steps.

![Image block](doc:8ea4708/tier:basic/page:18/block:5)

(c) ViperX 300, midtrained, 100 gradient steps.

![Image block](doc:8ea4708/tier:basic/page:18/block:7)

(d) ViperX 300, pretrained backbone, 2000 gradient steps.

![Image block](doc:8ea4708/tier:basic/page:18/block:9)

(e) Original Stanford UMI gripper [14], midtrained, 200 gradient steps.

![Image block](doc:8ea4708/tier:basic/page:18/block:11)

(f) Original Stanford UMI gripper [14], pretrained backbone, 2000 gradient steps.

Figure 7 Few-shot adaptation to unseen embodiments. Each pair shows the midtrained model above the pretrained backbone for one platform held out from training: a single 7-DoF arm (xArm 7) drawn from Open X-Embodiment [17], a bimanual platform (ViperX 300, as used by ALOHA [98]) and the original Stanford UMI gripper [14]. Similar embodiments do occur in the corpus, and they are what make transfer possible at all, since a wholly alien platform would leave nothing to build on. They are still new embodiments, however, and adaptation between embodiments does not happen for free: each brings its own kinematics, camera mounting and data distribution, which have to be acquired rather than inherited. All initialisations are finetuned under an identical recipe with single-trajectory batches at a batch size of 1, and none receives an embodiment identifier or any embodiment-specific parameters. The midtrained model reaches an action-following rollout within 100 to 200 gradient steps on all three platforms. The pretrained backbone, which must initialise the action-conditioning and multiview modules from scratch, does not reach the same regime even at ten to twenty times that budget.

For each platform we finetune two initialisations under an identical recipe with single-trajectory batches at a batch size of 1. Neither receives an embodiment identifier or any embodiment-specific parameters, so the only difference between them is the starting point: the midtrained model already carries the action-conditioning and multiview modules learned from our 30,110-hour cross-embodiment corpus (Sec. 3.2), whereas the pretrained Wan2.2-TI2V-5B backbone must initialise those modules from scratch. To avoid penalising the baseline for this, we deliberately grant it ten to twenty times the finetuning budget of the midtrained model: 2000 steps against 100 to 200.

The midtrained model reaches an action-following rollout on all three platforms within 100 to 200 steps

18

<!-- page 19 of 42 -->

(Fig. 7). The pretrained backbone, at 2000 steps, does not: its generations are neither visually plausible nor action-controllable, remaining blurry and exhibiting no plausible movement or contact.

We read this as transfer rather than fast fitting, for three reasons. The optimiser is barely used: 200 gradient steps at a batch size of 1 draw 200 training windows from the recorded episodes, which is far too few to re-learn a platform. No run receives an embodiment identifier or an embodiment-specific parameter, so the gain cannot come from a lookup on which body is present. And the same gap appears on a single arm, a bimanual platform and a handheld device, which rules out an explanation tied to one kinematic structure. The baseline, given ten to twenty times the optimisation, never reaches the regime at all, so the difference is not one of budget. Taken together, these leave little room for an account that does not involve knowledge already carried in the weights. What they do not separate is corpus <u>scale</u> from the presence of an action interface at all: the baseline begins without one, so the two cannot be disentangled here, and we make no claim about which contributes more.

## Scene and Camera-Pose Generalisation.

The 30,110-hour corpus spans a wide range of environments, and we ask whether that coverage transfers to a setting the model has never seen. We sample action trajectories and initial frames from the XVLA open-source dataset [100], which is held out of pre-training entirely. XVLA shares the AgileX Piper bimanual embodiment and includes clothes-folding tasks, so the robot itself is familiar; what differs is the camera configuration, the camera poses, and the setting as a whole. We generate multiview video conditioned on the sampled actions and assess it along three axes: <u>action controllability</u>, whether the arms follow the commanded trajectory; <u>object-motion plausibility</u>, whether the cloth deforms and travels as it should; and <u>cross-view coherence</u>, whether the head and both wrist views remain consistent with one another.

Zero-shot, PrimeSim controls the arms accurately and generates simple cloth motion faithfully, with quality degrading as the motion becomes contact-rich and the generated cloth state drifts from the commanded action. Table 5 makes that cost explicit: against a matched in-distribution baseline under an otherwise identical protocol, the model loses 3.97 dB PSNR and 0.105 LPIPS on XVLA. Finetuning for 200 steps at a batch size of 1 recovers most of the gap, in both the arm trajectories and the cloth deformation (Fig. 8). Seen platforms remain stronger than any transfer setting, which is the expected ordering and is the point: these results measure the cost of moving to a new platform, not a ceiling on what the simulator can generate.

The two results together say something more specific than that the model generalises. The same interface that carries to an unseen platform also carries to an unseen setting, with the task prompt but no embodiment identifier at inference. What transfers is therefore not a robot’s appearance but the mapping from actions to their visual consequences, which is what a simulator has to encode and what a video prior alone does not.

Table 5 Zero-shot transfer to a held-out dataset. Both rows share the checkpoint, sampler (50 steps, seed 42) and metrics at 192 × 256 over three jointly generated views; only the data differs. XR2 [82] is held-out episodes of repositories seen in pre-training; XVLA is excluded and differs in scene, garments and camera placement. Means ± std over eight validation windows. SSIM is not comparable across the two datasets, so only PSNR and LPIPS should be read as a degradation.

| Data | Regime | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- | --- |
| XR2 | in-distribution | 19.79 ± 1.64 | 0.731 ± 0.053 | 0.192 ± 0.052 |
| XVLA | zero-shot | 15.82 ± 2.24 | 0.762 ± 0.049 | 0.297 ± 0.072 |
| ∆ |  | -3.97 | +0.031 | +0.105 |

## 4.3 Posttraining Analysis

Posttraining turns the midtrained model into a causal autoregressive simulator through chunk-causal attention, few-step distillation and self-rollout training (Sec. 3.3). Posttraining primarily changes the model’s causal generation and conditioning behavior, so we evaluate both its short-horizon efficiency and fidelity, and its stability as the rollout horizon increases.

19

<!-- page 20 of 42 -->

![Image block](doc:8ea4708/tier:basic/page:20/block:1)

(a) Before adaptation: zero-shot.

![Image block](doc:8ea4708/tier:basic/page:20/block:3)

(b) After 200 gradient steps at batch size 1.

Figure 8 Zero-shot and few-shot generation on an unseen setting and camera pose. Both subfigures show the AgileX Piper bimanual arm on XVLA [100], held out of pre-training entirely: the embodiment is in the corpus, but the scene, the garments and the camera placements are not. Rows are cameras (head, left wrist, right wrist) and columns are seven frames sampled uniformly from a single rollout. (a) Without adaptation the model controls the arms and renders the cloth, but the generated cloth state drifts from the commanded action once the motion becomes contact-rich. (b) After 200 gradient steps the arm trajectories and the cloth deformation track the commanded actions much more closely and the three views stay mutually consistent, so most of the cost of the transfer is recovered.

Table 6 Distillation results. Stage 1 is the causal autoregressive teacher at 50 sampling steps. Stage 2 applies causal KV consistency distillation and Stage 3 causal KV distribution matching distillation (DMD), both reducing inference to 4 steps.

| Stage | Steps | FPS ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- | --- | --- |
| Stage 1 (Causal AR Diffusion) | 50 | 1.12 | 26.15 | 0.859 | 0.153 |
| Stage 2 (CD Distillation) | 4 | 10.04 | 24.40 | 0.841 | 0.193 |
| Stage 3 (Asymmetric DMD) | 4 | 10.12 | 24.87 | 0.838 | 0.176 |

## 4.3.1 Distillation Efficiency and Short-Horizon Fidelity

Distillation serves two purposes in PrimeSim: it reduces inference latency for interactive simulation, and it shortens the gradient graph during rollout training, where backpropagating through 20 to 50 denoising steps does not fit on current hardware. Table 6 ablates the sampler on the same RoboTwin 2.0 held-out split, with throughput measured on a single A100. Relative to the 50-step causal Stage 1 teacher, the four-step Stage 3 model improves throughput by approximately 9.0×; the measured rates for every stage and for the released model are in the table.

20

<!-- page 21 of 42 -->

Table 7 Long-horizon generation quality averaged over ten held-out RoboTwin 2.0 episodes. Stage 3 uses fixed-length self-rollout training, while Stage 4 extends self-rollout training to complete episodes. The Stage 4 result uses the rolling (3 + 18) cache configuration.

| Stage |  |  | Mean PSNR ↑ Mean SSIM ↑ Mean LPIPS ↓ | ∆PSNR↓ | ∆SSIM↓ | ∆LPIPS↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Stage 3 | 12.572 | 0.401 | 0.654 | 18.305 | 0.602 | 0.640 |
| Stage 4 | 18.709 | 0.695 | 0.403 | 8.427 | 0.120 | 0.233 |

Table 8 KV-cache organization ablation for long-horizon generation.

<table><tr><td rowspan="2">Memory configuration</td><td colspan="3">Full-rollout average</td><td colspan="3">Final window</td><td colspan="3">Degradation</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS</td><td>PSNR ↑</td><td>SSIM ↑ LPIPS ↓</td><td></td><td>∆PSNR↓</td><td>∆SSIM↓</td><td>ΔLPIPS↓</td></tr><tr><td>Rolling (3 + 18)</td><td>18.709</td><td>0.695</td><td>0.403</td><td>18.382</td><td>0.714</td><td>0.400</td><td>8.427</td><td>0.120</td><td>0.233</td></tr><tr><td>Rolling (0 + 21)</td><td>16.502</td><td>0.591</td><td>0.515</td><td>12.973</td><td>0.459</td><td>0.670</td><td>13.749</td><td>0.332</td><td>0.503</td></tr><tr><td>Rolling (0 + 9)</td><td>15.220</td><td>0.547</td><td>0.577</td><td>12.542</td><td>0.454</td><td>0.729</td><td>13.641</td><td>0.353</td><td>0.548</td></tr><tr><td>Hierarchical  $(3 + 9 + 9)$ </td><td>19.519</td><td>0.713</td><td>0.379</td><td>21.027</td><td>0.760</td><td>0.321</td><td>6.008</td><td>0.079</td><td>0.159</td></tr></table>

## 4.3.2 Long-Horizon Robustness

**Episode-level adaptation.** To evaluate the final long-horizon adaptation stage, we directly compare Stage 3, which performs self-rollout training within fixed-length clips, with Stage 4, which extends training to complete robot episodes. Both models are evaluated on the same ten episode-level held-out RoboTwin 2.0 trajectories using identical initial observations and robot actions.

Each rollout is initialized from the first ground-truth RGB frame and subsequently conditioned only on the model’s own generated history. The evaluated trajectories contain 610–1,046 generated frames, corresponding to 61.0–104.6 seconds at 10 FPS, and include three synchronized camera views.

Table 7 reports both the average quality over the complete rollout and the degradation between the first and last complete 32-frame generation windows. For PSNR and SSIM, degradation is computed as the first-window score minus the last-window score; for LPIPS, it is computed in the opposite direction. Lower degradation therefore indicates better long-horizon robustness for all three metrics.

**Memory configuration ablation.** We further compare rolling and hierarchical KV-cache organizations under the same long-horizon evaluation protocol. For rolling memory, the allocation denotes sink and rolling slots; for hierarchical memory, it denotes global, slow, and local slots.

Under the same 21-slot budget, hierarchical memory consistently outperforms the 3 + 18 rolling cache. It improves both average and final-window quality, while reducing ∆PSNR from 8.427 to 6.008 dB, ∆SSIM from 0.120 to 0.079, and ∆LPIPS from 0.233 to 0.159. This supports separating persistent, long-term, and recent context within the KV cache.

## 4.4 PrimeSim as Policy Simulator

The video metrics in the preceding sections measure how closely PrimeSim reproduces recorded trajectories. For a policy simulator, the more relevant question is whether conclusions drawn inside it carry over to the real environment. We examine this in two ways: whether PrimeSim predicts and ranks the success rates of fixed policies as a physics simulator does (Sec. 4.4.1), and whether it can serve as the training environment for reinforcement-learning post-training of a policy (Sec. 4.4.2).

**Setup.** We use the LIBERO-Spatial suite (10 tasks) in the LIBERO simulator as the reference, where success is LIBERO’s privileged-state task predicate. Policies are OpenVLA-OFT after supervised finetuning on LIBERO-Spatial demonstrations; they predict 8-step action chunks from the main camera view. Inside a world model, success is judged by a ResNet-based binary classifier finetuned on LIBERO-Spatial frames, which predicts from each generated frame whether the task has been completed; an episode receives a sparse

21

<!-- page 22 of 42 -->

![Chart block](doc:8ea4708/tier:basic/page:22/block:1)

(a) Success in world model vs. simulator

![Chart block](doc:8ea4708/tier:basic/page:22/block:3)

(b) Before vs. after RL post-training in PrimeSim

Figure 9 (a) Each point is one OpenVLA-OFT policy, evaluated on the same 300 LIBERO-Spatial initial states in the world model (x) and in the LIBERO simulator (y). The dotted line marks perfect agreement; points above it mean the world model under-estimates the policy. (b) Success rate of the SFT policy before and after 200 steps of GRPO post-training inside PrimeSim, measured in the simulator (solid) and in PrimeSim (hatched).

reward of 1 if any of its frames is classified as success and 0 otherwise.

## 4.4.1 Policy Evaluation Fidelity

We first freeze the policy and ask whether the world model predicts its success rate. We take eight OpenVLA-OFT policies of clearly different competence, whose simulator success ranges from 29.3% to 85.0%. Each policy is run with greedy actions from the same 300 initial states (10 tasks $\times 3 0$ trials), once in the LIBERO simulator and once closed-loop inside the world model. As a reference world model we use a publicly released Wan2.2-TI2V-5B [75] finetuned on LIBERO-Spatial (Wan2.2-FT). We report the Spearman rank correlation $\rho$ and the Pearson correlation r between world-model and simulator success, and the mean absolute gap |∆| between the two.

PrimeSim orders all eight policies exactly as the simulator does (Spearman $\rho = 1 . 0 0 ,$ , Pearson $r = 0 . 9 9 )$ and stays within 7.0 points of the simulator for every policy, with a mean absolute gap of 2.8 points $( \mathrm { F i g . ~ } 9 ( \mathsf { a } ) )$ Wan2.2-FT also tracks the overall trend $( r = 0 . 9 1 )$ but mis-orders several policies $( \rho = 0 . 9 0 )$ , and its errors grow with competence: it under-estimates the three strongest policies, whose simulator success is $7 9 . 0 \% ,$ 82.0% and $8 5 . 0 \mathrm { \hat { \% } } ,$ by 12.0, 12.7 and 18.3 points, for a mean absolute gap of 8.3 points. A world model that systematically under-rates good policies is of limited use for checkpoint selection, which is exactly where policy evaluation in a world model is needed.

## 4.4.2 RL Post-training inside PrimeSim

Beyond evaluation, PrimeSim can serve directly as the training environment for policy post-training. Starting from the SFT policy, we run GRPO (group size 8) entirely inside PrimeSim: rollouts are generated closed-loop by the world model and receive the sparse binary reward from the success classifier, and the physics simulator is never queried during training; it is used only to evaluate the resulting policy. After 200 RL steps, the policy’s success rate in the LIBERO simulator rises from 61.7% to 73.2%, and PrimeSim’s own estimate (60.1% to 71.9%) stays within 1.6 points of it $( \mathrm { F i g .   9 ( b ) } )$ , showing that improvements obtained purely inside PrimeSim transfer to the physics simulator.

**Scope of this experiment.** The aim here is to show that PrimeSim supports RL post-training, not to maximise

22

<!-- page 23 of 42 -->

absolute success on LIBERO. We deliberately use a short training budget and an off-the-shelf success classifier, so the numbers measure whether closed-loop policy optimisation inside the model works at all, not what the best attainable score is. What longer optimisation would buy, and what it would risk, is discussed under limitations in Section 6.

## 4.5 Ablation Study

We run component ablations rather than stage comparisons. Action representation, action conditioning and multiview consistency each isolate one midtraining choice, and we train all three from the pretrained checkpoint rather than the midtrained one, so that the stronger priors midtraining already supplies cannot mask the contribution of the component under test. The distillation ablation is the exception, since post-training begins from the midtrained model by construction and its variants are therefore compared from that checkpoint. The midtraining ablations use a small in-house dataset collected for folding clothes on one humanoid embodiment.

This setup has a scope worth stating once rather than in every subsection. Each ablation below is run on a single embodiment, an in-house humanoid for some and the simulated RoboTwin 2.0 [11] setup for others, so the comparisons isolate how one component behaves within one kinematic frame; none of them measures cross-embodiment behaviour, and the platform adaptation results in Sec. 4.2 are the evidence that speaks to that. A mechanism whose value is specifically cross-embodiment, such as the relative action convention cancelling each robot’s absolute origin, has nothing to act on in either setting, so these numbers are more likely to understate such components than to overstate them.

## 4.5.1 Action Representation

We compare three ways of representing actions: absolute poses, delta poses relative to the previous frame, and our relative action chunk anchored to the first frame of the window. To isolate the effect of the representation itself, we initialize all variants from the pretrained checkpoint rather than the midtrained one, so the comparison is not biased by stronger priors learned during midtraining. Results are shown in Table 9, reported in PSNR [37], SSIM [77], and LPIPS [96].

Table 9 Ablation study on action representation. Best result per metric in bold. ↑: higher is better; ↓: lower is better.

| Action Representation | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- |
| Absolute action | 22.23 | 0.79 | 0.14 |
| Delta action | 21.34 | 0.81 | 0.18 |
| Relative action chunk | 24.56 | 0.84 | 0.13 |

## 4.5.2 Action Conditioning

We ablate our two action conditioning mechanisms, framewise action conditioning and the frame-action adapter, by applying each in isolation and then together. Because prior work [39, 67] also uses two-channel action conditioning but relies on standard cross-attention and AdaLN-zero [62], we compare each proposed channel against its conventional counterpart: framewise conditioning against vanilla cross-attention, and the token-level adapter against AdaLN. The final row tests both proposed channels together. Results are shown in Table 10.

Reconstruction metrics do not measure controllability, so we test action following directly. We evaluate on counterfactual actions, meaning action sequences that do not correspond to the recorded trajectory: faithful generation therefore requires the model to respond to the commanded motion rather than replay the recorded one. PrimeSim follows these out-of-distribution commands, as shown in Fig. 10, where static scene elements remain stable across the rollout while the commanded motion is carried out.

A qualitative figure cannot by itself settle whether the motion is genuinely action-driven, and we do not ask it to. WorldSimProbe provides the independent quantitative test, and its T3 suite, action-source behaviour

23

<!-- page 24 of 42 -->

![Image block](doc:8ea4708/tier:basic/page:24/block:1)

(a)

![Image block](doc:8ea4708/tier:basic/page:24/block:3)

(b)

Figure 10 Counterfactual action following on LIBERO. Each row shows seven frames sampled uniformly from a single PrimeSim rollout, generated conditional on an action sequence rather than a task prompt. Our evaluation is built on counterfactual commands, meaning action trajectories that do not correspond to the demonstrated behaviour. Faithful generation therefore requires the model to respond to the commanded motion rather than replay the recorded one. Static scene elements remain stable across the rollout while the commanded motion is carried out, indicating that the generated future is driven by the action rather than by visual plausibility alone.

preservation, asks precisely the counterfactual question: whether behaviour survives when the actions originate from a different source than the one that produced the video. Against five open-source actionconditioned world models, PrimeSim ranks first on T3 (72.3) and on the coarse trajectory axis T2 (67.9), and second on local calibration T1 (70.2 against LingBot-VA’s 77.0) (Table 1). The two lines of evidence agree, and they agree on the residual weakness as well: action following holds at trajectory scale, while fine-grained local calibration is not yet the best in the field.

Table 10 Ablation study on action conditioning. The Baseline column uses vanilla cross-attention for framewise conditioning and Adaptive LayerNorm (AdaLN) for the adapter; Ours uses modified cross-attention and token fusion and shifting, respectively. The final row combines both mechanisms. Higher PSNR is better.

| Action Conditi | oning Mechanis | m PSNR | ↑ |
| --- | --- | --- | --- |
| Framewise | Adapter | Baseline | Ours |
| ✓ | × | 23.23 | 24.10 |
| × | ✓ | 20.10 | 20.24 |
| ✓ | ✓ | 23.31 | 24.56 |

## 4.5.3 Multiview Consistency

The three views share one transformer stack rather than separate per-camera networks. Each of the 30 DiT blocks attends within its own camera view, and every third block instead spans all three, so relations inside a view are established before information is periodically integrated across views. A learned camera token marks which view each token belongs to.

We perform an incremental ablation of the multiview consistency mechanisms, introducing each component in turn to isolate its contribution. The baseline concatenates frames along the height dimension without any conditioning. We then add a view token that conditions each view, and finally apply intra-view attention across the frame tokens. Results are shown in Table 11.

24

<!-- page 25 of 42 -->

![Image block](doc:8ea4708/tier:basic/page:25/block:1)

Figure 11 Joint multiview generation. Rows are cameras, from top to bottom head, left wrist and right wrist; columns are seven frames sampled uniformly across a single 33-frame PrimeSim rollout. The three views are generated together under a single action sequence rather than independently, and all respond to the same commanded motion. The wrist views change most over the interval, consistent with their proximity to the manipulation, while the head view tracks the arms throughout.

![Image block](doc:8ea4708/tier:basic/page:25/block:3)

Figure 12 Memory under occlusion. Five frames from a single RoboTwin 2.0 rollout, head camera. The arm sweeps across the workspace and its hand passes in front of the shoe in the lower left (fourth frame), so the shoe leaves the camera’s view entirely. When the hand clears, the shoe is present again at the same position, scale and orientation rather than being re-invented or dropped. Each window is conditioned on a selected KV-cache memory rather than on the previous frame alone, so an object occluded for several seconds is carried forward instead of being lost when it re-enters view.

Table 11 Ablation study on multiview consistency. Components are added incrementally from naive frame concatenation. Best result in bold.

| Multiview Mechanism | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- |
| Frame concatenation only | 16.30 | 0.62 | 0.58 |
| + View token conditioning | 23.78 | 0.83 | 0.18 |
| + Intra-view attention | 24.56 | 0.84 | 0.15 |

## 4.6 Qualitative Results

We show rollouts that are easier to judge by eye than by metric, and that the quantitative sections do not otherwise cover.

First, a sequence isolates what the temporal memory buys under occlusion (Fig. 12).

A second sequence shows rollouts on deformable cloth, the material class where long-horizon coherence is hardest to maintain (Fig. 13).

Two further sequences give a qualitative view of the EWMBench generations behind Table 2 (Fig. 14).

## 5 Related Work

**Video Generation in Robotics** Recent advances in general-purpose video generation [15, 75, 87] have catalyzed a growing body of work applying these models to robotics. One prominent direction leverages video generation for data expansion: by synthesizing diverse visual trajectories, these models augment limited teleoperation data, enabling broader coverage of environments, objects, and tasks [47, 57, 95, 99].

25

<!-- page 26 of 42 -->

![Image block](doc:8ea4708/tier:basic/page:26/block:1)

Figure 13 Cloth manipulation rollouts, seven frames sampled uniformly from each. All three are AgiBot G2 humanoid episodes shown from the head camera: a white cloth on a bamboo table beside a metal bowl (top two rows), and a blue cloth on a bedding surface with a folded garment to its right (bottom row). The garment deforms continuously across each rollout and stays distinguishable from the surface beneath it rather than dissolving into the background, which is the failure mode deformable objects expose over long horizons.

![Image block](doc:8ea4708/tier:basic/page:26/block:3)

Figure 14 Generated rollouts on two EWMBench tasks, the qualitative companion of Table 2 (Sec. 4.1.2). Each row shows seven frames sampled uniformly from a single rollout, generated open-loop at 10 FPS from the first frame, the task instruction and the ground-truth action trajectory, and shown at the 480 × 640 frame size used for scoring. Top: the arm picks a bottle from the conveyor and packs it into a cardboard box already holding other items. Bottom: the arm places a slice of bread onto the plate beside the toaster.

This mitigates a central bottleneck in robot learning—the scarcity of large-scale, diverse demonstrations. A complementary direction exploits the spatiotemporal physical understanding that video models acquire through large-scale pretraining. Rather than treating generated video merely as data, these works use the model’s internal representations as a backbone for action prediction and planning, demonstrating strong performance on downstream robotic tasks [5, 10, 21, 41, 46, 50, 58, 60, 88]. Together, these two lines of work reflect a broader shift: video generation models are evolving from passive data synthesizers into active components of the robot learning pipeline, serving simultaneously as sources of supervision and as engines of physical reasoning.

**Interactive World Models** Closely related to video generation is the emerging area of interactive world models, in which generation is conditioned on and steered by actions [4, 7, 20, 23, 26, 55, 66, 70, 73]. Much of this work targets interactive video games, from early action-conditioned systems [4, 7] to more recent higher-fidelity, longer-horizon ones [20, 23, 55, 66, 73]. More recently, the paradigm has extended to robotics [25, 28, 29, 39, 48, 49, 52, 53, 63, 68, 69, 72, 76, 84]. These robot-centric models differ from their game-oriented counterparts along several axes: they emphasize multi-view consistency [29, 63, 84] rather than a single fixed viewpoint; they operate on continuous, physically grounded control signals [29, 53] rather than discrete symbolic inputs [4, 7]; and they prioritize geometric and physical consistency [48, 49, 76] over visual aesthetics and responsiveness [23, 66]. Their use cases also diverge: game-based models target entertainment and content creation, whereas robot-based ones serve as learned simulators providing synthetic feedback for action evaluation, policy learning, and planning [25, 28, 39, 52, 68, 69, 72].

26

<!-- page 27 of 42 -->

## 6 Limitations and Conclusions

**Limitations.** Our evaluation is broad but not exhaustive, and we prefer to state the gaps rather than let them be inferred. One concerns the model itself: PrimeSim generates at 192 × 256 per view, so our claims concern simulator fidelity, whether actions are realized and the environment responds, rather than pixel-level realism, and we make no like-for-like visual fidelity comparison against models operating at higher resolution.

A further gap is a question of resources rather than of method. Adaptation to a new platform is few-shot, requiring 100 to 200 gradient steps on that platform’s own data, rather than zero-shot. We expect this gap to close with more midtraining GPU-hours, tighter alignment of each corpus’s action and state conventions, and a larger and more diverse corpus, since all three act on the same quantity that makes the interface transferable at all. We have not demonstrated it, and state it as an expectation rather than a result.

A final gap concerns scope, and it is the one we consider most important. Most of our quantitative evidence, and all of our closed-loop evidence, comes from simulation: the benchmark comparisons are on LIBERO, and policy training and evaluation inside PrimeSim are validated in simulation only. Our real-platform results establish that PrimeSim generates faithful video on real data, zero-shot on XVLA and after few-shot adaptation to xArm 7, ViperX 300 and a UMI gripper, and that it follows commanded action across the whole action space including chassis velocity. What they do not yet establish is the simulator role on physical hardware. Whether generated rollouts are faithful enough to stand in for simulator rollouts when improving a policy on a real platform remains open, and it is the direction we intend to pursue next.

**Conclusions.** We presented PrimeSim, a world model that turns a video generation backbone into an actionconditioned, multiview, causal simulator of robot manipulation. A three-stage curriculum does the work: midtraining on a 30,110-hour cross-embodiment corpus, midtraining that introduces action and multiview conditioning, and posttraining that converts the model into a causal few-step autoregressive generator. The result conditions on actions and the task prompt, but on no embodiment identifier at inference. Against five open-source action-conditioned world models, PrimeSim leads the WorldSimProbe axes that measure action realization at trajectory scale and interaction dynamics, and it reaches those results at 10.1 FPS through four-step distillation rather than the fifty steps the teacher requires. Its features transfer: three held-out platforms reach action-following rollouts within 100 to 200 gradient steps, where a general video prior given ten to twenty times that budget does not. We expect the combination of a large cross-embodiment corpus, an explicit action interface and a causal few-step sampler to be a practical route toward learned simulators that are fast and controllable enough to train policies inside.

27

<!-- page 28 of 42 -->

## References

[1] AgiBot World. Agibot world challenge 2026. [https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2026,](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2026) 2026.

[2] AgiBot-World-Contributors, Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, Shu Jiang, Yuxin Jiang, Cheng Jing, Hongyang Li, Jialu Li, Chiming Liu, Yi Liu, Yuxiang Lu, Jianlan Luo, Ping Luo, Yao Mu, Yuehan Niu, Yixuan Pan, Jiangmiao Pang, Yu Qiao, Guanghui Ren, Cheng Ruan, Jiaqi Shan, Yongjian Shen, Chengshi Shi, Mingkang Shi, Modi Shi, Chonghao Sima, Jianheng Song, Huijie Wang, Wenhao Wang, Dafeng Wei, Chengen Xie, Guo Xu, Junchi Yan, Cunbiao Yang, Lei Yang, Shukai Yang, Maoqing Yao, Jia Zeng, Chi Zhang, Qinglin Zhang, Bin Zhao, Chengyue Zhao, Jiaqi Zhao, and Jianchao Zhu. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems, 2025. URL [https://arxiv.org/abs/2503.06669.](https://arxiv.org/abs/2503.06669)

[3] Simple AI, :, Yuteng Wei, Jinming Ma, Jiawei Wang, Weitao Zhou, Yushen Zuo, Ke Rui, Minglei Li, Jinhao Zhang, Zhikang Pan, Xiang Wang, Haoran Jia, Huan Du, Zicheng Zeng, Jun Ma, Guiyu Qin, Di Zhang, and Xiaofei Li. Hifi-umi: Learning deployable manipulation policies from high-fidelity umi data alone, 2026. URL [https://arxiv.org/abs/2607.25895.](https://arxiv.org/abs/2607.25895)

[4] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in atari, 2024. URL [https://arxiv.org/abs/2405.12399.](https://arxiv.org/abs/2405.12399)

[5] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, and Jun Zhu. Motus: A unified latent action world model, 2025. URL [https://arxiv.org/abs/2512.13030.](https://arxiv.org/abs/2512.13030)

[6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakr ishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for real-world control at scale, 2023. URL [https://arxiv.org/abs/2212.06817.](https://arxiv.org/abs/2212.06817)

[7] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, and Tim Rocktäschel. Genie: Generative interactive environments, 2024. URL [https://arxiv.org/abs/2402.15391.](https://arxiv.org/abs/2402.15391)

[8] Remi Cadene, Simon Aliberts, Francesco Capuano, Michel Aractingi, Adil Zouitine, Pepijn Kooijmans, Jade Choghari, Martino Russi, Caroline Pascal, Steven Palma, Mustafa Shukor, Jess Moss, Alexander Soare, Dana Aubakirova, Quentin Lhoest, Quentin Gallouédec, and Thomas Wolf. Lerobot: An open-source library for end-to-end robot learning, 2026. URL [https://arxiv.org/abs/2602.22818.](https://arxiv.org/abs/2602.22818)

[9] Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion, 2024. URL [https://arxiv.org/abs/2407.01392.](https://arxiv.org/abs/2407.01392)

[10] Boyuan Chen, Tianyuan Zhang, Haoran Geng, Caiyi Zhang, Peihao Li, Kiwhan Song, William T. Freeman, Jitendra Malik, Pieter Abbeel, Russ Tedrake, Vincent Sitzmann, and Yilun Du. Large video planner enables generalizable robot control, 2026. URL [https://arxiv.org/abs/2512.15840.](https://arxiv.org/abs/2512.15840)

[11] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, and Yao Mu. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation, 2025. URL [https://arxiv.org/abs/2506.18088.](https://arxiv.org/abs/2506.18088)

[12] Xinyi Chen, Yilun Chen, Yanwei Fu, Ning Gao, Jiaya Jia, Weiyang Jin, Hao Li, Yao Mu, Jiangmiao Pang, Yu Qiao, Yang Tian, Bin Wang, Bolun Wang, Fangjing Wang, Hanqing Wang, Tai Wang, Ziqin Wang, Xueyuan Wei, Chao Wu, Shuai Yang, Jinhui Ye, Junqiu Yu, Jia Zeng, Jingjing Zhang, Jinyu Zhang, Shi Zhang, Feng Zheng, Bowen Zhou,

28

<!-- page 29 of 42 -->

and Yangkun Zhu. Internvla-m1: A spatially guided vision-language-action framework for generalist robot policy, 2025. URL [https://arxiv.org/abs/2510.13778.](https://arxiv.org/abs/2510.13778)

[13] Yuzhi Chen, Ronghan Chen, Dongjie Huo, Yandan Yang, Dekang Qi, Haoyun Liu, Tong Lin, Shuang Zeng, Junjin Xiao, Xinyuan Chang, Feng Xiong, Xing Wei, Zhiheng Ma, and Mu Xu. Abot-physworld: Interactive world foundation model for robotic manipulation with physics alignment, 2026. URL [https://arxiv.org/abs/2603.23376](https://arxiv.org/abs/2603.23376).

[14] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots, 2024. URL [https://arxiv.org/abs/2402.10329.](https://arxiv.org/abs/2402.10329)

[15] Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, Weishi Mi, Kevin Zhang, Zhiyuan Qin, Wanxin Tian, Kuangzhi Ge, Hao Li, Zezhong Qian, Anthony Chen, Qiang Zhou, Yueru Jia, Jiaming Liu, Yong Dai, Qingpo Wuwu, Chengyu Bai, Yu-Kai Wang, Ying Li, Lizhang Chen, Yong Bao, Zhiyuan Jiang, Jiacheng Zhu, Kai Tang, Ruichuan An, Yulin Luo, Qiuxuan Feng, Siyuan Zhou, Chi min Chan, Chengkai Hou, Wei Xue, Sirui Han, Yike Guo, Shanghang Zhang, and Jian Tang. Wow: Towards a world omniscient world model through embodied interaction, 2025. URL [https://arxiv.org/abs/2509.22642.](https://arxiv.org/abs/2509.22642)

[16] Peterson Co, Sicheng Hu, Chunxuan Jiao, Hongyang Cheng, Yulin Luo, Yijie Xu, Sixiang Chen, Zhongxia Zhao, Zihao Wang, DaFeng Chi, Peidong Liu, YuTong Chen, Henghua Liu, Zhihao Yuan, Huizhu Jia, Yuzheng Zhuang, Tianle Zhang, Liang Lin, Huajie Tan, and Shanghang Zhang. Worldsimprobe: Diagnosing simulator faithfulness in action-conditioned world models for embodied manipulation, 2026. URL [https://arxiv.org/abs/2608.09298.](https://arxiv.org/abs/2608.09298)

[17] Embodiment Collaboration, Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, Albert Tung, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anchit Gupta, Andrew Wang, Andrey Kolobov, Anikait Singh, Animesh Garg, Aniruddha Kembhavi, Annie Xie, Anthony Brohan, Antonin Raffin, Archit Sharma, Arefeh Yavary, Arhan Jain, Ashwin Balakrishna, Ayzaan Wahid, Ben Burgess-Limerick, Beomjoon Kim, Bernhard Schölkopf, Blake Wulfe, Brian Ichter, Cewu Lu, Charles Xu, Charlotte Le, Chelsea Finn, Chen Wang, Chenfeng Xu, Cheng Chi, Chenguang Huang, Christine Chan, Christopher Agia, Chuer Pan, Chuyuan Fu, Coline Devin, Danfei Xu, Daniel Morton, Danny Driess, Daphne Chen, Deepak Pathak, Dhruv Shah, Dieter Büchler, Dinesh Jayaraman, Dmitry Kalashnikov, Dorsa Sadigh, Edward Johns, Ethan Foster, Fangchen Liu, Federico Ceola, Fei Xia, Feiyu Zhao, Felipe Vieira Frujeri, Freek Stulp, Gaoyue Zhou, Gaurav S. Sukhatme, Gautam Salhotra, Ge Yan, Gilbert Feng, Giulio Schiavi, Glen Berseth, Gregory Kahn, Guangwen Yang, Guanzhi Wang, Hao Su, Hao-Shu Fang, Haochen Shi, Henghui Bao, Heni Ben Amor, Henrik I Christensen, Hiroki Furuta, Homanga Bharadhwaj, Homer Walke, Hongjie Fang, Huy Ha, Igor Mordatch, Ilija Radosavovic, Isabel Leal, Jacky Liang, Jad Abou-Chakra, Jaehyung Kim, Jaimyn Drake, Jan Peters, Jan Schneider, Jasmine Hsu, Jay Vakil, Jeannette Bohg, Jeffrey Bingham, Jeffrey Wu, Jensen Gao, Jiaheng Hu, Jiajun Wu, Jialin Wu, Jiankai Sun, Jianlan Luo, Jiayuan Gu, Jie Tan, Jihoon Oh, Jimmy Wu, Jingpei Lu, Jingyun Yang, Jitendra Malik, João Silvério, Joey Hejna, Jonathan Booher, Jonathan Tompson, Jonathan Yang, Jordi Salvador, Joseph J. Lim, Junhyek Han, Kaiyuan Wang, Kanishka Rao, Karl Pertsch, Karol Hausman, Keegan Go, Keerthana Gopalakrishnan, Ken Goldberg, Kendra Byrne, Kenneth Oslund, Kento Kawaharazuka, Kevin Black, Kevin Lin, Kevin Zhang, Kiana Ehsani, Kiran Lekkala, Kirsty Ellis, Krishan Rana, Krishnan Srinivasan, Kuan Fang, Kunal Pratap Singh, Kuo-Hao Zeng, Kyle Hatch, Kyle Hsu, Laurent Itti, Lawrence Yunliang Chen, Lerrel Pinto, Li Fei-Fei, Liam Tan, Linxi "Jim" Fan, Lionel Ott, Lisa Lee, Luca Weihs, Magnum Chen, Marion Lepert, Marius Memmel, Masayoshi Tomizuka, Masha Itkina, Mateo Guaman Castro, Max Spero, Maximilian Du, Michael Ahn, Michael C. Yip, Mingtong Zhang, Mingyu Ding, Minho Heo, Mohan Kumar Srirama, Mohit Sharma, Moo Jin Kim, Muhammad Zubair Irshad, Naoaki Kanazawa, Nicklas Hansen, Nicolas Heess, Nikhil J Joshi, Niko Suenderhauf, Ning Liu, Norman Di Palo, Nur Muhammad Mahi Shafiullah, Oier Mees, Oliver Kroemer, Osbert Bastani, Pannag R Sanketi, Patrick "Tree" Miller, Patrick Yin, Paul Wohlhart, Peng Xu, Peter David Fagan, Peter Mitrano, Pierre Sermanet, Pieter Abbeel, Priya Sundaresan, Qiuyu Chen, Quan Vuong, Rafael Rafailov, Ran Tian, Ria Doshi, Roberto Martín-Martín, Rohan Baijal, Rosario Scalise, Rose Hendrix, Roy Lin, Runjia Qian, Ruohan Zhang, Russell Mendonca, Rutav Shah, Ryan Hoque, Ryan Julian, Samuel Bustamante, Sean Kirmani, Sergey Levine, Shan Lin, Sherry Moore, Shikhar Bahl, Shivin Dass, Shubham Sonawani, Shubham Tulsiani, Shuran Song, Sichun Xu, Siddhant Haldar, Siddharth Karamcheti, Simeon Adebola, Simon Guist, Soroush Nasiriany, Stefan Schaal, Stefan Welker, Stephen Tian, Subramanian Ramamoorthy, Sudeep Dasari, Suneel Belkhale, Sungjae Park, Suraj Nair, Suvir Mirchandani, Takayuki Osa, Tanmay Gupta, Tatsuya Harada, Tatsuya Matsushima, Ted Xiao, Thomas Kollar, Tianhe Yu, Tianli Ding, Todor Davchev, Tony Z. Zhao, Travis Armstrong, Trevor Darrell, Trinity Chung, Vidhi Jain, Vikash Kumar, Vincent Vanhoucke, Vitor Guizilini, Wei Zhan, Wenxuan Zhou, Wolfram Burgard, Xi Chen, Xiangyu Chen, Xiaolong Wang, Xinghao Zhu, Xinyang Geng, Xiyuan Liu, Xu Liangwei, Xuanlin Li, Yansong Pang, Yao Lu, Yecheng Jason Ma, Yejin Kim, Yevgen Chebotar, Yifan Zhou, Yifeng Zhu, Yilin Wu, Ying Xu,

29

<!-- page 30 of 42 -->

Yixuan Wang, Yonatan Bisk, Yongqiang Dou, Yoonyoung Cho, Youngwoon Lee, Yuchen Cui, Yue Cao, Yueh-Hua Wu, Yujin Tang, Yuke Zhu, Yunchu Zhang, Yunfan Jiang, Yunshuang Li, Yunzhu Li, Yusuke Iwasawa, Yutaka Matsuo, Zehan Ma, Zhuo Xu, Zichen Jeff Cui, Zichen Zhang, Zipeng Fu, and Zipeng Lin. Open x-embodiment: Robotic learning datasets and rt-x models, 2025. URL [https://arxiv.org/abs/2310.08864.](https://arxiv.org/abs/2310.08864)

[18] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-forcing++: Towards minute-scale high-quality video generation, 2025. URL [https://arxiv.org/abs/2510.02283](https://arxiv.org/abs/2510.02283).

[19] Tianyuan Dai, Josiah Wong, Yunfan Jiang, Chen Wang, Cem Gokmen, Ruohan Zhang, Jiajun Wu, and Li Fei-Fei. Automated creation of digital cousins for robust policy learning, 2024. URL [https://arxiv.org/abs/2410.07408](https://arxiv.org/abs/2410.07408).

[20] Nan Duan, Haoyang Huang, Weiyang Jin, Haoran Li, Yaowei Li, Yuming Li, Yijun Liu, Xin Lu, Xiaoxiao Ma, Yanwen Ma, Yaofeng Su, Yilang Sun, Haoyu Wang, Zeyue Xue, Songchun Zhang, and Junhao Zhuang. Long-horizon audio visual generation for persistent stories and interactive worlds, 2026. URL [https://arxiv.org/abs/2608.23383.](https://arxiv.org/abs/2608.23383)

[21] Dyna Robotics. Dyna-2: A 1-million-hour scaling law for world-action models. August 2026. URL [https://dyna.co/dyna-2.](https://dyna.co/dyna-2)

[22] Gunnar Farnebäck. Two-frame motion estimation based on polynomial expansion. In Josef Bigun and Tomas Gustavsson, editors, Image Analysis, pages 363–370, Berlin, Heidelberg, 2003. Springer Berlin Heidelberg. ISBN 978-3-540-45103-7.

[23] Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei Shu, Zhiheng Liu, Andy Zheng, Yukun Huang, Yu Liu, and Hongyang Zhang. The matrix: Infinite-horizon world generation with real-time moving control, 2024. URL [https://arxiv.org/abs/2412.03568.](https://arxiv.org/abs/2412.03568)

[24] Peng Gao, Jichao Jiao, Chen Liao, Daqin Luo, Yiming Ma, Baiyu Pan, Hailin Shi, Jiyuan Wang, Xiangming Xu, and Xiongwei Zhao. Thinker-wm: A unified world model with shared multimodal representations for embodied intelligence, 2026. URL [https://arxiv.org/abs/2610.18447.](https://arxiv.org/abs/2610.18447)

[25] Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, Qianli Ma, Seungjun Nah, Loic Magne, Jiannan Xiang, Yuqi Xie, Ruijie Zheng, Dantong Niu, You Liang Tan, K. R. Zentner, George Kurian, Suneel Indupuru, Pooya Jannaty, Jinwei Gu, Jun Zhang, Jitendra Malik, Pieter Abbeel, Ming-Yu Liu, Yuke Zhu, Joel Jang, and Linxi "Jim" Fan. Dreamdojo: A generalist robot world model from large-scale human videos, 2026. URL [https://arxiv.org/abs/2602.06949.](https://arxiv.org/abs/2602.06949)

[26] Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, Ka Leong Cheng, Haojie Zhang, Jian Gao, Tianrui Feng, Yuzheng Liu, Yao Yao, Yinghao Xu, Xing Zhu, Yujun Shen, and Hao Ouyang. Infinite worlds with versatile interactions, 2026. URL [https://arxiv.org/abs/2607.07534.](https://arxiv.org/abs/2607.07534)

[27] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang.

30

<!-- page 31 of 42 -->

Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, Sept 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL [http://dx.doi.org/10.1038/s41586-025-09422-z.](http://dx.doi.org/10.1038/s41586-025-09422-z)

[28] Yanjiang Guo, Tony Lee, Lucy Xiaoyang Shi, Jianyu Chen, Percy Liang, and Chelsea Finn. Vlaw: Iterative co-improvement of vision-language-action policy and world model, 2026. URL [https://arxiv.org/abs/2602.12063](https://arxiv.org/abs/2602.12063).

[29] Yanjiang Guo, Lucy Xiaoyang Shi, Jianyu Chen, and Chelsea Finn. Ctrl-world: A controllable generative world model for robot manipulation, 2026. URL [https://arxiv.org/abs/2510.10125.](https://arxiv.org/abs/2510.10125)

[30] David Ha and Jürgen Schmidhuber. World models. 2018. doi: 10.5281/ZENODO.1207631. URL [https://zenodo.org/record/1207631.](https://zenodo.org/record/1207631)

[31] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination, 2020. URL [https://arxiv.org/abs/1912.01603.](https://arxiv.org/abs/1912.01603)

[32] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models, 2024. URL [https://arxiv.org/abs/2301.04104.](https://arxiv.org/abs/2301.04104)

[33] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL [https://arxiv.org/abs/2106.09685.](https://arxiv.org/abs/2106.09685)

[34] Mu Huang, Hui Wang, Kerui Ren, Linning Xu, Yunsong Zhou, Mulin Yu, Bo Dai, and Jiangmiao Pang. Soma: A real-to-sim neural simulator for robotic soft-body manipulation, 2026. URL [https://arxiv.org/abs/2602.02402](https://arxiv.org/abs/2602.02402).

[35] Siyuan Huang, Liliang Chen, Pengfei Zhou, Shengcong Chen, Yue Liao, Zhengkai Jiang, Yue Hu, Peng Gao, Hongsheng Li, Maoqing Yao, et al. Enerverse: Envisioning embodied future space for robotics manipulation. Advances in Neural Information Processing Systems, 38:37693–37720, 2026.

[36] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

[37] Q. Huynh-Thu and Mohammed Ghanbari. Scope of validity of psnr in image/video quality assessment. Electronics Letters, 44:800 – 801, 02 2008. doi: 10.1049/el:20080522.

[38] Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and g0 dual-system vla model, 2025. URL [https://arxiv.org/abs/2509.00576](https://arxiv.org/abs/2509.00576).

[39] Zhennan Jiang, Shangqing Zhou, Yutong Jiang, Zefang Huang, Mingjie Wei, Yuhui Chen, Tianxing Zhou, Zhen Guo, Hao Lin, Quanlu Zhang, Yu Wang, Haoran Li, Chao Yu, and Dongbin Zhao. Wovr: World models as reliable simulators for post-training vla policies with rl, 2026. URL [https://arxiv.org/abs/2602.13977.](https://arxiv.org/abs/2602.13977)

[40] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, Vitor Guizilini, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Muhammad Zubair Irshad, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Martín-Martín, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. Droid: A large-scale in-the-wild robot manipulation dataset, 2025. URL [https://arxiv.org/abs/2403.12945.](https://arxiv.org/abs/2403.12945)

[41] Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos policy: Fine-tuning video models for visuomotor control and planning, 2026. URL [https://arxiv.org/abs/2601.16163.](https://arxiv.org/abs/2601.16163)

[42] Kuaishou Technology. Kling: A progressive framework for video generation, 2024. URL [https://klingai.com.](https://klingai.com)

[43] Nathan Lambert. Reinforcement learning from human feedback, 2026. URL [https://arxiv.org/abs/2504.12501](https://arxiv.org/abs/2504.12501).

31

<!-- page 32 of 42 -->

[44] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Wensi Ai, Benjamin Martinez, Hang Yin, Michael Lingelbach, Minjune Hwang, Ayano Hiranaka, Sujay Garlanka, Arman Aydin, Sharon Lee, Jiankai Sun, Mona Anvari, Manasi Sharma, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan Lou, Caleb R Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Yunzhu Li, Silvio Savarese, Hyowon Gweon, C. Karen Liu, Jiajun Wu, and Li Fei-Fei. Behavior-1k: A human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation, 2024. URL [https://arxiv.org/abs/2403.09227.](https://arxiv.org/abs/2403.09227)

[45] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu. Causal world modeling for robot control, 2026. URL [https://arxiv.org/abs/2601.21998](https://arxiv.org/abs/2601.21998).

[46] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model, 2025. URL [https://arxiv.org/abs/2503.00200.](https://arxiv.org/abs/2503.00200)

[47] Xinghang Li, Jun Guo, Qiwei Li, Long Qian, Hang Lai, Yueze Wang, Hongyu Yan, Jiahang Cao, Xi Chen, Jingen Qu, Jiaxi Song, Nan Sun, Hanye Zhao, Futeng Liu, Wanli Peng, Heyun Wang, Yunhong Wang, Caoyu Xia, Jack Zhao, Diyun Xiang, Hangjun Ye, Heng Qu, Huaping Liu, and Jason Li. Xiaomi-robotics-u0: Unified embodied synthesis with world foundation model, 2026. URL [https://arxiv.org/abs/2607.11643.](https://arxiv.org/abs/2607.11643)

[48] Yaxuan Li, Yichen Zhu, Junjie Wen, Chaomin Shen, and Yi Xu. Worldeval: World model as real-world robot policies evaluator, 2025. URL [https://arxiv.org/abs/2505.19017.](https://arxiv.org/abs/2505.19017)

[49] Yaxuan Li, Zhongyi Zhou, Yefei Chen, Yaokai Xue, and Yichen Zhu. dworldeval: Scalable robotic policy evaluation via discrete diffusion world model, 2026. URL [https://arxiv.org/abs/2604.22152.](https://arxiv.org/abs/2604.22152)

[50] Yue Liao, Pengfei Zhou, Siyuan Huang, Donglin Yang, Shengcong Chen, Yuxin Jiang, Yue Hu, Jingbin Cai, Si Liu, Jianlan Luo, Liliang Chen, Shuicheng Yan, Maoqing Yao, and Guanghui Ren. Genie envisioner: A unified world foundation platform for robotic manipulation, 2025. URL [https://arxiv.org/abs/2508.05635.](https://arxiv.org/abs/2508.05635)

[51] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning, 2023. URL [https://arxiv.org/abs/2306.03310.](https://arxiv.org/abs/2306.03310)

[52] Xiaokang Liu, Zechen Bai, Hai Ci, Kevin Yuchen Ma, and Mike Zheng Shou. World-vla-loop: Closed-loop learning of video world model and vla policy, 2026. URL [https://arxiv.org/abs/2602.06508.](https://arxiv.org/abs/2602.06508)

[53] Yuejiang Liu, Fan Feng, Lingjing Kong, Weifeng Lu, Jinzhou Tang, Kun Zhang, Kevin Murphy, Chelsea Finn, and Yilun Du. World action verifier: Self-improving world models via forward-inverse asymmetry, 2026. URL [https://arxiv.org/abs/2604.01985.](https://arxiv.org/abs/2604.01985)

[54] Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations, 2023. URL [https://arxiv.org/abs/2310.17596.](https://arxiv.org/abs/2310.17596)

[55] Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, and Kaipeng Zhang. Yume-1.5: A text-controlled interactive world generation model, 2025. URL https://arxiv.org/abs[2512.22096](https://arxiv.org/abs/2512.22096).

[56] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots, 2024. URL [https://arxiv.org/abs/2406.02523.](https://arxiv.org/abs/2406.02523)

[57] NVIDIA, :, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, Daniel Dworakowski, Jiaojiao Fan, Michele Fenzi, Francesco Ferroni, Sanja Fidler, Dieter Fox, Songwei Ge, Yunhao Ge, Jinwei Gu, Siddharth Gururani, Ethan He, Jiahui Huang, Jacob Huffman, Pooya Jannaty, Jingyi Jin, Seung Wook Kim, Gergely Klár, Grace Lam, Shiyi Lan, Laura Leal-Taixe, Anqi Li, Zhaoshuo Li, Chen-Hsuan Lin, Tsung-Yi Lin, Huan Ling, Ming-Yu Liu, Xian Liu, Alice Luo, Qianli Ma, Hanzi Mao, Kaichun Mo, Arsalan Mousavian, Seungjun Nah, Sriharsha Niverty, David Page, Despoina Paschalidou, Zeeshan Patel, Lindsey Pavao, Morteza Ramezanali, Fitsum Reda, Xiaowei Ren, Vasanth Rao Naik Sabavat, Ed Schmerling, Stella Shi, Bartosz Stefaniak, Shitao Tang, Lyne Tchapmi, Przemek Tredak, Wei-Cheng Tseng, Jibin Varghese, Hao Wang, Haoxiang Wang, Heng Wang, Ting-Chun Wang, Fangyin Wei, Xinyue Wei, Jay Zhangjie Wu, Jiashu Xu, Wei Yang, Lin Yen-Chen, Xiaohui Zeng, Yu Zeng, Jing Zhang, Qinsheng Zhang, Yuxuan Zhang, Qingqing Zhao, and Artur Zolkowski. Cosmos world foundation model platform for physical ai, 2025. URL [https://arxiv.org/abs/2501.03575.](https://arxiv.org/abs/2501.03575)

32

<!-- page 33 of 42 -->

[58] NVIDIA, :, Aditi, Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, Aarti Basant, Mukesh Beladiya, Mohammad Qazim Bhat, Zaid Pervaiz Bhat, Dan Blick, Vanni Brighella, Han Cai, Tiffany Cai, Eric Cameracci, Jiaxin Cao, Yulong Cao, Mark Carlson, Carlos Casanova, Ting-Yun Chang, Yan Chang, Yu-Wei Chao, Prithvijit Chattopadhyay, Roshan Chaudhari, Chieh-Yun Chen, Junyu Chen, Ke Chen, Qizhi Chen, Wenkai Chen, Xiaotong Chen, Yu Chen, An-Chieh Cheng, Click Cheng, Xiu Chia, Jeana Choi, Chaeyeon Chung, Wenyan Cong, Yin Cui, Magdalena Dadela, Nalin Dadhich, Wenliang Dai, Joyjit Daw, Alperen Degirmenci, Rodrigo Vieira Del Monte, Robert Denomme, Sameer Dharur, Marco Di Lucca, Ke Ding, Wenhao Ding, Yifan Ding, Yuzhu Dong, Nicole Drumheller, Yilun Du, Aigul Dzhumamuratova, Aleksandr Efitorov, Hamid Eghbalzadeh, Naomi Eigbe, Imad El Hanafi, Hassan Eslami, Benedikt Falk, Jiaojiao Fan, Jim Fan, Amol Fasale, Sergiy Fefilatyev, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Vikram Fugro, Prashant Gaikwad, TJ Galda, Katelyn Gao, Yihuai Gao, Wenhang Ge, Sreyan Ghosh, Arushi Goel, Vivek Goel, Akash Gokul, Rama Govindaraju, Jinwei Gu, Miguel Guerrero, Elfie Guo, Aryaman Gupta, Siddharth Gururani, Hugo Hadfield, Song Han, Ankur Handa, Zekun Hao, Mohammad Harrim, Ali Hassani, Nathan Hayes-Roth, Yufan He, Chris Helvig, Cyrus Hogg, Madison Huang, Michael Huang, Sophia Huang, Yufan Huang, Jacob Huffman, DeLesley Hutchins, Suneel Indupuru, Boris Ivanovic, Arihant Jain, Joel Jang, Ryan Ji, Yanan Jian, Dongfu Jiang, Jingyi Jin, Atharva Joshi, Nikhilesh Joshi, Pranjali Joshi, Andy Ju, Jaehun Jung, Weiwei Kang, Scott Kassekert, Jan Kautz, Ashna Khetan, Julia Kiczka, Slawek Kierat, Gwanghyun Kim, Kuno Kim, Sunny Kim, Kezhi Kong, Xin Kong, Zhifeng Kong, Tomasz Kornuta, Egor Krivov, Hui Kuang, Saurav Kumar, Chia-Wen Kuo, George Kurian, Wojciech Kutak, JF Lafleche, Himangshu Lahkar, Omar Laymoun, Jayjun Lee, Sanggil Lee, Gabriele Leone, Boyi Li, Freya Li, Jiajun Li, Jinfeng Li, Ling Li, Pengcheng Li, Shangru Li, Tingle Li, Xiaolong Li, Xuan Li, Zhaoshuo Li, Zhiqi Li, Hao Liang, Maosheng Liao, Chen-Hsuan Lin, Tsung-Yi Lin, Ming-Yu Liu, Sifei Liu, Zihan Liu, Hai Loc Lu, Xiangyu Lu, Alice Luo, Ruipu Luo, Wenjie Luo, Jiangran Lyu, Martin Ding Ma, Nic Ma, Qianli Ma, Dawid Majchrowski, Louis Marcoux, Miguel Martin, Qing Miao, Ashkan Mirzaei, Shreyas Misra, Kaichun Mo, Durra Mohsin, Hyejin Moon, Pawel Morkisz, Saeid Motiian, Kirill Motkov, Seungjun Nah, Yashraj Narang, Deepak Narayanan, Thabang Ngazimbi, Julian Ouyang, Shubham Pachori, David Page, Yatian Pang, Sehwi Park, Mahesh Patekar, Mostofa Patwary, Marco Pavone, Trung Pham, Wei Ping, Soha Pouya, Shrimai Prabhumoye, Varun Praveen, Delin Qu, Hesam Rabeti, Morteza Ramezanali, Marilyn Reeb, Xuanchi Ren, Kristen Rumley, Wojciech Rymer, Jun Saito, Yeongho Seol, John Shao, Piyush Shekdar, Tianwei Shen, Humphrey Shi, Min Shi, Stella Shi, Kevin Shih, Mohammad Shoeybi, Mateusz Sieniawski, Shuran Song, Alexander Sotelo, Amir Sotoodeh, Sunil Srinivasa, Vignesh Srinivasakumar, Bartosz Stefaniak, Rahul Heinrich Steiger, Shangkun Sun, Jiaxiang Tang, Shitao Tang, Yangyang Tang, Yue Tang, Tolou Tavakkoli, Kayley Ting, Krzysztof Tomala, Wei-Cheng Tseng, Jibin Varghese, Sergei Vasilev, Thomas Volk, Raju Wagwani, Roger Waleffe, Andrew Z. Wang, Boxiang Wang, Haoxiang Wang, Qiao Wang, Shihao Wang, Shijie Wang, Ting-Chun Wang, Yan Wang, Yu Wang, Rohit Watve, David Wehr, Fangyin Wei, Xinshuo Weng, Jay Zhangjie Wu, Kedi Wu, Hongchi Xia, Summer Xiao, Tianjun Xiao, Kevin Xie, Daguang Xu, Jiashu Xu, Mengyao Xu, Ruqing Xu, Xingqian Xu, Yao Xu, Dinghao Yang, Dong Yang, Hans Yang, Xiaodong Yang, Xuning Yang, Yichu Yang, Yurong You, Zhiding Yu, Hao Yuan, Simon Yuen, Xiaohui Zeng, Pengcuo Zeren, Cindy Zha, Haotian Zhang, Jenny Zhang, Jing Zhang, Liangkai Zhang, Paris Zhang, Shun Zhang, Xuanmeng Zhang, Zhizheng Zhang, Ann Zhao, Yilin Zhao, Yuliya Zhautouskaya, Charles Zhou, Fengzhe Zhou, Shilin Zhu, Yuke Zhu, Dima Zhylko, and Artur Zolkowski. Cosmos 3: Omnimodal world models for physical ai, 2026. URL [https://arxiv.org/abs/2606.02800.](https://arxiv.org/abs/2606.02800)

[59] OpenAI. Sora 2, 2025. URL [https://openai.com/sora.](https://openai.com/sora)

[60] Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, and Elvis Nava. mimic-video: Video-action models for generalizable robot control beyond vlas, 2025. URL [https://arxiv.org/abs/2512.15692](https://arxiv.org/abs/2512.15692).

[61] Mingjie Pan, Siyuan Feng, Qinglin Zhang, Xinchen Li, Jianheng Song, Chendi Qu, Yi Wang, Chuankang Li, Ziyu Xiong, Zhi Chen, Yi Liu, and Jianlan Luo. Sop: A scalable online post-training system for vision-language-action models, 2026. URL [https://arxiv.org/abs/2601.03044.](https://arxiv.org/abs/2601.03044)

[62] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. URL [https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748).

[63] Boxiang Qiu, Liliang Chen, Yue Liao, Nan Wang, Lintao Wang, Jiayi Luo, Wenzhi Zhao, Shengcong Chen, Di Chen, Ye Li, Chen Gao, Shuicheng Yan, Si Liu, Maoqing Yao, and Guanghui Ren. Ge-sim 2.0: A roadmap towards comprehensive closed-loop video world simulators for robotic manipulation, 2026. URL [https://arxiv.org/abs/2605.27491](https://arxiv.org/abs/2605.27491).

[64] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL [https://arxiv.org/abs/1707.06347.](https://arxiv.org/abs/1707.06347)

33

<!-- page 34 of 42 -->

[65] Chen Tang, Ben Abbatematteo, Jiaheng Hu, Rohan Chandra, Roberto Martín-Martín, and Peter Stone. Deep reinforcement learning for robotics: A survey of real-world successes, 2024. URL [https://arxiv.org/abs/2408.03539](https://arxiv.org/abs/2408.03539).

[66] Junshu Tang, Jiacheng Liu, Jiaqi Li, Longhuang Wu, Haoyu Yang, Penghao Zhao, Siruis Gong, Xiang Yuan, Shuai Shao, Linfeng Zhang, and Qinglin Lu. Hunyuan-gamecraft-2: Instruction-following interactive game world model, 2026. URL [https://arxiv.org/abs/2511.23429.](https://arxiv.org/abs/2511.23429)

[67] BWM Team. Bwm: A low-cost high-fidelity world simulator for robot learning, 2026. URL [https://arxiv.org/abs/2607.29302](https://arxiv.org/abs/2607.29302).

[68] Gemini Robotics Team, Krzysztof Choromanski, Coline Devin, Yilun Du, Debidatta Dwibedi, Ruiqi Gao, Abhishek Jindal, Thomas Kipf, Sean Kirmani, Isabel Leal, Fangchen Liu, Anirudha Majumdar, Andrew Marmon, Carolina Parada, Yulia Rubanova, Dhruv Shah, Vikas Sindhwani, Jie Tan, Fei Xia, Ted Xiao, Sherry Yang, Wenhao Yu, and Allan Zhou. Evaluating gemini robotics policies in a veo world simulator, 2026. URL [https://arxiv.org/abs/2512.10675](https://arxiv.org/abs/2512.10675).

[69] GigaBrain Team, Boyuan Wang, Bohan Li, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Jie Li, Jindi Lv, Jingyu Liu, Lv Feng, Mingming Yu, Peng Li, Qiuping Deng, Tianze Liu, Xinyu Zhou, Xinze Chen, Xiaofeng Wang, Yang Wang, Yifan Li, Yifei Nie, Yilong Li, Yukun Zhou, Yun Ye, Zhichao Liu, and Zheng Zhu. Gigabrain-0.5m\*: a vla that learns from world model-based reinforcement learning, 2026. URL [https://arxiv.org/abs/2602.12099](https://arxiv.org/abs/2602.12099).

[70] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models, 2026. URL [https://arxiv.org/abs/2601.20540.](https://arxiv.org/abs/2601.20540)

[71] Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, Yaping Li, Ping Wang, Junhao Cai, Jia Zeng, Hao Dong, and Jiangmiao Pang. Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy, 2025. URL [https://arxiv.org/abs/2511.16651.](https://arxiv.org/abs/2511.16651)

[72] Wei-Cheng Tseng, Gashon Hussein, Yuzhu Dong, Allen Z. Ren, Lucy X. Shi, XuDong Wang, Sergey Levine, Zhaoshuo Li, Jinwei Gu, Florian Shkurti, Ming-Yu Liu, and Quan Vuong. Sc3-eval: Evaluating robot foundation models via self-consistent video generation, 2026. URL [https://arxiv.org/abs/2606.18610.](https://arxiv.org/abs/2606.18610)

[73] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines, 2025. URL [https://arxiv.org/abs/2408.14837.](https://arxiv.org/abs/2408.14837)

[74] Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. Bridgedata v2: A dataset for robot learning at scale, 2024. URL [https://arxiv.org/abs/2308.12952.](https://arxiv.org/abs/2308.12952)

[75] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models, 2025. URL [https://arxiv.org/abs/2503.20314](https://arxiv.org/abs/2503.20314).

[76] Yixuan Wang, Rhythm Syed, Fangyu Wu, Mengchao Zhang, Aykut Onol, Jose Barreiros, Hooshang Nayyeri, Tony Dear, Huan Zhang, and Yunzhu Li. Interactive world simulator for robot policy training and evaluation, 2026. URL [https://arxiv.org/abs/2603.08546.](https://arxiv.org/abs/2603.08546)

[77] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004. doi: 10.1109/TIP.2003.819861.

[78] Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, Shichao Fan, Xinhua Wang, Fei Liao, Zhen Zhao, Guangyu Li, Zhao Jin, Lecheng Wang, Jilei Mao, Ning Liu, Pei Ren, Qiang Zhang, Yaoxu Lyu, Mengzhen Liu, He Jingyang, Yulin Luo, Zeyu Gao, Chenxuan Li, Chenyang Gu, Yankai Fu, Di Wu, Xingyu Wang, Sixiang Chen, Zhenyu Wang, Pengju An, Siyuan Qian, Shanghang Zhang, and Jian Tang. Robomind: Benchmark on multi-embodiment intelligence normative data for

34

<!-- page 35 of 42 -->

robot manipulation. In Robotics: Science and Systems XXI, RSS2025. Robotics: Science and Systems Foundation, June 2025. doi: 10.15607/rss.2025.xxi.152. URL [http://dx.doi.org/10.15607/RSS.2025.XXI.152.](http://dx.doi.org/10.15607/RSS.2025.XXI.152)

[79] Philipp Wu, Alejandro Escontrela, Danijar Hafner, Ken Goldberg, and Pieter Abbeel. Daydreamer: World models for physical robot learning, 2022. URL [https://arxiv.org/abs/2206.14176.](https://arxiv.org/abs/2206.14176)

[80] Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, Zhaoye Long, Runtian Xu, Yue Wang, Chong Liu, Dihan Wang, Ziqiang Ni, Xiang Yang, You Liu, Ruoxuan Feng, Lei Zhang, Denghang Huang, Chenghao Jin, Anlan Yin, Xinlong Wang, Zhenguo Sun, Junkai Zhao, Mengfei Du, Mingyu Cao, Xiansheng Chen, Hongyang Cheng, Xiaojie Zhang, Yankai Fu, Ning Chen, Cheng Chi, Sixiang Chen, Huaihai Lyu, Xiaoshuai Hao, Yequan Wang, Bo Lei, Dong Liu, Xi Yang, Yance Jiao, Tengfei Pan, Yunyan Zhang, Songjing Wang, Ziqian Zhang, Xu Liu, Ji Zhang, Caowei Meng, Zhizheng Zhang, Jiyang Gao, Song Wang, Xiaokun Leng, Zhiqiang Xie, Zhenzhen Zhou, Peng Huang, Wu Yang, Yandong Guo, Yichao Zhu, Suibing Zheng, Hao Cheng, Xinmin Ding, Yang Yue, Huanqian Wang, Chi Chen, Jingrui Pang, YuXi Qian, Haoran Geng, Lianli Gao, Haiyuan Li, Bin Fang, Gao Huang, Yaodong Yang, Hao Dong, He Wang, Hang Zhao, Yadong Mu, Di Hu, Hao Zhao, Tiejun Huang, Shanghang Zhang, Yonghua Lin, Zhongyuan Wang, and Guocai Yao. Robocoin: An open-sourced bimanual robotic data collection for integrated manipulation, 2026. URL [https://arxiv.org/abs/2511.17441.](https://arxiv.org/abs/2511.17441)

[81] Xiaofei Wu, Guozhen Zhang, Zhiyong Xu, Yuan Zhou, Qinglin Lu, and Xuming He. Pack and force your memory: Long-form and consistent video generation, 2025. URL [https://arxiv.org/abs/2510.01784.](https://arxiv.org/abs/2510.01784)

[82] Jiafeng Xu, Qi Li, Yan Shen, Yiyu Ren, Travis Davies, Shaowen He, Ze Wang, Yifan Yang, Ran Cheng, and Hao Dong. Scaling bimanual household manipulation from 1,500 hours of demonstrations to on-policy corrections, 2026. URL [https://arxiv.org/abs/2609.03591.](https://arxiv.org/abs/2609.03591)

[83] Ziming Xu, Shuang Liang, Ruobing Han, Ziqiao Xi, Mingxing Rao, Kun Zhou, Zijun Zhang, Yuchen Yan, Yufan Wei, Junbo Huang, Yifei Shao, Fang Nan, and Biwei Huang. Causalwm: Causal chain-of-thought reasoning for embodied world model, 2026. URL [https://arxiv.org/abs/2609.23184.](https://arxiv.org/abs/2609.23184)

[84] Jiazhi Yang, Kunyang Lin, Jinwei Li, Wencong Zhang, Tianwei Lin, Longyan Wu, Zhizhong Su, Hao Zhao, Ya-Qin Zhang, Li Chen, Ping Luo, Xiangyu Yue, and Hongyang Li. Rise: Self-improving robot policy with compositional world model, 2026. URL [https://arxiv.org/abs/2602.11075.](https://arxiv.org/abs/2602.11075)

[85] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, and Yukang Chen. Longlive: Real-time interactive long video generation, 2025. URL [https://arxiv.org/abs/2509.22622.](https://arxiv.org/abs/2509.22622)

[86] Sizhe Yang, Juncheng Mu, Tianming Wei, Chenhao Lu, Xiaofan Li, Linning Xu, Zhengrong Xue, Zhecheng Yuan, Dahua Lin, Jiangmiao Pang, and Huazhe Xu. Memorywam: Efficient world action modeling with persistent memory, 2026. URL [https://arxiv.org/abs/2606.20562.](https://arxiv.org/abs/2606.20562)

[87] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer, 2025. URL [https://arxiv.org/abs/2408.06072.](https://arxiv.org/abs/2408.06072)

[88] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi "Jim" Fan, and Joel Jang. World action models are zero-shot policies, 2026. URL [https://arxiv.org/abs/2602.15922.](https://arxiv.org/abs/2602.15922)

[89] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis, 2024. URL [https://arxiv.org/abs/2405.14867](https://arxiv.org/abs/2405.14867).

[90] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation, 2024. URL [https://arxiv.org/abs/2311.18828](https://arxiv.org/abs/2311.18828).

[91] Checheng Yu, Chonghao Sima, Gangcheng Jiang, Hai Zhang, Haoguang Mai, Hongyang Li, Huijie Wang, Jin Chen, Kaiyang Wu, Li Chen, Lirui Zhao, Modi Shi, Ping Luo, Qingwen Bu, Shijia Peng, Tianyu Li, and Yibo

35

<!-- page 36 of 42 -->

Yuan. χ<sub>0</sub>: Resource-aware robust manipulation via taming distributional inconsistencies, 2026. URL [https://arxiv.org/abs/2602.09021.](https://arxiv.org/abs/2602.09021)

[92] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval, 2025. URL [https://arxiv.org/abs/2506.03141.](https://arxiv.org/abs/2506.03141)

[93] Hu Yue, Siyuan Huang, Yue Liao, Shengcong Chen, Pengfei Zhou, Liliang Chen, Maoqing Yao, and Guanghui Ren. Ewmbench: Evaluating scene, motion, and semantic quality in embodied world models. arXiv preprint arXiv:2505.09694, 2025.

[94] He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, Yongming Rao, Dongsheng Zhang, Wanjia He, Ling Chen, Kai Huang, Jiahao Chen, Sichang Su, Xumin Yu, Ziyi Wang, Chengwei Zhu, Xiao Teng, Yuchun Guo, Yufeng Zhang, Yuandong Liu, Rui Wang, Zisheng Lu, Han Hu, and Zhengyou Zhang. Hy-embodied-0.5-vla: From vision-language-action models to a real-world robot learning stack, 2026. URL [https://arxiv.org/abs/2606.14409.](https://arxiv.org/abs/2606.14409)

[95] Jie Zhang, Xiaoyue Chen, Anzhe Chen, Dayiheng Liu, Deqing Li, Gengze Zhou, Hale Yin, Haoqi Yuan, Haoyang Li, Jiahao Li, Jiazhao Zhang, Jingren Zhou, Kaiyuan Gao, Kun Yan, Lihan Jiang, Ningyuan Tang, Pei Lin, Qihang Peng, Shengming Yin, Tianhe Wu, Tianyi Yan, Xiao Xu, Yan Shu, Yanran Zhang, Ye Wang, Yi Wang, Yilei Chen, Yixian Xu, Yiyang Huang, Yuxiang Chen, Zekai Zhang, Zhendong Wang, Zixing Lei, Zhixuan Liang, Zihao Liu, Zikai Zhou, Chenxu Lv, Xiong-Hui Chen, and Chenfei Wu. Qwen-robotworld technical report: Unifying embodied world modeling through language-conditioned video generation, 2026. URL [https://arxiv.org/abs/2606.17030.](https://arxiv.org/abs/2606.17030)

[96] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018. doi: 10.1109/CVPR.2018.00068.

[97] Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive diffusion distillation for real-time interactive video generation, 2026. URL [https://arxiv.org/abs/2605.15141.](https://arxiv.org/abs/2605.15141)

[98] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware, 2023. URL [https://arxiv.org/abs/2304.13705.](https://arxiv.org/abs/2304.13705)

[99] Yujie Zhao, Hongwei Fan, Di Chen, Shengcong Chen, Liliang Chen, Xiaoqi Li, Guanghui Ren, and Hao Dong. Real2edit2real: Generating robotic demonstrations via a 3d control interface, 2026. URL [https://arxiv.org/abs/2512.19402](https://arxiv.org/abs/2512.19402).

[100] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, and Xianyuan Zhan. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model, 2025. URL [https://arxiv.org/abs/2510.10274](https://arxiv.org/abs/2510.10274).

[101] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation, 2026. URL [https://arxiv.org/abs/2602.02214.](https://arxiv.org/abs/2602.02214)

## .1 Corpus Composition

Our training corpus pools a large collection of open-source and proprietary robot demonstration datasets, listed in Table 12. Our guiding principle was to include all available data of reasonable quality and build a pipeline that converts heterogeneous sources into a unified format. Below we describe the conversion steps, the filtering criteria, and the final representation choices.

Our in-house data consists of teleoperation demonstrations collected on the AgiBot G2, AgiBot G2X, and PrimeBOT H2 humanoid platforms for household tasks such as laundry and room tidying.

## A Training Details

36

<!-- page 37 of 42 -->

Table 12 Training data mixture, measured from dataset metadata rather than from upstream release figures. Hours are derived as frames/fps/3600 and so are consistent with the frame counts by construction. The mixture spans 20 dataset families, 1,773 LeRobot repositories, 1.52M episodes and 22 distinct robot platforms; it is 84% real by trajectory time. Real and Sim split each family by data source, and two families contribute both.

|  |  | Hours |  |  | Frames |  |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset family | Real | Sim | Total | Real | Sim | Total |
| In-House Data | 9,540 | - | 9,540 | 375M | - | 375M |
| InternData-A1 [71] | 4,000 | 170 | 4,170 | 410M | 1M | 411M |
| AgiBot World [2] | 2,300 | - | 2,300 | 243M | - | 243M |
| Hy-Embodied (UMI) [94] | 2,100 | - | 2,100 | 226M | - | 226M |
| BEHAVIOR-1K 2026 [44] | - | 2,000 | 2,000 | - | 211M | 211M |
| HiFi-UMI-130k [3] | 2,000 | - | 2,000 | 180M | - | 180M |
| InternData-M1 [12] | 160 | 1,700 | 1,860 | 18M | 181M | 199M |
| XR2 [82] | 1,500 | - | 1,500 | 162M | - | 162M |
| RoboMIND [78] | 850 | - | 850 | 92M | - | 92M |
| RoboCOIN [80] | 700 | - | 700 | 77M | - | 77M |
| AgiBot World 2026 [1] | 610 | - | 610 | 66M | - | 66M |
| RoboCasa [56] | - | 610 | 610 | - | 44M | 44M |
| DROID [40] | 510 | - | 510 | 28M | - | 28M |
| Galaxea OpenWorld [38] | 490 | - | 490 | 26M | - | 26M |
| MimicGen [54] | - | 210 | 210 | - | 15M | 15M |
| Kai0 [91] | 130 | - | 130 | 15M | - | 15M |
| RoboTwin 2.0 [11] | - | 60 | 60 | - | 6M | 6M |
| RT-1 Fractal [6] | 350 | - | 350 | 4M | - | 4M |
| Bridge V2 [74] | 110 | - | 110 | 2M | - | 2M |
| LIBERO [51] | - | 10 | 10 | - | 1M | 1M |
| Total | 25,350 | 4,760 | 30,110 | 1,924M | 459M | 2,383M |

## A.1 Architecture Hyperparameters

Table 13 Model architecture.

| Component | Value |
| --- | --- |
| Backbone init | Wan2.2-TI2V-5B [75] |
| Parameters (total) | 5.05B |
| Hidden dim | 3072 |
| Transformer blocks | 30 |
| Attention heads | 24 |
| Patch size (spatial) | 2×2 |
| VAE spatial compression | 16× |
| VAE temporal compression | 4× |
| Latent channels | 48 |
| Text encoder | umT5-XXL, frozen |
| Action tokenizer | 2-layer MLP |

## A.2 Training Hyperparameters

We report the architecture and the midtraining configuration for PrimeSim. All downstream stages reuse the same optimizer, precision, and backbone unless noted elsewhere.

The architecture is initialized from Wan2.2-TI2V-5B [75] and retains its backbone dimensions. We add only some lightweight action conditioning modules and three learnable view tokens on top of the frozen text

37

<!-- page 38 of 42 -->

encoder, so the trainable parameter count is close to the backbone’s.

Midtraining uses a cosine-decayed learning rate with a short warmup and a small minimum learning rate, which we found necessary to avoid late-stage loss spikes on long training runs. The action classifier-freeguidance dropout is set to 0.1 to enable guidance at inference without retraining.

Table 14 Midtraining hyperparameters. The architecture is initialized from the Wan2.2-TI2V-5B checkpoint, whose own training is not reported here.

| Hyperparameter | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | 1 × 10<sup>-4</sup> |
| LR schedule | cosine decay |
| Warmup steps | 10,000 |
| Min LR ratio | 0.05 (5 × 10<sup>-6</sup>) |
| β<sub>1</sub>, β<sub>2</sub> | 0.9, 0.95 |
| Weight decay | 1 × 10<sup>-2</sup> |
| Precision | bfloat16 |
| Action CFG dropout | 0.1 |
| Resolution (per view) | 192×256 |
| Frames per sample | 33 |
| Sampling FPS | 10 |
| Train/val split | 0.99 / 0.01 |

## A.3 Distillation Hyperparameters

Stage 3 refines the causal consistency student of Stage 2 with the asymmetric DMD objective of Sec. 3.3.3. Three models take part. The <u>teacher</u> is the frozen bidirectional midtrained model and supplies the real-data score. The <u>student</u> is the causal few-step generator being distilled. The <u>critic</u> is initialised from the teacher and then trained with standard flow matching on the student’s own samples, so it tracks the distribution the student currently produces rather than the one the teacher defines. Only LoRA adapters are trained on the student, and the adapters are exported directly rather than as an EMA of the full weights, which is why the adapter rank appears in the table.

The critic is updated five times per generator update. It has to stay ahead of a generator that is itself moving, and an under-trained critic supplies a gradient that points at stale targets. The real guidance scale of 3.0 is applied to the teacher’s score, which sharpens the supervision the student is matched against; without it the teacher averages over modes and the student inherits that averaging. Both networks use the same learning rate, and the sigma shift is the resolution-dependent noise-schedule shift of the flow matching formulation. Because the surrogate loss is built so that its gradient is exactly Eq. 11, the KL objective carries no separate coefficient, and no balancing weight between the two losses appears in the table. Table 17 lists the settings used for every Stage 3 result reported in this work.

38

<!-- page 39 of 42 -->

Table 15 Stage 1 autoregressive modelling hyperparameters. The model is initialized from the midtrained PrimeSim checkpoint and adapted to chunk-causal generation using teacher forcing.

| Hyperparameter | Value |
| --- | --- |
| Initialization | Midtrained PrimeSim |
| Training objective | Flow matching with teacher forcing |
| Trainable parameters | LoRA adapters |
| Optimizer | AdamW |
| Learning rate | 1 × 10<sup>-4</sup> |
| LR schedule | Cosine decay |
| Warmup steps | 10,000 |
| Minimum LR ratio | 0.05 (5 × 10<sup>-6</sup>) |
| β<sub>1</sub>, β<sub>2</sub> | 0.9, 0.95 |
| Weight decay | 1 × 10<sup>-2</sup> |
| Precision | bfloat16 |
| Action CFG dropout | 0.1 |
| Resolution per view | 192 × 256 |
| RGB frames per sample | 33 |
| Latent frames per window, T | 9 |
| Sampling FPS | 10 |
| Chunk size, k | 3 latent timesteps |
| Number of chunks, T/k | 3 |
| Within-chunk attention | Bidirectional |
| Across-chunk attention | Causal |
| Teacher sampling steps | 50 |
| Train/validation split | 0.99 / 0.01 |

Table 16 Stage 2 causal consistency distillation hyperparameters. Stage 2 initializes both teacher and student from the Stage 1 causal model and distills the 50-step teacher into a four-step causal student using one online teacher ODE step per iteration.

| Hyperparameter | Value |
| --- | --- |
| Initialization | Stage 1 causal model |
| Training objective | Causal consistency distillation |
| Trainable parameters | Student LoRA adapters |
| Optimizer | AdamW |
| Learning rate | 1 × 10<sup>-4</sup> |
| Precision | bfloat16 |
| Resolution per view | 192 × 256 |
| RGB frames per sample | 33 |
| Latent frames per window, T | 9 |
| Sampling FPS | 10 |
| Chunk size, k | 3 latent timesteps |
| Number of chunks, T/k | 3 |
| Sampling steps | 4 |

39

<!-- page 40 of 42 -->

Table 17 Stage 3 asymmetric distribution matching distillation hyperparameters. Stage 3 initializes the generator from the four-step causal student produced by Stage 2 and refines it using a frozen bidirectional midtrained model as the real-score teacher and a trainable fake-score network that tracks the student’s generated distribution.

| Hyperparameter | Value |
| --- | --- |
| Initialization | Stage 2 causal student |
| Training objective | Asymmetric DMD |
| Trainable parameters | Student LoRA adapters and fake-score network |
| Optimizer | AdamW |
| Learning rate | 1 × 10<sup>-4</sup> |
| Precision | bfloat16 |
| Resolution per view | 192 × 256 |
| RGB frames per sample | 33 |
| Latent frames per window, T | 9 |
| Sampling FPS | 10 |
| Chunk size, k | 3 latent timesteps |
| Number of chunks, T/k | 3 |
| Sampling steps | 4 |
| Critic:generator update ratio | 5 : 1 |
| Generator learning rate | 1 × 10<sup>-4</sup> |
| Critic learning rate | 1 × 10<sup>-4</sup> |
| Student LoRA rank | 32 |

Table 18 Stage 4 episode-level rollout and hierarchical temporal memory hyperparameters. Cache capacities are measured in temporal latent slots.

| Hyperparameter | Value |
| --- | --- |
| RGB frames per sample | 33 |
| Latent frames per window, T | 9 |
| Sampling steps | 4 |
| Global sink capacity | 3 |
| Slow-memory capacity | 9 |
| Local-memory capacity | 9 |
| Total KV-cache capacity | 21 |
| Generator updates per episode | 1 |

## B Latency Benchmark Details

This appendix collects the protocol details behind Table 3: the timing environment, per-model warm-up and chunk semantics, the configuration of every measured system, and the latent-token proxy used for fairness discussion.

## B.1 Environment and Timing Scope

Table 19 Timing environment. All reported numbers use a single GPU at batch size one.

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB |
| Precision | bfloat16 |
| Batch size | 1 (single episode / prompt stream) |
| Timing API | time.perf_counter() with torch.cuda.synchronize() |
| Reported aggregate | steady-state mean, warm-up discarded |

40

<!-- page 41 of 42 -->

The unit of work is one autoregressive chunk of future frames at each model’s native chunk size. The timed region covers the chunk generation call, including GPU-to-CPU materialization of outputs when it is part of the public API: model.step() including numpy conversion for GE-Sim 2.0, pipe() plus .cpu() on future frames for BWM, text/observation encoding, chain-of-thought sampling and VAE decoding with preloaded weights for CausalWM, and run\_episode with checkpoint loading, text encoding and forward kinematics excluded for PrimeSim. Checkpoint loading, tokenizer/model construction and first-time CUDA kernel compilation are never timed; warm-up chunks absorb compilation and cache effects.

## B.2 Warm-up, Repeats and Chunk Semantics

**Table 20** Warm-up schedule, timed repetitions and aggregation per model.

| Model | Warm-up | Timed | Aggregation |
| --- | --- | --- | --- |
| PrimeSim Stage-3 | 1 window | 3 repeats | mean |
| GE-Sim 2.0 | 3 chunks | 12 chunks | mean of last 5 (steady state) |
| BWM (fair-aligned) | 1 chunk | 3 chunks | mean |
| BWM (official) | 1 chunk | 2 chunks | mean |
| Ctrl-World | bench default | 5 chunks | mean |
| CausalWM | weights preloaded | 2 chunks | mean |

Table 21 Chunk semantics per model: conditioning history, future frames per chunk T, number of camera views V and denoising steps.

| Model | History | Future T | Views V | Denoise steps |
| --- | --- | --- | --- | --- |
| PrimeSim Stage-3 | continuous KV | 32 (33-frame window) | 3 | 4 (1000, 750, 500, 250) |
| GE-Sim 2.0 | 4 (memory band) | 25 | 3 | 4 (distilled) |
| BWM (fair-aligned) | 4 | 25 | 1 | 50 |
| BWM (official) | 9 | 48 | 1 | 50 |
| Ctrl-World | 6 | 5 | 3 (stacked) | 50 |
| CausalWM | first frame + text | 25 | 1 | 4+4+4 per stage |

## B.3 Per-model Configurations

**PrimeSim Stage-3 (ours).** Wan2.2-TI2V-5B with Stage-3 asymmetric DMD, action-conditioned, continuous KV cache with local attention span 9. Three views at 288 × 384, 33-frame windows (32 usable frames) under the RoboTwin 2.0 Stage-3 evaluation contract, four-step consistency sampler at sigmas (1000, 750, 500, 250). Steady-state repeats measure 10.006, 10.011 and 10.329 TempFPS (mean 10.115, 3.16 s per window); a longer continuous-KV episode of about 72 model frames measures 7.63 TempFPS, and a cold-start run without warm-up measures 6.41 TempFPS — neither is used as the headline. Timed scope is run\_episode with loading, text encoding and forward kinematics excluded.

**GE-Sim 2.0.** Cosmos 2B with Pose Expert, released distilled checkpoint, action-conditioned with an episode band of intrinsics, extrinsics and trajectory. Three views at 384 × 512, history 4, future 25, four-step distilled sampling with sigma schedule (1.0, 0.9375, 0.8333, 0.625), triton\_rope enabled and Liger/SpargeAttention off. Twelve timed chunks follow three warm-up chunks; early chunks are noisy (9.78 s to 44.84 s), so the steady state is the mean of the last five, 11.15 s per chunk.

**BWM, fair-aligned.** Wan2.2-TI2V-5B with action AdaLN (BLM checkpoint, step-12000), conditioned on absolute end-effector actions. Single view at 384 × 512, history 4, future 25 (chunk of 29 frames total), 50-step sampling at CFG 1.0. Three timed chunks measure 6.23, 6.15 and 6.19 s (mean 6.19 s). The distilled BWM-fast weights were not publicly available at measurement time.

41

<!-- page 42 of 42 -->

**BWM, official configuration.** Vendor-default inference config: single view at 672 × 896, history 9, future 48, 50-step sampling. Mean chunk time 61.71 s. Reported as the heavy vendor-default point rather than as a fair peer.

**Ctrl-World.** SVD image-to-video UNet with CLIP text, conditioned on 7D actions. Three views stacked on the height axis into a 576 × 320 canvas (192 × 320 per view), history 6, future 5, 50-step sampling. Five timed chunks measure 5.76 to 5.80 s (mean 5.78 s).

**CausalWM.** LTX-2.3 22B with Gemma-3-12B, generating a flow → pointmap → RGB chain-of-thought from text and a first frame — not action-conditioned. Single view at 480 × 640, 25 future frames, four steps per chain-of-thought stage. Two timed chunks measure 39.23 and 36.72 s (mean 37.97 s), with weights preloaded and load time excluded.

## B.4 Latent-token Proxy

For fairness discussion we estimate the number of latent tokens per forward as

$$
\begin{array} { r } { \widehat { \# \mathrm { t o k e n s } } \; \approx \; V \times \Big ( \lfloor \frac { T _ { \mathrm { l a t } } - 1 } { 4 } \rfloor + 1 \Big ) \times \lfloor \frac { H } { 8 } \rfloor \times \lfloor \frac { W } { 8 } \rfloor , } \end{array}\tag{14}
$$

i.e. a spatial×temporal compression grid in the style of Cosmos/Wan tokenizers (spatial stride 8, temporal compression ≈ 4). This is a coarse grid proxy, not FLOPs.

**Table 22** Approximate latent tokens per forward (Eq. 14). GE-Sim 2.0 counts its effective temporal extent including memory; Ctrl-World uses its stacked 576 × 320 canvas; CausalWM’s temporal stride may differ, so its value is an estimate.

| Model | #\\tokens |
| --- | --- |
| PrimeSim Stage-3 | 46,656 |
| GE-Sim 2.0 | 73,728 |
| BWM (fair-aligned) | 24,576 |
| BWM (official) | 141,120 |
| Ctrl-World | 14,400 |
| CausalWM | ∼19,200 |

PrimeSim’s proxy uses the 33-frame window $( 3 \times 9 \times 3 6 \times 4 8 = 4 6 { , } 6 5 6 ) ,$ ; the continuous KV cache bounds the attention span but does not change this per-window grid count, so it is not inflated to a memory-plus-future figure.

42