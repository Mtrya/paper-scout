<!-- page 1 of 26 -->

arXiv:2610.40748v1 [cs.RO] 29 Oct 2026

JINGDONG

# WorldBridge: Bridging Video and Action for Task Generalization in World Action Models

**WorldBridge Team, Joy Future Academy**

## Abstract

Achieving the broad task generalization of embodied models in physical environments remains a central challenge in robot learning. We study this challenge in World Action Models (WAMs) through two complementary capabilities: video generalization, which predicts the visual evolution of new tasks, and action generalization, which translates these predictions into executable behavior through a video-to-action bridge. By varying video and action supervision for tasks outside an initial training set, we examine their respective contributions to task transfer. On RoboTwin 2.0, using only in-distribution video-action supervision yields a 25.07% success rate on out-of-distribution tasks. Introducing videos of these tasks without their action labels raises success to 55.90%, while adding just one action-labeled demonstration per task further improves it to 70.70%. These findings suggest that broader video coverage enables task transfer by leveraging existing inverse-dynamics capabilities, while even sparse action supervision further strengthens those capabilities. Guided by this perspective, we introduce **WorldBridge**, a WAM with separate video and action experts connected through structured attention and trained with joint flow matching. WorldBridge incorporates visual memory at two temporal scales and action forcing, alongside a unified data processing pipeline that standardizes and filters heterogeneous robot trajectories and egocentric hand-derived supervision. Across multiple manipulation benchmarks, WorldBridge achieves state-of-the-art performance among the compared methods, including 96.23% on RoboTwin 2.0 full data setting and 80.9% on clean to random setting, 82.8% on RoboCasa GR1 TableTop, and 57.6% on DOMINO Level-1. Our study highlights the complementary roles of video prediction and action grounding in developing robots that generalize across tasks.

**Project page:** [https://worldbridge-jd.github.io/](https://worldbridge-jd.github.io/)

## 1 Introduction

Task generalization marks a fundamental shift from earlier pretrained NLP models, such as BERT [16], to modern large language models (LLMs) [6, 15, 63]. Whereas earlier models typically require task-specific fine-tuning, a single LLM can perform diverse tasks specified through natural-language instructions or a few in-context examples, without updating its parameters for each new task. This capability has motivated the pursuit of generalist embodied agents that can translate human instructions into a broad repertoire of physical behaviors. Vision-language-action (VLA) models represent an important step toward this goal, transferring knowledge from large-scale vision-language pretraining into robot control [5, 47]. However, understanding an instruction does not automatically provide the physical competence needed to execute it. Bringing task generalization into robotics requires connecting semantic knowledge to the visual dynamics and motor behaviors of physical interaction.

1

<!-- page 2 of 26 -->

![Image block](doc:adc627e/tier:basic/page:2/block:1)

Figure 1 Overview of our WorldBridge, a foundational WAM for task generalization. WorldBridge utilizes heterogeneous data sources and is trained with a simple but effective model architecture in two stages. The bottom panels report task-generalization results under different video and action supervision settings, benchmark performance, and real-world deployment results.

Despite substantial progress, a prominent focus of embodied generalization research remains robustness to changes in the conditions under which a task is performed, including object arrangements, camera viewpoints, lighting, and visual appearance. Benchmarks such as LIBERO-Plus systematically evaluate these variations, exposing the brittleness of otherwise successful policies [19]. Task generalization introduces an additional challenge: a robot must execute behaviors beyond those covered by its original training tasks. Achieving this capability requires understanding how a newly requested task should unfold and producing the actions that realize it, even when corresponding robot demonstrations are scarce or unavailable.

World Action Models (WAMs) provide a promising foundation for addressing this challenge by coupling visual prediction with action generation. Video offers dense supervision of physical interactions, allowing models to learn how scenes evolve beyond the coverage of action-labeled robot data. DreamZero demonstrates that video priors can support unseen-task execution and transfer from video-only demonstrations [68], while Zero-WAM explores task generalization through in-context human video guidance [75]. Recent advances in native video-action pretraining [72], systematic WAM pretraining [58], and temporal context [24] further strengthen this paradigm. These developments motivate a more specific question: how do broader video coverage and additional action supervision contribute to task generalization, and when does the connection between them become the limiting factor?

In this work, we investigate task generalization in WAMs through two complementary capabilities: video generalization and action generalization. Video generalization enables a model to predict the visual evolution required by tasks beyond its initial training distribution. Action generalization refers to the ability to translate those visual predictions into executable actions beyond the tasks that supplied action supervision. This translation forms a video-to-action bridge, serving the role of an inverse dynamics model (IDM). A model may predict an appropriate visual future yet lack the action grounding needed to realize it. Conversely, a transferable bridge can allow knowledge acquired from additional videos to improve task execution without corresponding action labels, which motivates studying the availability of video and action supervision separately.

2

<!-- page 3 of 26 -->

![Chart block](doc:adc627e/tier:basic/page:3/block:1)

Figure 2 Task generalization under progressively expanded supervision. Compared with in-domain (ID) video-action training alone (25.07% OOD success rate), adding out-of-distribution (OOD) videos increases success to 55.90%, while adding a few paired OOD video-action examples further improves it to 70.70%.

Figure 2 illustrates this progression, which we examine by varying supervision for tasks outside an initial in-distribution (ID) training set. With only ID video-action pairs, the model achieves nontrivial but limited generalization, obtaining a 25.07% success rate on out-of-distribution (OOD) tasks. Adding videos of those OOD tasks while keeping action supervision restricted to the ID set increases OOD success to 55.90%. This improvement suggests that expanded video coverage can support new task execution through a bridge learned from ID action supervision. Keeping the video set fixed, adding only one action-labeled demonstration per OOD task further raises success to 70.70%. These rapid gains are consistent with sparse paired examples strengthening the bridge for visual transitions that are already covered by the video data. Together, these observations support a complementary account of task generalization: video supervision expands the range of task evolutions a WAM can model, while action supervision improves its ability to turn that knowledge into reliable control.

Guided by this perspective, we introduce **WorldBridge**, a WAM that couples video prediction and action generation through a Mixture-of-Transformers (MoT) architecture [35]. WorldBridge maintains separate video and action experts trained with a joint flow-matching objective. A structured attention mask establishes a directional information path from observed context to future video representations and then to action predictions, making predicted visual dynamics an explicit source of conditioning for control. Short-term and long-term visual memories provide information about local motion and task progression. We additionally employ action forcing, assigning independent noise levels to actions within a chunk during training so that the action expert learns to predict actions using temporal context at varying levels of corruption.

To support learning across heterogeneous sources, we develop a unified pretraining data pipeline for robot trajectories and egocentric human demonstrations. The pipeline standardizes control semantics and maps available robot controls and human hand-derived targets into a shared 46-dimensional action representation. Dimension-wise validity masks preserve partial supervision across embodiments with different control interfaces. Robot trajectories undergo kinematic and temporal consistency checks, while egocentric hand trajectories are standardized in a common reference frame and filtered using geometric and motion constraints. This processing supports broader visual coverage together with usable action supervision, providing a common interface for learning the video-to-action bridge across diverse data sources.

Our evaluations demonstrate strong performance across a range of manipulation tasks. WorldBridge achieves state-of-the-art results among the compared methods on multiple benchmarks, with average success rates of 96.23% on RoboTwin 2.0, 80.9% in the RoboTwin 2.0 C2R setting, 82.8% on RoboCasa GR1 TableTop, and 57.6% on DOMINO Level-1. On RMBench, it matches the best result in our comparison, with an 83.0% success rate. We further conduct real-world evaluations on the Agibot-G2 and AC-1 platforms, covering long-horizon execution, memory-dependent tasks, spatial localization, and bimanual coordination. Together

3

<!-- page 4 of 26 -->

![Image block](doc:adc627e/tier:basic/page:4/block:1)

Figure 3 Overview of the WorldBridge model architecture.

with our task-generalization study, these results support the value of jointly developing video prediction and action grounding.

Our contributions are threefold:

• We study task generalization in WAMs through the complementary roles of video generalization and action generalization, quantifying the gains from additional task videos and sparse action supervision.

• We introduce WorldBridge and a unified data processing pipeline that connect visual dynamics to executable control across heterogeneous robot and egocentric demonstrations.

• We evaluate WorldBridge across RoboTwin 2.0, RoboCasa GR1 TableTop, RMBench, DOMINO Level-1, SimplerEnv-WidowX, and RoboDojo, covering diverse manipulation demands and distribution shifts. We further deploy it on AgiBot G2 and ARX AC-1, fine-tuning the same jointly pretrained checkpoint for each platform and evaluating long-horizon execution, memory-dependent interaction, spatial alignment, and bimanual coordination.

## 2 Method

## 2.1 Model

## 2.1.1 Architecture

WorldBridge adopts a Mixture of Transformers (MoT) architecture comprising a video expert and an action expert. The two experts maintain modality-specific parameters and exchange information through joint self-attention, enabling coupled modeling of future visual observations and robot actions. Each expert contains its own feed-forward networks and output decoder, while the video expert additionally incorporates language features through cross-attention [56]. This architecture preserves modality-specific processing while allowing visual dynamics and action generation to inform one another. An overview is shown in Fig. 3.

Given visual observations O and a language instruction $\ell ,$ we construct the conditioning context c using a video VAE and a T5 text encoder, respectively.

$$
c = \left( \mathcal { E } _ { \mathrm { V A E } } ( \mathcal { O } ) , \mathcal { E } _ { \mathrm { T 5 } } ( \ell ) \right) .\tag{1}
$$

The clean prediction targets consist of future video latents $\mathbf { z } \; = \; \mathcal { E } _ { \mathrm { V A E } } ( \mathcal { V } )$ and an action chunk $\mathbf { A } _ { t } \mathbf { \nabla } = \mathbf { \nabla }$

4

<!-- page 5 of 26 -->

$[ \mathbf { a } _ { t } , \ldots , \mathbf { a } _ { t + H - 1 } ]$ , where t denotes the current environment timestep and H is the action prediction horizon.

We independently sample a video flow timestep $\tau _ { v }$ and an action flow timestep $\tau _ { a }$ in [0, 1]. Using a linear flow-matching path, we construct the noisy inputs as

$$
\begin{aligned} { \mathbf { z } ( \tau _ { v } ) } & { { } = ( 1 - \tau _ { v } ) \mathbf { z } + \tau _ { v } \mathbf { \epsilon } _ { v } , } \\ { \mathbf { A } _ { t } ( \tau _ { a } ) } & { { } = ( 1 - \tau _ { a } ) \mathbf { A } _ { t } + \tau _ { a } \mathbf { \epsilon } _ { a } , } \\ \end{aligned}\tag{2}
$$

where $\epsilon _ { v }$ and $\epsilon _ { a }$ are independent standard Gaussian noise tensors with shapes matching their respective targets. Under this convention, $\tau = 0$ corresponds to clean data and $\tau = 1$ to pure noise. Independent sampling of $\tau _ { v }$ and $\tau _ { a }$ allows the two experts to process inputs at different noise levels during training.

Conditioned on their respective flow timesteps, the two experts process the noisy inputs through L MoT blocks, exchanging information through joint self-attention. The video and action decoders produce the corresponding velocity predictions:

$$
\begin{array} { r } { ( \hat { \mathbf { u } } _ { v } , \hat { \mathbf { u } } _ { a } ) = f _ { \theta } \left( \mathbf { z } ( \tau _ { v } ) , \mathbf { A } _ { t } ( \tau _ { a } ) , \tau _ { v } , \tau _ { a } ; c \right) . } \end{array}\tag{3}
$$

The model is optimized through joint flow matching, with the predicted velocities regressed toward

$$
\begin{aligned}\mathbf{u}_{v}^{\star} &= \boldsymbol{\epsilon}_{v} - \mathbf{z}, \\\mathbf{u}_{a}^{\star} &= \boldsymbol{\epsilon}_{a} - \mathbf{A}_{t}.\end{aligned}\tag{4}
$$

At inference, future video latents and the action chunk are independently initialized from standard Gaussian noise:

$$
\mathbf { z } ( 1 ) \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) , \qquad \mathbf { A } _ { t } ( 1 ) \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) .\tag{5}
$$

Both are progressively denoised by integrating the learned velocity fields from $\tau   =   1$ toward $\tau   =   0$ . A first-order update takes the form

$$
\begin{aligned} { \mathbf { z } ( \tau _ { v } - \Delta \tau _ { v } ) } & { { } = \mathbf { z } ( \tau _ { v } ) - \Delta \tau _ { v } \: \hat { \mathbf { u } } _ { v } , } \\ { \mathbf { A } _ { t } ( \tau _ { a } - \Delta \tau _ { a } ) } & { { } = \mathbf { A } _ { t } ( \tau _ { a } ) - \Delta \tau _ { a } \: \hat { \mathbf { u } } _ { a } , } \\ \end{aligned}\tag{6}
$$

where the velocity predictions are evaluated at the current states and flow timesteps. Throughout denoising, the conditioning context c remains fixed. The final video latents $\mathbf { z } ( 0 )$ can be reconstructed through the VAE decoder, while $\mathbf { A } _ { t } ( 0 )$ constitutes the predicted action chunk.

## 2.1.2 Memory

To provide temporal context beyond the current observation, WorldBridge maintains visual memory at two temporal scales. Short-term memory provides recent observations for capturing local motion, while long-term memory provides a broader view of task progression. Let $\mathbf { o } _ { t }$ denote the visual observation at the current environment timestep. We construct two chronologically ordered observation sequences:

$$
\begin{aligned} { \mathcal { O } _ { t } ^ { \operatorname { s h o r t } } } & { { } = [ \mathbf { o } _ { t - 4 \delta _ { s } } , \mathbf { o } _ { t - 3 \delta _ { s } } , \mathbf { o } _ { t - 2 \delta _ { s } } , \mathbf { o } _ { t - \delta _ { s } } ] , } \\ { \mathcal { O } _ { t } ^ { \operatorname { l o n g } } } & { { } = [ \mathbf { o } _ { t - 4 \delta _ { l } } , \mathbf { o } _ { t - 3 \delta _ { l } } , \mathbf { o } _ { t - 2 \delta _ { l } } , \mathbf { o } _ { t - \delta _ { l } } ] , } \\ \end{aligned}\tag{7}
$$

where $\delta _ { s }$ and $\delta _ { l }$ are the short-term and long-term sampling strides. The long-term stride corresponds to the temporal span of one predicted action chunk. Observations preceding the episode boundary are replaced with the first observation of that episode.

We encode both memory sequences together with the current observation using the causal video VAE. To align with its temporal compression factor of four, we repeat the current observation four times and construct

$$
\mathcal { P } _ { t } = [ \mathbf { p } , \mathcal { O } _ { t } ^ { \operatorname { l o n g } } , \mathcal { O } _ { t } ^ { \operatorname { s h o r t } } , \operatorname { R e p e a t } _ { 4 } ( \mathbf { o } _ { t } ) ] ,\tag{8}
$$

5

<!-- page 6 of 26 -->

where p is a zero-valued temporal padding frame. After VAE encoding, we discard the leading latent associated with this padding frame and retain three latent frames:

$$
[ \mathbf { m } _ { t } ^ { \mathrm { l o n g } } , \mathbf { m } _ { t } ^ { \mathrm { s h o r t } } , \mathbf { z } _ { t } ^ { \mathrm { c u r } } ] = \mathrm { D r o p F i r s t } \left( \mathcal { E } _ { \mathrm { V A E } } ( \mathcal { P } _ { t } ) \right) .\tag{9}
$$

These latent frames provide a compact representation of the long-term history, short-term history, and current observation. They are spatially tokenized and prepended to the noisy future video tokens.

## 2.1.3 Action Forcing

When a single flow timestep is shared across an entire action chunk, all action tokens are corrupted to the same noise level. Action forcing instead exposes the action expert to chunks containing actions at different noise levels. Since action-token inputs can vary substantially in their noise levels and are difficult to model in isolation, this design encourages each token to rely more on contextual visual information.

Specifically, we extend the scalar action flow timestep $\tau _ { a }$ to a vector $\boldsymbol { \tau } _ { a } = [ \tau _ { a , 0 } , \ldots , \tau _ { a , H - 1 } ]$ . For a group size g, we independently sample a flow timestep $s _ { a , k }$ for each group $k = 0 , \ldots , \lceil H / g \rceil - 1$ from the training flow-matching schedule, independently of the video timestep $\tau _ { v }$ . The timestep assigned to action $j$ is

$$
\tau _ { a , j } = s _ { a , \lfloor j / g \rfloor } , \qquad j = 0 , \ldots , H - 1 .\tag{10}
$$

For the j-th action in the chunk, the noisy input is

$$
\widetilde { \mathbf { a } } _ { t + j } = ( 1 - \tau _ { a , j } ) \mathbf { a } _ { t + j } + \tau _ { a , j } \mathbf { \epsilon } _ { a , j } , \qquad j = 0 , \ldots , H - 1 ,\tag{11}
$$

where the noise vectors $\mathbf { \epsilon } _ { a , j } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ are sampled independently, even when actions share a flow timestep. Here, actions are represented in the normalized unified action space.

Each action is encoded as one token and receives its corresponding flow-time embedding. The action expert therefore processes a mixture of relatively clean and highly corrupted actions within the same chunk. Through attention within the action branch, each prediction can use the other actions at their respective noise levels, while also conditioning on future video features. The prediction target remains the flow-matching velocity:

$$
\mathbf { u } _ { a , j } ^ { \star } = \mathbf { \epsilon } _ { a , j } - \mathbf { a } _ { t + j } .\tag{12}
$$

At inference, all action tokens are initialized with independent standard Gaussian noise and denoised jointly using a shared action flow timestep. Thus, varying noise levels across actions or groups are only used during training, while the complete action chunk is generated through joint denoising at inference.

## 2.1.4 Attention Mask

We adopt stage-specific attention masks to control information exchange between video and action tokens within each MoT block. As illustrated in Figure 4, the joint sequence comprises four groups: memory tokens, current frame tokens, future frame tokens, and action tokens. We use a densely connected video mask during pretraining and introduce more structured dependencies during post-training.

We use stage-specific attention masks to balance flexible multimodal learning during pretraining with an explicit video-to-action pathway during post-training. During pretraining, video tokens attend to all video tokens but not to action tokens, preserving actionindependent video generation. Action tokens attend to both video and action tokens, allowing the action

![Chart block](doc:adc627e/tier:basic/page:6/block:17)

![Chart block](doc:adc627e/tier:basic/page:6/block:18)

Figure 4 Attention mask design. (a) Attention mask for pretraining. (b) Attention mask for post-training.

6

<!-- page 7 of 26 -->

expert to learn from the full visual and action context. During post-training, memory and current-frame tokens attend to one another; future-video tokens attend to the current-frame and future-video tokens, and action tokens attend to future-video and action tokens. This structure routes information from memory and current observations through predicted future features to action predictions, encouraging actions to be grounded in the modeled visual evolution. For tasks with strong memory demands, we use the pretraining mask instead, allowing action tokens to access memory tokens directly.

## 2.2 Data Processing

## 2.2.1 Robot Data Processing

We curate heterogeneous robot demonstrations through a unified pipeline comprising semantic auditing, representation standardization, and trajectory filtering. We first map source-specific state and action fields into a shared representation with explicit validity masks. We then trim stationary episode boundaries, remove short trajectories, and apply kinematic and temporal consistency checks. Processing rules are configured for each source and embodiment to accommodate differences in annotation availability, control semantics, and signal characteristics.

Semantic Auditing and Representation Standardization. We first check what each source field represents, including joint order, EEF coordinates and rotations, gripper opening direction, and the meaning of recorded actions. State and action are mapped separately because a measured state and a control target are not interchangeable. Available values are placed in the shared 46-dimensional layout described in Section 2.3.2. Separate state and action masks identify which entries are available; missing or non-finite values are set to zero and masked out.

EEF rotations are converted to unit quaternions in $( x , y , z , w )$ order. Gripper values are calibrated for each source so that 0 means fully closed and 1 means fully open. Each dexterous hand has six slots: bending of the little, ring, middle, index, and thumb fingers, followed by thumb rotation. Depending on the robot, we reorder the original channels, combine coupled finger joints, or map a shared hand-closure signal to the corresponding finger-bending slots. These slots therefore do not always represent independently measured finger joints. Unsupported hand motions remain masked. Four additional slots store supported waist and torso signals, such as waist pitch and torso lift, with a mapping specified for each robot; unused slots remain masked. EEF positions retain their original coordinate frames and units. Relative action targets and training normalization are defined separately in Section 2.3.2.

Static Boundary Trimming and Length Filtering. We detect motion from adjacent state differences over dimensions that are valid and finite at both timesteps. A transition is considered active if any comparable dimension changes by more than $1 0 ^ { - 4 }$ in the stored feature space. We retain the interval spanning the first and last active transitions, including their endpoint frames. This removes stationary prefixes and suffixes while preserving pauses within task execution. Fully stationary episodes with observable states are discarded, whereas episodes without comparable valid state dimensions are left unchanged by this check. We subsequently remove trajectories shorter than 48 frames and update episode and video-window metadata to reflect the retained intervals.

Kinematic Consistency Filtering. For sources with comparable joint and EEF annotations, we compute the EEF pose from recorded joint positions using forward kinematics (FK). We account for source-specific base frames and tool definitions before comparing the FK pose with the recorded pose. Let $( \mathbf { p } _ { \mathrm { F K } } , \mathbf { R } _ { \mathrm { F K } } )$ and $\left( \mathbf { p } _ { \mathrm { d a t a } } , \mathbf { R } _ { \mathrm { d a t a } } \right)$ denote these two poses at the same timestep. We measure their position and orientation errors as

$$
e _ { \operatorname { p o s } } = \| \mathbf { p } _ { \operatorname { F K } } - \mathbf { p } _ { \operatorname { d a t a } } \| _ { 2 } , \qquad e _ { \operatorname { r o t } } = \operatorname { a r c c o s } \biggl ( \frac { \operatorname { t r } ( \mathbf { R } _ { \operatorname { F K } } ^ { \top } \mathbf { R } _ { \operatorname { d a t a } } ) - 1 } { 2 } \biggr ) \: .\tag{13}
$$

The argument of arccos is clipped to [−1, 1] for numerical stability. A frame passes when the applicable errors are below embodiment-specific thresholds, and an episode is retained when a sufficient fraction of evaluated frames passes. Invalid required poses count as failures, while position-only configurations omit the orientation

7

<!-- page 8 of 26 -->

![Image block](doc:adc627e/tier:basic/page:8/block:1)

Figure 5 Overview of the WorldBridge data processing pipeline.

check. For selected sources with an unknown fixed frame relationship, we estimate a single rigid transformation per episode before computing the errors. This alignment is used only for consistency assessment.

Signal Anomaly and State–Action Consistency Filtering. We apply three complementary signal checks and reject episodes that fail an applicable criterion.

Sudden-change detection. We estimate a smooth trajectory using median and Savitzky–Golay filtering, then evaluate the residual together with second- and third-order temporal differences. A sample is flagged when an excessive residual coincides with an excessive higher-order difference. Robust, channel-specific thresholds distinguish abrupt anomalies from ordinary motion. Quaternion signs are aligned temporarily during detection to avoid artificial discontinuities between equivalent rotations. The smoothing and sign alignment are used for detection and do not replace the stored signals.

State–action consistency. For comparable joint dimensions with sufficient motion, we compare the directions of changes in smoothed state and action signals. We search offsets within ±15 frames and retain the best directional agreement for each eligible dimension. Episodes are rejected when the minimum agreement across eligible dimensions falls below the source-specific threshold. This procedure assesses temporal consistency without shifting the saved action sequences.

Extreme-value detection. We estimate the 1st and 99th percentiles from valid samples within each repository. The admissible interval extends beyond these quantiles by the larger of five times their span and a channelspecific minimum margin. Episodes containing out-of-range or non-finite active values are rejected. Quaternion, gripper, and dexterous-hand channels are exempt from this range test. Source-specific configurations determine the applicable checks and exemptions.

## 2.2.2 Ego Data Processing

We process egocentric demonstrations into synchronized visual observations and hand-derived action targets. Each sequence contains egocentric RGB frames, camera-to-world poses, and 21 three-dimensional keypoints

8

<!-- page 9 of 26 -->

per hand in a consistent MANO ordering. Available task and subtask annotations provide the language context. Our pipeline standardizes the coordinate representation, maps hand geometry into the shared action layout, and filters unreliable supervision using geometric and temporal consistency checks.

Coordinate Standardization. To account for the moving egocentric camera, we express future hand geometry in the camera frame of the current observation. Let $\mathbf { T } _ { c _ { t } } ^ { w } \in S E ( 3 )$ denote the camera-to-world transformation at timestep t, and let $\bar { \mathbf { x } } _ { t + k , j } ^ { c _ { t + k } }$ be the homogeneous coordinate of hand keypoint j observed at a future timestep. Its coordinate in the current camera frame is

$$
\begin{array} { r } { \bar { \mathbf { x } } _ { t + k , j } ^ { c _ { t } } = \left( \mathbf { T } _ { c _ { t } } ^ { w } \right) ^ { - 1 } \mathbf { T } _ { c _ { t + k } } ^ { w } \bar { \mathbf { x } } _ { t + k , j } ^ { c _ { t + k } } . } \end{array}\tag{14}
$$

The reference frame remains fixed throughout the prediction chunk, placing future hand trajectories in a common coordinate system despite changes in camera pose. Video and geometry are sampled on a shared physical-time schedule to preserve their temporal correspondence.

Hand-to-Action Mapping. We derive a canonical wrist pose from each transformed hand. The wrist keypoint defines its position, while the wrist and metacarpophalangeal joints define its orientation. Specifically, the forward axis points from the wrist toward the centroid of the four finger bases, and a least-squares palm plane provides the palm normal. A handedness correction gives the left and right hands consistent axis semantics.

We additionally extract a continuous gripper-openness target from the distance between the thumb tip and a virtual opposing finger formed from the index and middle fingertips. Normalizing this distance by palm width reduces dependence on hand size. For dexterous-hand supervision, finger bending and thumb opposition are mapped to six bounded joint-position targets per hand.

Quality Filtering and Validity Masks. We assess hand annotations using biomechanical plausibility, coordinate bounds, bone lengths, and temporal motion statistics, including velocity, acceleration, and jerk. Motion derivatives are computed using actual timestamps and world-coordinate hand trajectories. Camera rotations are checked for orthogonality and determinant consistency. Dataset-specific thresholds accommodate differences in motion statistics across sources.

Quality annotations are consumed at the sampled action timesteps. Targets associated with rejected frames are masked, while windows with more than 20% rejected action timesteps are discarded. An invalid reference camera pose invalidates the corresponding action chunk. During action construction, non-finite coordinates, padded frames, and degenerate hand geometry are further excluded through validity masks. Missing observations of one hand do not automatically invalidate the other. This preserves usable visual context and partial hand supervision without treating unavailable action dimensions as valid zero-valued targets.

## 2.3 Pretraining Recipe

We pretrain WorldBridge jointly on robot demonstrations and egocentric human demonstrations processed by the pipeline in Section 2.2. Below, we describe the training corpus, the shared action representation, and the pretraining setup.

## 2.3.1 Robot and Ego Data

Robot Data. Our corpus combines real-world and simulated robot demonstrations. Real-world sources include ABC-130K [2], AgiBotWorld-Beta [7], AgiBotWorld2026 [1], DROID [27], Galaxea Open-World [25], RoboMIND [59], RoboCOIN [60], and RoboMIND 2.0 [23]. Simulation sources include GenieSim 3.0 [69], and InternData-A1 [54], RoboMIND 2.0 also contributes simulation subsets. Together, these datasets cover single-arm, bimanual, and humanoid platforms equipped with parallel-jaw grippers or dexterous hands. Table 1 summarizes the curated sources.

9

<!-- page 10 of 26 -->

Table 1 Robot and ego corpus statistics before and after curation, excluding RoboCasa and RoboTwin. For robot data, hour retention includes subset selection, boundary trimming, length filtering, and quality filtering. For ego data, after-curation episode counts reflect episode-level annotation filtering, and hours count frames passing all seven configured quality checks within retained episodes. Durations are computed using repository-specific frame rates.

|  | Before c | uration | After c | uration | Retention |
| --- | --- | --- | --- | --- | --- |
| Dataset | Episodes | Hours | Episodes | Hours | (% hours) |
| Robot datasets |  |  |  |  |  |
| ABC-130K [2] | 130,919 | 4,130.93 | 66,426 | 1,642.30 | 39.76 |
| AgiBotWorld-Beta [7] | 158,776 | 2,392.17 | 155,587 | 2,286.54 | 95.58 |
| AgiBotWorld2026 [1] | 22,436 | 456.85 | 18,233 | 390.66 | 85.51 |
| DROID [27] | 57,774 | 273.12 | 56,279 | 263.80 | 96.59 |
| Galaxea [25] | 18,983 | 454.66 | 17,883 | 423.37 | 93.12 |
| RoboCOIN [60] | 98,580 | 688.84 | 94,877 | 653.23 | 94.83 |
| RoboMIND [59] | 73,882 | 218.31 | 25,171 | 36.91 | 16.91 |
| RoboMIND 2.0 [23] | 187,590 | 1,218.70 | 116,901 | 262.37 | 21.53 |
| GenieSim 3.0 [69] | 139,591 | 252.34 | 133,801 | 211.38 | 83.77 |
| InternData-A1 [54] | 613,119 | 3,890.74 | 590,557 | 3,559.02 | 91.47 |
| Robot subtotal | 1,501,650 | 13,976.65 | 1,275,715 | 9,729.58 | 69.61 |
| Ego datasets |  |  |  |  |  |
| EgoStandard | 1,623,216 | 8,170.37 | 1,623,216 | 7,656.85 | 93.71 |
| Lightwheel | 31,576 | 1,049.92 | 31,576 | 987.62 | 94.07 |
| XDOF | 42,059 | 1,000.44 | 42,059 | 971.46 | 97.10 |
| CyberOrigin | 35,062 | 1,026.51 | 35,062 | 985.22 | 95.98 |
| Xperience | 312,743 | 1,156.00 | 123,138 | 477.29 | 41.29 |
| EgoDex | 314,839 | 724.84 | 222,982 | 531.15 | 73.28 |
| GenRobot | 27,813 | 1,000.14 | 27,813 | 982.05 | 98.19 |
| Ego subtotal | 2,387,308 | 14,128.22 | 2,105,846 | 12,591.63 | 89.12 |
| Total | 3,888,958 | 28,104.87 | 3,381,561 | 22,321.21 | 79.42 |

Ego Data. Our egocentric corpus comprises EgoStandard [36], Lightwheel [36], XDOF, CyberOrigin, Gen-Robot, Xperience [52], and EgoDex [22]. These sources provide first-person visual observations and handderived supervision, which we convert into the shared action representation through the ego processing pipeline. Validity masks retain available hand supervision while excluding unreliable or missing targets.

Dataset Statistics. Table 1 reports episode counts and durations before and after curation for the listed sources. Durations are computed using their specific frame rates, and retention is the ratio of output to input hours. The lower retention of ABC-130K and RoboMIND primarily reflects invalid or unavailable EEF supervision and embodiment-level subset selection.

After curation, the listed corpus contains 1,275,715 robot episodes (9,729.58 hours) and 2,105,846 ego episodes (12,591.63 hours), totaling 3,381,561 episodes and 22,321.21 hours. The contribution of each source during training additionally depends on the sampling mixture.

## 2.3.2 Unified Action Space

To accommodate heterogeneous robot embodiments within a shared action expert, we organize actions into a fixed 46-dimensional representation comprising arm joints (14D), end-effector poses (14D), grippers (2D), dexterous hands (12D), and waist controls (4D). Each embodiment populates its available fields, accompanied by a dimension-wise validity mask. This shared layout preserves embodiment-specific control variables while allowing a common prediction interface. Unavailable dimensions are zero-filled and excluded from action supervision.

10

<!-- page 11 of 26 -->

End-Effector-Relative Representation. While the standardized data stores end-effector poses as positions and quaternions, we express their prediction targets relative to the corresponding end-effector frame at the current timestep. For each end effector, let $( \mathbf { p } _ { t } , \mathbf { R } _ { t } )$ denote the reference pose in the current action record, and $( \mathbf { p } _ { t , k } , \mathbf { R } _ { t , k } )$ the pose of the k-th target in the prediction chunk. We compute

$$
\begin{aligned} { \Delta \mathbf { p } _ { t , k } } & { { } = \mathbf { R } _ { t } ^ { \top } ( \mathbf { p } _ { t , k } - \mathbf { p } _ { t } ) , } \\ { \Delta \phi _ { t , k } } & { { } = \operatorname { L o g } ( \mathbf { R } _ { t } ^ { \top } \mathbf { R } _ { t , k } ) ^ { \vee } , } \\ \end{aligned}\tag{15}
$$

where $\log ( \cdot ) ^ { \vee }$ maps a rotation matrix to its three-dimensional rotation vector in radians. The reference frame remains fixed throughout the chunk. Each original seven-dimensional pose slot therefore contains three translation components, three rotation components, and one masked padding dimension. The remaining action fields retain their standardized values. This representation expresses end-effector motion in a local frame while retaining the shared 46-dimensional layout.

Masked Normalization and Supervision. We normalize action targets using dimension-wise statistics computed separately for each dataset–robot-type group. For an action component $a _ { t , k , d }$ with validity $m _ { t , k , d } \in \{ 0 , 1 \}$ the normalized target is

$$
\widetilde { a } _ { t , k , d } = \left\{ \begin{aligned} { } & { { } \frac { a _ { t , k , d } - \mu _ { g , d } } { \operatorname* { m a x } ( \sigma _ { g , d } , \epsilon ) } , } & { m _ { t , k , d } = 1 , } \\ { } & { { } 0 , } & { m _ { t , k , d } = 0 , } \\ \end{aligned} \right.\tag{16}
$$

where g identifies the dataset–robot-type group. The statistics are computed in the same action representation used for training. Validity masks also account for the reference pose required by the relative transformation. During flow matching, the masks are applied to the clean actions, sampled noise, and velocity targets. The action loss is first averaged over valid dimensions within each action token, and then over tokens containing valid supervision.

State Conditioning. We additionally encode the current frame’s absolute 46-dimensional action record as a state token for the action expert. This state token retains the absolute end-effector poses and is normalized using separate current-frame statistics, with unavailable dimensions set to zero. It provides the reference configuration for predicting the relative action chunk and is excluded from the action prediction loss.

## 2.3.3 Pretraining Setup

Optimization. We jointly optimize the video and action experts with equally weighted video and action flow-matching losses, using AdamW with $\beta _ { 1 } = 0 . 9 ,   \beta _ { 2 } = 0 . 9 5$ , weight decay 0.01, and gradient clipping at a norm of 0.5. Both experts use a base learning rate of $1 0 ^ { - 4 }$ . We pretrain for 60,000 optimizer steps, linearly increasing the learning rate from $1 0 ^ { - 1 0 }$ to $9 . 9 \times 1 0 ^ { - 5 }$ during the first 2,000 steps. Training uses BF16 on 64 GPUs, with 12 samples per GPU and two gradient-accumulation steps, giving an effective batch size of $12 \times 64 \times 2 = 1,536$

Sampling and Training Windows. We sample robot and ego data with probabilities 0.7 and 0.3, respectively, and weight repositories within each source category by usable physical duration. Each chunk contains H = 32 action targets and eight future video frames, with one video frame per four action timesteps. Robot and ego action grids are sampled at 15 Hz and 30 Hz, respectively. For sources configured with three camera views, we spatially compose synchronized main-camera and left/right wrist-camera images into a single frame at each timestep. The main view occupies the top region, while the two wrist views, each resized to half the main view’s width and height, are placed side by side below it, forming a T-shaped layout. The composite frame is then bilinearly resized to 320 × 384 pixels (width × height) without added padding and encoded as one video stream by the shared video VAE. Sources configured for a single view use the selected camera image resized to the same resolution. The same camera layout is applied to memory, current, and future frames. The current absolute action record is encoded as a 46-dimensional state token; action targets use the normalized unified representation.

11

<!-- page 12 of 26 -->

Attention Mask and Noise Sampling. We use the pretraining mask in Fig. 4 (a). Video tokens attend to all video tokens and do not attend to the action branch. Action-branch tokens attend to all video tokens and to one another. Thus, video prediction is independent of action inputs, while action prediction can use the full video context. The video and action flow timesteps are sampled independently. Action forcing is disabled in this pretraining configuration: all actions within a chunk share one action flow timestep.

Memory Configuration. Both short-term and long-term visual memory are enabled, with four observations at each scale. On the source-specific action time grid, short-term observations have offsets $\left[ - 8 , - 6 , - 4 , - 2 \right]$ while long-term observations have offsets $\left[ -128, -96, -64, -32 \right]$ relative to the current observation. Thus, $\delta _ { s } = 2$ and $\delta _ { l } = H = 3 2$ in Section 2.1.2. The causal video VAE compresses these histories into two memory latent frames alongside the current-frame latent. All three provide clean conditioning context, with flow timestep zero, under the pretraining attention mask.

## 2.4 Efficiency Implementation

Inference acceleration. We accelerate inference by decoupling video and action denoising. At deployment, a naive implementation performs N denoising steps in both the video and action branches, typically with $N = 1 0$ . Multi-step denoising in the computationally intensive video branch substantially increases inference cost. For acceleration, we perform n synchronized denoising steps that update both branches and construct per-layer video key–value features; the remaining $N - n$ action denoising steps reuse the features from the final synchronized step. The directional attention mask permits this reuse without feedback from action tokens to video tokens, although freezing early video features approximates joint denoising. Visual features are refreshed for every observation, and future-video RGB decoding is omitted. Action chunks, comprising 48 steps in the paper setting, amortize observation encoding across the predicted sequence.

We further accelerate action inference through operator fusion and CUDA Graph replay. The action denoising computation is compiled with torch.compile into optimized GPU kernels, fusing compatible operations to reduce intermediate memory traffic and kernel launches. CUDA Graphs capture the resulting GPU execution sequence and replay it during subsequent denoising steps, reducing Python dispatch and CPU-side kernel-launch overhead. The policy runs in BF16 under inference mode. We also reuse language context, scheduler metadata, and timestep-dependent AdaLN tensors, and evaluate QKV projection, KV-buffer, and fixed-shape attention optimizations. Compilation and graph capture are completed during warmup.

Asynchronous execution. To maintain smooth robot motion despite model inference latency, we decouple policy inference from action execution in real-world experiments. The server computes the next action chunk on the GPU while the client continues executing the current chunk. The client prefetches new predictions based on the remaining execution time of its current plan, with at most one outstanding request to avoid accumulating stale observations. Each predicted chunk is timestamped relative to its input observation. Upon receipt, expired actions are discarded, while the remaining actions retain their original execution times. The client resamples the chunk once onto the 100 Hz servo grid, and the C++ controller handles trajectory splicing and final amplitude limiting.

## 3 Experiments

## 3.1 Task Generalization

We investigate how video and action supervision contribute to task generalization in WAMs. Table 2 compares different supervision settings on RoboTwin 2.0. Across all settings, we keep the ID video and action data fixed, while varying the availability of OOD videos and action labels. Here, ID and OOD refer to a fixed task partition defined before introducing additional supervision.

**Generalization from ID supervision alone.** When trained with only ID video-action supervision, the model achieves a high ID success rate of 91.85%, but its OOD success rate is limited to 25.07%. The nonzero OOD performance indicates that the model can transfer some capabilities beyond the tasks providing supervision.

12

<!-- page 13 of 26 -->

Table 2 Average success rates (%) on RoboTwin 2.0 under different amounts of OOD action supervision. OOD action counts denote the number of demonstrations per task.

| Exp. | # ID | # ID | # OOD | # OOD | ID | OOD | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Index | Videos | Actions | Videos | Actions | SR (%) | SR (%) | SR (%) |
| 1 | 550 | 550 | 0 | 0 | 91.85 | 25.07 | 71.81 |
| 2 | 550 | 550 | 550 | 0 | 91.96 | 55.90 (+30.83) | 81.14 |
| 3 | 550 | 550 | 550 | 1 | 91.70 | 70.70 (+14.80) | 85.40 |
| 4 | 550 | 550 | 550 | 5 | 92.30 | 84.67 (+13.97) | 90.01 |
| 5 | 550 | 550 | 550 | 20 | 92.25 | 90.03 (+5.36) | 91.58 |
| 6 | 550 | 550 | 550 | 50 | 90.22 | 92.27 (+2.24) | 90.83 |
| 7 | 550 | 550 | 550 | 100 | 92.50 | 93.17 (+0.90) | 92.70 |
| 8 | 550 | 550 | 100 | 100 | 93.21 | 88.83 | 91.90 |
| 9 | 550 | 550 | 550 | 550 | 91.55 | 92.83 | 91.93 |

However, the substantial gap between ID and OOD performance shows that this transfer alone is insufficient for reliable execution across the OOD task set.

**Additional videos enable transfer without OOD action labels.** Introducing OOD videos while retaining action supervision exclusively from ID tasks raises OOD success from 25.07% to 55.90%, an absolute improvement of 30.83 percentage points. Meanwhile, ID performance remains nearly unchanged at 91.96%. The resulting overall success rate increases from 71.81% to 81.14%. This improvement demonstrates that additional task knowledge supplied through videos can translate into better robot execution without corresponding OOD action labels. It is consistent with the perspective illustrated in Figure 2: broader video coverage supports task transfer through a video-to-action bridge whose action supervision comes from ID tasks. Nevertheless, the remaining gap between ID and OOD success suggests that video supervision alone does not fully resolve the challenges of executing these tasks.

**Sparse action supervision strengthens task-specific grounding.** Keeping the video data fixed, adding only one action-labeled demonstration per OOD task increases OOD success from 55.90% to 70.70%. With five demonstrations per task, success further rises to 84.67%, yielding a 28.77-point improvement over the setting with OOD videos but no OOD action labels. These gains are consistent with sparse paired supervision strengthening the inverse-dynamics capability that connects visual task evolution to executable actions. In this regime, the additional demonstrations provide action grounding for tasks already covered by the video data. Increasing the action budget to 20 demonstrations per task yields 90.03% OOD success, only 2.80 points below the 550-demonstration setting while using approximately 3.6% of its OOD action-label budget. With 100 demonstrations, OOD success reaches 93.17%; increasing the budget to 550 produces a comparable result of 92.83%, suggesting diminishing returns at higher supervision levels.

**Scaling videos yields further gains with limited action supervision.** Experiments 5–9 further illustrate the complementary roles of video and action supervision. With 100 action-labeled demonstrations per OOD task, increasing OOD videos from 100 to 550 raises OOD success from 88.83% to 93.17%, a gain of 4.34 percentage points. In contrast, with 550 OOD videos fixed, increasing the action budget from 100 to 550 demonstrations yields no improvement, from 93.17% to 92.83%. Moreover, the setting with 550 videos and 20 action-labeled demonstrations (Exp. 5) outperforms that with 100 videos and 100 demonstrations (Exp.8), despite using one-fifth as many action labels. These results suggest that a relatively small amount of paired supervision may suffice to learn an effective video-to-action bridge, or inverse dynamics model. Once this action grounding is established, additional videos can further improve OOD task execution, whereas increasing action supervision offers limited additional gains in this regime.

## 3.2 Simulation Benchmarks

**RoboTwin 2.0.** We evaluate on 50 language-conditioned manipulation tasks using the ALOHA-AgileX bimanual robot in RoboTwin 2.0 [12], covering object rearrangement, articulated-object manipulation, and coordinated two-arm control. We consider two settings: Full RoboTwin uses 50 Clean and 500 Randomized demonstrations per task, while Clean2Random uses only the 50 Clean demonstrations to assess generalization

13

<!-- page 14 of 26 -->

Table 3 Success rates (%) on RoboTwin 2.0 [12] under two training-data settings. Each setting is evaluated in Clean and Randomized scenes, with their average reported. Clean-only training corresponds to the Clean2Random protocol.

| Method | Trai Clean | ning: Clean + Ra Randomized | ndomized Average | T Clean | raining: Clean Randomized | Only Average |
| --- | --- | --- | --- | --- | --- | --- |
| GR00T-N1.7 [45] | - | - | - | 43.6 | 20.7 | 32.2 |
| StarVLA [53] | - | - | - | 58.1 | 10.6 | 34.4 |
| Xiaomi-Robotics-0 [9] | - | - | - | 62.9 | 18.2 | 40.6 |
| π<sub>0</sub> [4] | 65.92 | 58.40 | 62.16 | - | - | - |
| X-VLA [74] | 72.80 | 72.84 | 72.82 | 68.0 | 20.9 | 44.5 |
| Spatial Forcing [29] | - | - | - | 77.2 | 26.7 | 52.0 |
| π<sub>0.5</sub> [47] | 82.74 | 76.76 | 79.75 | 73.1 | 47.9 | 60.5 |
| ABot-M0 [67] | 86.06 | 85.08 | 85.57 | 70.7 | 36.0 | 53.4 |
| Qwen-VLA [57] | 86.10 | 87.20 | 86.65 | - | - | - |
| Motus [3] | 88.66 | 87.02 | 87.84 | - | - | - |
| StarVLA-OFT [53] | 88.20 | 88.30 | 88.25 | - | - | - |
| StarVLA-GR00T [53] | 88.00 | 88.50 | 88.25 | - | - | - |
| StarVLA-π [53] | 88.10 | 88.80 | 88.45 | - | - | - |
| JoyAI-RA 0.1 [73] | 90.48 | 89.28 | 89.88 | - | - | - |
| Being-H0.7 [41] | 90.20 | 89.60 | 89.90 | - | - | - |
| X-WAM [21] | 89.80 | 90.70 | 90.25 | 70.0 | 25.8 | 47.9 |
| Hy-Embodied-0.5-VLA [71] | 90.90 | 90.10 | 90.50 | - | - | - |
| ACE-Ego-0 [32] | 91.12 | 90.62 | 90.87 | - | - | - |
| 4D-WAM [65] | 91.92 | 90.76 | 91.34 | 81.5 | 41.8 | 61.7 |
| Fast-WAM [70] | 91.88 | 91.78 | 91.83 | 77.8 | 1.9 | 39.9 |
| LingBot-VA [33] | 92.93 | 91.55 | 92.24 | 80.7 | 34.6 | 57.7 |
| AHA-WAM [8] | 93.40 | 92.20 | 92.80 | 64.3 | 3.2 | 33.8 |
| InternVLA-A1.5 [42] | 93.30 | 93.00 | 93.20 | - | - | - |
| GalaxeaVLA (G0.5) [38] | 93.70 | 92.80 | 93.30 | - | - | - |
| LingBot-VA 2.0 [72] | 93.80 | 93.40 | 93.60 | - | - | - |
| GigaBrain-0.7 [20] | - | - | - | 66.8 | 67.9 | 67.4 |
| OpenWAM-α [58] | 93.74 | 93.46 | 93.60 | 89.4 | 48.7 | 69.0 |
| Qwen-RobotManip [50] | 93.70 | 94.00 | 93.85 | - | - | - |
| Long-WAM (IDM) [24] | 94.70 | 94.20 | 94.40 | - | - | - |
| MotuBrain [43] | 95.80 | 96.08 | 95.94 | - | - | - |
| Kairos [26] | 96.90 | 95.20 | 96.05 | - | - | - |
| Qwen-RobotManip-Context [50] | - | - | - | 84.7 | 69.4 | 77.1 |
| WorldBridge (Ours) | 96.16 | 96.30 | 96.23 | 88.3 | 73.4 | 80.9 |

to randomized scenes of the same tasks. The policy conditions on head and wrist RGB observations, language instructions, and proprioception, and incorporates short- and long-term visual memory to predict joint-position and gripper commands. We train for 60,000 steps on the combined Clean and Randomized data in Full RoboTwin, and for 20,000 steps on Clean data alone in Clean2Random. We evaluate each setting over 100 episodes per task per scene condition, totaling 10,000 trials. As shown in Table 3, WorldBridge achieves 96.16% success on Clean and 96.30% on Randomized scenes under Full RoboTwin, averaging 96.23%. Under Clean2Random, it achieves 88.3% on Clean and 73.4% on Randomized scenes, averaging 80.9%. These results demonstrate reliable execution across diverse manipulation tasks and robustness to scene variations, including transfer to randomized environments when downstream training uses only Clean demonstrations.

**RoboCasa GR1.** We train on 24,000 teleoperated demonstrations spanning 24 language-conditioned manipulation tasks with the Fourier GR1 humanoid robot in RoboCasa [44], with 1,000 episodes per task. These tasks cover pick-and-place and articulated-object manipulation. We find that video generation quality is important for downstream policy performance. We therefore replay the original demonstrations at a higher rendering resolution, crop out the black borders at the top and bottom, and resize the resulting observations to 416 × 288 pixels (width × height). We fine-tune the model for 100K steps on 16 GPUs, using learning rates of $5 \times 1 0 ^ { - 5 }$ for the video backbone and $1 \times 1 0 ^ { - 5 }$ for the action expert. For evaluation, we perform 50 rollouts on each of the 24 tasks, totaling 1,200 trials, and achieve an overall success rate of 82.75%, outperforming the strongest VLA [32] and WAM [24] baselines in our comparison by 10.0 and 4.1 percentage points, respectively.

**RoboDojo.** We evaluate our policy on RoboDojo [11], an Isaac Sim-based benchmark for bimanual manipulation using the ARX X5 platform. The benchmark comprises 42 tasks spanning five complementary capability

14

<!-- page 15 of 26 -->

Table 4 Results on RoboCasa GR1 TableTop, RMBench, DOMINO Level-1, and SimplerEnv-WidowX. We report success rates (%) and the DOMINO manipulation score (MS)

<table><tr><td rowspan="2">Method</td><td>RoboCasa GR1</td><td>RMBench</td><td colspan="2">DOMINO</td><td colspan="5">SimplerEnv-WidowX</td></tr><tr><td>SR</td><td>SR</td><td>SR</td><td>MS</td><td>Spoon</td><td>Carrot</td><td>Cube</td><td>Eggplant</td><td>Avg.</td></tr><tr><td colspan="10">VLA</td></tr><tr><td>OpenVLA [28]</td><td></td><td></td><td>1.5</td><td>6.1</td><td>4.2</td><td>0.0</td><td>0.0</td><td>12.5</td><td>4.2</td></tr><tr><td>X-VLA [13, 74]</td><td></td><td>9.8</td><td></td><td></td><td>100.0</td><td>91.7</td><td>95.8</td><td>95.8</td><td>95.8</td></tr><tr><td>π0 [4]</td><td>13.6</td><td></td><td>8.2</td><td>24.0</td><td>29.1</td><td>0.0</td><td>16.6</td><td>62.5</td><td>27.1</td></tr><tr><td>PUMA [18]</td><td></td><td></td><td>17.2</td><td>35.0</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Magma [64]</td><td></td><td></td><td></td><td></td><td>37.5</td><td>31.0</td><td>12.7</td><td>60.5</td><td>35.8</td></tr><tr><td>π0.5 [47]</td><td>37.0</td><td>10.4</td><td>9.6</td><td>26.2</td><td>49.3</td><td>64.7</td><td>44.7</td><td>69.7</td><td>57.1</td></tr><tr><td>Mem-0 [13]</td><td></td><td>42.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SpatialVLA [48]</td><td></td><td></td><td></td><td></td><td>16.7</td><td>25.0</td><td>29.2</td><td>100.0</td><td>42.7</td></tr><tr><td>CronusVLA [30]</td><td></td><td></td><td></td><td></td><td>66.7</td><td>54.2</td><td>20.8</td><td>100.0</td><td>60.4</td></tr><tr><td>GR00T-N1.6 [46]</td><td>47.6</td><td></td><td></td><td></td><td>55.4</td><td>46.0</td><td>5.0</td><td>89.0</td><td>48.9</td></tr><tr><td>StarVLA-π [53]</td><td>48.9</td><td></td><td></td><td></td><td>78.1</td><td>46.9</td><td>30.2</td><td>88.5</td><td>60.9</td></tr><tr><td>RoboInter [31]</td><td></td><td></td><td></td><td></td><td>54.2</td><td>25.0</td><td>29.2</td><td>91.7</td><td>50.0</td></tr><tr><td>StarVLA-GR00T [53]</td><td>52.8</td><td></td><td></td><td></td><td>83.0</td><td>59.4</td><td>18.8</td><td>100.0</td><td>65.3</td></tr><tr><td>StarVLA-OFT [53]</td><td>53.8</td><td></td><td>10.9</td><td>30.5</td><td>90.3</td><td>38.5</td><td>29.7</td><td>100.0</td><td>64.6</td></tr><tr><td>Qwen-VLA-Instruct [57]</td><td>56.7</td><td></td><td>26.6</td><td>39.5</td><td></td><td></td><td></td><td></td><td>73.7</td></tr><tr><td>ABot-M0 [67]</td><td>58.3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GR00T-N1.7 [45]</td><td></td><td></td><td></td><td></td><td>78.0</td><td>58.0</td><td>48.0</td><td>53.0</td><td>59.3</td></tr><tr><td>JoyAI-RA 0.1 [73]</td><td>63.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DIAL [14]</td><td>70.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FutureVLA [62]</td><td></td><td></td><td></td><td></td><td>83.3</td><td>75.0</td><td>29.2</td><td>100.0</td><td>71.9</td></tr><tr><td>ACE-Ego-0 [32]</td><td>72.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LaMem-VLA [49]</td><td></td><td></td><td></td><td></td><td>83.3</td><td>75.0</td><td>41.7</td><td>95.8</td><td>73.9</td></tr><tr><td>InternVLA-A1.5 [42]</td><td></td><td></td><td>29.3</td><td>42.5</td><td>92.4</td><td>85.4</td><td>69.5</td><td>75.7</td><td>80.8</td></tr><tr><td>Xiaomi-Robotics-0 [9]</td><td></td><td></td><td></td><td></td><td>95.8</td><td>62.5</td><td>75.0</td><td>83.3</td><td>79.2</td></tr><tr><td>GTA-VLA [37]</td><td></td><td></td><td></td><td></td><td>95.8</td><td>87.5</td><td>66.7</td><td>75.0</td><td>81.2</td></tr><tr><td>GalaxeaVLA (G0.5) [38]</td><td></td><td></td><td></td><td></td><td>97.5</td><td>75.0</td><td>83.3</td><td>93.3</td><td>87.3</td></tr><tr><td colspan="10">WAM</td></tr><tr><td>Fast-WAM [70]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OA-WAM [39]</td><td></td><td>5.9</td><td>19.9</td><td>33.3</td><td>83.0</td><td>71.1</td><td>65.0</td><td>98.2</td><td>79.3</td></tr><tr><td>DynamicWAM [40]</td><td></td><td></td><td>38.2</td><td>53.2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Being-H0.7 [41]</td><td>49.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OpenWAM-α [58]</td><td>60.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LingBot-VA [33]</td><td></td><td>78.2</td><td>24.1</td><td>36.1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Long-WAM [24]</td><td>78.7</td><td></td><td>34.9</td><td>45.1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MemoryWAM [66]</td><td></td><td>83.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WorldBridge (Ours)</td><td>82.8</td><td>83.0</td><td>57.6</td><td>64.6</td><td>95.8</td><td>83.3</td><td>91.7</td><td>66.7</td><td>84.4</td></tr></table>

dimensions: Generalization, Precision, Long-Horizon, Memory, and Open. These dimensions assess robustness to scene variations, fine-grained manipulation accuracy, multi-step execution, reasoning over past observations, and transfer to unseen task specifications, respectively. Our policy achieves an overall Score of 26.03 and a success rate of 19.15% on RoboDojo, with Scores of 23.18, 35.85, 37.29, 31.35, and 2.50 across the five capability dimensions: Generalization, Precision, Long-Horizon, Memory, and Open, respectively. These results demonstrate capabilities spanning fine-grained control, multi-stage manipulation, and reasoning over observation history. We fine-tune the policy using AdamW with a learning rate of $5 \times 1 0 ^ { - 5 }$ , a global batch size of 256, and bfloat16 precision. We use a cosine learning-rate schedule with 2,000 warm-up steps and report results from the checkpoint after 50,000 fine-tuning steps.

**SimplerEnv-WidowX.** We evaluate language-conditioned manipulation policy transfer on the WidowX suite of SimplerEnv [34], a real-to-sim benchmark that instantiates the BridgeData V2/WidowX setup for scalable simulated policy evaluation. We follow the official SimplerEnv WidowX task suite and evaluate four Bridge style manipulation tasks: putting a spoon on a towel, putting a carrot on a plate, stacking a green cube on

15

<!-- page 16 of 26 -->

Table 5 Performance comparison on the RoboDojo benchmark. Score and success rate (SR, %) are reported across six evaluation settings and averaged over the five capability dimensions. Additional entries are taken from the official leaderboard (October 9, 2026) [51].

<table><tr><td rowspan="2">Model</td><td colspan="2">Gen-Std</td><td colspan="2">Gen-Rand</td><td colspan="2">Precision</td><td colspan="2">Long</td><td colspan="2">Memory</td><td colspan="2">Open</td><td colspan="2">Average</td></tr><tr><td>Score</td><td>SR</td><td>Score</td><td>SR</td><td>Score</td><td>SR</td><td>Score</td><td>SR</td><td>Score</td><td>SR</td><td>Score</td><td>SR</td><td>Score</td><td>SR</td></tr><tr><td colspan="10">VLA</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GR00T-N1.7 [45]</td><td>3.97</td><td>2.44</td><td>0.35</td><td>0.00</td><td>2.54</td><td>0.67</td><td>8.30</td><td>3.58</td><td>1.06</td><td>0.89</td><td>0.18</td><td>0.17</td><td>2.85</td><td>1.31</td></tr><tr><td>π0 [4]</td><td>7.18</td><td>4.89</td><td>0.71</td><td>0.22</td><td>3.56</td><td>0.75</td><td>6.19</td><td>2.00</td><td>3.47</td><td>2.11</td><td>0.25</td><td>0.25</td><td>3.48</td><td>1.53</td></tr><tr><td>ABot-M0 [67]</td><td>9.20</td><td>5.44</td><td>2.26</td><td>1.56</td><td>5.50</td><td>1.75</td><td>3.96</td><td>0.50</td><td>2.44</td><td>2.22</td><td>0.72</td><td>0.67</td><td>3.67</td><td>1.73</td></tr><tr><td>StarVLA [53]</td><td>7.54</td><td>4.67</td><td>0.33</td><td>0.00</td><td>9.90</td><td>4.33</td><td>14.15</td><td>6.50</td><td>3.34</td><td>2.44</td><td>0.67</td><td>0.58</td><td>6.40</td><td>3.24</td></tr><tr><td>Xiaomi-Robotics-0 [9]</td><td>13.81</td><td>10.78</td><td>1.05</td><td>0.33</td><td>8.42</td><td>4.58</td><td>13.51</td><td>6.92</td><td>5.07</td><td>3.67</td><td>0.22</td><td>0.17</td><td>6.93</td><td>4.18</td></tr><tr><td>StarVLA-OFT [53]</td><td>13.42</td><td>9.00</td><td>1.39</td><td>0.56</td><td>16.08</td><td>10.17</td><td>14.28</td><td>6.17</td><td>1.74</td><td>1.67</td><td>1.41</td><td>1.33</td><td>8.18</td><td>4.82</td></tr><tr><td>StarVLA-GR00T [53]</td><td>13.94</td><td>9.56</td><td>1.97</td><td>1.00</td><td>9.56</td><td>4.67</td><td>16.18</td><td>7.17</td><td>5.60</td><td>4.33</td><td>1.95</td><td>1.92</td><td>8.25</td><td>4.67</td></tr><tr><td>X-VLA [74]</td><td>17.90</td><td>12.22</td><td>3.04</td><td>1.33</td><td>18.32</td><td>12.00</td><td>16.53</td><td>9.75</td><td>4.76</td><td>3.56</td><td>0.55</td><td>0.50</td><td>10.13</td><td>6.52</td></tr><tr><td>StarVLA-PI v3 [53]</td><td>19.08</td><td>14.44</td><td>3.36</td><td>1.67</td><td>17.77</td><td>12.50</td><td>18.46</td><td>11.00</td><td>4.59</td><td>4.00</td><td>2.03</td><td>2.00</td><td>10.81</td><td>7.51</td></tr><tr><td>InternVLA-A1.5 [42]</td><td>16.81</td><td>11.78</td><td>3.90 5.82</td><td>1.89</td><td>15.23</td><td>10.17</td><td>23.80</td><td>13.75</td><td>4.93</td><td>3.56</td><td>1.43</td><td>1.42</td><td>11.15</td><td>7.14</td></tr><tr><td>π0.5 [47]</td><td>20.93</td><td>14.89</td><td></td><td>1.44</td><td>12.40</td><td>5.50</td><td>23.54</td><td>14.67</td><td>5.78</td><td>4.56</td><td>1.98</td><td>1.67</td><td>11.41</td><td>6.91</td></tr><tr><td>Spatial Forcing [29]</td><td>21.25</td><td>14.89 16.56</td><td>6.98 1.57</td><td>3.78</td><td>17.32</td><td>10.58</td><td>23.26</td><td>14.58</td><td>5.43</td><td>4.11</td><td>1.78</td><td>1.58</td><td>12.38</td><td>8.04</td></tr><tr><td>Hy-Embodied-0.5-VLA [71] Xiaomi-Robotics-1 [61]</td><td>21.98 35.65</td><td>28.00</td><td>11.44</td><td>0.22 6.00</td><td>13.81</td><td>8.00</td><td>25.74</td><td>14.92</td><td>13.37</td><td>12.11</td><td>0.65</td><td>0.58</td><td>13.07</td><td>8.80</td></tr><tr><td>GalaxeaVLA (G0.5) [38]</td><td>27.72</td><td>21.44</td><td>9.20</td><td>4.22</td><td>26.69 28.25</td><td>18.83 20.42</td><td>38.39 44.12</td><td>23.67 32.25</td><td>7.81 8.61</td><td>6.56</td><td>3.94 1.73</td><td>3.58</td><td>20.07</td><td>13.93</td></tr><tr><td>DM0.5 [17]</td><td>23.49</td><td>17.89</td><td>8.06</td><td>4.00</td><td>24.82</td><td>16.75</td><td>33.70</td><td>19.50</td><td>47.74</td><td>7.33 47.44</td><td>2.43</td><td>1.58 2.08</td><td>20.23</td><td>14.88</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>WAM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>24.90</td><td>19.34</td></tr><tr><td colspan="10"></td><td colspan="7"></td></tr><tr><td colspan="10"></td><td colspan="7"></td></tr><tr><td>Fast-WAM [70] AHA-WAM [8]</td><td>4.33 10.32</td><td>2.11 6.22</td><td>0.34 1.26</td><td>0.11 0.33</td><td>1.96 5.86</td><td>0.00 2.42</td><td>9.14 8.61</td><td>5.17 2.67</td><td>3.55 2.97</td><td>3.44 2.78</td><td>0.42 0.88</td><td>0.42 0.83</td><td>3.48 4.82</td><td>2.03 2.39</td></tr><tr><td>X-WAM [21]</td><td>11.24</td><td>5.33</td><td>3.54</td><td>1.33</td><td>6.72</td><td>1.83</td><td>17.47</td><td>9.08</td><td>6.32</td><td>4.67</td><td>0.57</td><td>0.25</td><td>7.69</td><td>3.83</td></tr><tr><td>OpenWAM-α [58]</td><td>33.16</td><td>25.56</td><td>8.26</td><td>4.11</td><td>18.45</td><td>9.25</td><td>34.93</td><td>25.33</td><td>10.41</td><td>9.11</td><td>1.41</td><td>1.08</td><td>17.18</td><td>11.92</td></tr><tr><td>WorldBridge (Ours)</td><td>36.38</td><td>28.00</td><td>9.97</td><td>4.00</td><td>35.85</td><td>26.50</td><td>37.29</td><td>20.50</td><td>31.35</td><td>31.00</td><td>2.50</td><td>1.75</td><td>26.03</td><td>19.15</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

a yellow cube, and putting an eggplant into a yellow basket. WorldBridge is fine-tuned on the real-world dataset BridgeData V2 [55] for 60K gradient steps, without additional simulation demonstrations during this adaptation. The policy conditions on RGB observations, language instructions, and proprioception, with shortand long-term visual histories, and predicts 16 consecutive 7-D actions. As shown in Table 4, WorldBridge achieves 84.4% average success, with task-wise rates of 95.8%, 83.3%, 91.7%, and 66.7%, respectively. It exceeds the OA-WAM, InternVLA-A1.5 and the most VLAs with a reported result in this suite.

**RMBench.** RMBench [13] assesses dual-arm manipulation across nine non-Markovian tasks where decisionmaking relies heavily on past temporal cues rather than instantaneous observations. Following the standard protocol, we train the policy with 50 expert trajectories per task and evaluate over 100 rollouts. Crucially, during both training and inference, we cache all historical frames to serve as the conditioning context. As reported in Table 4, methods limited by bounded observation windows or reactive mappings struggle severely under such temporal dependencies, with Fast-WAM [70] (5.9%), X-VLA [74] (9.8%), and π<sub>0</sub>.5 [47] (10.4%) achieving marginal success. In contrast, WorldBridge establishes state-of-the-art performance with an **83.0%** average success rate—surpassing strong full-history baselines such as LingBot-VA [33] (78.2%) and matching MemoryWAM [66]—demonstrating its capability to leverage complete temporal contexts for complex robotic manipulation.

**DOMINO.** This benchmark [18] evaluates manipulation with moving targets, extending static-scene evaluation to dynamic interception and tracking. We evaluate the ALOHA-AgileX on all 35 Level-1 tasks, which use constant-velocity target motion with dynamics coefficient α = 0.1, and evaluate 7,000 episodes in total across the Clean and Randomized dynamic configurations, following the PUMA evaluation protocol [18]. WorldBridge directly reuses the checkpoint trained on RoboTwin 2.0 and is evaluated on DOMINO without additional dynamic-task fine-tuning. The policy uses head and wrist RGB observations, language, and proprioception; it predicts 16 actions and executes the first ten before replanning. Randomized evaluation additionally varies backgrounds, lighting, table height, and clutter. We report success rate (SR) and manipulation score (MS), averaged over both configurations, and MS measures normalized end-effector progress toward the target, with penalties for out-of-bounds motion and clutter collisions. Table 4 reports 57.6% SR and 64.6 MS for WorldBridge, exceeding the reported results of DynamicWAM, the second-best method in this comparison, by 19.4 percentage points in SR and 11.4 points in MS. Within our evaluation, Clean yields 59.1% SR and 67.7 MS, while Randomized yields 56.1% SR and 61.6 MS, showing a modest success-rate decrease under

16

<!-- page 17 of 26 -->

Table 6 Ablation study of action forcing and memory length on three robotic manipulation benchmarks.

| Exp. | Action | Short | Long | RoboTwin | RoboCasa | RoboDojo |
| --- | --- | --- | --- | --- | --- | --- |
| Index | Forcing | Memory | Memory | SR (%) | SR (%) | SR (%) |
| 1 | ✗ | ✗ | ✗ | 91.28 | 67.50 | 14.50 |
| 2 | ✓ | ✗ | ✗ | 93.31 | 69.17 | 16.52 |
| 3 | ✓ | ✓ | ✗ | 93.98 | 74.33 | 16.90 |
| 4 | ✓ | ✓ | ✓ | 95.20 | 78.58 | 21.07 |

Table 7 Effect of robot and egocentric pretraining on RoboTwin Clean-to-Random.

<table><tr><td rowspan="2">Exp. Index</td><td colspan="2">Pretraining</td><td colspan="3">RoboTwin Clean-to-Random</td></tr><tr><td>Robot</td><td>Egocentric</td><td>Clean (%)</td><td>Rand (%)</td><td>Avg (%)</td></tr><tr><td>1</td><td>x</td><td>x</td><td>82.9</td><td>42.6</td><td>62.7</td></tr><tr><td>2</td><td>√</td><td>X</td><td>87.0</td><td>67.7</td><td>77.4</td></tr><tr><td>3</td><td></td><td></td><td>88.3</td><td>73.4</td><td>80.9</td></tr></table>

additional scene variation. These results demonstrate strong static-to-dynamic transfer of our model without dynamic-task fine-tuning. This capability benefits from the complementary roles of short- and long-term memory, which provide temporal cues and world-modeling priors acquired through pretraining.

**Benefits of action forcing and memory.** Table 6 presents a cumulative ablation, adding action forcing, short-term memory, and long-term memory in sequence. Following the joint video–action architecture in Section 2.1.1, action forcing trains the action expert with actions at different noise levels within a chunk (Section 2.1.3), while the two memory scales provide recent motion cues and broader task history (Section 2.1.2). Adding action forcing alone improves success from 91.28% to 93.31% on RoboTwin, from 67.50% to 69.17% on RoboCasa, and from 14.50% to 16.52% on RoboDojo. Adding short-term memory next yields further gains of 0.67, 5.16, and 0.38 percentage points, respectively; adding long-term memory then contributes another 1.22, 4.25, and 4.17 points. The full configuration reaches 95.20%, 78.58%, and 21.07%, improving over the configuration without action forcing or memory by 3.92, 11.08, and 6.57 points. These sequential results support the complementary value of action forcing and temporal context, with memory additions particularly beneficial on RoboCasa and RoboDojo.

**Benefits of pretraining.** Table 7 isolates the contributions of robot and egocentric pretraining on RoboTwin Clean-to-Random. Without either source, the model achieves 62.7% average success (82.9% on Clean and 42.6% on Randomized scenes). Adding robot pretraining raises the average to 77.4% (+14.7 percentage points), with the largest gain on Randomized scenes, from 42.6% to 67.7%. Adding egocentric pretraining on top further improves the Clean and Randomized results to 88.3% and 73.4%, respectively, raising the average to 80.9% (+3.5 points). Overall, combining both sources improves average success by 18.2 points over training without pretraining, suggesting that egocentric data provides complementary gains, particularly under scene randomization.

## 3.3 Real-Robot Experiments

Tasks and evaluation procedure. We evaluate WorldBridge on six complementary manipulation tasks across AgiBot G2 and ARX AC-1, covering long-horizon execution, precise manipulation, spatial alignment, bimanual coordination, and memory-dependent interaction. Figure 6 presents representative executions, with annotations highlighting the operations associated with these capability requirements. On G2, arranging flowers requires relocating vases, successively grasping and inserting two slender flowers, and placing the completed arrangement. It tests long-horizon sequencing together with precision insertion, since the robot must preserve task progress while repeatedly aligning a thin stem with a vase opening. Scanning drinks interleaves sequential manipulation with human-induced scene changes, testing whether the robot can maintain task continuity and the required processing order using visual memory. Hanging cups requires grasping two mugs and placing their handles onto rack hooks at different heights. It tests spatial understanding and orientation-sensitive placement, coupling

17

<!-- page 18 of 26 -->

Long horizon and precision insertion

![Image block](doc:adc627e/tier:basic/page:18/block:2)

Figure 6 Representative real-robot executions on six manipulation tasks. From top to bottom: arranging flowers, scanning drinks, separating LEGO bricks, packing books, sorting screws, and hanging cups. Arrows and circles highlight key operations, including sequential flower insertion, bimanual brick separation, and precise placement. The drink-scanning task probes three aspects of memory: remembering post-scan placement destinations, tracking previously scanned drinks, and processing the newly introduced drink last.

grasp selection with alignment of the handle and hook.

On AC-1, separating LEGO bricks requires repeated bimanual disassembly of connected bricks followed by sorting into bins by color. It tests long-horizon coordination, stable grasping during separation, and classification-guided placement as the assembly changes throughout execution. Sorting screws requires grasping small screws and nuts, placing them into their designated compartments, and closing the box. It combines fine-grained object discrimination with precise grasping and placement, requiring correct part selection and accurate manipulation within a single sequence. Packing books requires opening a schoolbag, inserting a book while maintaining access to the opening, and final closing the bag. It tests long-horizon manipulation of a deformable container: the robot must coordinate bag opening with subsequent insertion despite changes in the bag’s shape and accessible geometry.

We compare WorldBridge with $\pi _ { 0 . 5 }$ [47], Motus [3], FastWAM [70], and LingBot-VA [33], and evaluate a WorldBridge variant without our joint video–action pretraining. All methods are fine-tuned on the same demonstration set for each task and evaluated over 20 trials with varied initial object positions or orientations. Success requires completing the entire task without human assistance to the robot; introducing a drink during scanning is a prescribed interaction. We report per-task and average success rates across all tasks.

18

<!-- page 19 of 26 -->

![Chart block](doc:adc627e/tier:basic/page:19/block:1)

Figure 7 Task success rates across five methods.

Overall real-robot performance. Figure 7 summarizes the quantitative results. WorldBridge achieves an average success rate of 72.5%, compared with 57.5% for $\pi _ { 0 . 5 } ,$ 39.2% for LingBot-VA, 29.2% for FastWAM, and 10.8% for Motus. It achieves the highest success rate on five tasks and ties $\pi _ { 0 . 5 }$ at 80% on separating LEGO bricks. On G2, WorldBridge reaches 70% on arranging flowers and 85% on hanging cups, compared with 65% and 75% for $\pi _ { 0 . 5 }$ , respectively. These results cover both extended manipulation sequences and precise spatial placement; the larger gap on scanning drinks is discussed below. On AC-1, WorldBridge achieves 80% on packing books and 55% on sorting screws, exceeding $\pi _ { 0 . 5 }$ at 75% and 45%, respectively, while matching its LEGO disassembly performance. Sorting screws remains the least successful task for WorldBridge, identifying precise small-part sorting as a remaining challenge within this suite.

Cross-embodiment experiments. Cross-embodiment adaptation follows our unified 46-dimensional action representation, which maps robot-specific control variables into fixed semantic slots and uses dimension-wise validity masks to exclude unavailable components from pretraining supervision. This common interface allows the action expert to learn from heterogeneous control data while preserving the semantics of the available controls. For the real-robot experiments, we fine-tune the same jointly pretrained checkpoint separately for G2 and AC-1, transferring the shared video and action experts and adapting the state and action interfaces to each platform. G2 uses 16-dimensional native commands comprising seven joints per arm and two grippers, whereas AC-1 uses 14-dimensional commands comprising six joints per arm and two grippers. Platform-specific input and output adapters connect the common pretrained experts to these native interfaces.

Benefits of joint video–action pretraining. The paired WorldBridge results in Fig. 7 show that omitting joint video–action pretraining reduces average success to 46.7%, an absolute decrease of 25.8 percentage points. This variant retains the pretrained Wan video backbone [56], so the comparison measures the additional benefit of our joint pretraining stage. Pretraining improves all six tasks. The largest gain occurs on sorting screws, where success increases from 10% to 55%; packing books and separating LEGO bricks each improve from 50% to 80%. The screw-sorting sequence in Fig. 6 illustrates the fine grasping and compartment placement required by the task with the largest pretraining gain. On G2, the gains are 15 percentage points for arranging flowers, 20 for hanging cups, and 15 for scanning drinks. The improvements therefore span precise small-object handling, spatial alignment, coordinated manipulation, and memory-dependent execution. In particular, the substantial gains on the AC-1 tasks support the value of joint pretraining for downstream adaptation under the available task demonstrations.

Memory-dependent drink scanning. In this task, the robot must grasp a scanner, scan the original drinks in order, and finally scan an additional drink introduced by a human during execution. Correct execution requires three aspects of memory. First, the robot must remember each drink’s placement destination after scanning, preserving the spatial association across grasping, scanning, and placement. Second, it must track which drinks have already been scanned to select the next drink without repeating or omitting a scan. Third,

19

<!-- page 20 of 26 -->

Table 8 Cumulative ablation of model-call latency for real-world deployment. Timings are measured after warmup with CUDA synchronization using BF16, recorded observations, precomputed T5 embeddings, and ten action denoising steps. Each configuration uses 60 calls across three measurement rounds. The checkpoint uses no historical observations. Timing includes conditioning VAE encoding and denoising, excluding JPEG processing, text encoding, communication, and robot execution.

| Configuration | P50 (ms) | P95 (ms) | Speedup |
| --- | --- | --- | --- |
| Joint video/action, 10/10 | 957.2 | 1085.0 | 1.00× |
| Decoupled video/action, 1/10 | 269.7 | 290.5 | 3.55× |
| + Inference mode and runtime caches | 218.7 | 260.0 | 4.38× |
| + QKV projection and KV buffers | 204.1 | 212.1 | 4.69× |
| + Attention switches (full eager profile) | 227.2 | 232.0 | 4.21× |
| + Compilation, without CUDA Graphs | 126.6 | 134.7 | 7.56× |
| + CUDA Graph replay | 103.9 | 105.6 | 9.21× |

it must identify the newly introduced drink and defer its scan until the original drinks have been processed, preserving the order of task events despite the scene change. The drink-scanning row in Fig. 6 visualizes these dependencies through placement markers, highlighted scanned drinks, and the human introduction of a drink designated for the final scan. A drink’s current pose does not necessarily reveal its scanning status or when it entered the scene; visually similar observations can therefore demand different next actions depending on the execution history. WorldBridge conditions action prediction on short- and long-term visual history, providing context for recent manipulation transitions and earlier task events. As quantified in Fig. 7, WorldBridge achieves 65% success, compared with 35% for LingBot-VA and 5% for $\pi _ { 0 . 5 }$ . This advantage is consistent with visual memory helping resolve ambiguity about task state and maintain the required execution order during interaction.

Latency ablation. Table 8 reports a cumulative ablation of model-call latency. Reducing video denoising from ten steps to one while retaining ten action denoising steps lowers median latency from 957.2 to 269.7 ms (3.55×). Runtime caching and QKV/KV-buffer optimizations further reduce it to 204.1 ms, while the full eager attention-switch profile raises it to 227.2 ms. Compilation without CUDA Graphs lowers median latency to 126.6 ms (7.56× speedup over baseline), and CUDA Graph replay reduces it further to 103.9 ms (9.21×), with a final P95 of 105.6 ms. The explicit action FlashAttention switch does not improve eager latency, and some eager-path optimizations are superseded by compilation. With these optimizations, the deployed system sustains nearly 7 Hz during real-world execution, including data processing and communication, while maintaining strong task success rates and demonstrating practical real-time usability.

## 4 Related Work

Vision-language-action (VLA) models use pretrained vision-language models to learn robot policies that follow language instructions. RT-2 transfers web knowledge to robot control [5], and OpenVLA provides an open-source model trained on diverse robot demonstrations [28]. $\pi _ { 0 }$ uses flow matching to generate continuous actions [4], while $\pi _ { 0 . 5 }$ uses heterogeneous training data to improve generalization across environments [47]. However, these models still struggle with new tasks when action-labeled demonstrations are limited or unavailable. Combining action prediction with world modeling offers a way to learn both how tasks unfold and how to execute them. WorldVLA unifies image and action understanding and generation in an autoregressive model, showing that jointly learning the two can improve both action prediction and visual generation [10]. Video-based World Action Models (WAMs) further use video priors to support robot control beyond the coverage of action-labeled data. DreamZero shows that these priors can support unseen-task execution and transfer from video-only demonstrations [68], while Zero-WAM uses human videos as in-context guidance for new tasks [75]. Other studies explore video-action pretraining [58, 72] and temporal context through longer observation histories or persistent memory [24, 66]. Our work examines how video coverage and action supervision contribute to task generalization, and how the video-to-action bridge turns predicted visual dynamics into executable actions.

20

<!-- page 21 of 26 -->

## 5 Conclusion

We presented WorldBridge, a World Action Model that connects visual dynamics to executable control for task generalization. Our study highlights the complementary roles of video coverage and action grounding: additional task videos enable transfer beyond the tasks providing action supervision, while even a single actionlabeled demonstration per new task substantially improves execution. Guided by these findings, WorldBridge couples video and action experts through structured attention and joint flow matching, incorporating temporal memory, action forcing, and unified pre-training on heterogeneous robot and egocentric demonstrations. Experiments across simulation benchmarks and two real-robot platforms demonstrate strong manipulation performance and the benefits of joint video–action pretraining. These results suggest that advancing task generalization requires both broader visual experience and a transferable video-to-action bridge. Extending this bridge to more diverse tasks with less action supervision remains an important direction for future work.

## 6 Contributions

## Core Contributors

Jun Cen<sup>5</sup>, Hao Li<sup>2,3,4</sup>, Liangkai Li<sup>1,4</sup>, Yaoxu Lyu<sup>1,2,3,4</sup>, Meng Wei<sup>3</sup>, Zeqi Xiao<sup>3</sup>, Sizhe Yang<sup>3,4</sup>, Yifan Ye<sup>3,5</sup>, Duo Zheng<sup>1,2,3</sup>

Authors are listed in alphabetical order. The superscripts denote: <sup>1</sup> Data, <sup>2</sup> Pretraining, <sup>3</sup> Post-training, Real-world Infra and Deployment, <sup>5</sup> Model Architecture.

## Contributors

Yubo Guo, Yuqiang Yang, Shanghang Zhang

Authors are listed in alphabetical order.

## Supervision

Jun Cen†

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">† Corresponding author: cenjun.1@jd.com</span></small>

21

<!-- page 22 of 26 -->

## References

[1] AgiBot World Team. AgiBot World 2026. [https://huggingface.co/datasets/agibot-world/AgiBotWorld2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026),2026.

[2] Arthur Allshire, Himanshu Gaurav Singh, Ritvik Singh, Adam Rashid, Hongsuk Choi, David McAllister, Justin Yu, Yiyuan Chen, Huang Huang, Pieter Abbeel, Xi Chen, Rocky Duan, Phillip Isola, Jitendra Malik, Fred Shentu, Guanya Shi, Philipp Wu, and Angjoo Kanazawa. Scalable behavior cloning with open data, training, and evaluation. arXiv preprint arXiv:2606.27375, 2026. [https://arxiv.org/abs/2606.27375](https://arxiv.org/abs/2606.27375).

[3] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, and Jun Zhu. Motus: A unified latent action world model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 35101–35113, 2026.

[4] Kevin Black et al. π0: A vision-language-action flow model for general robot control, 2024.

[5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. RT-2: Vision-language-action models transfer web knowledge to robotic control, 2023. [https://arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818).

[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, 2020. [https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html).

[7] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xu Huang, et al. AgiBot World Colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025. [https://arxiv.org/abs/2503.06669](https://arxiv.org/abs/2503.06669).

[8] Jisong Cai, Long Ling, Shiwei Chu, Zhongshan Liu, Jiayue Kang, Zhixuan Liang, Wenjie Xu, Yinan Mao, Weinan Zhang, Xiaokang Yang, Ru Ying, Ran Zheng, and Yao Mu. Aha-wam: Asynchronous horizon-adaptive world-action modeling with observation-guided context routing. arXiv preprint arXiv:2606.09811, 2026.

[9] Rui Cai, Jun Guo, Xinze He, Piaopiao Jin, et al. Xiaomi-robotics-0: An open-sourced vision-language-action model with real-time execution, 2026.

[10] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, and Hao Chen. Worldvla: Towards autoregressive action world model, 2025. [https://arxiv.org/abs/2506.21539](https://arxiv.org/abs/2506.21539).

[11] Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Weijie Wan, Baijun Chen, Haoran Lu, Haowen Yan, Honghao Su, et al. RoboDojo: A unified sim-and-real benchmark for comprehensive evaluation of generalist robot manipulation policies. arXiv preprint arXiv:2607.04434, 2026. [https://arxiv.org/abs/2607.04434.](https://arxiv.org/abs/2607.04434)

[12] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan-Ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, and Yao Mu. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 13673–13699. PMLR, 2026. [https://proceedings.mlr.press/v306/chen26j.html](https://proceedings.mlr.press/v306/chen26j.html).

22

<!-- page 23 of 26 -->

[13] Tianxing Chen, Yuran Wang, Mingleyang Li, Yan Qin, Hao Shi, Zixuan Li, Yifan Hu, Yingsheng Zhang, Kaixuan Wang, Yue Chen, et al. Rmbench: Memory-dependent robotic manipulation benchmark with insights into policy design. arXiv preprint arXiv:2603.01229, 2026.

[14] Yi Chen, Yuying Ge, Hui Zhou, Mingyu Ding, Yixiao Ge, and Xihui Liu. Dial: Decoupling intent and action via latent world modeling for end-to-end vla, 2026.

[15] DeepSeek-AI. Deepseek-v3 technical report, 2024. [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437).

[16] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186. Association for Computational Linguistics, 2019. doi: 10.18653/v1/N19-1423. [https://aclanthology.org/N19-1423/](https://aclanthology.org/N19-1423/).

[17] Dexmal Team. Opendm: Dm0.5 vision-language-action model. [https://github.com/dexmal/opendm](https://github.com/dexmal/opendm), 2026. Official model repository.

[18] Heng Fang, Shangru Li, Shuhan Wang, Xuanyang Xi, Dingkang Liang, and Xiang Bai. Towards generalizable robotic manipulation in dynamic environments, 2026. [https://arxiv.org/abs/2603.15620](https://arxiv.org/abs/2603.15620).

[19] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. LIBERO-Plus: A progressive robustness benchmark for visual-language-action models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. [https://openaccess.thecvf.com/content/CVPR2026/papers/Fei\_LIBERO-Plus\_A\_Progressive\_Robustness\_Benchmark\_for\_Visual-Language-Action\_Models\_CVPR\_2026\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf).

[20] GigaBrain Team, Angen Ye, Axiang Sun, Can Jin, Chenxi Cheng, Chong Shi, et al. Gigabrain-0.7: Scaling embodied foundation models to emergent capabilities with a three-system architecture, 2026.

[21] Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, and Huaping Liu. Unified 4d world action modeling from video priors with asynchronous denoising. arXiv preprint arXiv:2604.26694, 2026.

[22] Ryan Hoque, Peide Huang, David Yoon, Jian Zhang, et al. Egodex: Learning dexterous manipulation from large-scale egocentric video. In International Conference on Learning Representations, volume 2026, 2026.

[23] Chengkai Hou, Kun Wu, Jiaming Liu, Zhengping Che, Di Wu, Fei Liao, et al. RoboMIND 2.0: A multimodal, bimanual mobile manipulation dataset for generalizable embodied intelligence. arXiv preprint arXiv:2512.24653, 2025. [https://arxiv.org/abs/2512.24653](https://arxiv.org/abs/2512.24653).

[24] Wei Huang, Bohan Zhang, Chenzhi Liu, Isabella Liu, Shuai Yang, Weian Mao, Luozhou Wang, Yicheng Xiao, Weifeng Lin, Qixin Hu, Bryan Chu, Sifei Liu, Linxi Fan, Xiaojuan Qi, Song Han, and Yukang Chen. Long-wam: Scaling the context of world-action models. arXiv preprint arXiv:2610.10528, 2026.

[25] Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea Open-World Dataset and G0 dual-system VLA model. arXiv preprint arXiv:2509.00576, 2025. [https://arxiv.org/abs/2509.00576](https://arxiv.org/abs/2509.00576).

[26] Kairos Team, Fei Wang, Shan You, Qiming Zhang, Tao Huang, Zuoyi Fu, Zhisheng Zheng, Yunlong Xi, Feng Lv, Xiaoming Wu, Zeyu Liu, Cong Wan, Pu Li, Ruiqing Yang, Xiaoou Li, Wei Wang, Kangkang Zhu, Yuwei Zhang, Shi Fu, Zheng Zhang, Xiaoning Wu, Xuzeng Fan, Dacheng Tao, and Xiaogang Wang. Kairos: A regret-aware native world-action model stack for physical ai, 2026.

[27] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, et al. DROID: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. [https://arxiv.org/abs/2403.12945](https://arxiv.org/abs/2403.12945).

[28] Moo Jin Kim et al. Openvla: An open-source vision-language-action model, 2024.

[29] Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial forcing: Implicit spatial representation alignment for vision-language-action model, 2026. [https://spatial-forcing.github.io/](https://spatial-forcing.github.io/). ICLR 2026.

23

<!-- page 24 of 26 -->

[30] Hao Li, Shuai Yang, Yilun Chen, Yang Tian, Xiaoda Yang, Xinyi Chen, Hanqing Wang, Tai Wang, Feng Zhao, Dahua Lin, et al. Cronusvla: Transferring latent motion across time for multi-frame prediction in manipulation. arXiv e-prints, pages arXiv–2506, 2025.

[31] Hao Li, Ziqin Wang, Zi-han Ding, Shuai Yang, Yilun Chen, Yang Tian, Xiaolin Hu, Tai Wang, Dahua Lin, Feng Zhao, et al. Robointer: A holistic intermediate representation suite towards robotic manipulation. arXiv preprint arXiv:2602.09973, 2026.

[32] Hao Li, Ganlong Zhao, Yufei Liu, Haotian Hou, Guoquan Ye, Tongyan Fang, Chunxiao Liu, Siyuan Huang, Jianbo Liu, Xiaogang Wang, and Hongsheng Li. Ace-ego-0: Unifying egocentric human and robotic data for vla pretraining, 2026.

[33] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu. Causal world modeling for robot control, 2026. [https://arxiv.org/abs/2601.21998](https://arxiv.org/abs/2601.21998).

[34] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024.

[35] Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, and Xi Victoria Lin. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models, 2024. [https://arxiv.org/abs/2411.04996](https://arxiv.org/abs/2411.04996).

[36] Lightwheel. EgoSuite-Open100K: A Fully Annotated Open Egocentric Human Dataset. [https://huggingface.co/collections/LightwheelAI/egosuite-open100k](https://huggingface.co/collections/LightwheelAI/egosuite-open100k), August 2026. Dataset.

[37] Yiran Ling, Qing Lian, Jinghang Li, Qing Jiang, Tianming Zhang, Xiaoke Jiang, Chuanxiu Liu, Jie Liu, and Lei Zhang. Guide, think, act: Interactive embodied reasoning in vision-language-action models, 2026.

[38] Yicheng Liu, Zibin Dong, Baijun Ye, Tianyuan Yuan, Tao Jiang, Anqi Yang, Shicheng Cao, Haonan Liu, Yue Sun, Zihan Guo, Xiao Liu, Dong Ke, Changxun Pan, Chenru Wu, Tailai Cheng, Xiaoshu Ren, Xinlei Zhang, Jianning Cui, Zijie Zhao, Haoyu Zhang, Kaiming Xu, Haodong Yang, Bowen Zhang, Jiahui Niu, Shaoting Zhu, Shiduo Zhang, and Hang Zhao. G0.5: One autoregressive stream for robot reasoning and action, 2026.

[39] Yushan Liu, Peibo Sun, Shoujie Li, Yifan Xie, Lingfeng Zhang, Xintao Chao, Shiyuan Dong, Fang Chen, Xiao-Ping Zhang, and Wenbo Ding. Oa-wam: Object-addressable world action model for robust robot manipulation. arXiv preprint arXiv:2605.06481, 2026.

[40] Yunfan Lou, Hewen Gao, Xiyu Zhu, Zhuoran Qiao, Xuan Han, Yifan Yang, Yifan Ye, Boxian Yao, and Zhibo Pang. Dynamicwam: Dual-path motion conditioning for world-action models in dynamic manipulation, 2026. [https://arxiv.org/abs/2608.00793](https://arxiv.org/abs/2608.00793).

[41] Hao Luo, Wanpeng Zhang, Yicheng Feng, Sipeng Zheng, Haiweng Xu, Chaoyi Xu, Ziheng Xi, Yuhui Fu, and Zongqing Lu. Being-h0.7: A latent world-action model from egocentric videos, 2026.

[42] Haoxiang Ma, Junhao Cai, Xiaoxu Xu, Hao Li, Yuyin Yang, Yang Tian, Jiafei Cao, Hongrui Zhu, Zherui Qiu, Zhaxizhuoma, Yuqiang Yang, Jiaqi Peng, Xueyuan Wei, Yangkun Zhu, Jiahao Jiang, Xing Gao, Hanqing Wang, Feng Yuan, Kailin Li, Xueyue Zhu, Tai Wang, Yan Ding, Jiangmiao Pang, Jia Zeng, Jingjing Zhang, Zhou Bowen, Yao Mu, Chunhua Shen, and Weinan Zhang. Internvla-a1.5: Unifying understanding, latent foresight, and action for compositional generalization, 2026.

[43] MotuBrain Team, Chendong Xiang, Fan Bao, Haitian Liu, Hengkai Tan, Hongzhe Bi, James Li, Jiabao Liu, Jingrui Pang, Kiro Jing, Louis Liu, Mengchen Cai, Rongxu Cui, Ruowen Zhao, Runqing Wang, Shuhe Huang, Yao Feng, Yinze Rong, Zeyuan Wang, and Jun Zhu. Motubrain: An advanced world action model for robot control, 2026.

[44] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. RoboCasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems (RSS), 2024. [https://arxiv.org/abs/2406.02523](https://arxiv.org/abs/2406.02523).

[45] NVIDIA. NVIDIA Isaac GR00T N1.7-3B. Hugging Face model card, 2026. [https://huggingface.co/nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B).

[46] NVIDIA, Jim Fan, et al. GR00T N1: An open foundation model for generalist humanoid robots, 2025. [https://research.nvidia.com/labs/gear/gr00t-n1/](https://research.nvidia.com/labs/gear/gr00t-n1/). NVIDIA GR00T technical report.

24

<!-- page 25 of 26 -->

[47] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[48] Delin Qu et al. Spatialvla: Exploring spatial representations for visual-language-action model, 2025.

[49] Hongyu Qu, Jianzhe Gao, Xiaobin Hu, Shaohuan Yang, Xinlei Yu, Rui Yan, Wenguan Wang, Xiangbo Shu, and Shuicheng Yan. Dual latent memory in vision-language-action models for robotic manipulation, 2026.

[50] Qwen Team. Qwen-robotmanip technical report: Alignment unlocks scale for robotic manipulation foundation models, 2026.

[51] RoboDojo Team. RoboDojo: Official simulation leaderboard, 2026. [https://robodojo-benchmark.com/LeaderBoard](https://robodojo-benchmark.com/LeaderBoard). Snapshot updated October 9, 2026; accessed October 13, 2026. Numerical data: https://robodojo-benchmark.com/data/leaderboard.generated.json.

[52] Ropedia. Xperience-10m: A large-scale egocentric multimodal dataset with structured 3d/4d annotations, 2026. Dataset.

[53] StarVLA Community. Starvla: A lego-like codebase for vision-language-action model developing, 2026.

[54] Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, Yaping Li, Ping Wang, Junhao Cai, Jia Zeng, Hao Dong, and Jiangmiao Pang. InternData-A1: Pioneering high-fidelity synthetic data for pre-training generalist policy. arXiv preprint arXiv:2511.16651, 2025. [https://arxiv.org/abs/2511.16651](https://arxiv.org/abs/2511.16651).

[55] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on robot learning, pages 1723–1736. PMLR, 2023.

[56] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[57] Qiuyue Wang, Mingsheng Li, Jian Guan, Jinhui Ye, Sicheng Xie, Yitao Liu, Junhao Chen, Zhixuan Liang, Jie Zhang, Xintong Hu, Xuhong Huang, Pei Lin, Junyang Lin, Dayiheng Liu, Shuai Bai, Jingren Zhou, Jiazhao Zhang, Haoqi Yuan, Gengze Zhou, Hang Yin, Ye Wang, Yiyang Huang, Zixing Lei, Wujian Peng, Delin Chen, Yingming Zheng, Jingyang Fan, Xianwei Zhuang, Xin Zhou, Haoyang Li, Anzhe Chen, Tong Zhang, Xuejing Liu, Yuchong Sun, Ruizhe Chen, Zhaohai Li, Chenxu Lü, Zhibo Yang, Tao Yu, and Xionghui Chen. Qwen-vla: Unifying vision-language-action modeling across tasks, environments, and robot embodiments, 2026.

[58] Yuran Wang, Siqiao Huang, Mingleyang Li, Chenhao Zhang, Jiaqi Liang, Weiyang Jin, Yue Chen, Xuemin Chi, Donghao Zhou, Qize Yu, Yu-Kai Wang, Yuhan Rui, Shenzhe Yao, Zhen Yuan, Zhenhao Shen, Kefei Zhu, Zijie Zhu, Ning Gao, Xiaowei Chi, Guanqi He, Shanghang Zhang, Hao Dong, Lin Shao, and Hang Zhao. OpenWAM: An open, modular exploration towards systematic world-action model pretraining, 2026. [https://arxiv.org/abs/2609.07398](https://arxiv.org/abs/2609.07398).

[59] Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, et al. RoboMIND: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877, 2024. [https://arxiv.org/abs/2412.13877](https://arxiv.org/abs/2412.13877).

[60] Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, et al. RoboCOIN: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025. [https://arxiv.org/abs/2511.17441](https://arxiv.org/abs/2511.17441).

[61] Xiaomi Robotics Team, Jun Guo, Piaopiao Jin, Jason Li, Peiyan Li, Yingyan Li, Futeng Liu, Wanli Peng, Optimus Qin, Yifei Su, et al. Xiaomi-robotics-1: Scaling vision-language-action models with over 100k hours of real-world trajectories. arXiv preprint arXiv:2607.15330, 2026. [https://arxiv.org/abs/2607.15330](https://arxiv.org/abs/2607.15330).

25

<!-- page 26 of 26 -->

[62] Xiaoxu Xu, Hao Li, Jinhui Ye, Yilun Chen, Jia Zeng, Xinyi Chen, Linning Xu, Dahua Lin, Weixin Li, and Jiangmiao Pang. Futurevla: Joint visuomotor prediction for vision-language-action model. arXiv preprint arXiv:2603.10712, 2026.

[63] An Yang et al. Qwen3 technical report, 2025. [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388).

[64] Jianwei Yang et al. Magma: A foundation model for multimodal ai agents, 2025.

[65] Lishan Yang, Wenxuan Song, Xi Wang, Pingyue Sheng, Zheng Fang, Ziyang Zhou, Junjie He, Haodong Yan, Jiayi Chen, Nan Sun, Qiao Sun, Pengwei Wang, Lingqiao Liu, Yan Wang, Yuxiang Gao, Feras Dayoub, and Haoang Li. 4D-WAM: Infusing spatiotemporal awareness into world action models through trajectory fields, 2026.

[66] Sizhe Yang, Juncheng Mu, Tianming Wei, Chenhao Lu, Xiaofan Li, Linning Xu, Zhengrong Xue, Zhecheng Yuan, Dahua Lin, Jiangmiao Pang, and Huazhe Xu. Memorywam: Efficient world action modeling with persistent memory. arXiv preprint arXiv:2606.20562, 2026.

[67] Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, Feng Xiong, Xing Wei, Zhiheng Ma, and Mu Xu. Abot-m0: Vla foundation model for robotic manipulation with action manifold learning, 2026.

[68] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi “Jim” Fan, and Joel Jang. World action models are zero-shot policies, 2026. [https://arxiv.org/abs/2602.15922](https://arxiv.org/abs/2602.15922).

[69] Chenghao Yin, Da Huang, Di Yang, Jichao Wang, Nanshu Zhao, Chen Xu, Wenjun Sun, Linjie Hou, Zhijun Li, Junhui Wu, Zhaobo Liu, Zhen Xiao, Sheng Zhang, Lei Bao, Rui Feng, Zhenquan Pang, Jiayu Li, Qian Wang, and Maoqing Yao. Genie Sim 3.0: A high-fidelity comprehensive simulation platform for humanoid robot. arXiv preprint arXiv:2601.02078, 2026. [https://arxiv.org/abs/2601.02078](https://arxiv.org/abs/2601.02078).

[70] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination?, 2026.

[71] He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, Yongming Rao, Dongsheng Zhang, Wanjia He, Ling Chen, Kai Huang, Jiahao Chen, Sichang Su, Xumin Yu, Ziyi Wang, Chengwei Zhu, Xiao Teng, Yuchun Guo, Yufeng Zhang, Yuandong Liu, Rui Wang, Zisheng Lu, Han Hu, and Zhengyou Zhang. Hy-embodied-0.5-vla: From vision-language-action models to a real-world robot learning stack, 2026.

[72] Qihang Zhang, Lin Li, Luyao Zhang, Shuai Yang, Yiming Luo, Shuaiting Li, Ruilin Wang, Junke Wang, Jiahao Shao, Gangwei Xu, Jiaming Zhou, Yishu Shen, Yudong Jin, Fangyi Xu, Shuailei Ma, Jiaqi Liao, Guanxing Lu, Zifan Shi, Yongkun Wen, Yujie Zhao, Weixuan Tang, Xinyang Wang, Chaojian Li, Jiapeng Zhu, Ka Leong Cheng, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu. Native video-action pretraining for generalizable robot control, 2026. [https://arxiv.org/abs/2607.08639](https://arxiv.org/abs/2607.08639).

[73] Tianle Zhang, Zhihao Yuan, Dafeng Chi, Peidong Liu, Dongwei Li, Kejun Hu, Likui Zhang, Junnan Nie, Ziming Wei, Zengjue Chen, Yili Tang, et al. Joyai-ra 0.1: A foundation model for robotic autonomy, 2026.

[74] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, and Xianyuan Zhan. X-vla: Softprompted transformer as scalable cross-embodiment vision-language-action model. In International Conference on Learning Representations, 2026.

[75] Jiaming Zhou, Qihang Zhang, Gangwei Xu, Cunxin Fan, Yujie Zhao, Ruilin Wang, Yiming Luo, Shuai Yang, Xing Zhu, Yujun Shen, Junwei Liang, and Yinghao Xu. Zero-WAM: In-context world-action modeling from human videos for open-ended task generalization, 2026. [https://arxiv.org/abs/2608.26103](https://arxiv.org/abs/2608.26103).

26