<!-- page 1 of 26 -->

![Image block](doc:57f42af/tier:basic/page:1/block:1)

DAMO TECH TO THE FUTURE 湖畔实验室

arXiv:2610.36791v1 [cs.RO] 27 Oct 2026

# RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions

**Haoyu Zhao**<sup>∗1,2,3</sup>, **Yinzhou Tang**<sup>∗1</sup>, **Zixuan Wang**<sup>∗1</sup>, **Minghao Zhu**<sup>1</sup>, **Dongchi Huang**<sup>1</sup>, **Xingyue Zhao**<sup>1</sup>, **Qin Zhao**<sup>2,3</sup>, **Kehan Li**<sup>1,4</sup>, **Siteng Huang**<sup>†1,4</sup>, **Xin Li**<sup>1,4</sup>, **Zhenyu Zeng**<sup>1,4</sup>, **Mingxiu Chen**<sup>1,4</sup>, **Deli Zhao**<sup>1,4</sup>, **Zhongyu Li**<sup>†2,3</sup>

<sup>1</sup>DAMO Academy, Alibaba Group, <sup>2</sup>Hong Kong Embodied AI Lab, <sup>3</sup>CUHK, <sup>4</sup>Hupan Lab, <sup>∗</sup>Core contributors,<sup>†</sup>Corresponding author

Learning controllable world dynamics from heterogeneous interaction videos is challenging because robot embodiments use incompatible action spaces, while human videos lack native control labels. We present RynnWorld-Latent, a cross-embodiment video world model that uses the 3D-grounded latent actions introduced by RynnLAM as a shared conditioning interface. This representation enables joint pretraining on human interaction videos, robot demonstrations, and simulation rollouts without requiring a common native action space. Built on a pretrained video transformer, RynnWorld-Latent combines global action tokens with an identity-initialized local pathway that aligns actions with future video latents. We pretrain on 23 datasets comprising approximately 553.5M frames and adapt the model to target robots through post-training. The framework supports two data-generation routes: digital teleoperation uses target robot commands to generate action-video demonstrations, while latent-action synthesis transfers source motion to an edited initial frame and pairs the resulting rollout with retargeted pseudo robot actions. Experiments show improved video prediction metrics on public benchmarks and two real-robot platforms, alongside qualitative action following when different action sequences share the same initial observation. Synthetic mid-training improves policy success rates on LIBERO and LIBERO-Plus, and both generation routes improve real-world policy success rates over real-demonstration-only training in the evaluated settings. These results support universal latent actions as a scalable interface for learning controllable visual dynamics across heterogeneous embodiments.

[https://alibaba-damo-academy.github.io/RynnWorld-Latent.github.io](https://alibaba-damo-academy.github.io/RynnWorld-Latent.github.io)

[https://github.com/alibaba-damo-academy/RynnWorld-Latent](https://github.com/alibaba-damo-academy/RynnWorld-Latent)

[https://huggingface.co/Alibaba-DAMO-Academy/RynnWorld-Latent](https://huggingface.co/Alibaba-DAMO-Academy/RynnWorld-Latent)

[https://www.modelscope.cn/models/DAMO\_Academy/RynnWorld-Latent](https://www.modelscope.cn/models/DAMO_Academy/RynnWorld-Latent)

**Date:** October 28, 2026

## Introduction

World models, which predict future states conditioned on actions, have long been theorized as a cornerstone of generalist robot intelligence (Sutton, 1991; LeCun et al., 2022; Richens et al., 2025). The recent progress of large-scale video generation has catalyzed a new class of video world models, in which future dynamics are represented directly in visual space (Zhao et al., 2026b,c; Agarwal et al., 2025; Ali et al., 2025; Zhao et al., 2026e,a, 2025). These models achieve impressive visual fidelity and provide a promising substrate for learning physical dynamics from large-scale video corpora. However, when applied to robotic manipulation, language-conditioned video generators face an interface mismatch: their primary control signal is a task-level description, while physical interaction is governed by continuous, frame-level causal interventions.

Language describes semantic intentions at a coarse temporal granularity. An instruction such as “pick up the cup” specifies a task outcome, but does not determine the precise trajectory by which the outcome is achieved. In contact-rich manipulation, small differences in timing, contact order, gripper state, or camera motion can lead to substantially different futures. Without fine-grained motion or action conditioning, a

<!-- page 2 of 26 -->

text-conditioned model must represent multiple plausible trajectories for the same instruction, making precise control over manipulation dynamics challenging. Moreover, natural language alone is not a convenient interface for specifying temporally precise interventions, such as closing the gripper earlier while maintaining the same preceding motion. These limitations motivate an explicit action-conditioning interface for controllable video world modeling.

A natural solution is to condition world models on robot actions (Gao et al., 2026). However, directly using raw robot actions introduces a severe alignment bottleneck. Different embodiments operate in heterogeneous kinematic spaces, with different joint counts, coordinate conventions, control frequencies, and hardware limits. As a result, a world model conditioned on raw actions is difficult to scale across platforms. At the same time, large-scale human videos contain abundant interaction dynamics without robot-specific control labels, while simulation actions need not match a target robot’s control space. What is needed is not a stronger action encoder for a particular robot, but a shared action interface that represents interaction-related visual change while abstracting away platform-specific control conventions.

We argue that latent actions provide such an interface. Instead of representing actions as robot commands, latent actions describe what changed between observations. Distilled from video in a self-supervised manner, continuous latent actions provide a shared description of visual transitions without requiring identical hardware control interfaces. Human hands, robot grippers, and simulated agents may differ drastically in their lowlevel control signals, yet their visual transitions can share motion and interaction patterns. By encoding heterogeneous videos into a shared latent-action representation, a single world model can be pretrained on human interactions, robot demonstrations, and simulation rollouts under a unified action-conditioning objective, without requiring their native control spaces to be aligned.

In this work, we introduce **RynnWorld-Latent**, a cross-embodiment video world model built upon **RynnLAM** (Zhao et al., 2026d), the 3D-grounded latent-action representation introduced in RynnVLA-Latent (Zhao et al., 2026d). While RynnVLA-Latent (Zhao et al., 2026d) uses these descriptors as supervision targets for policy pretraining, our work investigates their complementary role as conditioning signals for learning controllable visual dynamics. RynnWorld-Latent predicts future video from an initial observation and a sequence of continuous latent actions, without requiring text conditioning or a common native robot action space. Technically, we combine global action tokens with an identity-initialized, frame-aligned local conditioning pathway within a pretrained video diffusion transformer. The local pathway exploits the temporal correspondence between latent-action extraction and video-latent compression, allowing each action descriptor to condition its corresponding future video latent.

A central idea of this work is that RynnWorld-Latent should not be viewed only as a future predictor, but as a **data-generation engine** for scalable robot learning. This leads to two complementary generation modes. First, after target-robot post-training, RynnWorld-Latent can be conditioned on target robot actions and support interactive digital teleoperation (Zhao et al., 2026c): operator control signals are mapped to target actions and drive video generation, producing paired action-video data without requiring additional physical robot execution. Second, the latent-action-conditioned model enables data synthesis from existing videos. Given a source video, we extract its latent-action sequence, edit the first frame to replace the embodiment or environment, and generate a new rollout conditioned on the edited first frame and the original latent actions. First-frame editing specifies the new visual embodiment or environment, while the latent-action sequence guides the interaction dynamics. The resulting rollout has motion close to the source video, but appears under a new embodiment or environment. These generated rollouts can be paired with retargeted pseudo robot actions for downstream VLA training.

We evaluate RynnWorld-Latent from three complementary perspectives. First, we assess action-conditioned video prediction on public world-model benchmarks and held-out demonstrations from two real-robot platforms, measuring prediction fidelity after target-robot adaptation. Second, we investigate latent-action controllability through qualitative comparisons that vary the supplied latent-action sequences while holding the initial observation fixed, and illustrate cross-embodiment motion transfer through first-frame editing. Third, we examine the utility of world-model-generated data for downstream policy learning. Synthetic mid-training improves success rates on LIBERO and LIBERO-Plus, while both latent-action-driven synthesis and digital teleoperation improve observed closed-loop policy success on the evaluated real-world manipulation tasks. These evaluations jointly characterize prediction fidelity, qualitative action following, and downstream training

<!-- page 3 of 26 -->

utility.

Our main contributions are as follows:

• We present RynnWorld-Latent, a text-free, latent-action-conditioned video world model, which uses frozen RynnLAM descriptors as a shared conditioning interface for learning visual dynamics from human interaction videos, heterogeneous robot demonstrations, and simulation rollouts without aligning their native control spaces.

• We design an action-conditioning mechanism that combines global action tokens with locally identityinitialized, frame-aligned injection, associating transition descriptors with the corresponding future video latents.

• We enable latent-action-guided generation under edited robot embodiments and environments, and adapt the world model to native robot-action conditioning. The resulting latent-action synthesis and digital-teleoperation pipelines generate data that improves downstream policy learning in the evaluated settings.

## 2 Related Work

## 2.1 World Models

World models predict how environments change under actions and support planning and policy learning through simulated experience (Finn et al., 2016; Ebert et al., 2018; Hafner et al., 2019, 2020). Advances in video generation have enabled large-scale pretraining of visual dynamics models and their adaptation to interactive simulation (Agarwal et al., 2025; Yang et al., 2024; Wu et al., 2024b; Rigter et al., 2024; Huang et al., 2025). Language-conditioned video and image generation have also been used to produce visual plans and synthetic demonstrations for robot control (Du et al., 2023; Zhou et al., 2024; Black et al., 2024b; Jang et al., 2025). However, task-level instructions leave the timing and trajectory of individual movements largely unspecified. To simulate fine-grained manipulation, action-conditioned world models incorporate robot controls into future prediction (Zhu et al., 2025b; Guo et al., 2026; Wang et al., 2025b). Such predictions support policy evaluation (Li et al., 2025a; Quevedo et al., 2026; Wang et al., 2026c) and planning (Ebert et al., 2018; Zhou et al., 2025; Assran et al., 2025; Kim et al., 2026), while related approaches combine future prediction with action learning to improve robot policies (Wu et al., 2024a; Cheang et al., 2024; Zhu et al., 2025a; Ye et al., 2026; Sun et al., 2026).

Extending action-conditioned modeling across embodiments requires handling differences in action dimensions, coordinate conventions, and control semantics. Existing approaches address this through heterogeneous action modeling (Wang et al., 2025b) or geometric motion representations shared by humans and robots (Wu and Gao, 2026; Zhao et al., 2026c). For pretraining on diverse robot and human videos, a further challenge is that many data sources lack corresponding control labels. More closely related to our setting, latent-action pretraining infers conditioning variables from observed transitions, allowing videos without native action labels to contribute to world-model learning (Gao et al., 2025; Garrido et al., 2026; Gao et al., 2026; Liu and Shorinwa, 2026). Building on this direction, RynnWorld-Latent uses continuous latent actions as a shared conditioning interface for cross-embodiment world-model pretraining, without requiring direct alignment of embodiment-specific control spaces.

## 2.2 Latent Action Learning

Learning across embodiments requires representations that connect different forms of observation and control. Prior work studies shared policies trained on heterogeneous robot data (O’Neill et al., 2024; Octo Model Team, 2024; Kim et al., 2024; Wang et al., 2024) and action-space alignment using recorded controls (Zheng et al., 2025; Zhang et al., 2025c). For videos without corresponding robot actions, transferable skills and geometric motion cues provide intermediate representations for learning (Xu et al., 2023; Kim et al., 2025; Wen et al., 2024; Bharadhwaj et al., 2024; Collins et al., 2025). Latent action learning instead infers compact action variables from transitions between observations (Rybkin et al., 2019; Schmidt and Jiang, 2024). These variables have been used as supervision for policy pretraining and transfer (Ye et al., 2025; Chen et al., 2025b;

<!-- page 4 of 26 -->

![Image block](doc:57f42af/tier:basic/page:4/block:1)

![Image block](doc:57f42af/tier:basic/page:4/block:2)

![Image block](doc:57f42af/tier:basic/page:4/block:3)

![Chart block](doc:57f42af/tier:basic/page:4/block:4)

Figure 1 Pretraining corpus overview. The corpus contains 23 datasets, approximately 6,190 hours, and 2.57M episodes. The figure summarizes the distribution of human, real-robot, and simulation video, dataset contributions by duration, and object and action vocabulary.

Bu et al., 2025b; Chen et al., 2025a; Li et al., 2025b; Tharwat et al., 2025), as well as conditioning inputs for predicting future states (Menapace et al., 2021; Bruce et al., 2024; Alles et al., 2025; Bi et al., 2025; Garrido et al., 2026).

Both discrete codes (Bruce et al., 2024; Ye et al., 2025; Chen et al., 2025b) and continuous representations (Gao et al., 2025; Liang et al., 2026; Yang et al., 2026a; Garrido et al., 2026; Gao et al., 2026) have been explored. Recent work further studies alignment with available action supervision (Zhang et al., 2026a; Luo et al., 2026), the separation of motion from appearance (Yang et al., 2026a; Xu et al., 2026; Zhang et al., 2026b), and geometric constraints on the learned action space (Yang et al., 2026b). These distinctions matter because visual changes can include camera motion, background variation, and other effects unrelated to the agent’s control (Nikulin et al., 2025; Zhang et al., 2025a; Garrido et al., 2026). Building on these studies, we use a pretrained and frozen encoder RynnLAM (Zhao et al., 2026d) to extract 3D-grounded continuous latent-action descriptors from heterogeneous visual transitions. Unlike methods that require native control annotations or explicit motion retargeting during world-model pretraining, our approach uses these descriptors as a shared conditioning interface for human, robot, and simulation videos. This enables joint learning of visual dynamics without requiring the original action parameterizations to be aligned across embodiments.

## 3 RynnWorld-Latent

## 3.1 Overview

RynnWorld-Latent is a cross-embodiment latent-action-conditioned video world model that can further serve as a data-generation engine for robot learning. It predicts future video latents from an initial latent frame and a sequence of continuous latent actions, without relying on text conditioning. The pretraining corpus is summarized in Tab. 1 and Fig. 1. The framework consists of three stages: (i) cross-embodiment pretraining, where human egocentric videos, robot teleoperation data, and simulation rollouts are relabeled into a shared latent-action space; (ii) target-robot post-training, which adapts RynnWorld-Latent to a target embodiment and introduces a robot-action conditioning interface; and (iii) data generation, where the post-

<!-- page 5 of 26 -->

Table 1 Dataset-level statistics of the pretraining data. FPS denotes the frame rate declared in the training data metadata; comma-separated values indicate subsets with different frame rates. Hours and frame counts are rounded to two decimal places; episode counts are integers. Frame rates describe metadata rather than a uniform sampling rate across all sources.

| Dataset | FPS | Hours | Episodes | Frames (M) |
| --- | --- | --- | --- | --- |
| Robot data |  |  |  |  |
| DROID (Khazatsky et al., 2024) | 15 | 518.55 | 82,585 | 28.00 |
| RoboMIND (Wu et al., 2024c) | 20 | 200.21 | 42,919 | 14.42 |
| RoboMIND2.0 (Hou et al., 2025) | 15 | 569.26 | 71,226 | 30.74 |
| Galaxea (Jiang et al., 2025) | 15 | 320.49 | 17,013 | 17.36 |
| RoVid-X (Deng et al., 2026) | 30 | 317.93 | 390,614 | 34.34 |
| AGIBOT-Beta (Bu et al., 2025a) | 30 | 310.89 | 128,955 | 33.58 |
| AgiBotWorld2026 (Bu et al., 2025a) | 30 | 295.00 | 20,704 | 31.86 |
| MolmoAct2 (Fang et al., 2026) | 30 | 308.62 | 32,182 | 33.33 |
| Table30 (Yakefu et al., 2025) | 30 | 270.44 | 25,585 | 29.21 |
| Table30v2 (Yakefu et al., 2025) | 30 | 290.92 | 32,875 | 31.42 |
| RoboCOIN (Wu et al., 2025b) | 30 | 262.95 | 92,422 | 28.67 |
| BridgeV2 (Walke et al., 2023) | 5 | 135.25 | 13,454 | 2.43 |
| RDT-1B (Liu et al., 2025) | 30 | 27.69 | 6,076 | 2.99 |
| OpenX-Embodiment (O'Neill et al., 2024) | 30 | 17.30 | 3,550 | 0.82 |
| Subtotal | - | 3,845.50 | 960,160 | 319.16 |
| Robot data (simulation) |  |  |  |  |
| RoboCasa (Nasiriany et al., 2024) | 20 | 478.90 | 263,919 | 34.48 |
| BEHAVIOR-1K (Li et al., 2023) | 30 | 324.53 | 19,971 | 35.05 |
| InternData-A1 (Tian et al., 2026) | 30 | 322.39 | 215,443 | 34.82 |
| Subtotal | - | 1,125.82 | 499,333 | 104.35 |
| Human interaction video |  |  |  |  |
| HowTo100M (Miech et al., 2019) | 30 | 311.77 | 423,975 | 32.04 |
| EgoVid (Wang et al., 2026a) | 30 | 299.00 | 397,849 | 32.29 |
| EgoVerse (Punamiya et al., 2026) | 30 | 291.57 | 50,212 | 31.49 |
| EgoDex (Hoque et al., 2025) | 30 | 265.11 | 192,175 | 28.63 |
| SthSthV2 (Goyal et al., 2017) | 12 | 29.71 | 26,747 | 1.28 |
| EPIC_KITCHENS (Damen et al., 2020) | 50 | 21.51 | 16,905 | 4.26 |
| Subtotal | - | 1,218.67 | 1,107,863 | 129.99 |
| Total (23 datasets) | - | 6,189.99 | 2,567,356 | 553.50 |

trained model supports interactive digital teleoperation when conditioned on target robot actions, and enables latent-action-driven data synthesis when conditioned on transferred latent actions.

We index video latent frames from 0 to $L { - } 1$ , with the initial latent frame $x _ { 0 } = c$ held fixed. Formally, given this conditioning frame and latent actions $A = \{ \mathbf { z } _ { j } \} _ { j = 1 } ^ { J }$ , RynnWorld-Latent learns

$$
p _ { \theta } ( x _ { 1 : L - 1 } \mid c , \: A ) \: .\tag{1}
$$

The latent-action interface is used for source-motion transfer and data synthesis; target-robot adaptation introduces robot-action conditioning for prediction and digital teleoperation.

## 3.2 Latent-Action Interface

Raw robot actions are embodiment-bound: joint counts, control modes, and coordinate conventions differ across platforms. We instead use continuous latent actions as a unified interface. For $j \in \{ 1 , \ldots , J \}$ , given frames $v _ { 4 ( j - 1 ) }$ and $v _ { 4 j } ,$ the frozen RynnLAM (Zhao et al., 2026d) produces

$$
\mathbf { z } _ { j } = f _ { \phi } ( v _ { 4 ( j - 1 ) } , v _ { 4 j } ) = \left[ \mathbf { z } _ { j } ^ { m } ; \mathbf { z } _ { j } ^ { a } ; \mathbf { z } _ { j } ^ { c } \right] \in \mathbb { R } ^ { 6 0 8 } .\tag{2}
$$

<!-- page 6 of 26 -->

Here $\mathbf { z } _ { j } ^ { m } \in \mathbb { R } ^ { 5 1 2 }$ contains motion-token features, $\mathbf { z } _ { j } ^ { a } \in \mathbb { R } ^ { 6 4 }$ is the global action latent, and $\mathbf { z } _ { j } ^ { c } \in \mathbb { R } ^ { 3 2 }$ is the camera component. We group $[ \mathbf { z } _ { j } ^ { m } ; \mathbf { z } _ { j } ^ { a } ]$ into a 576-dimensional manipulation-related descriptor. Each action dimension is quantile-normalized to $\left[ \stackrel { \circ } { - 1 } , 1 \right]$ using the 1st and 99th percentiles estimated from an 8,000-episode sample, and then clipped.

A full training chunk contains $T { = } 8 1$ frames. A causal video VAE with $4   \times   1 6   \times   1 6$ compression (Wang et al., 2025a) maps it to $L { = } 2 1$ latent frames. Since the LAM stride equals the VAE temporal compression factor, each latent action is index-aligned with one latent-frame transition, giving $J { = } L { - } 1 { = } 2 0$ . This correspondence motivates our frame-aligned conditioning design in Sec. 3.3, allowing each action descriptor to condition its associated future video latent. For shorter episodes, we retain variable-length chunks with $T \in \{ 3 3 , 3 7 , \ldots , 8 1 \}$ corresponding to $J \in [ 8 , 2 0 ]$

## 3.3 Latent-Action-Conditioned World Model

RynnWorld-Latent builds on the Cosmos3 mixture-of-transformers video foundation model (Agarwal et al., 2026), which denoises video latents with rectified flow (Liu et al., 2022). The backbone does not use adaptive layer normalization, so action conditioning must enter through attention tokens or explicit feature modulation.

We encode each latent action with two linear projections and a shared residual MLP:

$$
\begin{array} { r } { h _ { j } = W ^ { \mathrm { a c t } } \left[ \mathbf { z } _ { j } ^ { m } ; \mathbf { z } _ { j } ^ { a } \right] + W ^ { \mathrm { c a m } }   \mathbf { z } _ { j } ^ { c } , \qquad \tau _ { j } = h _ { j } + \mathrm { M L P } ( h _ { j } ) . } \end{array}\tag{3}
$$

The MLP output projection is zero-initialized. All corpora use a single shared domain slot, keeping the encoder embodiment-agnostic. The action tokens $\tau _ { j }$ participate in joint self-attention, providing a global conditioning pathway.

Before the first transformer block, we additionally apply a local, frame-aligned pathway to the vision tokens of future latent frame $j ;$

$$
q \leftarrow \big ( q \odot ( 1 + \gamma _ { j } ) + \beta _ { j } \big ) + g \odot \tau _ { j } , \qquad q \in \mathcal { V } _ { j } .\tag{4}
$$

Here $\mathcal { V } _ { j }$ denotes the set of vision tokens belonging to future latent frame $j ,$ so the first action conditions frame 1 and the last action conditions frame L−1. A single zero-initialized linear head maps $\tau _ { j }$ to the modulation parameters $( \gamma _ { j } , \beta _ { j } )$ , and $g$ is a zero-initialized per-channel gate. The conditioning frame c is not modulated, and the local pathway is applied only under exact index alignment. Thus the two added vision-facing pathways are identity maps at initialization. This follows the zero-init convention of control-augmented diffusion models and makes the local action-dependent correction initially zero. Because the re-initialized action tokens also participate in joint attention, local identity initialization does not imply that the complete conditioned model is identical to the pretrained generator. The action-token stream itself is re-initialized because the pretrained action encoder is shape-incompatible with the 608-dimensional latent action space; re-initialized action-interface modules are trained with a 5× learning-rate multiplier.

## 3.4 Training Objective

When present, actions are supplied as clean conditioning, while only future video latents are denoised. Let $X ^ { \mathrm { d a t a } } = ( x _ { 1 } , \ldots , x _ { L - 1 } )$ denote the clean future latent sequence and ϵ Gaussian noise of the same shape. We follow the released rectified-flow parameterization, where t=1 corresponds to pure noise and t=0 corresponds to clean data:

$$
X _ { t } = t   \epsilon + \left( 1 - t \right) X ^ { \mathrm { d a t a } } , \qquad v _ { t } = \epsilon - X ^ { \mathrm { d a t a } } .\tag{5}
$$

The training objective is

$$
\mathcal { L } = \mathbb { E } _ { X ^ { \operatorname { d a t a } } , c , A , \epsilon , t } \left\| u _ { \theta } ( X _ { t } , t , c , A ) - v _ { t } \right\| ^ { 2 } .\tag{6}
$$

Only the first latent frame c is kept clean as conditioning. This training objective is consistent with inference, where latent actions are also clean and only the video stream is denoised. Training timesteps are sampled from a logit-normal distribution with resolution shift 5 for the 480p tier.

To support classifier-free guidance, we randomly drop action conditioning during training. With probability 0.1, we drop the entire action sequence. Independently, with probability 0.1 we drop the manipulation-related 576-dimensional slice, and with probability 0.1 we drop the 32-dimensional camera slice. These branches

<!-- page 7 of 26 -->

support joint and slice-specific guidance, which we apply at inference with a guidance scale of 1.15 (Sec. 4). During generation, denoising proceeds from t=1 toward t=0, while the initial frame and supplied action inputs remain fixed.

## 3.5 World-Model Data Engine

![Image block](doc:57f42af/tier:basic/page:7/block:3)

![Image block](doc:57f42af/tier:basic/page:7/block:4)

Figure 2 Overview of the world-model data engine. Top: Scalable data editing. RynnLAM extracts latent actions from a source human video, while its first frame is edited to depict the target robot. Conditioned on the edited frame and the original latent actions, RynnWorld-Latent generates robot video rollouts. The generated observations are paired with pseudo-actions obtained through motion retargeting and inverse kinematics. Bottom: Digital teleoperation. Human control signals are retargeted to robot actions, which condition RynnWorld-Latent together with a reference scene. Generated frames provide visual feedback to the operator, while the recorded actions and generated observations form synthetic demonstrations without physical robot execution.

To generate trajectories for policy learning, we combine the world model with interfaces for capturing interaction intent and producing action-video episodes. In our design, RynnWorld-Latent provides action-conditioned visual dynamics, while the surrounding pipeline provides target-robot adaptation, action retargeting, temporal alignment, and filtering. Compared with asset-based simulation, this generative pipeline can reduce the need for explicit 3D assets and manual system identification: scenes are specified implicitly by a conditioning image, visual outputs are produced in a learned video distribution, and interaction priors are drawn from large-scale video pretraining. We instantiate two complementary data-generation pipelines: interactive digital teleoperation and latent-action data synthesis.

**Target-robot adaptation.** Before using RynnWorld-Latent for data generation, we post-train it on target robot demonstrations. Our real-robot corpus is shared with RynnVLA-Latent (Zhao et al., 2026d) and contains 3,850 collected episodes across six tasks (Appendix B). For each task, 5% of the episodes are randomly held out for testing and the remaining 95% form the target-robot post-training split; the 3,850-episode count refers to the full collected corpus. The target data is processed with the same chunking protocol and latent-action labeling pipeline. This stage adapts the model to the target embodiment’s visual appearance, camera configuration, and interaction distribution. Robot-action conditioning is introduced through a separate target-specific encoder. When target robot actions are available, we introduce a lightweight action encoder $E _ { \psi }$ that maps a target robot action $u _ { j }$ into an action token:

$$
\tau _ { j } ^ { u } = E _ { \psi } ( u _ { j } ) .\tag{7}
$$

<!-- page 8 of 26 -->

![Image block](doc:57f42af/tier:basic/page:8/block:1)

Figure 3 RynnEdit. Two routes construct a robot-centric reference frame: explicit hand removal, retargeting, and robot rendering; or direct replacement of human hands with robotic counterparts using an image-editing model. The resulting first frame conditions subsequent world-model generation.

The token $\tau _ { j } ^ { u }$ is injected through the same conditioning pathway as the latent-action token $\tau _ { j }$ . We train this interface with the same rectified-flow video objective, replacing latent-action tokens with robot-action tokens. Here $u _ { j }$ denotes the robot-action input synchronized with the interval represented by latent action $\mathbf { z } _ { j } ;$ its parameterization depends on the target robot. The encoder provides an embodiment-specific control interface without requiring commands to share RynnLAM’s descriptor dimensions. After this adaptation, RynnWorld-Latent can be used not only as a latent-action-conditioned world model, but also as an action-conditioned world model for interactive generation.

**RynnEdit: Robot-centric reference-frame construction pipeline.** The data engine requires a reference scene frame $I _ { \mathrm { r e f } }$ compatible with the target robot’s appearance and workspace. However, collecting real-world robot data in new environments is costly. We therefore synthesize scenes featuring the target robot from large-scale egocentric human hand videos. Given an egocentric interaction video, we construct a robot-centric initial observation by replacing human hands with the target robot while preserving the background and initial interaction configuration. The resulting first frame conditions the data engine; the source latentaction sequence supplies the subsequent motion. Our pipeline shown in Fig. 3 consists of egocentric data preprocessing followed by two alternative routes for generating robot-centric initial scenes: robot inpainting like Ego2Robot (Wang et al., 2026b) and image editing using an end-to-end model. During preprocessing, we verify the integrity of the input videos and motion data, standardize their formats, and extract egocentric frames and human hand poses. We temporally align the video and motion sequences using zero-order hold or interpolation. We also use SAM3 (Carion et al., 2026), prompted with human hand keypoints and text descriptions, to segment the hands and combine the resulting masks. In the robot inpainting route, we perform video inpainting within the hand masks to reconstruct the background after removing the human hands. We retarget the extracted human hand motion to the robotic hand and estimate the robot base pose and joint configurations through motion post-processing and inverse kinematics with base-pose search in MuJoCo (Todorov et al., 2012). We then render the robot in the estimated poses and composite the rendered images onto the inpainted background to obtain robotic manipulation observations corresponding to the original human demonstrations. In the image editing route, we instead provide the original egocentric image and its hand mask to an end-to-end image-editing model, like Qwen-Image-Edit (Wu et al., 2025a), which replaces the human hands with robotic counterparts to generate an initial scene. The two routes thus produce synthetic robot observations through explicit retargeting and rendering or direct image editing, respectively. Finally, we apply quality gates to filter the generated observations and retain accepted samples as initial scenes for subsequent world-model-based data generation.

**Interactive digital teleoperation.** The teleoperation pipeline converts operator motion into target robot actions and uses RynnWorld-Latent as the visual rollout engine (Fig. 2). Given an initial conditioning frame $I _ { \mathrm { r e f } }$ and a stream of operator control signals, we retarget the operator motion to the target robot action space, producing a time-stamped action sequence. In chunk-based generation, the robot commands are synchronized with the video timeline and encoded through $E _ { \psi }$ . RynnWorld-Latent conditions on the current frame and the corresponding sequence of robot-action tokens to generate the future video latents. The last generated frame can then be used as the conditioning frame for the next chunk, enabling long-horizon interactive rollouts. Since

<!-- page 9 of 26 -->

the system records both the retargeted actions and the generated frames, interactive digital teleoperation directly produces paired action-video episodes without requiring physical robot execution.

**Latent-action data synthesis.** The second pipeline synthesizes new robot data from existing videos using latent-action transfer. Given a source video $V ^ { s }$ , we first extract its latent-action sequence $A ^ { s }$ using the frozen LAM. We then edit the first frame $I _ { 0 } ^ { s }$ to obtain a new reference frame $\tilde { I } _ { 0 }$ that replaces the embodiment or environment while preserving the scene layout, object arrangement, and camera viewpoint as much as possible. The edited frame is encoded as the conditioning latent $\tilde { c } ,$ and RynnWorld-Latent generates a rollout conditioned on this edited frame and the original latent-action sequence:

$$
\tilde { V } \sim p _ { \theta } ( \cdot \mid \tilde { c } ,   A ^ { s } )   .\tag{8}
$$

In this pipeline, first-frame editing specifies the target visual embodiment or environment, while the transferred latent-action sequence guides the generated interaction dynamics. The edited frame implicitly specifies the initial scene configuration without requiring explicit numerical object or robot states. The resulting rollout is encouraged to exhibit motion patterns related to the source interaction under the edited visual conditions. This transfer is approximate, since changes in robot geometry, workspace constraints, contact configurations, or viewpoint may affect whether the source motion remains physically feasible. The world model itself remains text-free: any prompts used by the external image editor only construct the reference frame $\tilde { I } _ { 0 }$ and do not condition the subsequent video generation.

**Action relabeling, synchronization, and filtering.** For interactive digital teleoperation, target robot actions can be logged directly. For latent-action data synthesis, robot actions can be obtained by relabeling the source motion. When the source video already has robot actions compatible with the target embodiment, those actions can be reused after temporal alignment. Otherwise, we derive pseudo robot actions by estimating the relevant source motion, retargeting it to the target robot workspace, and applying inverse kinematics; implementation details are provided in Appendix C. The pseudo actions are temporally aligned with the generated video in frame-index space. We emphasize that temporal synchronization does not by itself establish physical consistency: the actions come from retargeting or inverse kinematics, while the observations are produced by an independently trained generator. We therefore treat the derived actions as pseudo-labels and filter episodes based on joint limits, motion smoothness, visual quality, temporal consistency, and plausibility of object interactions. The resulting synthetic episodes can be combined with real robot data for downstream VLA training.

## 4 Experiments

## 4.1 Experimental Setup

**World-model implementation.** RynnWorld-Latent is built on Cosmos3-Edge (Agarwal et al., 2026), a mixtureof-transformers video foundation model. We train for 20K optimizer steps with a global batch size of 512, using AdamW with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 )$ , weight decay 0.05, gradient-norm clipping 1.0, a peak learning rate of $2 . 2 6 \times 1 0 ^ { - 4 }$ , 500 warmup steps, and linear decay to 10% of the peak. The newly introduced action-facing modules are re-initialized for the 608-dimensional latent-action space and use a 5× learning-rate multiplier. Training uses bf16 with FP32 master weights and optimizer states, and a power-function EMA with $s = 0 . 1$ Parallelism uses hybrid-sharded FSDP with shard degree 8 per node and replication degree 4 across nodes. All videos are trained in 480p aspect-ratio buckets (832 × 480, 640 × 640, 736 × 544, 544 × 736, and 480 × 832).

**Policy backbones.** For policy experiments, we use $\pi _ { 0 }$ (Black et al., 2024a) and $\pi _ { 0 . 5 }$ (Intelligence et al., 2025) as base VLA models. For all settings, we use the same downstream training protocol unless otherwise specified. For synthetic pretraining experiments, we use the term **mid-training** to denote continued pretraining on synthetic data, and **post-training** to denote subsequent fine-tuning on target task or robot data.

**Real-robot system setup.** Our real-world experiments are conducted on two distinct robotic platforms to evaluate world-model adaptation and downstream policy learning, as shown in Fig. 4. For downstream policy deployment on both platforms, the inference frequency is set at 50 $\mathrm { H z } ,$ and control commands are sent with a delay kept between 18 and 30 milliseconds. The low-level interface operates at a frequency of 500 Hz, ensuring smooth real-time control. Communication between the control policy and the low-level interface uses LCM

<!-- page 10 of 26 -->

![Image block](doc:57f42af/tier:basic/page:10/block:1)

Figure 4 Standardized Experimental Platforms. Two dual-arm embodiments used for real-world evaluation: the stationary Marvin-WUJI (dual 7-DOF arms with 20-DOF dexterous hands) and the mobile Astribot-S1 (dual 7-DOF arms with parallel grippers on a wheeled base).

(Lightweight Communications and Marshalling). These frequencies describe physical-robot policy deployment, rather than world-model video-generation throughput.

• **Marvin-WUJI.** Dual-Arm Dexterous Hand System. The first embodiment is built on the TIANJI M6 arms and WUJI dexterous hands. The hardware setup consists of dual Tianji 7-DOF robotic arms and dual Wuji 20-DOF dexterous hands, yielding a total of 54 degrees of freedom.

• **Astribot-S1.** Astribot-S1 Mobile Manipulator. The second embodiment utilizes the Astribot S1, an AI robot platform featuring a wheeled mobile base and dual 7-DOF robotic arms equipped with parallel grippers. Each arm provides 7 DOFs with a rated payload of 5 kg (horizontal extension) and an end-effector repeatability of ±0.1 mm. With a height of 170 cm and an arm span of 194 cm, Astribot-S1 provides a highly dynamic, mobile bimanual manipulation baseline compared to the stationary dexterous setup.

**Real-world policy tasks.** We evaluate downstream policy learning on four tasks that span varying levels of bimanual coordination, contact richness, and precision. These tasks are distributed across two distinct dual-arm robotic platforms: **Marvin-WUJI**, a stationary dual-arm robot equipped with dexterous hands, and **Astribot-S1**, a wheeled mobile dual-arm robot equipped with parallel grippers. The four tasks are defined as follows:

## Tasks performed on Marvin-WUJI (Dual-Arm Dexterous Hands):

• **(1) Plate-Placement:** The robot reaches toward a shelf, uses its dexterous hands to securely grasp a plate, and places it beside the oven on the tabletop, requiring precise spatial alignment and stable grasping of thin, fragile objects.

• **(2) Plate-Wiping:** A highly dexterous bimanual coordination task in which one hand grasps and stabilizes a plate while the other hand holds a cloth to wipe the plate’s surface before returning the plate to the shelf, requiring coordinated bimanual manipulation.

## Tasks performed on Astribot-S1 (Wheeled Mobile Dual-Arm Grippers):

• **(3) Steak-Serving:** The robot approaches the dining area, uses its dual grippers to stably lift a plate containing a cooked steak, and accurately serves it onto the designated spot on the dining table, integrating mobile navigation with precise placement.

• **(4) Plate-Retrieval:** Following the meal, the robot identifies and grasps used, dirty plates left on the dining table, collects the associated cutlery, and returns the items to their designated positions.

**Evaluation metrics.** For world-model video prediction quality, we report PSNR, SSIM, and LPIPS, where higher PSNR/SSIM and lower LPIPS are better. Following DreamDojo (Gao et al., 2026), we evaluate open-loop rollouts conditioned on an initial frame and native robot-action sequences through the target-robot action encoder, autoregressively replacing the conditioning frame with the last predicted frame. At inference we apply

<!-- page 11 of 26 -->

Table 2 Results on the DreamDojo GR-1 evaluation sets (Gao et $\mathrm { a l . }$ , 2026). Baseline values are literature-reported measurements after robot post-training.

| Model | In PSNR↑ | -lab Ev SSIM↑ | al LPIPS↓ | Eg PSNR↑ | oDex E SSIM↑ | val LPIPS↓ | Dream PSNR↑ | Dojo-H SSIM↑ | V Eval LPIPS↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cosmos-Predict2.5 (Ali et al., 2025) | 20.58 | 0.77 | 0.22 | 19.95 | 0.79 | 0.22 | 18.27 | 0.75 | 0.24 |
| DreamDojo-2B (Gao et al., 2026) | 21.11 | 0.77 | 0.22 | 20.41 | 0.78 | 0.23 | 18.81 | 0.75 | 0.24 |
| DreamDojo-14B (Gao et al., 2026) | 21.41 | 0.79 | 0.21 | 20.53 | 0.79 | 0.21 | 18.92 | 0.75 | 0.23 |
| RynnWorld-Latent | 21.42 | 0.82 | 0.20 | 20.89 | 0.81 | 0.19 | 19.22 | 0.79 | 0.21 |

Table 3 Results and ablation study on our real-robot evaluation sets. $\mathrm { ^ { \circ } N / A ^ { \circ } }$ denotes baseline implementations without support for Astribot-S1 mobile-base motion. PSNR, SSIM, and LPIPS are interpreted jointly as measures of video prediction fidelity.

<table><tr><td rowspan="2">Method</td><td colspan="3">Marvin-WUJI Eval</td><td colspan="3">Astribot-S1 Eval</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Cosmos3-Edge (Agarwal et al., 2026)</td><td>20.03</td><td>0.78</td><td>0.31</td><td>13.99</td><td>0.42</td><td>0.54</td></tr><tr><td>Cosmos3-Nano (Agarwal et al., 2026)</td><td>20.09</td><td>0.80</td><td>0.26</td><td>14.22</td><td>0.53</td><td>0.45</td></tr><tr><td>RynnWorld-Teleop (Zhao et al., 2026c)</td><td>20.89</td><td>0.76</td><td>0.19</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>RynnWorld-Latent</td><td>21.86</td><td>0.80</td><td>0.12</td><td>15.96</td><td>0.60</td><td>0.34</td></tr><tr><td colspan="7">Ablation Study</td></tr><tr><td>RynnWorld-Latent text-conditioned</td><td>19.80</td><td>0.74</td><td>0.30</td><td>13.60</td><td>0.40</td><td>0.55</td></tr><tr><td>RynnWorld-Latent w/o latent-action pretraining</td><td>21.32</td><td>0.78</td><td>0.16</td><td>15.31</td><td>0.55</td><td>0.39</td></tr><tr><td>RynnWorld-Latent w/o frame-aligned injection</td><td>21.14</td><td>0.76</td><td>0.18</td><td>15.08</td><td>0.54</td><td>0.41</td></tr><tr><td>local modulation RynnWorld-Latent  $\mathbf { w } / \mathbf { o }$ </td><td>21.62</td><td>0.77</td><td>0.14</td><td>15.71</td><td>0.58</td><td>0.36</td></tr></table>

classifier-free guidance on the action conditioning, $\hat { u } = u _ { \varnothing } + 1 . 1 5 \left( u _ { A } - u _ { \varnothing } \right)$ , where $u _ { A }$ is the velocity field conditioned on the action sequence and $u _ { \mathcal { Q } }$ the same field with the sequence dropped, as defined in Sec. 3. For these quantitative benchmarks, the robot-action-conditioned model receives recorded control commands rather than latent actions extracted from future ground-truth frames. The qualitative action-following experiment instead uses source-video latent actions. These image-based metrics measure prediction fidelity rather than physical validity or action following in isolation. We therefore also examine qualitative action following under a fixed initial observation (Sec. 4.3). For policy learning, we report task success rate. For simulation policy benchmarks, we report average success rate under the official evaluation protocol. For real-world experiments, we report closed-loop success rate on the physical robots.

## 4.2 Action-Conditioned World-Model Benchmarks

We first evaluate RynnWorld-Latent as an action-conditioned world model on both public benchmarks and our own real-robot benchmarks.

**DreamDojo benchmarks.** We consider three evaluation sets introduced by DreamDojo (Gao et al., 2026): **In-lab Eval**, **EgoDex Eval**, and **DreamDojo-HV Eval**. These sets probe whether human-video pretraining transfers to object interactions unseen in the robot post-training data.

**RynnWorld-Latent real-robot benchmarks.** To test simulation fidelity on our own hardware, we construct evaluation sets on both robotic platforms by recording ground-truth rollouts across a diverse range of bimanual manipulation scenarios and asking the world model to reproduce them from the same initial frame and action sequence. The evaluation episodes cover varied objects, scenes, and contact patterns, spanning both the stationary dexterous setup (**Marvin-WUJI**) and the mobile gripper setup (**Astribot-S1**), and use an episode-level split: for each task, we randomly reserve 5% of the recorded episodes for testing and use the remaining 95% for target-robot post-training. Episodes are split before temporal chunking so that chunks from the same demonstration do not appear in both sets. This task-stratified split evaluates held-out demonstrations rather than held-out tasks.

**Results.** Tab. 2 reports predictive quality on the three GR-1 evaluation sets. RynnWorld-Latent achieves better

<!-- page 12 of 26 -->

Table 4 Closed-loop success rates on real-world manipulation tasks. The upper block evaluates policies trained with digital-teleoperation data. The lower block evaluates policies trained with large-scale synthetic mid-training. Each applicable policy setting is evaluated with 35 trials per task on the designated robot. Success rates are percentages; bold values indicate the best result per task within each table block. ‘N/A’ denotes baseline implementations without support for Astribot-S1 mobile-base motion. The Real only rows are the reference settings for both blocks.

| Method | Training Setting | Marvin-Plate Placement | WUJI Plate Wiping | Astri Steak Serving | bot-S1Plate Retrieval |
| --- | --- | --- | --- | --- | --- |
| π<sub>0</sub> (Black et al., 2024a) | Digital teleoperation Real only | 80.00 | 71.43 | 37.14 | 45.71 |
| π0 | Real + Digital Teleop | 88.57 | 94.29 | 51.43 | 51.43 |
| π<sub>0.5</sub> (Intelligence et al., 2025) | Real only | 82.86 | 77.14 | 40.00 | 45.71 |
| π0.5 | Digital Teleop only | 57.14 | 54.29 | 28.57 | 31.43 |
| π0.5 | Real + RynnWorld-Teleop (Zhao et al., 2026c) | 85.71 | 82.86 | N/A | N/A |
| π0.5 | Real + Digital Teleop | 91.43 | 91.43 | 54.29 | 57.14 |
| π0 | Large-scale synthetic mid-training RynnWorld-Latent-80M + Real | 85.71 | 82.86 | 48.57 | 54.29 |
| π0.5 | Real + Ego2Robot (Wang et al., 2026b) | 88.57 | 91.43 | N/A | N/A |
| π0.5 | RynnWorld-Latent-80M + Real | 94.29 | 97.14 | 60.00 | 68.57 |

reported PSNR, SSIM, and LPIPS than the listed baselines on all three evaluation sets. For example, LPIPS decreases from 0.21 to 0.19 relative to DreamDojo-14B on EgoDex Eval. These results support improved perceptual prediction fidelity. Since baseline measurements are taken from the literature, this comparison concerns complete systems and does not isolate the contribution of the latent-action interface from differences in backbone, data, or training.

Tab. 3 reports video prediction quality on our dual-arm platforms. These evaluation sets cover dexterous-hand manipulation and mobile bimanual coordination, extending the evaluation beyond the GR-1 benchmark. RynnWorld-Latent improves PSNR and LPIPS over the listed baselines on both platforms after target-robot adaptation, and ties the best SSIM on Marvin-WUJI. These results demonstrate applicability to stationary dexterous and mobile gripper embodiments; they do not establish zero-shot transfer to an unadapted robot. Ablations of the conditioning interface and the injection architecture are discussed in Sec. 4.6.

## 4.3 Qualitative Action Following

We examine whether supplied latent actions guide the generated motion beyond the scene information provided by the first frame. For each comparison, we extract two latent-action sequences, $A ^ { ( 1 ) }$ and $A ^ { ( 2 ) }$ , from different source videos and pair them with the same initial observation $I _ { 0 } .$ Encoding this observation as $c _ { 0 } ,$ , we generate

$$
\begin{array} { r } { \hat { V } ^ { ( k ) } \sim p _ { \theta } ( \cdot \mid c _ { 0 } , A ^ { ( k ) } ) , \qquad k \in \{ 1 , 2 \} . } \end{array}\tag{9}
$$

Fig. 5 shows source clips and the corresponding generated rollouts. Although each pair shares an initial scene, its generated motion changes with the supplied action sequence and qualitatively resembles the corresponding source motion. These examples provide evidence of latent-action following under a shared initial observation. Unlike data editing, which changes the initial appearance while reusing source actions, this comparison holds the initial observation fixed and changes the action sequence. It is a qualitative assessment of visual transition control rather than a quantitative measure of physical accuracy or executable robot-action tracking.

## 4.4 Digital-Teleoperation Data for Real-World Policy Learning

We next test whether data collected through interactive digital teleoperation can train effective real-world policies.

**Data collection.** An operator provides control signals through a teleoperation interface. These signals are retargeted to the target robot action space and used to condition RynnWorld-Latent. Given an initial conditioning frame and an action chunk, RynnWorld-Latent generates the corresponding future video latent

<!-- page 13 of 26 -->

![Image block](doc:57f42af/tier:basic/page:13/block:1)

Figure 5 Action following under a shared initial observation. Each comparison uses latent-action sequences extracted from two different source videos. Both sequences are paired with the same initial frame to generate separate rollouts. Blue and green arrows associate source clips with their corresponding generated rollouts. The generated motion qualitatively follows the respective source sequence despite the shared initial scene.

sequence. The system records the synchronized robot actions and generated video frames, producing synthetic demonstrations without physical robot execution. Fig. 7 illustrates this loop. The digital-teleoperation episodes are synthetic data and are distinct from the physically collected demonstrations summarized in Appendix B.

**Experimental settings.** We compare **Real only** and **Real + Digital Teleop** for both π<sub>0</sub> (Black et al., 2024a) and π<sub>0</sub>.5 (Intelligence et al., 2025). The former uses real robot demonstrations, while the latter additionally uses generated digital-teleoperation episodes. For $\pi _ { 0 . 5 } ,$ we also evaluate **Digital Teleop only**, using synthetic demonstrations without task-specific real-demonstration training. We additionally compare against **Real + RynnWorld-Teleop**, which combines real demonstrations with synthetic teleoperation data generated by RynnWorld-Teleop (Zhao et al., 2026c). Our RynnWorld-Teleop baseline implementation does not support Astribot-S1 mobile-base motion, so results are reported only on Marvin-WUJI. All policies are evaluated closed-loop on the real robot platforms.

<!-- page 14 of 26 -->

Table 5 Synthetic corpus by source dataset. The RynnWorld-Latent data engine processes these human interaction videos into robot-action-labeled synthetic episodes. After filtering, the total corpus contains approximately 800 hours.

| Source Dataset | Synthetic Frames |
| --- | --- |
| EPIC-KITCHENS (Damen et al., 2020) | 2.5M |
| EgoDex (Hoque et al., 2025) | 80.8M |
| Something-Something (Goyal et al., 2017) | 1.5M |
| RynnWorld-Latent-80M (total) | 84.8M |

**Results.** The upper block of Tab. 4 reports closed-loop success rates on the real-world tasks. Adding digitalteleoperation data improves success rates on all four tasks for both backbones. For $\pi _ { 0 . 5 } ,$ Plate-Placement improves from 82.86% to 91.43%, and Steak-Serving from 40.00% to 54.29%. The **Digital Teleop only** setting achieves nonzero success on all four tasks but remains below Real only on every task, supporting synthetic data as an augmentation rather than a complete replacement for real demonstrations in this setting. On Marvin-WUJI, the $\pi _ { 0 . 5 }$ Real + Digital Teleop setting also exceeds Real + RynnWorld-Teleop by approximately 5.71 and 8.57 percentage points on Plate-Placement and Plate-Wiping, corresponding to two and three additional successful trials. Each reported rate is based on 35 trials; we interpret these as observed improvements rather than claims of statistical significance.

## 4.5 Scalable Data Editing and Synthesis for Policy Learning

Beyond interactive teleoperation, we use RynnWorld-Latent as a scalable data engine that converts large-scale human interaction videos into robot-action-labeled data.

**Synthetic corpus.** Given a source human video, we extract its latent-action sequence using the frozen RynnLAM, edit the first frame to replace the human embodiment or original environment with the target robot embodiment or a robot-compatible scene, and condition RynnWorld-Latent on the edited frame and the original latent-action sequence to generate a robotized rollout. Robot commands are obtained by relabeling the source motion through retargeting and inverse kinematics; implementation details are provided in Appendix C. Fig. 6 illustrates this pipeline.

**Large-scale synthesis corpus.** We apply this pipeline to EPIC-KITCHENS (Damen et al., 2020), EgoDex (Hoque et al., 2025), and Something-Something (Goyal et al., 2017). After filtering for visual quality, temporal consistency, action feasibility, and interaction plausibility, we obtain approximately 84.8M frames, corresponding to roughly 800 hours of synthetic robot video-action data. We refer to this corpus as RynnWorld-Latent-80M. Tab. 5 groups the generated frames by their source dataset.

**Simulation policy benchmarks.** We examine whether synthetic mid-training improves policy learning in simulation. Starting from pretrained $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ checkpoints, we perform mid-training on RynnWorld-Latent-80M and then evaluate on standard policy benchmarks. We use LIBERO (Liu et al., 2023) and LIBERO-Plus. LIBERO evaluates language-conditioned manipulation, while LIBERO-Plus evaluates robustness under perturbations.

**Real-world policy learning.** We further test whether synthetic mid-training improves real-world policy learning. For each policy backbone, we compare two settings: (i) **Real post-training only**, where the policy is fine-tuned solely on real robot demonstrations; and (ii) **RynnWorld-Latent-80M mid-training + Real post-training**, where the policy is first mid-trained on the synthetic corpus and then fine-tuned on the same real robot demonstrations. We additionally compare with **Real + Ego2Robot**, which combines real demonstrations with synthetic data generated by Ego2Robot (Wang et al., 2026b). Our Ego2Robot baseline implementation does not support Astribot-S1 mobile-base motion, so results are reported only on Marvin-WUJI. All policies are evaluated closed-loop on the real robot platforms.

**Results.** Tab. 6 reports simulation benchmark results after synthetic mid-training. Synthetic mid-training improves $\pi _ { 0 }$ from 94.4% to 96.1% on LIBERO and from 53.6% to 79.4% on LIBERO-Plus. For $\pi _ { 0 . 5 } .$ , the corresponding changes are 96.9% to 97.8% and 84.4% to 84.8%; the observed gains are therefore larger for $\pi _ { 0 }$ than for $\pi _ { 0 . 5 }$ . The lower block of Tab. 4 reports improvements over Real only on all four real-world tasks for both backbones. For $\pi _ { 0 . 5 }$ , Steak-Serving rises from 40.00% to 60.00%, and Plate-Retrieval from 45.71%

<!-- page 15 of 26 -->

Table 6 Policy success rates on LIBERO (Liu et al., 2023) and LIBERO-Plus. Values are percentages. Each synthetic-data variant uses RynnWorld-Latent-80M for mid-training before downstream evaluation.

| Method | LIBERO | LIBERO-Plus |
| --- | --- | --- |
| π<sub>0</sub> (Black et al., 2024a) | 94.4 | 53.6 |
| π<sub>0</sub> +RynnWorld-Latent-80M | 96.1 | 79.4 |
| π<sub>0.5</sub> (Intelligence et al., 2025) | 96.9 | 84.4 |
| π<sub>0.5</sub> +RynnWorld-Latent-80M | 97.8 | 84.8 |

to 68.57%. On Marvin-WUJI, $\pi _ { 0 . 5 }$ with RynnWorld-Latent-80M + Real also exceeds Real + Ego2Robot by approximately 5.71 percentage points on each task, corresponding to two additional successful trials. These comparisons support the utility of the synthetic corpus in the evaluated training pipelines. They do not isolate generation quality from differences in synthetic-data quantity or additional training compute.

![Image block](doc:57f42af/tier:basic/page:15/block:4)

Figure 6 Scalable data editing. Given a source video, we edit its first frame to replace the embodiment or environment, then condition RynnWorld-Latent on the edited frame and the source latent-action sequence to synthesize a robotized rollout.

## 4.6 Ablation Studies

**Action-conditioning interface.** Tab. 3 compares conditioning interfaces on the real-robot evaluation sets. Replacing the action-conditioning interface with task-level text reduces PSNR from 21.86 to 19.80 on Marvin-WUJI and from 15.96 to 13.60 on Astribot-S1; LPIPS increases from 0.12 to 0.30 and from 0.34 to 0.55, respectively. SSIM also decreases on both platforms. These results are consistent with frame-level action

<!-- page 16 of 26 -->

![Image block](doc:57f42af/tier:basic/page:16/block:1)

Figure 7 Interactive digital teleoperation. Operator control signals are retargeted to target-robot actions and condition RynnWorld-Latent together with a reference scene; the recorded actions and generated frames form synthetic demonstrations without physical robot execution.

information providing a more specific prediction signal than task-level language. The variant without latentaction pretraining performs worse than the full model on all reported video prediction metrics Native robot commands are also used to evaluate the full model after target-robot adaptation; the latent interface serves its cross-embodiment pretraining stage. The latent interface additionally allows videos without native robot controls to contribute to pretraining; the present ablation does not separately identify how much of the observed gain comes from representation design versus access to additional video data.

**Conditioning architecture.** Removing frame-aligned injection while retaining global action tokens decreases PSNR by 0.72 on Marvin-WUJI and 0.88 on Astribot-S1. SSIM decreases from 0.80 to 0.76 and from 0.60 to 0.54, while LPIPS increases from 0.12 to 0.18 and from 0.34 to 0.41. Removing only local feature modulation produces smaller degradations: PSNR decreases by 0.24 and 0.25, respectively. The full model achieves the best metrics among these variants on both platforms. These results support combining global action tokens with local frame-aligned conditioning, while indicating that the additive pathway retains useful conditioning when modulation is removed. Image-level metrics alone do not determine whether the errors arise from contact timing, camera motion, or other visual changes.

## Conclusion

We presented RynnWorld-Latent, a cross-embodiment video world model that uses shared continuous latent actions as a conditioning interface for learning visual dynamics from heterogeneous interaction videos. Building on the 3D-grounded RynnLAM representation, the model combines global action tokens with temporally aligned local conditioning to connect motion descriptors with future video latents. This enables world-model pretraining across human, robot, and simulation videos without requiring their native action spaces to be aligned, followed by target-robot adaptation through an embodiment-specific action encoder.

Experiments show strong video prediction fidelity on public benchmarks and two real-robot platforms, while qualitative comparisons demonstrate that generated motions respond to different supplied latent-action sequences under the same initial observation. The model also supports approximate cross-embodiment motion transfer and native-action-conditioned digital teleoperation. Synthetic data generated through both routes improves downstream policy success in the evaluated settings. Together, these findings support shared latent actions as a practical interface for cross-embodiment world-model learning and controllable robot-data generation.

**Limitations.** Latent actions are inferred from observed visual transitions and may encode camera motion or appearance information as well as manipulation. Qualitative action following and image-based prediction metrics therefore do not establish causal identification or physical correctness. In the synthesis pipeline, retargeted robot commands and generated observations are obtained through separate processes; temporal

<!-- page 17 of 26 -->

alignment and filtering cannot guarantee their physical consistency. The reported real-world success rates use 35 trials per task, and small differences should be interpreted accordingly. Moreover, synthetic mid-training adds both data and optimization compute, whose contributions are not separated by the current comparisons. Future work will address long-horizon consistency, action-observation alignment, and broader transfer across embodiments.

<!-- page 18 of 26 -->

## References

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, et al. Cosmos 3: Omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800, 2026.

Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.

Marvin Alles, Xingyuan Zhang, Patrick van der Smagt, and Philip Becker-Ehmck. Latent Action World Models for Control with Unlabeled Trajectories. arXiv preprint arXiv:2512.10016, 2025. [https://arxiv.org/abs/2512.10016](https://arxiv.org/abs/2512.10016)

Mido Assran, Adrien Bardes, David Fan, et al. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv preprint arXiv:2506.09985, 2025. [https://arxiv.org/abs/2506.09985.](https://arxiv.org/abs/2506.09985)

Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani. Track2Act: Predicting Point Tracks from Internet Videos enables Generalizable Robot Manipulation. In Proc. of European Conf. on Computer Vision, 2024. [https://arxiv.org/abs/2405.01527.](https://arxiv.org/abs/2405.01527)

Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, and Jun Zhu. Motus: A Unified Latent Action World Model. arXiv preprint arXiv:2512.13030, 2025. [https://arxiv.org/abs/2512.13030.](https://arxiv.org/abs/2512.13030)

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024a.

Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, and Sergey Levine. Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models. In Proc. of International Conference on Learning Representations, 2024b. [https://arxiv.org/abs/2310.10639.](https://arxiv.org/abs/2310.10639)

Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, and Tim Rocktäschel. Genie: Generative Interactive Environments. arXiv preprint arXiv:2402.15391, 2024. [https://arxiv.org/abs/2402.15391.](https://arxiv.org/abs/2402.15391)

Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025a.

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. UniVLA: Learning to Act Anywhere with Task-centric Latent Actions. In Robotics: Science and Systems, 2025b. [https://arxiv.org/abs/2505.06111](https://arxiv.org/abs/2505.06111).

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In Proc. of International Conference on Learning Representations, volume 2026, pages 138846–138923, 2026.

Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, and Minzhao Zhu. GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation. arXiv preprint arXiv:2410.06158, 2024. [https://arxiv.org/abs/2410.06158](https://arxiv.org/abs/2410.06158).

Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, Yanjiang Guo, Rushuai Yang, Yucen Wang, Xinquan Xiao, Li Zhao, Jianyu Chen, and Jiang Bian. villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models. arXiv preprint arXiv:2507.23682, 2025a. [https://arxiv.org/abs/2507.23682.](https://arxiv.org/abs/2507.23682)

Yi Chen, Yuying Ge, Weiliang Tang, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos. In Proc. of IEEE Intl. Conf. on Computer Vision, 2025b. [https://arxiv.org/abs/2412.04445.](https://arxiv.org/abs/2412.04445)

<!-- page 19 of 26 -->

Jeremy A. Collins, Loránd Cheng, Kunal Aneja, Albert Wilcox, Benjamin Joffe, and Animesh Garg. AMPLIFY: Actionless Motion Priors for Robot Learning from Videos. arXiv preprint arXiv:2506.14198, 2025. [https://arxiv.org/abs/2506.14198](https://arxiv.org/abs/2506.14198).

Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. The epic-kitchens dataset: Collection, challenges and baselines. IEEE Trans. on Pattern Anal. and Mach. Intell., 43(11):4125–4141, 2020.

Yufan Deng, Zilin Pan, Hongyu Zhang, Xiaojie Li, Ruoqing Hu, Yufei Ding, Yiming Zou, Yan Zeng, and Daquan Zhou. Rethinking video generation model for the embodied world. In Proc. of Intl. Conf. on Machine Learning, 2026.

Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B. Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning Universal Policies via Text-Guided Video Generation. arXiv preprint arXiv:2302.00111, 2023. [https://arxiv.org/abs/2302.00111](https://arxiv.org/abs/2302.00111).

Frederik Ebert, Chelsea Finn, Sudeep Dasari, Annie Xie, Alex Lee, and Sergey Levine. Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control. arXiv preprint arXiv:1812.00568, 2018. [https://arxiv.org/abs/1812.00568](https://arxiv.org/abs/1812.00568).

Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, Shirui Chen, Yi Ru Wang, et al. Molmoact2: Action reasoning models for real-world deployment. arXiv preprint arXiv:2605.02881, 2026.

Chelsea Finn, Ian Goodfellow, and Sergey Levine. Unsupervised Learning for Physical Interaction through Video Prediction. In Proc. of Advances in Neural Information Processing Systems, 2016. [https://arxiv.org/abs/1605.07157](https://arxiv.org/abs/1605.07157).

Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, and Chuang Gan. AdaWorld: Learning Adaptable World Models with Latent Actions. In Proc. of Intl. Conf. on Machine Learning, 2025. [https://arxiv.org/abs/2503.18938.](https://arxiv.org/abs/2503.18938)

Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, Qianli Ma, Seungjun Nah, Loic Magne, Jiannan Xiang, Yuqi Xie, Ruijie Zheng, Dantong Niu, You Liang Tan, K. R. Zentner, George Kurian, Suneel Indupuru, Pooya Jannaty, Jinwei Gu, Jun Zhang, Jitendra Malik, Pieter Abbeel, Ming-Yu Liu, Yuke Zhu, Joel Jang, and Linxi Jim Fan. DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos. arXiv preprint arXiv:2602.06949, 2026. [https://arxiv.org/abs/2602.06949](https://arxiv.org/abs/2602.06949).

Quentin Garrido, Tushar Nagarajan, Basile Terver, Nicolas Ballas, Yann LeCun, and Michael Rabbat. Learning Latent Action World Models In The Wild. arXiv preprint arXiv:2601.05230, 2026. [https://arxiv.org/abs/2601.05230.](https://arxiv.org/abs/2601.05230)

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The “something something” video database for learning and evaluating visual common sense. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 5843–5851. IEEE, 2017.

Yanjiang Guo, Lucy Xiaoyang Shi, Jianyu Chen, and Chelsea Finn. Ctrl-World: A Controllable Generative World Model for Robot Manipulation. In International Conference on Learning Representations, 2026. [https://arxiv.org/abs/2510.10125](https://arxiv.org/abs/2510.10125).

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning Latent Dynamics for Planning from Pixels. In Proc. of Intl. Conf. on Machine Learning, 2019. [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. In Proc. of International Conference on Learning Representations, 2020. [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

Ryan Hoque, Peide Huang, David J Yoon, Mouli Sivapurapu, and Jian Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

Chengkai Hou, Kun Wu, Jiaming Liu, Zhengping Che, Di Wu, Fei Liao, Guangrun Li, Jingyang He, Qiuxuan Feng, Zhao Jin, et al. Robomind 2.0: A multimodal, bimanual mobile manipulation dataset for generalizable embodied intelligence. arXiv preprint arXiv:2512.24653, 2025.

Siqiao Huang, Jialong Wu, Qixing Zhou, Shangchen Miao, and Mingsheng Long. Vid2World: Crafting Video Diffusion Models to Interactive World Models. arXiv preprint arXiv:2505.14357, 2025. [https://arxiv.org/abs/2505.14357](https://arxiv.org/abs/2505.14357).

<!-- page 20 of 26 -->

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: a vision-language-action model with open-world generalization, 2025. [https://arxiv.org/abs/2504.16054.](https://arxiv.org/abs/2504.16054)

Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, Kaushil Kundalia, Yen-Chen Lin, Loic Magne, Ajay Mandlekar, Avnish Narayan, You Liang Tan, Guanzhi Wang, Jing Wang, Qi Wang, Yinzhen Xu, Xiaohui Zeng, Kaiyuan Zheng, Ruijie Zheng, Ming-Yu Liu, Luke Zettlemoyer, Dieter Fox, Jan Kautz, Scott Reed, Yuke Zhu, and Linxi Fan. DreamGen: Unlocking Generalization in Robot Learning through Video World Models. arXiv preprint arXiv:2505.12705, 2025. [https://arxiv.org/abs/2505.12705.](https://arxiv.org/abs/2505.12705)

Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and g0 dual-system vla model. arXiv preprint arXiv:2509.00576, 2025.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Hanjung Kim, Jaehyun Kang, Hyolim Kang, Meedeum Cho, Seon Joo Kim, and Youngwoon Lee. UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations. In Conference on Robot Learning, 2025. [https://arxiv.org/abs/2505.08787](https://arxiv.org/abs/2505.08787).

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv preprint arXiv:2406.09246, 2024. [https://arxiv.org/abs/2406.09246.](https://arxiv.org/abs/2406.09246)

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning. arXiv preprint arXiv:2601.16163, 2026. [https://arxiv.org/abs/2601.16163.](https://arxiv.org/abs/2601.16163)

Yann LeCun et al. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1): 1–62, 2022.

Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, et al. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning, pages 80–93. PMLR, 2023.

Yaxuan Li, Yichen Zhu, Junjie Wen, Chaomin Shen, and Yi Xu. WorldEval: World Model as Real-World Robot Policies Evaluator. arXiv preprint arXiv:2505.19017, 2025a. [https://arxiv.org/abs/2505.19017.](https://arxiv.org/abs/2505.19017)

Zuolei Li, Xingyu Gao, Xiaofan Wang, and Jianlong Fu. LatBot: Distilling Universal Latent Actions for Vision-Language-Action Models. arXiv preprint arXiv:2511.23034, 2025b. [https://arxiv.org/abs/2511.23034.](https://arxiv.org/abs/2511.23034)

Anthony Liang, Pavel Czempin, Matthew M. Hong, Yutai Zhou, Jingzhen Wang, Erdem Biyik, and Stephen Tu. CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2026. [https://arxiv.org/abs/2505.04999.](https://arxiv.org/abs/2505.04999)

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Proc. of Advances in Neural Information Processing Systems, 36: 44776–44791, 2023.

Kechen Liu and Ola Shorinwa. CLAP: Cross-Embodiment Video World Models are Zero-Shot Physical Simulators. arXiv preprint arXiv:2608.27406, 2026. [https://arxiv.org/abs/2608.27406.](https://arxiv.org/abs/2608.27406)

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In Proc. of International Conference on Learning Representations, volume 2025, pages 29982–30009, 2025.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Hao Luo, Ye Wang, Wanpeng Zhang, Haoqi Yuan, Yicheng Feng, Haiweng Xu, Sipeng Zheng, and Zongqing Lu. Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 2026. [https://arxiv.org/abs/2602.21736.](https://arxiv.org/abs/2602.21736)

<!-- page 21 of 26 -->

Willi Menapace, Stephane Lathuiliere, Sergey Tulyakov, Aliaksandr Siarohin, and Elisa Ricci. Playable Video Generation. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 2021. [https://arxiv.org/abs/2101.12195](https://arxiv.org/abs/2101.12195).

Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 2630–2640. IEEE, 2019.

Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024.

Alexander Nikulin, Ilya Zisman, Denis Tarasov, Nikita Lyubaykin, Andrei Polubarov, Igor Kiselev, and Vladislav Kurenkov. Latent Action Learning Requires Supervision in the Presence of Distractors. arXiv preprint arXiv:2502.00379, 2025. [https://arxiv.org/abs/2502.00379.](https://arxiv.org/abs/2502.00379)

Octo Model Team. Octo: An Open-Source Generalist Robot Policy. arXiv preprint arXiv:2405.12213, 2024. [https://arxiv.org/abs/2405.12213](https://arxiv.org/abs/2405.12213).

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In IEEE International Conference on Robotics and Automation, pages 6892–6903. IEEE, 2024.

Ryan Punamiya, Simar Kareer, Zeyi Liu, Josh Citron, Ri-Zhao Qiu, Xiongyi Cai, Alexey Gavryushin, Jiaqi Chen, Davide Liconti, Lawrence Y Zhu, et al. Egoverse: An egocentric human dataset for robot learning from around the world. arXiv preprint arXiv:2604.07607, 2026.

Julian Quevedo, Ansh Kumar Sharma, Yixiang Sun, Varad Suryavanshi, Percy Liang, and Sherry Yang. Worldgym: World model as an environment for policy evaluation. In Proc. of International Conference on Learning Representations, volume 2026, pages 78932–78957, 2026.

Jonathan Richens, David Abel, Alexis Bellot, and Tom Everitt. General agents contain world models. arXiv preprint arXiv:2506.01622, 2025.

Marc Rigter, Tarun Gupta, Agrin Hilmkil, and Chao Ma. AVID: Adapting Video Diffusion Models to World Models. arXiv preprint arXiv:2410.12822, 2024. [https://arxiv.org/abs/2410.12822.](https://arxiv.org/abs/2410.12822)

Oleh Rybkin, Karl Pertsch, Konstantinos G. Derpanis, Kostas Daniilidis, and Andrew Jaegle. Learning what you can do before doing anything. In Proc. of International Conference on Learning Representations, 2019. [https://arxiv.org/abs/1806.09655](https://arxiv.org/abs/1806.09655).

Dominik Schmidt and Minqi Jiang. Learning to Act without Actions. In Proc. of International Conference on Learning Representations, 2024. [https://arxiv.org/abs/2312.10812.](https://arxiv.org/abs/2312.10812)

Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and Zhibo Chen. VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model. arXiv preprint arXiv:2602.10098, 2026. [https://arxiv.org/abs/2602.10098.](https://arxiv.org/abs/2602.10098)

Richard S Sutton. Dyna, an integrated architecture for learning, planning, and reacting. ACM Sigart Bulletin, 2(4): 160–163, 1991.

Bahey Tharwat, Yara Nasser, Ali Abouzeid, and Ian Reid. Latent Action Pretraining Through World Modeling. arXiv preprint arXiv:2509.18428, 2025. [https://arxiv.org/abs/2509.18428.](https://arxiv.org/abs/2509.18428)

Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, et al. Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, pages 976–985, 2026.

Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 5026–5033. IEEE, 2012.

Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on robot learning, pages 1723–1736. PMLR, 2023.

Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025a.

<!-- page 22 of 26 -->

Lirui Wang, Xinlei Chen, Jialiang Zhao, and Kaiming He. Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers. In Proc. of Advances in Neural Information Processing Systems, 2024. [https://arxiv.org/abs/2409.20537](https://arxiv.org/abs/2409.20537).

Lirui Wang, Kevin Zhao, Chaoqi Liu, and Xinlei Chen. Learning Real-World Action-Video Dynamics with Heterogeneous Masked Autoregression. arXiv preprint arXiv:2502.04296, 2025b. [https://arxiv.org/abs/2502.04296.](https://arxiv.org/abs/2502.04296)

Xiaofeng Wang, Kang Zhao, Feng Liu, Jiayu Wang, Guosheng Zhao, Xiaoyi Bao, Zheng Zhu, and Yingya Zhang. Egovid-5m: A large-scale video-action dataset for egocentric videos generation. Proc. of Advances in Neural Information Processing Systems, 38, 2026a.

Ye Wang, Pei Lin, Xiong-Hui Chen, Haoqi Yuan, Zhixuan Liang, Yiyang Huang, Anzhe Chen, Zixing Lei, Jie Zhang, Tao Zhang, et al. Ego2robot: Scalable robot data synthesis from egocentric human data. arXiv preprint arXiv:2608.02580, 2026b.

Yixuan Wang, Rhythm Syed, Fangyu Wu, Mengchao Zhang, Aykut Onol, Jose Barreiros, Hooshang Nayyeri, Tony Dear, Huan Zhang, and Yunzhu Li. Interactive World Simulator for Robot Policy Training and Evaluation. In Robotics: Science and Systems, 2026c. [https://arxiv.org/abs/2603.08546.](https://arxiv.org/abs/2603.08546)

Chuan Wen, Xingyu Lin, John So, Kai Chen, Qi Dou, Yang Gao, and Pieter Abbeel. Any-point Trajectory Modeling for Policy Learning. In Robotics: Science and Systems, 2024. [https://arxiv.org/abs/2401.00025.](https://arxiv.org/abs/2401.00025)

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025a. [https://arxiv.org/abs/2508.02324](https://arxiv.org/abs/2508.02324).

Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation. In Proc. of International Conference on Learning Representations, 2024a. [https://arxiv.org/abs/2312.13139.](https://arxiv.org/abs/2312.13139)

Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, and Mingsheng Long. iVideoGPT: Interactive VideoGPTs are Scalable World Models. In Proc. of Advances in Neural Information Processing Systems, 2024b. [https://arxiv.org/abs/2405.15223](https://arxiv.org/abs/2405.15223).

Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, et al. Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877, 2024c.

Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, et al. Robocoin: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025b.

Zhuoyuan Wu and Jun Gao. OSCAR: Omni-Embodiment Action-Conditioned World Model for Robotics. arXiv preprint arXiv:2606.04463, 2026. [https://arxiv.org/abs/2606.04463.](https://arxiv.org/abs/2606.04463)

Mengda Xu, Zhenjia Xu, Cheng Chi, Manuela Veloso, and Shuran Song. XSkill: Cross Embodiment Skill Discovery. In Conference on Robot Learning, 2023. [https://arxiv.org/abs/2307.09955.](https://arxiv.org/abs/2307.09955)

Runze Xu, Yiluo Zhang, Jian Wang, Yu Wang, and Jincheng Yu. Motion-Focused Latent Action Enables Cross-Embodiment VLA Training from Human EgoVideos. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2026. [https://arxiv.org/abs/2606.18955.](https://arxiv.org/abs/2606.18955)

Adina Yakefu, Bin Xie, Chongyang Xu, Enwen Zhang, Erjin Zhou, Fan Jia, Haitao Yang, Haoqiang Fan, Haowei Zhang, Hongyang Peng, et al. Robochallenge: Large-scale real-robot evaluation of embodied policies. arXiv preprint arXiv:2510.17950, 2025.

Jiange Yang, Yansong Shi, Haoyi Zhu, Mingyu Liu, Kaijing Ma, Yating Wang, Gangshan Wu, Tong He, and Limin Wang. CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 2026a. [https://arxiv.org/abs/2505.17006.](https://arxiv.org/abs/2505.17006)

Jiarui Yang, Jiale Zhange, Jiawei Li, Hang Guo, Wen Huang, Jinpeng Wang, Peidong Liu, and Shu-Tao Xia. LAWM-3D: Learning 3D-Aware Latent Actions from Human Videos for Generalizable Robot World Models. arXiv preprint arXiv:2608.05706, 2026b. [https://arxiv.org/abs/2608.05706.](https://arxiv.org/abs/2608.05706)

<!-- page 23 of 26 -->

Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning Interactive Real-World Simulators. In Proc. of International Conference on Learning Representations, 2024. [https://arxiv.org/abs/2310.06114.](https://arxiv.org/abs/2310.06114)

Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent Action Pretraining from Videos. In Proc. of International Conference on Learning Representations, 2025. [https://arxiv.org/abs/2410.11758.](https://arxiv.org/abs/2410.11758)

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi Jim Fan, and Joel Jang. World Action Models are Zero-shot Policies. arXiv preprint arXiv:2602.15922, 2026. [https://arxiv.org/abs/2602.15922.](https://arxiv.org/abs/2602.15922)

Chubin Zhang, Jianan Wang, Zifeng Gao, Yue Su, Tianru Dai, Cai Zhou, Jiwen Lu, and Yansong Tang. CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Action Models from Human Videos. arXiv preprint arXiv:2601.04061, 2026a. [https://arxiv.org/abs/2601.04061.](https://arxiv.org/abs/2601.04061)

Chuheng Zhang, Tim Pearce, Pushi Zhang, Kaixin Wang, Xiaoyu Chen, Wei Shen, Li Zhao, and Jiang Bian. What Do Latent Action Models Actually Learn? arXiv preprint arXiv:2506.15691, 2025a. [https://arxiv.org/abs/2506.15691](https://arxiv.org/abs/2506.15691).

Jinglei Zhang, Jiankang Deng, Chao Ma, and Rolandos Alexandros Potamias. Hawor: World-space hand motion reconstruction from egocentric videos. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, pages 1805–1815. IEEE, 2025b.

Tianqiu Zhang, Muyang Lyu, Yufan Zhang, Fang Fang, and Si Wu. DiLA: Disentangled Latent Action World Models. arXiv preprint arXiv:2605.15725, 2026b. [https://arxiv.org/abs/2605.15725.](https://arxiv.org/abs/2605.15725)

Yang Zhang, Chenwei Wang, Ouyang Lu, Yuan Zhao, Yunfei Ge, Zhenglong Sun, Xiu Li, Chi Zhang, Chenjia Bai, and Xuelong Li. Align-Then-stEer: Adapting the Vision-Language Action Models through Unified Latent Guidance. arXiv preprint arXiv:2509.02055, 2025c. [https://arxiv.org/abs/2509.02055.](https://arxiv.org/abs/2509.02055)

Haoyu Zhao, Hao Wang, Xingyue Zhao, Hao Fei, Hongqiu Wang, Chengjiang Long, and Hua Zou. Physsplat: Efficient physics simulation for 3d scenes via mllm-guided gaussian splatting. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 5242–5252, 2025.

Haoyu Zhao, Cheng Zeng, Linghao Zhuang, Yaxi Zhao, Shengke Xue, Hao Wang, Xingyue Zhao, Zhongyu Li, Kehan Li, Siteng Huang, et al. High-fidelity simulated data generation for real-world zero-shot robotic manipulation learning with gaussian splatting. IEEE Robotics and Automation Letters, 2026a.

Haoyu Zhao, Xingyue Zhao, Siteng Huang, Xin Li, Deli Zhao, and Zhongyu Li. RynnWorld-4D: 4d embodied world models for robotic manipulation. arXiv preprint arXiv:2607.06559, 2026b.

Haoyu Zhao, Xingyue Zhao, Hangyu Li, Biao Gong, Kehan Li, Siteng Huang, Xin Li, Deli Zhao, and Zhongyu Li. RynnWorld-Teleop: An action-conditioned world model for digital teleoperation. arXiv preprint arXiv:2607.06558, 2026c.

Haoyu Zhao, Xingyue Zhao, Zhian Su, Bohan Hou, Yinzhou Tang, Fengyi Wang, Kehan Li, Yuming Jiang, Siteng Huang, Xin Li, Zhenyu Zeng, Mingxiu Chen, Deli Zhao, and Zhongyu Li. Rynnvla-latent: Scaling robotic manipulation foundation models with 3d-grounded latent actions. arXiv preprint arXiv:2610.34356, 2026d.

Haoyu Zhao, Linghao Zhuang, Xingyue Zhao, Cheng Zeng, Haoran Xu, Yuming Jiang, Jun Cen, Kexiang Wang, Jiayan Guo, Siteng Huang, et al. Towards affordance-aware robotic dexterous grasping with human-like priors. In Proc. of the AAAI Conf. on Artificial Intelligence, pages 13126–13134, 2026e.

Jinliang Zheng, Jianxiong Li, Dongxiu Liu, Yinan Zheng, Zhihao Wang, Zhonghong Ou, Yu Liu, Jingjing Liu, Ya-Qin Zhang, and Xianyuan Zhan. Universal Actions for Enhanced Embodied Foundation Models. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 2025. [https://arxiv.org/abs/2501.10105.](https://arxiv.org/abs/2501.10105)

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning. arXiv preprint arXiv:2411.04983, 2025. [https://arxiv.org/abs/2411.04983.](https://arxiv.org/abs/2411.04983)

Shangchen Zhou, Chongyi Li, Kelvin CK Chan, and Chen Change Loy. Propainter: Improving propagation and transformer for video inpainting. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 10477–10486, 2023.

<!-- page 24 of 26 -->

Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. RoboDreamer: Learning Compositional World Models for Robot Imagination. arXiv preprint arXiv:2404.12377, 2024. [https://arxiv.org/abs/2404.12377](https://arxiv.org/abs/2404.12377).

Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets. arXiv preprint arXiv:2504.02792, 2025a. [https://arxiv.org/abs/2504.02792.](https://arxiv.org/abs/2504.02792)

Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong. IRASim: A Fine-Grained World Model for Robot Manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025b. [https://arxiv.org/abs/2406.14540.](https://arxiv.org/abs/2406.14540)

Bojia Zi, Weixuan Peng, Xianbiao Qi, Jianan Wang, Shihao Zhao, Rong Xiao, and Kam-Fai Wong. Minimax-remover: Taming bad noise helps video object removal. Proc. of Advances in Neural Information Processing Systems, 38: 75518–75547, 2026.

<!-- page 25 of 26 -->

![Image block](doc:57f42af/tier:basic/page:25/block:1)

Figure 8 Marvin-WUJI demonstration collection. An operator controls dual robot arms and dexterous hands using motion trackers and data gloves.

## Appendix

## A Teleoperation Setup

For Marvin-WUJI, we collect real-world demonstrations through the teleoperation system shown in Fig. 8. The hardware setup consists of dual Tianji 7-DOF robotic arms and dual Wuji 20-DOF dexterous hands, yielding a total of 54 degrees of freedom.

For arm control, the human operator wears five HTC Vive trackers (mounted on the chest, both wrists, and both upper arms). The system computes wrist-to-chest relative transforms at 100-120 Hz and feeds them into a Pinocchio-based inverse kinematics solver running in a separate process. The resulting joint commands are further smoothed by a Ruckig trajectory generator with velocity, acceleration, and jerk constraints before being sent to the robot arms at 200 Hz.

For hand control, the operator wears Manus data gloves. The raw glove signals are converted to a 21-point MediaPipe hand skeleton format and retargeted to the 20-DOF Wuji hand joint space via a dedicated retargeting module, with an exponential moving average filter applied for motion smoothing.

## B Real-World Demonstration Corpus

The physically collected demonstration corpus is the same as that used in RynnVLA-Latent (?). It contains 3,850 episodes across six manipulation tasks on the real-robot platforms described in Sec. 4. Tab. 7 reports the collected episode counts. These counts describe the full corpus, not the number of episodes in an individual post-training split or the amount of generated digital-teleoperation data. For world-model adaptation and evaluation, we randomly split episodes within each task into 95% for post-training and 5% for testing. The split is defined at the episode level before video chunking. Thus all chunks from a held-out episode remain outside the post-training split, and the test tasks themselves are represented in training. The downstream policy experiments in this paper evaluate four of the six collected tasks.

We use the world-model chunking protocol from Sec. 3: RynnLAM descriptors are extracted at a four-frame

<!-- page 26 of 26 -->

Table 7 Collected real-world demonstrations. The corpus is shared with RynnVLA-Latent. Counts refer to physically collected episodes, not synthetic data.

| Task | Description | Episodes |
| --- | --- | --- |
| Plate-Placement | Take a plate from a shelf and place it beside the oven. | 500 |
| Steak-Cooking | Flip a steak in a pan. | 800 |
| Plate-Wiping | Take a dirty plate from a shelf, wipe it with a cloth, and return it. | 500 |
| Table-Cleaning | Collect scattered items into a box and place the box on a shelf. | 850 |
| Steak-Serving | Carry a steak plate to the dining area and place it on a placemat. | 600 |
| Plate-Retrieval | Collect a dirty plate and cutlery and return them to their designated positions. | 600 |
| Total |  | 3,850 |

stride, with up to 20 latent-action transitions in an 81-frame video chunk. This differs from the policy-specific action-chunk protocol in the companion VLA work. Sharing the demonstration corpus does not imply identical training objectives or temporal chunk lengths.

## C Details of Pseudo Robot Action Relabeling

For source videos without target robot actions, we estimate the relevant hand or end-effector trajectory from the source motion. The estimated trajectory is retargeted to the target robot workspace through coordinate mapping, workspace scaling, and embodiment-specific retargeting. We then solve inverse kinematics to obtain target robot joint commands. During IK solving, we enforce joint limits, velocity limits, and temporal smoothness. The resulting actions are treated as pseudo-labels rather than ground-truth robot commands. These pseudo-labels are subsequently filtered together with the generated videos before being used for policy training.

## D Baseline Implementation Details

**Ego2Robot.** We reproduce Ego2Robot (Wang et al., 2026b) in our pipeline. Specifically, we audit each raw video and its corresponding HaWoR (Zhang et al., 2025b) motion-capture trajectory, geometrically retarget the HaWoR hand poses to the 20-degree-of-freedom (DoF) Wuji hand, and resample the trajectories using zero-order hold or interpolation. At the source resolution, we use the projected HaWoR keypoints as prompts for SAM3 (Carion et al., 2026) to segment both hands and refine the segmentation using text prompts. We then take the union of the left- and right-hand masks, dilate it by 4 pixels, and remove the hands using a video object removal model, such as MiniMax-Remover (Zi et al., 2026) or ProPainter (Zhou et al., 2023). In parallel, on the CPU, we estimate the robot base pose through a coarse-to-fine inverse kinematics (IK) search in MuJoCo, evaluating 81 candidates in the coarse stage. The visual strategy is decoupled from the segmentation masks, and this branch outputs qpos54 and an IK report. Finally, we render the TIANJI M6 arm and Wuji hand using MuJoCo’s EGL-based off-screen rasterization and composite the rendered robot onto the inpainted video using an end-effector-locked projection strategy.

We further introduce quality gates for sample selection. A sample is marked as suitable for training only if it passes both the inpainting content quality checks and the strict IK and projection thresholds. The inpainting checks require a peak signal-to-noise ratio (PSNR) outside the removal mask of at least 30 dB, relaxed to 22 dB for videos with camera motion; the absence of entirely black frames; and successful checks of per-frame spatial standard deviation and temporal response. The rendering criteria require a position error of at most 5 mm, a rotation error of at most 5, an end-effector projection error of at most 3 pixels, a 95th-percentile retargeting landmark error of at most 0.1, and a finger-direction consistency score of at least 0.9.