<!-- page 1 of 29 -->

arXiv:2610.34356v1 [cs.RO] 25 Oct 2026

DAMO TECH TO THE FUTURE 湖畔实验室

# RynnVLA-Latent: Scaling Robotic Manipulation Foundation Models with 3D-Grounded Latent Actions

**Haoyu Zhao**<sup>∗1,2,3</sup>, **Xingyue Zhao**<sup>∗1,6</sup>, **Zhian Su**<sup>∗1</sup>, **Bohan Hou**<sup>∗1</sup>, **Yinzhou Tang**<sup>∗1</sup>, **Fengyi Wang**<sup>2,3</sup>, **Kehan Li**<sup>1</sup>, **Yuming Jiang**<sup>5</sup>, **Siteng Huang**<sup>†1,4</sup>, **Xin Li**<sup>1,4</sup>, **Zhenyu Zeng**<sup>1,4</sup>, **Mingxiu Chen**<sup>1,4</sup>, **Deli Zhao**<sup>1,4</sup>, **Zhongyu Li**<sup>†2,3</sup>

<sup>1</sup>DAMO Academy, Alibaba Group, <sup>2</sup>Hong Kong Embodied AI Lab, <sup>3</sup>CUHK, <sup>4</sup>Hupan Lab, <sup>5</sup>Alibaba Group, <sup>6</sup>PUMC

<sup>∗</sup>Core contributors,<sup>†</sup>Corresponding author

Scaling vision-language-action (VLA) models is constrained by scarce action-labeled demonstrations and incompatible control spaces across robot embodiments. We present **RynnVLA-Latent**, a framework that turns heterogeneous interaction videos into scalable supervision for robot policy pretraining. At its core, **RynnLAM** learns continuous, 3D-grounded latent actions from visual transitions without requiring native robot controls. By combining residual future-feature prediction, camera-compensated 3D-flow supervision, and camera-motion decoupling, RynnLAM encourages representations that capture control-relevant motion rather than appearance changes alone. We apply RynnLAM to 30.1K hours of human, robot, and simulation videos, producing 745.52M latent-action descriptors. RynnVLA-Latent then learns to predict latent-action chunks from visual observations and language through flow matching, before adapting its pretrained backbone and action expert to embodiment-specific robot actions. The resulting 4B policy achieves 98.8% success on LIBERO, 84.7% on LIBERO-Plus without benchmark-specific adaptation, 51.9% average success across five VLABench tracks, and 88.6% average success on the RoboTwin 2.0 bimanual benchmark. It also achieves the highest observed success rates among the compared methods on six real-world manipulation tasks across dexterous-hand and mobile-gripper platforms. Together with representation ablations and data-scaling results, these findings demonstrate the potential of latent-action supervision for scaling robot policy learning beyond native action-labeled data.

[https://alibaba-damo-academy.github.io/RynnVLA-Latent.github.io](https://alibaba-damo-academy.github.io/RynnVLA-Latent.github.io)

[https://github.com/alibaba-damo-academy/RynnVLA-Latent](https://github.com/alibaba-damo-academy/RynnVLA-Latent)

[https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnvla-latent](https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnvla-latent)

[https://www.modelscope.cn/collections/DAMO\_Academy/RynnVLA-Latent](https://www.modelscope.cn/collections/DAMO_Academy/RynnVLA-Latent)

**Date:** October 27, 2026

DAMO TECH TO THE FUTURE

## Introduction

Large-scale pretraining has enabled broad capabilities in language and vision-language models (Brown et al., 2020; Hoffmann et al., 2022; Kaplan et al., 2020; Achiam et al., 2023; Chen et al., 2022; Bai et al., 2025; Li et al., 2026a). Recent vision-language-action (VLA) and world-action models extend this approach to robot control (Black et al., 2024; Intelligence et al., 2025; Bjorck et al., 2025; Wu et al., 2026; Li et al., 2026b; Ye et al., 2026; Yuan et al., 2026b; Bi et al., 2026; Zhao et al., 2026d). Scaling policy learning, however, requires supervision that connects visual observations and task instructions to actions across diverse embodiments Zhao et al. (2026a, 2025).

Robot demonstrations are expensive to collect, and their native controls differ in dimensions, coordinate frames, and semantics (Zhao et al., 2026b,c). Human and web videos provide much broader interaction coverage but lack corresponding robot commands. These sources cannot simply be pooled under one native-action

<!-- page 2 of 29 -->

prediction objective. A shared supervision interface is needed to exploit their visual transitions before adapting a policy to a particular robot.

Latent actions provide such an interface by encoding changes between observations without native control labels (Ye et al., 2025; Chen et al., 2025c; Bu et al., 2025b; Gao et al., 2026, 2025; Schmidt and Jiang, 2024; Chen et al., 2024). Existing approaches demonstrate the value of transition reconstruction and feature-space prediction, but visual change can include camera motion, static appearance, and effects unrelated to the agent’s control. A representation that reconstructs images well therefore need not be equally useful for predicting low-level manipulation actions.

We address this through geometric supervision and explicit camera-motion separation. We introduce **RynnLAM**, which encodes a frame pair into motion tokens, a global action latent, and a camera latent. Residual futurefeature prediction preserves transition information through the bottleneck, while camera-compensated 3D-flow pseudo-labels and contrastive feature warping encourage motion-sensitive representations. Camera-pose prediction and gradient reversal further encourage separation of ego-motion from manipulation-related cues. These objectives provide geometric grounding without requiring manually annotated robot actions; they encourage camera decoupling rather than guaranteeing exact viewpoint invariance or physical identifiability.

Using RynnLAM as an offline labeler, we construct shared latent-action supervision for 25 video sources spanning human interaction, real robots, and simulation. The corpus contains approximately 30.1K hours and 3.057B frames and yields 745.52M latent-action descriptors. We then pretrain **RynnVLA-Latent**, a Mixture-of-Transformers policy that couples a pretrained vision-language backbone with a narrow action expert. The policy predicts latent-action chunks from multi-view observations and language via flow matching, without native-action or proprioceptive supervision. In robot post-training, we reuse both the backbone and action-expert trunk, introduce proprioception, and retarget the output projection to native robot commands. This separates video-based motion supervision from embodiment-specific action alignment.

Our evaluation follows the same path from representation learning to robot deployment. On LARY, frozen representation probes test whether the descriptors predict low-level control targets, with the full descriptor achieving average relative-action MSE 0.38. After native-action post-training, the 4B policy obtains 98.8% on LIBERO, 84.7% on LIBERO-Plus, 51.9% average success across five VLABench tracks, and 88.3% average success under the RoboTwin 2.0 Native protocol. These benchmarks respectively assess task execution, perturbation robustness, semantic grounding, and bimanual manipulation. Six real-world tasks on stationary dexterous and mobile gripper platforms test the interface after adaptation to different controls. Data-scaling and component studies further characterize where the approach helps and where performance remains limited.

Our contributions are as follows:

• **3D-grounded latent-action supervision.** We introduce RynnLAM, which combines residual feature prediction, camera-compensated 3D-flow supervision, and camera-motion decoupling to construct continuous transition representations without native robot-action labels.

• **Scalable VLA pretraining from heterogeneous video.** We develop RynnVLA-Latent, which learns shared latent-action targets from 30.1K hours of human, robot, and simulation videos, and transfers the pretrained policy to native robot controls through embodiment-specific post-training.

• **Representation andpolicy-levelvalidation.** We demonstrate control predictability through frozen-representation probes, competitive manipulation performance across simulation and real-world platforms, and improvements with increasing latent-action pretraining data.

## 2 Related Work

## 2.1 Latent Action Learning

Latent action learning provides a way to exploit videos without native action annotations by representing the transformations between observations. LAPO (Schmidt and Jiang, 2024) and Genie (Bruce et al., 2024) demonstrate that latent action spaces can support policy learning and controllable prediction from unlabeled videos. LAPA (Ye et al., 2025) learns discrete latent actions through frame-transition reconstruction and uses them as VLA pretraining targets, while Moto (Chen et al., 2025c) learns motion tokens for autoregressive

<!-- page 3 of 29 -->

pretraining and transfers the resulting motion priors to robot control. IGOR (Chen et al., 2024) similarly compresses image-goal transitions into shared action representations for learning across human and robot videos. These approaches establish visual transitions as a scalable source of action-like supervision beyond robot demonstrations.

Subsequent work investigates how to make such representations more useful for control. UniVLA (Bu et al., 2025b) introduces language-guided, task-centric latent actions in DINO feature space, while villa-X (Chen et al., 2026) augments visual reconstruction with a proprioceptive forward-dynamics objective to ground latent actions in robot states and actions. Latent actions also serve as transferable conditioning signals for world models: AdaWorld (Gao et al., 2025) uses video-derived latent actions for adaptable future prediction, and DreamDojo (Gao et al., 2026) learns continuous proxy actions from large-scale human videos. Garrido et al. (Garrido et al., 2026) further investigate constrained continuous latent actions for representation-space world modeling on in-the-wild videos.

Our focus is the geometric grounding of latent actions without robot-native action or proprioceptive supervision. RynnLAM combines residual future-feature prediction, camera-compensated dynamic 3D-flow supervision, and contrastive feature warping to learn a factorized continuous transition representation. A dedicated camera-pose pathway and an adversarial objective further encourage separation of camera ego-motion from manipulation-relevant motion. Rather than relying on continuity or feature-space prediction alone, these complementary objectives encourage the latent representation to capture physical scene changes and provide shared supervision across human, simulated, and real-robot videos.

## 2.2 Vision-Language-Action Models

Vision-language-action (VLA) models transfer pretrained vision-language knowledge to robot control. Open-VLA (Kim et al., 2024) and FAST (Pertsch et al., 2025) explore action-token prediction, while $\pi _ { 0 }$ (Black et al., 2024), GR00T N1 (Bjorck et al., 2025), Xiaomi-Robotics-0 (Cai et al., 2026), and LingBot-VLA (Wu et al., 2026) couple vision-language backbones with continuous action generators. Open X-Embodiment and RT-X (O’Neill et al., 2024), together with $\pi _ { 0 . 5 }$ (Intelligence et al., 2025), demonstrate the value of cross-embodiment and heterogeneous training. To handle incompatible action spaces, UniAct (Zheng et al., 2025) learns universal action representations with embodiment-specific decoding, X-VLA (Zheng et al., 2026) uses soft prompts and domain-specific projections, and Qwen-RobotManip (Yuan et al., 2026a) aligns canonical state-action representations and camera-frame motion. Human videos provide additional supervision through latent-action labels in GR00T N1 (Bjorck et al., 2025) and reconstructed hand motion in VITRA (Li et al., 2025).

Our work focuses on constructing a shared supervision interface for scalable VLA pretraining, rather than aligning heterogeneous native control parameterizations. RynnVLA-Latent learns continuous RynnLAMgenerated latent-action chunks from human, robot, and simulation videos through flow matching. The same descriptors also condition cross-embodiment video generation in the companion RynnWorld-Latent model. Native robot actions and proprioception are introduced only during embodiment-specific post-training, allowing large-scale video-based pretraining to be separated from executable control alignment.

## 3 RynnVLA-Latent

A central obstacle to scaling robotic manipulation is that the most abundant sources of manipulation knowledge—human videos, web videos, simulation, and heterogeneous robot datasets—do not share a common action representation. We address this problem by learning a shared latent-action space supervised by visual transitions and geometric pseudo-labels, without requiring identical native control spaces. This latent space serves both as an automatic annotation for large-scale unlabeled video and as an intermediate control representation that can later be aligned with concrete robot actions. Our framework consists of three components. We first assemble and preprocess the multi-embodiment video corpus (Sec. 3.1). We then train RynnLAM (Sec. 3.2) with future-feature prediction, dynamic 3D-flow supervision, and camera-motion decoupling, and use its encoder to label the corpus offline. Finally, we train RynnVLA-Latent (Sec. 3.3), a vision-language-action policy that learns to generate these latent actions from visual observations and language via flow matching. In post-training, the same flow-matching objective is retargeted from latent actions to

<!-- page 4 of 29 -->

![Image block](doc:4137b11/tier:basic/page:4/block:1)

![Chart block](doc:4137b11/tier:basic/page:4/block:2)

![Chart block](doc:4137b11/tier:basic/page:4/block:3)

![Image block](doc:4137b11/tier:basic/page:4/block:4)

Figure 1 Overview of the RynnVLA-Latent pretraining corpus. We aggregate approximately 30.1K hours from 25 datasets, comprising 16.63M episodes and 3.057B frames. Human video and real-robot video each contribute approximately 40% of the duration, with simulation accounting for the remaining 20%. The figure summarizes source composition, dataset contributions by duration, and object and action vocabulary from public tasks.

![Image block](doc:4137b11/tier:basic/page:4/block:6)

Figure 2 Data curation pipeline. The video data is collected from diverse sources and partitioned into short clips during data preprocessing. Each clip undergoes a multi-modal annotation process: (1) Video Captioning: Qwen3-VL (Bai et al., 2025) generates detailed natural language descriptions of the video content; (2) Optical Flow Estimation: DPFlow (Morimitsu et al., 2025) computes dense per-frame motion fields; (3) Depth Estimation: Depth Anything 3 (Lin et al., 2025) produces monocular depth predictions.

the target robot’s action space, so that costly action-labeled demonstrations are required only for the final embodiment alignment.

## 3.1 Pretraining Data

We collect a multi-embodiment corpus from **25** video sources, covering real-world robot demonstrations, simulated rollouts, and human interaction video (egocentric and web sources) across diverse embodiments (Fig. 1). Every source is processed by the same pipeline without the need for action labels (Fig. 2).

We process the entire dataset in parallel across 250 shards using RynnLAM. Specifically, we extract latent

<!-- page 5 of 29 -->

Table 1 Dataset-level statistics of the multi-source video corpus: all 25 sources listed below. Hours, episode counts, and frame counts summarize the source data. FPS denotes the frame rate declared in the source metadata rather than a uniform sampling rate, and can vary within a dataset; hours and frame counts are therefore not always consistent with this column.

| Dataset | FPS | Hours | Episodes | Frames (M) |
| --- | --- | --- | --- | --- |
| Robot data |  |  |  |  |
| RoVid-X (Deng et al., 2026) | 30 | 5,416.98 | 2,820,534 | 585.034 |
| AGIBOT-Beta (Bu et al., 2025a) | 30 | 2,341.93 | 141,959 | 252.929 |
| MolmoAct2 (Fang et al., 2026) | 30 | 703.44 | 32,239 | 75.972 |
| RoboCOIN (Wu et al., 2025) | 30 | 702.50 | 102,017 | 76.619 |
| DROID (Khazatsky et al., 2024) | 15 | 507.25 | 94,031 | 27.391 |
| AgiBotWorld2026 (Bu et al., 2025a) | 30 | 443.24 | 20,723 | 47.870 |
| RoboMIND2.0 (Hou et al., 2025) | 26 | 424.78 | 76,745 | 34.124 |
| Table30v2 (Yakefu et al., 2025) | 30 | 423.75 | 32,939 | 45.765 |
| Galaxea (Jiang et al., 2025) | 15 | 400.33 | 17,040 | 21.688 |
| Table30 (Yakefu et al., 2025) | 30 | 288.40 | 25,606 | 31.147 |
| RoboMIND (Wu et al., 2024) | 20 | 155.16 | 43,611 | 11.172 |
| DreamDojo (Gao et al., 2026) | 20 | 87.90 | 22,185 | 6.329 |
| OpenX-Embodiment (O'Neill et al., 2024) | 30 | 76.16 | 213,497 | 3.011 |
| BridgeV2 (Walke et al., 2023) | 5 | 37.58 | 16,218 | 0.676 |
| RDT-1B (Liu et al., 2025) | 30 | 29.57 | 6,089 | 3.193 |
| Subtotal | - | 12,038.97 | 3,665,433 | 1,222.920 |
| Robot data (simulation) |  |  |  |  |
| RoboCasa (Nasiriany et al., 2024) | 20 | 2,168.75 | 582,567 | 156.150 |
| BEHAVIOR-1K (Li et al., 2023) | 30 | 1,952.63 | 20,000 | 210.884 |
| InternData-A1 (Tian et al., 2026) | 30 | 1,828.77 | 320,910 | 197.508 |
| RoboTwin 2.0 (Chen et al., 2025a) | 30 | 189.03 | 125,501 | 20.415 |
| Subtotal | - | 6,139.18 | 1,048,978 | 584.957 |
| Human interaction video |  |  |  |  |
| HowTo100M (Miech et al., 2019) | 30 | 7,633.42 | 9,493,227 | 784.274 |
| EgoVid (Wang et al., 2025) | 30 | 2,208.80 | 1,963,777 | 238.550 |
| EgoVerse (Punamiya et al., 2026) | 30 | 1,213.30 | 57,621 | 131.036 |
| EgoDex (Hoque et al., 2025) | 30 | 823.93 | 338,181 | 88.984 |
| SthSthV2 (Goyal et al., 2017) | 12 | 33.34 | 32,968 | 1.440 |
| EPIC_KITCHENS (Damen et al., 2020) | 60 | 26.74 | 31,875 | 5.288 |
| Subtotal | - | 11,939.53 | 11,917,649 | 1,249.572 |
| Total corpus (25 datasets) | - | 30,117.68 | 16,632,060 | 3,057.449 |

actions from consecutive frame pairs sampled every 4 frames, i.e., pairs (0, 4), (4, 8), (8, 12), . . . , yielding one latent-action descriptor per adjacent pair in the four-frame sampling grid.

The source corpus contains approximately **30.1K hours**, **16.63M episodes**, and **3.057B frames** (Tab. 1). Offline labeling produces **745.52M latent actions**. For policy pretraining, we form chunks containing C=6 latent-action targets from the offline annotation index. The RGB-frame stride used to extract a descriptor and the temporal spacing used to form a policy chunk are distinct indexing choices. The resulting policy training index contains **121.06M samples**, corresponding to approximately **946K optimizer steps** per epoch at a global batch size of 128.

The corpus varies widely in embodiment, duration, and episode length. HowTo100M alone contributes more episodes than all real-robot sources combined, even though the human and robot groups have similar total duration. Shared latent-action labels make these sources compatible with one prediction objective, while source-weighted sampling controls their relative contribution during policy pretraining (Sec. 4.1).

<!-- page 6 of 29 -->

![Image block](doc:4137b11/tier:basic/page:6/block:1)

Figure 3 Architecture of RynnLAM and comparison with existing methods. (a) Conventional latent action models compress a frame transition into a single latent representation and reconstruct the future observation conditioned on the source frame. (b) RynnLAM factorizes transition information into motion tokens $z _ { t } ^ { m }$ , an action latent $z _ { t } ^ { a } ,$ and a camera latent $z _ { t } ^ { c }$ The three components are trained through complementary objectives: feature-space reconstruction, 3D dynamic-flow prediction, and relative camera-pose estimation, respectively. This factorization encourages the action representation to capture physical scene dynamics while encouraging camera ego-motion to be represented separately.

## 3.2 RynnLAM

To enable action alignment across diverse embodiments, including heterogeneous robots and humans, we adopt latent actions as shared proxy representations of motion between a video frame pair. Several prior works learn a quantized codebook of latent actions (Ye et al., 2025; Chen et al., 2025c). Transition reconstruction provides a shared learning target, but does not explicitly distinguish object motion from camera-induced visual change. Our goal is to add geometric supervision that makes the representation more useful for control. To this end, RynnLAM requires the latent representation to support three complementary objectives: predicting **future visual features**, recovering **dynamic 3D motion**, and disentangling **camera ego-motion from scene motion**.

RynnLAM is built upon Depth-Anything-3 (Lin et al., 2025); Fig. 3 contrasts this factorized design with conventional latent-action models that reconstruct the future observation. Given a pair of RGB frames $( \mathbf { O } _ { t } , \mathbf { O } _ { t + \Delta } )$ , the encoder outputs a factorized continuous latent action

$$
\mathbf { z } _ { t } = [ \mathbf { z } _ { t } ^ { m } ; \mathbf { z } _ { t } ^ { a } ; \mathbf { z } _ { t } ^ { c } ] ,\tag{1}
$$

where $\mathbf { z } _ { t } ^ { m } \in \mathbb { R } ^ { 5 1 2 }$ collects 8 local motion tokens $( 8 \times 6 4 ) ,   \mathbf { z } _ { t } ^ { a } \in \mathbb { R } ^ { 6 4 }$ is a global action latent, and $\mathbf { z } _ { t } ^ { c } \in \mathbb { R } ^ { 3 2 }$ is a dedicated camera latent. The encoder first computes a gated temporal difference between backbone features of the two frames, making the visual change explicit while suppressing irrelevant static appearance. Learned readout tokens summarize the global transition into $\mathbf { z } _ { t } ^ { a }$ and $\mathbf { z } _ { t } ^ { c }$ , while learned motion queries attend to the target-frame features and produce compact motion tokens $\mathbf { z } _ { t } ^ { m }$ . This design allows the latent action to encode both global transition cues and local motion information without relying on explicit action labels.

The latent action is trained with self-supervised objectives that require no manual annotation. We minimize a combined objective of the form:

$$
\begin{aligned} { \mathcal { L } _ { \operatorname { L A M } } } & { { } = \lambda _ { \operatorname { r e c } } \: \mathcal { L } _ { \operatorname { r e c } } + \lambda _ { \operatorname { f l o w } } \: \mathcal { L } _ { \operatorname { f l o w } } + \lambda _ { \operatorname { p o s e } } \: \mathcal { L } _ { \operatorname { p o s e } } } \\ { } & { { } \quad + \lambda _ { \operatorname { f e a t } } \: \mathcal { L } _ { \operatorname { f e a t } } + \lambda _ { \operatorname { g r l } } \: \mathcal { L } _ { \operatorname { g r l } } , } \\ \end{aligned}\tag{2}
$$

<!-- page 7 of 29 -->

with $$\lambda _ { \mathrm { r e c } } { = } \lambda _ { \mathrm { f l o w } } { = } \lambda _ { \mathrm { p o s e } } { = } 5 \text {, } \lambda _ { \mathrm { f e a t } } { = } 3 \text {, and } \lambda _ { \mathrm { g r l } } { = } 2 \text {. }$$

The reconstruction term $\mathcal { L } _ { \mathrm { r e c } }$ predicts the residual target-feature change $\mathbf { X } _ { t + \Delta } - \mathbf { X } _ { t }$ from the source features and the motion tokens $\mathbf { z } _ { t } ^ { m }$ . Although $\mathbf { z } _ { t } ^ { m }$ is produced from target-frame features, it enters the reconstruction decoder as a compact latent bottleneck and serves as the decoder’s only cross-frame input. The $8 \times 6 4 \cdot$ dimensional motion-token bottleneck compresses target-frame information, encouraging the representation to preserve compact transition cues rather than directly reproducing the full target feature map. Together with residual future-feature prediction, this bottleneck encourages the motion tokens to encode changes between observations while reducing reliance on static appearance. The resulting compact transition representation can then serve as part of the full latent-action descriptor predicted by the downstream policy without access to future frames at inference time. Predicting a zero residual incurs an error wherever the target features differ from the source features. Normalizing by the magnitude of the target change emphasizes transition reconstruction relative to its scale. Concretely, we use a normalized residual error,

$$
\mathcal { L } _ { \operatorname { r e c } } = \frac { \left\| \left( \hat { \mathbf { X } } _ { t + \Delta } - \mathbf { X } _ { t } \right) - \left( \mathbf { X } _ { t + \Delta } - \mathbf { X } _ { t } \right) \right\| ^ { 2 } } { \left\| \mathbf { X } _ { t + \Delta } - \mathbf { X } _ { t } \right\| ^ { 2 } + \epsilon _ { 0 } } ,\tag{3}
$$

which supervises the predicted change rather than absolute feature appearance, and $\epsilon _ { 0 }$ is a small constant that stabilizes the denominator when the two frames are nearly identical.

The flow term $\mathcal { L } _ { \mathrm { f l o w } }$ predicts dynamic 3D scene flow in the source camera frame, using source features, monocular depth, and the action latent. Pseudo 3D-flow supervision is obtained from estimated depth, camera poses, and off-the-shelf optical flow. After compensating for estimated camera motion, static regions should have approximately zero dynamic flow, encouraging the latent to capture object-level physical motion rather than 2D pixel shifts. To further reduce reliance on static source appearance, we additionally mask a portion of source patch tokens during flow decoding.

The feature term $\mathcal { L } _ { \mathrm { f e a t } }$ is contrastive rather than a plain consistency loss. Given the flow-warped source features $\tilde { \mathbf { X } } _ { t } ,$ we require them to be closer to the target features $\mathbf { X } _ { t + \Delta }$ than to the un-warped source features $\mathbf { X } _ { t } .$

$$
\mathcal { L } _ { \operatorname { f e a t } } = \mathbb { E } \left[ w _ { t } \; \operatorname* { m a x } \Bigl ( 0 , \; \mu - s \bigl ( \tilde { \mathbf { X } } _ { t } , \mathbf { X } _ { t + \Delta } \bigr ) + s \bigl ( \tilde { \mathbf { X } } _ { t } , \mathbf { X } _ { t } \bigr ) \Bigr ) \right] ,\tag{4}
$$

where $s ( \cdot , \cdot )$ denotes cosine similarity between the corresponding per-location feature vectors, $\mu$ is a margin, and $w _ { t }$ weights each spatial location by its flow magnitude. Using the un-warped source as a negative discourages identity warping in regions where the source and target differ, while the motion-magnitude weighting focuses supervision on regions with non-trivial dynamics.

To separate viewpoint changes from manipulation-relevant dynamics, we introduce explicit camera-pose decoupling. Let $g _ { \mathrm { p o s e } } ( \cdot )$ denote a pose decoder that maps a latent to the relative 6-DoF camera pose $\mathbf { p } _ { t } ^ { \mathrm { c a m } }$ between the two frames. The camera latent $\mathbf { z } _ { t } ^ { c }$ is supervised to decode this pose, $\mathcal { L } _ { \mathrm { p o s e } } = \left\| g _ { \mathrm { p o s e } } ( \mathbf { z } _ { t } ^ { c } ) - \mathbf { p } _ { t } ^ { \mathrm { c a m } } \right\| ^ { 2 }$ while a gradient-reversal objective drives the same decoder to fail on the action latent $\mathbf { z } _ { t } ^ { a }$

$$
\mathcal { L } _ { \operatorname { g r l } } = \left\| g _ { \operatorname { p o s e } } \big ( { \operatorname { G R L } } ( \mathbf { z } _ { t } ^ { a } ) \big ) - \mathbf { p } _ { t } ^ { \operatorname { c a m } } \right\| ^ { 2 } ,\tag{5}
$$

where GRL(·) forwards its input unchanged and negates the gradient during backpropagation. As a result, $\mathbf { z } _ { t } ^ { a }$ is encouraged to retain interaction- and motion-relevant cues, while ego-motion is routed into the separate camera latent.

For offline labeling, only the encoder is required. It processes multi-embodiment frame pairs and extracts the concatenated descriptor $\left[ \mathbf { z } ^ { m } ; \mathbf { z } ^ { a } ; \mathbf { z } ^ { c } \right]$ . Camera-pose decoupling acts specifically on the global action latent; the full descriptor retains the camera latent and is therefore not assumed to be camera-invariant. The motion tokens and global latents provide complementary transition information, with the full 608-dimensional descriptor achieving better control predictability than any individual component in our representation probes (Appendix E). We therefore use the complete descriptor as the shared supervision target for policy pretraining, without requiring manual alignment of native control spaces. The decoders are retained only for qualitative visualization.

<!-- page 8 of 29 -->

![Image block](doc:4137b11/tier:basic/page:8/block:1)

Figure 4 Two-stage training pipeline of RynnVLA-Latent. (a) Latent-action pretraining. RynnLAM provides shared transition descriptors for human and heterogeneous robot videos. The policy predicts these descriptors from observations and language. (b) Robot post-training. Both the pretrained RynnBrain backbone and action-expert trunk are reused. The output projection is retargeted to native robot actions, and proprioceptive state is introduced. Post-training adapts the pretrained policy to the target control space without adding a separate latent-to-action decoder.

## 3.3 RynnVLA-Latent

The latent actions produced by RynnLAM provide an embodiment-agnostic interface between video under standing and control; Fig. 4 summarizes the resulting training pipeline. Because the same encoder maps heterogeneous videos into a common descriptor layout, human, robot, and simulation videos can contribute to one policy objective without reconciling their native action conventions. Building on this interface, RynnVLA-Latent separates the learning of generalizable manipulation dynamics from embodiment-specific action alignment. During pre-training, the policy learns to generate latent actions from raw observations and language; during post-training, the same flow-matching objective is retargeted from latent actions to the target robot’s action space. This decoupling allows the model to scale on unlabeled multi-embodiment videos, while requiring costly action-labeled demonstrations only for the final embodiment alignment.

**Architecture.** RynnVLA-Latent adopts a Mixture-of-Transformers (MoT) architecture (Liang et al., 2024) that couples a pretrained vision-language model with a narrow action expert. The VLM backbone, RynnBrain (Li et al., 2026a), jointly encodes multi-view images and the language instruction into a causal prefix. The action expert is a randomly initialized transformer stream with the same depth as the VLM but a smaller hidden width. At each layer, tokens from the two streams are jointly attended, while their output projections and feed-forward pathways remain separate. The VLM prefix remains causal and cannot attend to the action suffix, whereas the action suffix is bidirectional and attends to the full VLM prefix. This design allows the action expert to exploit rich visual-language context without duplicating the full VLM capacity, and permits action-prediction gradients to refine the VLM representation. We instantiate RynnVLA-Latent with 2B and 4B VLM backbones while keeping the action expert narrow. The 2B and 4B names refer to the backbone variants; total policy parameter counts, including the action expert, are reported in Tab. 2.

**Pre-training: flow matching on latent actions.** We use half-open chunk notation: $\mathbf { z } _ { t : t + C }   =   ( \mathbf { z } _ { t } , \ldots , \mathbf { z } _ { t + C - 1 } )$ contains exactly C latent actions. For each sample i and each available view $v \in \mathcal { V } _ { i }$ , RynnLAM provides a latent-action chunk $\mathbf { z } _ { t : t + C : } ^ { ( v ) }$ where $\mathcal { V } _ { i }$ denotes the set of views present in that sample. The pre-training objective is to learn a conditional velocity field that generates these chunks from Gaussian noise. Given

$$
\mathbf { z } _ { t : t + C } ^ { \tau , ( v ) } = \tau \: \mathbf { \epsilon } ^ { ( v ) } + \left( 1 - \tau \right) \mathbf { z } _ { t : t + C } ^ { ( v ) } , \qquad \mathbf { \epsilon } ^ { ( v ) } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ,\tag{6}
$$

where $\tau { = } 1$ corresponds to pure noise and $\tau { = } 0$ to the clean latent, the expert predicts the derivative of this interpolation with respect to $\tau ,   \mathbf { \epsilon } ^ { ( v ) } - \mathbf { z } _ { t : t + C } ^ { ( v ) } ,$ conditioned on all present views, the language instruction, and the flow timestep τ . The training objective is

$$
\mathcal { L } _ { \operatorname { F M } } = \mathbb { E } _ { i , \tau , \mathbf { \epsilon } } \frac { 1 } { | \mathcal { V } _ { i } | } \sum _ { v \in \mathcal { V } _ { i } } \frac { 1 } { C d _ { z } } \left\| \mathbf { v } _ { \theta } ^ { ( v ) } - \big ( \mathbf { \epsilon } ^ { ( v ) } - \mathbf { z } _ { t : t + C } ^ { ( v ) } \big ) \right\| _ { 2 } ^ { 2 } ,\tag{7}
$$

where $d _ { z } { = } 6 0 8$ is the width of the concatenated latent action $\mathbf { z } _ { t }$ in Eq. (1). By defining $\mathcal { V } _ { i }$ as the set of present views, missing views contribute neither to attention nor to the loss. Timestep information is injected

<!-- page 9 of 29 -->

Table 2 Model configurations for the scaling variants of RynnVLA-Latent. Total parameters denote the total number of model parameters.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Layers</td><td colspan="2">VLM</td><td colspan="2">Action Expert</td><td rowspan="2">Total Params</td></tr><tr><td>Hidden Size</td><td>Params</td><td>Hidden Size</td><td>Params</td></tr><tr><td>RynnVLA-Latent-2B</td><td>28</td><td>2048</td><td>2.13 B</td><td>768</td><td>0.38 B</td><td>2.51B</td></tr><tr><td>RynnVLA-Latent-4B</td><td>36</td><td>2560</td><td>4.44 B</td><td>768</td><td>0.60 B</td><td>5.04 B</td></tr></table>

through adaptive normalization and input embeddings. We sample u ∼ Beta(1.5, 1) and linearly map u to [0.001, 0.999], biasing training toward noisier timesteps. Both the VLM and the action expert are updated. This stage requires neither robot actions nor proprioception; its supervision is available for any video from which RynnLAM can extract latent actions.

**Multi-view conditioning.** Embodiments differ not only in action spaces but also in camera configurations. To avoid hard-coding a fixed set of views, RynnVLA-Latent organizes latent-action generation using canonical camera slots. For each present view, its VLM patch embeddings are pooled into a slot token and prepended to the action expert. Empty slots are masked in attention and in the loss. This slot-based interface accommodates samples with different available views without supervising absent outputs. Camera configurations are mapped to the canonical slots; the interface alone does not guarantee invariance to an unseen viewpoint.

**Post-training: embodiment-specific action alignment.** Pre-training yields a model that generates embodimentagnostic latent actions but does not yet produce executable robot commands. In post-training, we retarget the same flow-matching objective from RynnLAM latents to the target robot’s action space and fine-tune the full model on demonstrations from that embodiment. Proprioceptive state is introduced only at this stage. No separate action decoder is added: the readout trunk pretrained during pre-training is retained and terminated by a thin newly initialized projection to the target action dimension, so the expert’s pretrained capacity is reused rather than duplicated. We use a unified action layout and mask dimensions not used by the target embodiment, allowing one interface to support different robots.

**Inference.** At inference, RynnVLA-Latent initializes the output sequence from Gaussian noise and integrates the learned velocity field from τ=1 toward τ=0 using Euler steps. In pre-training this output is a latent-action chunk; after post-training it is directly the robot action chunk. The VLM prefix is computed once and reused across integration steps. The resulting action chunk is executed in full before replanning. Post-training updates the policy parameters and changes the prediction target to native actions while retaining this sampling procedure; execution requires no separate latent-to-action decoder.

## 4 Experiments

We evaluate the proposed framework from three complementary perspectives. First, we assess the quality of the latent action representation itself, asking whether RynnLAM learns a representation that is predictive of low-level control signals rather than merely encoding visual semantics or appearance change. Second, we report completed downstream policy evaluations on simulation benchmarks covering manipulation success, robustness to perturbations, and language grounding. We additionally evaluate bimanual manipulation on RoboTwin 2.0 under the Native protocol. Third, we evaluate post-trained policies on real-world dual-arm platforms with different end-effectors and mobility, testing applicability across these deployment settings after embodiment-specific adaptation.

## 4.1 Implementation Details

**RynnLAM.** RynnLAM is built upon Depth-Anything-3 (Lin et al., 2025) and trained on video frame pairs without native-action labels, using visual-feature targets and geometric pseudo-labels. The model is optimized with the combined objective in Eq. (2), which supervises the latent action through residual feature prediction, dynamic 3D flow regression, contrastive warped-feature alignment, camera pose decoding, and gradientreversal-based camera decoupling. Training proceeds in two stages: Stage 1 (36k steps, encoder frozen) and

<!-- page 10 of 29 -->

![Image block](doc:4137b11/tier:basic/page:10/block:1)

Figure 5 Standardized Experimental Platforms. Two dual-arm embodiments used for real-world evaluation: the stationary Marvin-WUJI (dual 7-DOF arms with 20-DOF dexterous hands) and the mobile Astribot-S1 (dual 7-DOF arms with parallel grippers on a wheeled base).

Stage 2 (20k steps, encoder unfrozen). The final 608-dimensional descriptor comprises 8 motion tokens (512-D), a global latent action (64-D), and a relative camera-pose latent (32-D). RynnLAM is applied offline to the entire pretraining corpus (temporal stride 4 frames) to generate the training index for RynnVLA-Latent.

**RynnVLA-Latent.** RynnVLA-Latent adopts a Mixture-of-Transformers (MoT) architecture (Liang et al., 2024) consisting of a pretrained vision-language model and a narrow action expert. The VLM backbone is RynnBrain (Li et al., 2026a). The action expert matches the VLM depth but uses a smaller width $( d _ { \mathrm { m o d e l } } { = } 7 6 8 ,   d _ { \mathrm { f f } } { = } 2 7 5 2 )$ , interacting via per-layer joint attention. The observation interface supports 6 viewconditioned slots, with observation frames sampled at 2 fps, a maximum of 64 frames, and a 4096-token budget. Native end-effector rotations use a 6-D representation during robot-action post-training. During pretraining, RynnVLA-Latent is trained with flow matching to predict chunks of 6 latent-action targets conditioned on multi-view observations and language instructions, where each latent is the 608-dimensional RynnLAM descriptor. Proprioception is excluded. Both the VLM and action expert are updated using AdamW $( \beta _ { 1 } { = } 0 . 9 ,   \beta _ { 2 } { = } 0 . 9 5$ , weight decay= 0, gradient clipping= 1.0). Sources are sampled with probability $p _ { d } \propto W _ { d } ^ { 0 . \dot { 7 } }$ , where $W _ { d }$ is the configured weight for source $d ,$ to balance internet video and robot data. During post-training, the same flow-matching objective is retargeted from latent actions to the target robot’s action space, and proprioceptive state is introduced. No separate decoder is added; the pretrained readout trunk is retained and terminated by a thin projection to the target action dimension.

## 4.2 Real-Robot System Setup

Our real-world experiments use two robotic platforms to evaluate deployment after embodiment-specific post-training (Fig. 5). The policy predicts native action chunks, which are passed to the robot control interface for execution. Communication uses LCM (Lightweight Communications and Marshalling). This evaluation measures closed-loop task success; inference speed is not an evaluation metric.

**Marvin-WUJI.** Dual-Arm Dexterous Hand System. The first embodiment is built on the TIANJI M6 arms and WUJI dexterous hands. The hardware setup consists of dual Tianji 7-DOF robotic arms and dual Wuji 20-DOF dexterous hands, yielding a total of 54 degrees of freedom.

**Astribot-S1.** Astribot-S1 Mobile Manipulator. The second embodiment utilizes the Astribot S1, an AI robot platform featuring a wheeled mobile base and dual 7-DOF robotic arms equipped with parallel grippers. Each arm provides 7 DOFs with a rated payload of 5 kg (horizontal extension) and an end-effector repeatability of ±0.1 mm. With a height of 170 cm and an arm span of 194 cm, Astribot-S1 complements the stationary dexterous platform with mobile bimanual manipulation and parallel grippers.

<!-- page 11 of 29 -->

Table 3 Results and Ablation Study on the LARY benchmark (Nie et al., 2026). Low-level control regression (MSE, lower is better). ‘RynnLAM (Raw Features)’ probes the latent representation produced directly by the RynnLAM encoder without assembling the 608-dimensional descriptor, following the general-encoder protocol; the ‘RynnLAM’ rows probe the full 608-dimensional latent-action descriptor.

| Method | Avg. | CALVIN | VLABench | RoboCOIN | AgiBotWorld-B |
| --- | --- | --- | --- | --- | --- |
| General Encoders (Absolute Action) |  |  |  |  |  |
| V-JEPA 2 (Assran et al., 2025) | 0.25 | 0.27 | 0.07 | 0.32 | 0.33 |
| DINOv3 (Siméoni et al., 2025) | 0.19 | 0.22 | 0.06 | 0.22 | 0.24 |
| Wan2.2 (Wan et al., 2025) | 0.30 | 0.39 | 0.09 | 0.34 | 0.39 |
| FLUX.2-dev (Labs, 2025) | 0.35 | 0.25 | 0.04 | 0.47 | 0.62 |
| RynnLAM (Raw Features) | 0.14 | 0.08 | 0.01 | 0.24 | 0.22 |
| Embodied LAMs (Absolute Action) |  |  |  |  |  |
| LAPA (Ye et al., 2025) | 0.97 | 0.96 | 0.95 | 0.96 | 1.00 |
| Moto (Chen et al., 2025c) | 0.67 | 0.24 | 0.60 | 0.88 | 0.97 |
| UniVLA (Bu et al., 2025b) | 0.87 | 0.82 | 0.74 | 0.94 | 0.97 |
| villa-X (Chen et al., 2026) | 0.87 | 0.86 | 0.72 | 0.94 | 0.97 |
| DreamDojo (Gao et al., 2026) | 0.49 | 0.34 | 0.14 | 0.74 | 0.74 |
| RynnLAM | 0.27 | 0.34 | 0.01 | 0.34 | 0.39 |
| Embodied LAMs (Relative Action) |  |  |  |  |  |
| LAPA (Ye et al., 2025) | 0.93 | 0.72 | 1.00 | 1.01 | 0.98 |
| Moto (Chen et al., 2025c) | 0.77 | 0.65 | 0.64 | 0.91 | 0.89 |
| DreamDojo (Gao et al., 2026) | 0.55 | 0.35 | 0.33 | 0.81 | 0.70 |
| RynnLAM | 0.38 | 0.40 | 0.05 | 0.42 | 0.64 |
| Ablation Study (Absolute Action) |  |  |  |  |  |
| RynnLAM with frozen encoder | 0.37 | 0.41 | 0.10 | 0.43 | 0.52 |
| RynnLAM w/o 3D flow | 0.37 | 0.46 | 0.10 | 0.42 | 0.48 |
| RynnLAM w/o camera | 0.32 | 0.39 | 0.07 | 0.39 | 0.42 |
| RynnLAM w/o motion-token residual forcing | 0.37 | 0.45 | 0.09 | 0.43 | 0.52 |
| RynnLAM w/o GRL | 0.32 | 0.37 | 0.09 | 0.38 | 0.43 |
| RynnLAM | 0.27 | 0.34 | 0.01 | 0.34 | 0.39 |
| Ablation Study (Relative Action) |  |  |  |  |  |
| RynnLAM with frozen encoder | 0.52 | 0.43 | 0.12 | 0.72 | 0.79 |
| RynnLAM w/o 3D flow | 0.50 | 0.46 | 0.10 | 0.70 | 0.75 |
| RynnLAM w/o camera | 0.47 | 0.43 | 0.09 | 0.65 | 0.69 |
| RynnLAM w/o motion-token residual forcing | 0.57 | 0.50 | 0.14 | 0.80 | 0.85 |
| RynnLAM w/o GRL | 0.49 | 0.43 | 0.07 | 0.72 | 0.73 |
| RynnLAM | 0.38 | 0.40 | 0.05 | 0.42 | 0.64 |

## 4.3 Latent Action Representation Quality

We evaluate the quality of the learned latent representation using the LARY benchmark (Nie et al., 2026) We focus on its low-level control regression track, which decodes end-effector trajectories from a frozen representation using a lightweight MLP. Since the official absolute targets can be partially predicted from the currently visible robot pose under short temporal strides, we additionally report a relative setting that predicts end-effector displacements from the first frame. This setting more directly measures whether the representation encodes motion rather than static pose. We compare with general semantic encoders, generative pixel encoders, embodied latent-action models, and general latent-action models, following Nie et al. (2026).

## 4.4 Simulation Benchmarks

We report completed policy evaluations on LIBERO, LIBERO-Plus, and VLABench, covering manipulation success, robustness to perturbations, and semantic grounding. RoboTwin 2.0 Native evaluation additionally assesses bimanual manipulation. Detailed benchmark statistics and evaluation protocols are provided in

<!-- page 12 of 29 -->

![Image block](doc:4137b11/tier:basic/page:12/block:1)

Figure 6 Latent-action similarity visualization. We retrieve the nearest neighbors of several query latent actions across the multi-embodiment corpus and display the corresponding video frames. The displayed high-similarity neighbors exhibit related motion patterns across different embodiments and visual domains, providing a qualitative illustration of cross-domain transition similarity.

## Appendix A.

**LIBERO and LIBERO-Plus.** LIBERO (Liu et al., 2023) evaluates language-conditioned single-arm manipulation across spatial, object, goal, and long-horizon task suites. LIBERO-Plus (Fei et al., 2025) uses the same tasks but evaluates robustness under perturbations such as viewpoint, robot state, language, lighting, background, layout, and sensor noise. We use LIBERO to verify basic manipulation competence and LIBERO-Plus to measure robustness beyond in-distribution execution.

**VLABench.** VLABench (Zhang et al., 2025) evaluates semantic grounding and instruction understanding in manipulation. Its tasks require resolving target objects and intentions under implicit instructions, unseen categories, and visual variations. We report success rate together with the benchmark’s progress and intention scores.

**RoboTwin 2.0.** RoboTwin 2.0 (Chen et al., 2025a) evaluates bimanual manipulation under clean and randomized scenes. We use the Native protocol, which includes both clean and randomized demonstrations during training. Tab. 6 reports results for RynnVLA-Latent alongside the listed baselines.

## 4.5 Real-World Robot Tasks

We evaluate post-trained policies on six tasks spanning bimanual coordination, tool use, and mobile manipulation. The tasks are distributed across two dual-arm platforms: **Marvin-WUJI**, a stationary dual-arm robot equipped with dexterous hands, and **Astribot-S1**, a wheeled mobile dual-arm robot equipped with parallel grippers. The demonstrations are collected through the teleoperation setup described in Appendix B, with per-task episode counts reported in Tab. 11. The six tasks are defined as follows:

## Tasks performed on Marvin-WUJI (Dual-Arm Dexterous Hands):

• **(1) Plate-Placement:** The robot reaches toward a shelf, uses its dexterous hands to securely grasp a plate, and places it on the tabletop to the left of the oven, requiring precise spatial alignment and stable grasping of thin, fragile objects.

<!-- page 13 of 29 -->

Table 4 Results on the LIBERO benchmark.

| Method | Spatial | Object | Goal | Long | Avg. |
| --- | --- | --- | --- | --- | --- |
| OpenVLA-OFT (Kim et al., 2025) | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π<sub>0</sub>-FAST (Pertsch et al., 2025) | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π<sub>0</sub> (Black et al., 2024) | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| π<sub>0.5</sub> (Intelligence et al., 2025) | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T-N1.7 (Bjorck et al., 2025) | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| InternVLA-M1 (Chen et al., 2025b) | 98.0 | 99.0 | 93.8 | 92.6 | 95.9 |
| Motus (Bi et al., 2026) | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| Fast-WAM (Yuan et al., 2026b) | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| Xiaomi-Robotics-0 (Cai et al., 2026) | 98.8 | 100.0 | 98.8 | 97.2 | 98.7 |
| LingBot-VA (Li et al., 2026b) | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| RynnVLA-Latent-2B | 98.6 | 99.6 | 97.0 | 97.8 | 98.3 |
| RynnVLA-Latent-4B | 99.4 | 99.8 | 98.8 | 97.0 | 98.8 |

Table 5 Results on LIBERO-Plus, evaluated zero-shot with the LIBERO-finetuned checkpoint.

| Method | Cam. | Robot | Lang. | Light | Bkg. | Noise | Layout | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenVLA (Kim et al., 2024) | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT (Kim et al., 2025) | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π<sub>0</sub> (Black et al., 2024) | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π<sub>0</sub>-FAST (Pertsch et al., 2025) | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| π<sub>0.5</sub> (Intelligence et al., 2025) | 78.4 | 73.6 | 80.8 | 96.2 | 94.1 | 89.0 | 84.5 | 84.4 |
| StarVLA (Community, 2026) | 52.5 | 49.8 | 88.5 | 95.7 | 95.7 | 73.0 | 76.9 | 74.1 |
| ABot-M0 (Yang et al., 2026) | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 80.5 |
| Cosmos Policy (Kim et al., 2026) | 75.8 | 63.3 | 81.7 | 96.5 | 88.9 | 92.7 | 82.2 | 82.2 |
| RynnVLA-Latent-2B | 76.4 | 59.3 | 88.3 | 96.2 | 94.6 | 89.6 | 83.2 | 82.9 |
| RynnVLA-Latent-4B | 72.9 | 64.0 | 91.7 | 97.1 | 95.4 | 93.2 | 85.2 | 84.7 |

• **(2) Steak-Cooking:** The robot uses a spatula to flip a steak in a pan, requiring coordinated tool manipulation.

• **(3) Plate-Wiping:** A highly dexterous bimanual coordination task in which one hand grasps and stabilizes a plate while the other hand holds a cloth to wipe the plate’s surface before returning the plate to the shelf, requiring coordinated manipulation.

## Tasks performed on Astribot-S1 (Wheeled Mobile Dual-Arm Grippers):

• **(4) Table-Cleaning:** The mobile robot collects scattered items on the table into a box and then places the box on a shelf, combining object collection with mobile manipulation.

• **(5) Steak-Serving:** The robot approaches the dining area, uses its dual grippers to stably lift a plate containing a cooked steak, and accurately serves it onto the designated spot on the dining table, integrating mobile navigation with precise placement.

• **(6) Plate-Retrieval:** Following the meal, the robot identifies and grasps used, dirty plates left on the dining table, collects the associated cutlery, and returns the items to their designated positions.

## 4.6 Results

**Latent-action representation.** Tab. 3 reports low-level control regression on LARY (Nie et al., 2026), with extended analyses under the relative protocol reported in Appendix E. In the absolute setting, the un-assembled encoder output (RynnLAM (Raw Features)) achieves average MSE 0.14, but its RoboCOIN error (0.24) remains above DINOv3 (0.22). The full descriptor achieves the lowest average MSE among the listed embodied LAMs in both target settings: 0.27 absolute and 0.38 relative. Relative MSE is lower than DreamDojo (0.55), Moto (0.77), and LAPA (0.93) on average, with gains on VLABench, RoboCOIN, and AgiBotWorld-Beta. DreamDojo remains better on relative CALVIN (0.35 versus 0.40), and Moto has the lowest absolute CALVIN

<!-- page 14 of 29 -->

Table 6 RoboTwin 2.0 Native protocol. Values are success rates in percent. Baseline values are taken from the respective reports. Avg. in our rows is the arithmetic mean of Clean and Randomized.

| Method | Clean | Randomized | Avg. |
| --- | --- | --- | --- |
| π<sub>0</sub> (Black et al., 2024) | 65.9 | 58.4 | 62.2 |
| π<sub>0.5</sub> (Intelligence et al., 2025) | 82.7 | 76.8 | 79.8 |
| X-VLA (Zheng et al., 2026) | 72.9 | 72.8 | 72.9 |
| StarVLA-FAST (Community, 2026) | 72.5 | 83.2 | 77.9 |
| ABot-M0 (Yang et al., 2026) | 86.1 | 85.1 | 85.6 |
| LingBot-VLA (Wu et al., 2026) | 88.6 | 86.7 | 87.7 |
| RynnVLA-Latent-2B | 87.9 | 87.2 | 87.6 |
| RynnVLA-Latent-4B | 88.7 | 88.4 | 88.6 |

Table 7 Results on VLABench. SR, PS, and IS denote success rate, progress score, and intention score. The Average columns report arithmetic means across the five tracks when all five values are available. Dashes denote unreported measurements.

<table><tr><td rowspan="2">Method</td><td colspan="3">In-Distribution</td><td colspan="3">Cross-Category</td><td colspan="3">Common Sense</td></tr><tr><td>SR</td><td>PS</td><td>IS</td><td>SR</td><td>PS</td><td>IS</td><td>SR</td><td>PS</td><td>IS</td></tr><tr><td>π0 (Black et al., 2024)</td><td>47.0</td><td>62.7</td><td>67.8</td><td>21.2</td><td>33.6</td><td>44.0</td><td>29.1</td><td>43.0</td><td>54.9</td></tr><tr><td>ACoT-VLA (Zhong et al., 2026)</td><td></td><td>66.1</td><td>79.8</td><td></td><td>38.9</td><td>54.1</td><td>1</td><td>37.8</td><td>52.3</td></tr><tr><td>π₀-FAST (Pertsch et al., 2025)</td><td>56.2</td><td>66.8</td><td>72.4</td><td>31.0</td><td>38.2</td><td>47.8</td><td>38.0</td><td>48.6</td><td>56.8</td></tr><tr><td>X-VLA (Zheng et al., 2026)</td><td>1</td><td>67.8</td><td></td><td></td><td>25.1</td><td></td><td></td><td>48.2</td><td></td></tr><tr><td>(Intelligence et al., 2025) π0.5</td><td>65.4</td><td>77.8</td><td>80.4</td><td>38.2</td><td>49.7</td><td>52.0</td><td>43.9</td><td>57.3</td><td>60.0</td></tr><tr><td>RynnVLA-Latent-2B RynnVLA-Latent-4B</td><td>58.5 67.8</td><td>71.2 80.3</td><td>81.2 83.9</td><td>30.3 48.0</td><td>40.1 60.7</td><td>53.3 68.2</td><td>26.2 44.6</td><td>39.9 53.8</td><td>59.8 54.1</td></tr><tr><td></td><td>Semantic Instruction</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Method</td><td>SR</td><td>PS</td><td>IS</td><td>SR</td><td>Unseen Texture PS</td><td>IS</td><td></td><td>Average PS</td><td>IS</td></tr><tr><td>π0 (Black et al., 2024)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>SR</td><td></td><td></td></tr><tr><td>ACoT-VLA (Zhong et al., 2026)</td><td>17.3</td><td>38.7</td><td>58.0</td><td>32.2</td><td>42.5</td><td>50.6</td><td>29.4</td><td>44.1</td><td>55.0 63.5</td></tr><tr><td></td><td>1 35.0</td><td>39.6</td><td>56.8</td><td>1</td><td>54.6</td><td>74.6</td><td></td><td>47.4</td><td>58.6</td></tr><tr><td>π₀-FAST (Pertsch et al., 2025) X-VLA (Zheng et al., 2026)</td><td></td><td>45.0 63.1</td><td>59.4</td><td>39.0</td><td>49.0</td><td>56.8</td><td>39.8</td><td>49.5</td><td></td></tr><tr><td>(Intelligence et al., 2025) π0.5</td><td>1 48.2</td><td>64.2</td><td>67.0</td><td>44.9</td><td>62.3</td><td>65.0</td><td>一 48.1</td><td>51.1 62.3</td><td>64.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RynnVLA-Latent-2B</td><td>34.0</td><td>51.0</td><td>73.0</td><td>37.1</td><td>50.7</td><td>74.5</td><td>37.2</td><td>50.6</td><td>68.4</td></tr><tr><td>RynnVLA-Latent-4B</td><td>53.0</td><td>69.8</td><td>71.1</td><td>45.9</td><td>62.1</td><td>72.9</td><td>51.9</td><td>65.3</td><td>70.0</td></tr></table>

error in that block. Thus the representation improves average control predictability without uniformly dominating every dataset. Relative targets emphasize displacement prediction, while absolute targets also reward encoding the currently visible pose.

**LIBERO and robustness.** Tab. 4 reports average LIBERO success rates of 98.3% and 98.8% for the 2B and 4B variants. The 4B result is the highest reported average in the table, exceeding Xiaomi-Robotics-0 by 0.1 percentage point; this small difference should be interpreted as a reported score difference rather than evidence of statistical significance. The 4B variant leads Spatial and ties the highest Goal score, while other methods lead Object and Long. The 2B variant also exceeds 4B on Long (97.8% versus 97.0%), so increasing model size does not improve every suite.

On LIBERO-Plus (Tab. 5), the same LIBERO-finetuned policies obtain 82.9% and 84.7% without further benchmark-specific adaptation. The 4B variant has the highest total among the listed comparisons, narrowly above π<sub>0</sub>.5 (84.4%), and leads the language, lighting, noise, and layout perturbations. However, π<sub>0</sub>.5 performs better on camera and robot-state perturbations, while StarVLA leads background perturbations. These results show strong overall robustness with remaining weaknesses under viewpoint and initial-state changes.

**Bimanual manipulation.** RoboTwin 2.0 Native results (Tab. 6) demonstrate strong performance, with the

<!-- page 15 of 29 -->

Table 8 Success rates on real-world robotic manipulation tasks. Values are closed-loop success rates in percent on six tasks after robot-specific post-training. Each setting is evaluated with 35 trials per task. Bold indicates the highest observed rate per task among the listed comparisons.

<table><tbody><tr><td rowspan="2">Model</td><td colspan="3">Marvin-WUJI</td><td colspan="3">Astribot-S1</td></tr><tr><td>Plate Placement</td><td>Steak Cooking</td><td>Plate Wiping</td><td>Table Cleaning</td><td>Steak Serving</td><td>Plate Retrieval</td></tr><tr><td>π<sub>0</sub> (Black et al., 2024)</td><td>80.00</td><td>34.29</td><td>71.43</td><td>22.86</td><td>37.14</td><td>45.71</td></tr><tr><td>π<sub>0.5</sub> (Intelligence et al., 2025)</td><td>82.86</td><td>28.57</td><td>77.14</td><td>25.71</td><td>40.00</td><td>45.71</td></tr><tr><td>GR00T-N1.7 (Bjorck et al., 2025)</td><td>74.29</td><td>28.57</td><td>74.29</td><td>17.14</td><td>45.71</td><td>37.14</td></tr><tr><td>RynnVLA-Latent-2B</td><td>80.00</td><td>34.29</td><td>68.57</td><td>34.29</td><td>42.86</td><td>48.57</td></tr><tr><td>RynnVLA-Latent-4B</td><td>85.71</td><td>42.86</td><td>82.86</td><td>40.00</td><td>57.14</td><td>54.29</td></tr></tbody></table>

Table 9 Policy variants on LIBERO and LIBERO-Plus. Values are success rates in percent. Avg. is the arithmetic mean of the two benchmark scores. All rows, including the latent-action references, are trained on the same approximately 1,000-hour subset and are therefore directly comparable; they differ from the full-corpus results in Tabs. 4 and 5.

| Method | LIBERO | LIBERO-Plus | Avg. |
| --- | --- | --- | --- |
| RynnVLA-Latent-2B-QwenVL (Bai et al., 2025) | 98.1 | 78.8 | 88.5 |
| RynnVLA-Latent-4B-QwenVL (Bai et al., 2025) | 98.6 | 79.7 | 89.2 |
| RynnVLA-Latent-2B w/o camera slots | 98.5 | 81.9 | 90.2 |
| RynnVLA-Latent-4B w/o camera slots | 98.3 | 82.9 | 90.6 |
| RynnVLA-Latent-2B Shared vector representation | 98.7 | 80.5 | 89.6 |
| RynnVLA-Latent-4B Shared vector representation | 99.2 | 83.5 | 91.4 |
| RynnVLA-Latent-2B Domain-specific projection | 99.0 | 82.1 | 90.6 |
| RynnVLA-Latent-4B Domain-specific projection | 99.1 | 83.6 | 91.4 |
| RynnVLA-Latent-2B-LAPA (Ye et al., 2025) | 98.2 | 80.2 | 89.2 |
| RynnVLA-Latent-4B-LAPA (Ye et al., 2025) | 98.0 | 81.3 | 89.7 |
| RynnVLA-Latent-2B (RynnLAM) | 98.1 | 82.3 | 90.2 |
| RynnVLA-Latent-4B (RynnLAM) | 98.7 | 84.2 | 91.5 |

4B variant achieving the highest success rates among all listed methods. Specifically, RynnVLA-Latent-4B achieves 88.7% on Clean and 88.4% on Randomized scenes, yielding an average success rate of 88.6%. The 2B variant also performs competitively, scoring 87.9% Clean, 87.2% Randomized, and 87.6% on average. The 4B variant exceeds the 2B variant by 0.8 percentage points in Clean and 1.2 in Randomized, yielding a 1.1-point gain in average success rate. Notably, the 4B variant’s Clean score surpasses LingBot-VLA (88.6%), and its Randomized score exceeds both LingBot-VLA (86.7%) and StarVLA-FAST (83.2%), establishing a new leading result on this benchmark under the Native protocol. Because the Native protocol includes randomized demonstrations during training, these results assess performance within that protocol rather than zero-shot transfer to randomized scenes.

**Semantic grounding.** Tab. 7 reports five VLABench tracks. The 4B variant achieves the highest listed success rate on each track and arithmetic-mean SR, PS, and IS of 51.9, 65.3, and 70.0, respectively. Its cross-category SR is 48.0%, compared with 38.2% for $\pi _ { 0 . 5 }$ . The gains are not uniform across auxiliary scores: $\pi _ { 0 . 5 }$ leads common-sense PS and IS, and narrowly leads unseen-texture PS. The 2B variant has lower average SR and PS than $\pi _ { 0 . 5 }$ but higher average IS, illustrating that interpreting an instruction and completing the corresponding manipulation can remain separate challenges.

**Real-world manipulation.** Tab. 8 reports closed-loop results on the two platforms. The 4B variant has the highest observed success rate among the listed methods on all six tasks. Relative to $\pi _ { 0 . 5 }$ , Table-Cleaning improves from 25.71% to 40.00% and Steak-Serving from 40.00% to 57.14%. The 2B variant provides a more mixed comparison: it improves Table-Cleaning and Plate-Retrieval over the listed baselines, but trails $\pi _ { 0 . 5 }$ on Plate-Placement and Plate-Wiping and trails GR00T-N1.7 on Steak-Serving. These results support the utility of the pretrained policy after target-robot post-training, rather than zero-shot deployment on an unseen embodiment. Fig. 7 shows representative executions.

<!-- page 16 of 29 -->

"Collect all scattered items on the table into the box, then place the box onto the shelf.."

"Take the plate from the shelf and place it to the left of the oven."

"Take the dirty plate from the shelf, wipe it with a cloth, and return it to the same shelf. "

![Image block](doc:4137b11/tier:basic/page:16/block:4)

"Carry the steak plate horizontally to the designated dining spot and place it gently on the placemat."

"Collect the dirty plate and the knife and fork from the dining table, and return them to their original positions."

Figure 7 Post-training evaluation on real-world tasks. Each row shows the language instruction and representative execution frames for one task on the corresponding platform.

**Qualitative interpretation of the latent space.** Fig. 6 illustrates related motion patterns among retrieved latent neighbors across domains. To visualize policy-predicted latents, we pass a predicted chunk to the frozen RynnLAM 3D-flow decoder (Fig. 9). The decoded fields provide a qualitative interpretation of the motion implied by the latent predictions. They do not independently establish metric physical accuracy or executable-action consistency.

## 4.7 Component Studies and Data Scaling

## 4.7.1 RynnLAM

We further ablate the main designs of RynnLAM in Tab. 3. Since the absolute setting can be partially affected by the currently visible end-effector pose, we mainly discuss the relative setting, while the absolute results show consistent trends.

**End-to-end training.** Training with a frozen encoder increases the relative average MSE from 0.38 to 0.52, and increases the absolute average MSE from 0.27 to 0.37. This gap supports updating the backbone during latent-action learning to improve the representation’s utility for the evaluated control-regression tasks.

**3D flow grounding.** Removing 3D flow supervision increases the relative average MSE from 0.38 to 0.50, and

<!-- page 17 of 29 -->

![Chart block](doc:4137b11/tier:basic/page:17/block:1)

Figure 8 Scaling of latent-action pretraining. We pretrain RynnVLA-Latent-2B and RynnVLA-Latent-4B with 10%, 30%, 50%, 70% and 100% of the latent-action pretraining corpus, and evaluate the resulting policies on LIBERO-Plus (Fei et al., 2025). Both models achieve their highest success rates with the full corpus, although improvements are not strictly monotonic across all data fractions.

increases the absolute average MSE to 0.37. The error increases on all four datasets in both settings. This supports geometric scene-flow supervision as a useful signal for learning control-predictive descriptors; the probe does not directly measure the metric accuracy of recovered 3D motion.

**Camera-motion decoupling.** Removing the camera branch or the GRL objective increases the relative average MSE to 0.47 and 0.49, respectively, compared with 0.38 for the full model. The corresponding absolute averages are both 0.32, compared with 0.27 for the full model. The error increases in both target settings support camera-pose supervision and the adversarial objective as useful components for control predictability. These ablations do not independently establish viewpoint invariance, which is assessed separately through downstream perturbation results.

**Motion-token residual forcing.** Removing motion-token residual forcing causes the largest degradation in the relative setting, increasing the average MSE from 0.38 to 0.57. Its absolute average also increases to 0.37. The increase on every dataset supports residual future-feature prediction as a useful objective for preserving control-relevant information in the compact motion tokens. It is consistent with the intended role of the bottleneck, although the probe does not identify the specific shortcut used by a variant without this objective.

## 4.7.2 RynnVLA-Latent Policy Variants

To isolate the effects of model architecture and action-supervision design, we conduct controlled comparisons on an approximately 1,000-hour subset drawn from RoboCOIN (Wu et al., 2025), EgoDex (Hoque et al., 2025), EPIC-KITCHENS (Damen et al., 2020), and Something-Something (Goyal et al., 2017). All variants in Tab. 9, including the RynnLAM-based reference models, use this subset rather than the full 30.1K-hour corpus. We therefore distinguish these controlled configuration comparisons from the full-scale results reported in Tabs. 4 and 5.

**Latent Actions as Cross-Embodiment Action Alignment.** We compare RynnLAM-based supervision against two strategies for aligning heterogeneous action spaces: a shared canonical action vector with masking (Yuan et al., 2026a; Intelligence et al., 2025) and embodiment-specific input and output projections (Agarwal et al., 2026; Zheng et al., 2026). For human videos without native action labels, both explicit alignment baselines use motion-capture-derived wrist poses and fingertip positions as pseudo-action supervision, with implementation details provided in Appendix C. As shown in Tab. 9, under the same approximately 1,000-hour pretraining subset, RynnLAM-based pretraining achieves higher LIBERO-Plus success than both alternatives: 82.3% versus 80.5%/82.1% for 2B, and 84.2% versus 83.5%/83.6% for 4B. In contrast, the explicit action-alignment variants achieve higher LIBERO success. These results suggest that latent-action supervision can favor

<!-- page 18 of 29 -->

"Picks cucumber, tomato and corn from a supermarket shelf and places them into a plastic bag in the shopping cart."

![Image block](doc:4137b11/tier:basic/page:18/block:2)

"Lifts a steam iron and irons both sides of a hanging shirt, then places the iron back on the garment steamer."

![Image block](doc:4137b11/tier:basic/page:18/block:4)

"Picks up the yellow whiteboard eraser from the rack with its right arm, and wipes the red marker writing off the whiteboard."

![Image block](doc:4137b11/tier:basic/page:18/block:6)

"Picks up the blue bowl and places it onto the yellow plate."

![Image block](doc:4137b11/tier:basic/page:18/block:8)

"Puts the strings into the white basket."

![Image block](doc:4137b11/tier:basic/page:18/block:10)

"Picks the green block up from the table and places it into the black bowl."

![Image block](doc:4137b11/tier:basic/page:18/block:12)

"Turns on the electric stove and places the moka pot onto the hot burner."

![Image block](doc:4137b11/tier:basic/page:18/block:14)

Figure 9 Qualitative visualization of 3D scene flow decoded from latent actions predicted by RynnVLA-Latent. Given the current observation and language instruction, RynnVLA-Latent predicts a latent-action chunk, which is fed into the RynnLAM flow decoder to visualize the implied 3D motion.

<!-- page 19 of 29 -->

robustness to distribution shifts, while explicit native-action alignment remains competitive for in-distribution task execution.

**Backbone, camera slots, and latent-action targets.** Under the shared training subset, the RynnBrain-based models outperform their Qwen-VL counterparts on LIBERO-Plus by 3.5 and 4.5 percentage points for 2B and 4B, respectively. Removing canonical camera slots reduces LIBERO-Plus success by 0.4 and 1.3 percentage points, while replacing RynnLAM descriptors with LAPA targets reduces it by 2.1 and 2.9 percentage points. These results support the combined design of the vision-language backbone, multi-view conditioning interface, and continuous latent-action supervision, with the most consistent differences appearing under perturbation-based evaluation.

**Scaling Behavior.** We investigate the impact of latent-action pretraining data scale by training RynnVLA-Latent with 10%, 30%, 50%, 70%, and 100% of the video corpus and evaluating the resulting policies on LIBERO-Plus under the same downstream protocol. As shown in Fig. 8, both 2B and 4B models achieve their highest success rates with the full pretraining corpus, demonstrating an overall benefit from scaling up heterogeneous video supervision. While the 2B model improves monotonically across the evaluated data fractions, the 4B model exhibits a slight decrease from 10% to 30% before improving with larger data scales. These results suggest that increasing latent-action pretraining data generally improves downstream robustness within the evaluated range, although the relationship is not strictly monotonic for every model size.

## 5 Conclusion

We presented a framework for latent-action pretraining of vision-language-action policies from heterogeneous interaction video. RynnLAM combines residual feature prediction, camera-compensated 3D-flow pseudo-labels, contrastive warping, and camera-pose decoupling to produce continuous transition descriptors without native robot-action labels. RynnVLA-Latent learns to predict these descriptors through flow matching, then reuses its pretrained backbone and action expert while retargeting the output space to robot commands.

The evaluation supports this interface at two levels. Frozen probes show that the descriptor is predictive of low-level controls, with average relative-action MSE 0.38 on LARY. After native-action post-training, the 4B policy achieves 98.8% on LIBERO, 84.7% on LIBERO-Plus, 51.9% average success on the five reported VLABench tracks, and 88.3% under the RoboTwin 2.0 Native protocol. It also attains the best success rate among the compared baselines on all six real-world tasks across stationary dexterous and mobile gripper platforms. Performance remains task-dependent, and RoboTwin scores remain below the strongest listed baselines. Together with the data-scaling trend, these findings support shared transition supervision as a practical route to policy pretraining across incompatible control spaces.

**Limitations.** Depth, camera poses, and flow are estimated pseudo-labels, so geometric supervision does not guarantee calibrated metric motion or complete separation of camera and manipulation information. Low-level control probes measure predictability, and decoded flow provides qualitative interpretation, rather than proof of causal action identifiability. Policy performance varies across tasks and perturbations, and deployment still requires embodiment-specific action-labeled demonstrations. The current results also do not establish a general scaling law or isolate pretraining-data volume from all differences in optimization exposure. Future work will improve geometric consistency, viewpoint robustness, and transfer with less target-robot supervision.

<!-- page 20 of 29 -->

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, et al. Cosmos 3: Omnimodal world models for physical AI. arXiv preprint arXiv:2606.02800, 2026.

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 35101–35113, 2026.

Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. GR00T n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, pages 1877–1901, 2020.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In International Conference on Machine Learning, pages 4603–4623, 2024.

Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. AgiBot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025a.

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. UniVLA: Learning to act anywhere with task-centric latent actions. In Robotics: Science and Systems, 2025b.

Rui Cai, Jun Guo, Xinze He, Piaopiao Jin, Jie Li, Bingxuan Lin, Futeng Liu, Wei Liu, Fei Ma, Kun Ma, Feng Qiu, Heng Qu, Yifei Su, Qiao Sun, Dong Wang, Donghao Wang, Yunhong Wang, Rujie Wu, Diyun Xiang, Yu Yang, Hangjun Ye, Yuan Zhang, and Quanyun Zhou. Xiaomi-robotics-0: An open-sourced vision-language-action model with real-time execution. arXiv preprint arXiv:2602.12684, 2026.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025a.

Xi Chen, Xiao Wang, Soravit Changpinyo, Anthony J Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. PaLI: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022.

Xiaoyu Chen, Junliang Guo, Tianyu He, Chuheng Zhang, Pushi Zhang, Derek Cathera Yang, Li Zhao, and Jiang Bian. Igor: Image-goal representations are the atomic control units for foundation models in embodied AI. arXiv preprint arXiv:2411.00785, 2024.

<!-- page 21 of 29 -->

Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, Yanjiang Guo, Rushuai Yang, Yucen Wang, Xinquan Xiao, Li Zhao, et al. Villa-x: Enhancing latent action modeling in vision-language-action models. In International Conference on Learning Representations, pages 70673–70703, 2026.

Xinyi Chen, Yilun Chen, Yanwei Fu, Ning Gao, Jiaya Jia, Weiyang Jin, Hao Li, Yao Mu, Jiangmiao Pang, Yu Qiao, et al. InternVLA-m1: A spatially guided vision-language-action framework for generalist robot policy. arXiv preprint arXiv:2510.13778, 2025b.

Yi Chen, Yuying Ge, Weiliang Tang, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for learning robot manipulation from videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19752–19763, 2025c.

StarVLA Community. StarVLA: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026.

Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. The EPIC-KITCHENS dataset: Collection, challenges and baselines. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020.

Yufan Deng, Zilin Pan, Hongyu Zhang, Xiaojie Li, Ruoqing Hu, Yufei Ding, Yiming Zou, Yan Zeng, and Daquan Zhou. Rethinking video generation model for the embodied world. In International Conference on Machine Learning, 2026.

Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, Shirui Chen, Yi Ru Wang, et al. MolmoAct2: Action reasoning models for real-world deployment. arXiv preprint arXiv:2605.02881, 2026.

Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. LIBERO-Plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, and Chuang Gan. AdaWorld: Learning adaptable world models with latent actions. arXiv preprint arXiv:2503.18938, 2025.

Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, et al. DreamDojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949, 2026.

Quentin Garrido, Tushar Nagarajan, Basile Terver, Nicolas Ballas, Yann LeCun, and Michael Rabbat. Learning latent action world models in the wild. In International Conference on Machine Learning, 2026.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The “something something” video database for learning and evaluating visual common sense. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5842–5850, 2017.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Ryan Hoque, Peide Huang, David J Yoon, Mouli Sivapurapu, and Jian Zhang. EgoDex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

Chengkai Hou, Kun Wu, Jiaming Liu, Zhengping Che, Di Wu, Fei Liao, Guangrun Li, Jingyang He, Qiuxuan Feng, Zhao Jin, et al. RoboMIND 2.0: A multimodal, bimanual mobile manipulation dataset for generalizable embodied intelligence. arXiv preprint arXiv:2512.24653, 2025.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and g0 dual-system VLA model. arXiv preprint arXiv:2509.00576, 2025.

<!-- page 22 of 29 -->

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026.

Black Forest Labs. FLUX.2: Frontier visual intelligence. [https://bfl.ai/blog/flux-2,](https://bfl.ai/blog/flux-2) 2025.

Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, et al. Behavior-1k: A benchmark for embodied AI with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning, pages 80–93, 2023.

Kehan Li, Bohan Hou, Minghao Zhu, Tianyi Zhang, Zesen Cheng, Zhikai Wang, Sicong Leng, Xin Li, Xiao Lin, Biying Yao, et al. RynnBrain 1.1: Towards more capable and generalizable embodied foundation model. arXiv preprint arXiv:2607.17977, 2026a.

Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026b.

Qixiu Li, Yu Deng, Yaobo Liang, Lin Luo, Lei Zhou, Chengtang Yao, Lingqi Zeng, Zhiyuan Feng, Huizhi Liang, Sicheng Xu, Yizhong Zhang, Xi Chen, Hao Chen, Lily Sun, Dong Chen, Jiaolong Yang, and Baining Guo. Scalable vision-language-action model pretraining for robotic manipulation with real-life human activity videos. arXiv preprint arXiv:2510.21571, 2025.

Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, et al. Mixture-of-Transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.

Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems, pages 44776–44791, 2023.

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. RDT-1B: A diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, pages 29982–30009, 2025.

Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 2022.

Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. HowTo100M: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2630–2640, 2019.

Henrique Morimitsu, Xiaobin Zhu, Roberto M Cesar, Xiangyang Ji, and Xu-Cheng Yin. DPFlow: Adaptive optical flow estimation with a dual-pyramid framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17810–17820, 2025.

Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. RoboCasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024.

<!-- page 23 of 29 -->

Dujun Nie, Fengjiao Chen, Qi Lv, Jun Kuang, Xiaoyu Li, Xuezhi Cao, and Xunliang Cai. LARY: A latent action representation yielding benchmark for generalizable vision-to-action alignment. In European Conference on Computer Vision, 2026.

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and RT-X models: Open x-embodiment collaboration 0. In IEEE International Conference on Robotics and Automation, pages 6892–6903, 2024.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. FAST: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

Ryan Punamiya, Simar Kareer, Zeyi Liu, Josh Citron, Ri-Zhao Qiu, Xiongyi Cai, Alexey Gavryushin, Jiaqi Chen, Davide Liconti, Lawrence Y Zhu, et al. EgoVerse: An egocentric human dataset for robot learning from around the world. arXiv preprint arXiv:2604.07607, 2026.

Dominik Schmidt and Minqi Jiang. Learning to act without actions. In International Conference on Learning Representations, pages 9379–9395, 2024.

Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, et al. InternData-A1: Pioneering high-fidelity synthetic data for pre-training generalist policy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 976–985, 2026.

Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. BridgeData v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736, 2023.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Xiaofeng Wang, Kang Zhao, Feng Liu, Jiayu Wang, Guosheng Zhao, Xiaoyi Bao, Zheng Zhu, and Yingya Zhang. EgoVid-5M: A large-scale video-action dataset for egocentric videos generation. In Advances in Neural Information Processing Systems, 2025.

Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, et al. RoboMIND: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877, 2024.

Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, et al. RoboCOIN: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025.

Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, et al. A pragmatic VLA foundation model. arXiv preprint arXiv:2601.18692, 2026.

Adina Yakefu, Bin Xie, Chongyang Xu, Enwen Zhang, Erjin Zhou, Fan Jia, Haitao Yang, Haoqiang Fan, Haowei Zhang, Hongyang Peng, et al. RoboChallenge: Large-scale real-robot evaluation of embodied policies. arXiv preprint arXiv:2510.17950, 2025.

Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, et al. ABot-M0: VLA foundation model for robotic manipulation with action manifold learning. arXiv preprint arXiv:2602.11236, 2026.

<!-- page 24 of 29 -->

Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Se June Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. In International Conference on Learning Representations, pages 28213–28239, 2025.

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

Haoqi Yuan, Zhixuan Liang, Anzhe Chen, Ye Wang, Haoyang Li, Pei Lin, Yiyang Huang, Zixing Lei, Tong Zhang, Jiazhao Zhang, et al. Qwen-RobotManip technical report: Alignment unlocks scale for robotic manipulation foundation models. arXiv preprint arXiv:2606.17846, 2026a.

Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026b.

Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, Yu-Gang Jiang, et al. VLABench: A large-scale benchmark for language-conditioned robotics manipulation with long-horizon reasoning tasks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11142–11152, 2025.

Haoyu Zhao, Hao Wang, Xingyue Zhao, Hao Fei, Hongqiu Wang, Chengjiang Long, and Hua Zou. Physsplat: Efficient physics simulation for 3d scenes via mllm-guided gaussian splatting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5242–5252, 2025.

Haoyu Zhao, Cheng Zeng, Linghao Zhuang, Yaxi Zhao, Shengke Xue, Hao Wang, Xingyue Zhao, Zhongyu Li, Kehan Li, Siteng Huang, et al. High-fidelity simulated data generation for real-world zero-shot robotic manipulation learning with gaussian splatting. IEEE Robotics and Automation Letters, 2026a.

Haoyu Zhao, Xingyue Zhao, Siteng Huang, Xin Li, Deli Zhao, and Zhongyu Li. Rynnworld-4d: 4d embodied world models for robotic manipulation. arXiv preprint arXiv:2607.06559, 2026b.

Haoyu Zhao, Xingyue Zhao, Hangyu Li, Biao Gong, Kehan Li, Siteng Huang, Xin Li, Deli Zhao, and Zhongyu Li. Rynnworld-teleop: An action-conditioned world model for digital teleoperation. arXiv preprint arXiv:2607.06558, 2026c.

Haoyu Zhao, Linghao Zhuang, Xingyue Zhao, Cheng Zeng, Haoran Xu, Yuming Jiang, Jun Cen, Kexiang Wang, Jiayan Guo, Siteng Huang, et al. Towards affordance-aware robotic dexterous grasping with human-like priors. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 13126–13134, 2026d.

Jinliang Zheng, Jianxiong Li, Dongxiu Liu, Yinan Zheng, Zhihao Wang, Zhonghong Ou, Yu Liu, Jingjing Liu, Ya-Qin Zhang, and Xianyuan Zhan. Universal actions for enhanced embodied foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22508–22519, 2025.

Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. In International Conference on Learning Representations, pages 60580–60606, 2026.

Linqing Zhong, Yi Liu, Yifei Wei, Ziyu Xiong, Si Liu, and Guanghui Ren. ACoT-VLA: Action chain-of-thought for vision-language-action models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8152–8162, 2026.

<!-- page 25 of 29 -->

## Appendix

## A Benchmark Details

## A.1 Latent Action Representation Quality

We first evaluate the latent action representation itself. Rather than judging latent actions by downstream manipulation success, which entangles representation quality with the rest of the VLA stack, the LARY benchmark (Nie et al., 2026) probes the frozen representation directly. We adopt its low-level control regression track, which decodes end-effector trajectories from a frozen representation through a lightweight MLP action expert, scored by MSE across CALVIN (Mees et al., 2022) and VLABench (Zhang et al., 2025) (7-DoF Franka), RoboCOIN (Wu et al., 2025) (12-DoF bimanual) and AgiBotWorld-Beta (Bu et al., 2025a) (16-DoF AgiBot G1); we leave aside its semantic classification track, which scores category separability rather than physical grounding. The official targets are absolute end-effector states in a fixed base frame. At the short stride (s=5) used to form each image pair, the currently visible pose can help predict these targets. Absolute-target performance may therefore reflect static pose information as well as transition information. We therefore also report a relative setting that re-expresses the same targets as displacements from the end-effector state at the first frame, leaving the probe, splits, stride and metric untouched, so that the probe emphasizes displacement prediction rather than absolute pose reconstruction. We compare against three of the paradigms of (Nie et al., 2026): general semantic encoders (V-JEPA 2 (Assran et al., 2025), DINOv3 (Siméoni et al., 2025)), generative pixel encoders (Wan2.2 (Wan et al., 2025), FLUX.2-dev (Labs, 2025)), and Embodied LAMs (LAPA (Ye et al., 2025), UniVLA (Bu et al., 2025b), villa-X (Chen et al., 2026)), plus Moto (Chen et al., 2025c) and DreamDojo (Gao et al., 2026) evaluated under the identical protocol.

**LIBERO and LIBERO-Plus.** LIBERO (Liu et al., 2023) is a language-conditioned tabletop benchmark on a Franka Emika Panda. Its four suites—Spatial, Object, Goal, Long, 10 tasks each with 50 human-teleoperated demonstrations—isolate spatial grounding, target disambiguation, goal following, and long-horizon chaining. We fine-tune a single policy jointly across all 40 tasks and evaluate each suite over 500 rollouts (50 per task). LIBERO-Plus (Fei et al., 2025) reuses the same 40 tasks as a robustness diagnostic, perturbing seven factors—layout, viewpoint, robot initial state, language, lighting, background, and sensor noise—over 10,030 curated episodes. We evaluate the LIBERO-finetuned checkpoint without additional LIBERO-Plus adaptation to measure robustness under these perturbations.

**VLABench.** VLABench (Zhang et al., 2025) returns to a fixed Franka Panda but shifts the difficulty from control to grounding: task categories spanning primitive and composite interactions, built in MuJoCo over 2,164 objects in 163 categories, with instructions written to convey intent implicitly rather than by template. We fine-tune on the benchmark’s official primitive-task split and evaluate on the fixed episode sets of the benchmark’s public tracks, which hold the tasks constant while varying what the policy must resolve to locate the target: in-distribution, cross-category, common-sense, semantic-instruction, and unseen-texture. We omit the benchmark’s cross-task track, whose train/test partition is left to the user and whose numbers are therefore not comparable across papers.

**RoboTwin 2.0.** RoboTwin 2.0 (Chen et al., 2025a) contains 50 bimanual manipulation tasks, including handovers, cooperative lifting, and stabilization followed by manipulation. The benchmark provides 731 object instances in 147 categories and supports multiple dual-arm embodiments. We use ALOHA-AgileX and the Native protocol, with 50 clean and 500 randomized demonstrations per task (27,500 in total). A single policy is trained jointly across all 50 tasks and evaluated in clean and randomized scenes. Randomization covers clutter, lighting, background, tabletop height, and language. Because randomized demonstrations are included during training, the two evaluation domains measure performance within this training protocol rather than zero-shot transfer from clean training to randomized scenes. Results for this protocol are reported in Tab. 6.

## B Teleoperation Setup

For Marvin-WUJI, we collect demonstrations through the teleoperation system shown in Fig. 10. The hardware setup consists of dual Tianji 7-DOF robotic arms and dual Wuji 20-DOF dexterous hands, yielding a total of

<!-- page 26 of 29 -->

![Image block](doc:4137b11/tier:basic/page:26/block:1)

Figure 10 Marvin-WUJI demonstration collection. Motion trackers control the dual robot arms, and data gloves control the dexterous hands.

54 degrees of freedom.

For arm control, the human operator wears five HTC Vive trackers (mounted on the chest, both wrists, and both upper arms). The system computes wrist-to-chest relative transforms at 100-120 Hz and feeds them into a Pinocchio-based inverse kinematics solver running in a separate process. The resulting joint commands are further smoothed by a Ruckig trajectory generator with velocity, acceleration, and jerk constraints before being sent to the robot arms at 200 Hz.

For hand control, the operator wears Manus data gloves. The raw glove signals are converted to a 21-point MediaPipe hand skeleton format and retargeted to the 20-DOF Wuji hand joint space via a dedicated retargeting module, with an exponential moving average filter applied for motion smoothing.

## C Details of Action-Alignment Baselines

To compare our latent-action-based cross-embodiment alignment with conventional action-alignment strategies, we implement two baselines commonly used in VLA pretraining.

## C.1 Shared Vector Representation

Following prior VLA research (Yuan et al., 2026a; Intelligence et al., 2025), we map heterogeneous state and action spaces from different data sources and embodiments into a unified 85-dimensional canonical vector representation. The layout is designed for dual-arm embodiments. For single-arm robots, only the first half of each bilateral slot group is populated. The vector is organized into the following semantic slots:

1. **Joint Position (0–13):** dual-arm joint positions, with 7 dimensions allocated to each arm.

2. **EEF Pose (14–31):** Cartesian end-effector position (3 dimensions) and orientation represented using a continuous 6-D rotation representation (6 dimensions) for each arm.

3. **Gripper Position (32–33):** one gripper-position dimension for each arm.

4. **Dexterous Hand (34–73):** 20 dimensions reserved for each dexterous hand.

5. **Torso (74–77):** 4 dimensions for torso degrees of freedom.

6. **Head (78–80):** 3 dimensions for head and neck motion.

<!-- page 27 of 29 -->

7. **Base (81–84):** 4 dimensions reserved for mobile-base degrees of freedom.

Different embodiments populate different subsets of this unified vector according to their available degrees of freedom. Unused dimensions are excluded from the training loss through a per-dimension binary mask, preventing spurious supervision on structurally absent action dimensions.

## C.2 Domain-specific Projection

Following prior work (Agarwal et al., 2026; Zheng et al., 2026), this variant retains a shared action expert but replaces the common input and output projections with a bank of domain-specific projections. Each physical robot embodiment is assigned a domain identifier, which selects the corresponding projection parameters for that sample. In this design, the main policy remains shared across embodiments, while the lightweight boundary projections capture embodiment-specific differences in action parameterization and distribution.

## C.3 Human Egocentric Datasets

For human egocentric datasets, we treat human motion as a separate embodiment type when using domainspecific projections. These videos carry no native control labels; wrist poses and the positions of five fingertips are mapped to the corresponding slots of the unified vector.

## C.4 Delta Actions and Normalization

During training, all action slots use a delta-action representation: each action in an action chunk is expressed relative to the robot’s current state. For normalization, the 6-D end-effector rotation slots are left unchanged. All other action slots are normalized using the q<sub>0</sub>.01 and q<sub>0</sub>.99 quantiles computed from the training data.

## D RynnVLA-Latent Dataset

**Video captioning.** We use Qwen3-VL (Bai et al., 2025) to generate captions for the video data. Specifically, we leverage the model’s strong video-language understanding capabilities to produce detailed, structured descriptions of each video clip. The videos are first sampled at a frame rate of 1 FPS and split into segments <u>of 5 seconds. For each segment, we provide the following prompt to the model:</u>

Please describe this video in detail. Include the following aspects:

1. The main subject and action in the video.

2. The environment and background.

3. Any objects and their interactions.

4. The overall scene context and atmosphere.

Provide a concise but comprehensive caption in one paragraph.

The generation is performed with a maximum output length of 512 tokens and a temperature of 0.7 to balance creativity and coherence. The generated captions are then collected and stored in JSON format for downstream tasks.

**Optical flow annotation.** We employ DPFlow (Morimitsu et al., 2025), a state-of-the-art optical flow estimation model. For each video, frame pairs are processed sequentially at native resolution, and the estimated flow fields are visualized via color encoding and saved as MP4 videos at 25 FPS.

**Depth annotation.** We employ Depth Anything 3 (Lin et al., 2025), specifically the DA3NESTED-GIANT-LARGE-1.1 checkpoint, which provides dense per-frame depth predictions along with camera pose estimation. Each video is sampled at 30 FPS and processed at a working resolution of 392 pixels (short side, upper-bound resize).

<!-- page 28 of 29 -->

## E Extended Ablations of RynnLAM

To complement the main ablations, we further evaluate RynnLAM under the relative low-level control protocol of the LARY benchmark (Nie et al., 2026). We focus on the relative action setting because it reduces reliance on reconstructing the currently visible absolute end-effector pose, and thus more directly reflects whether the frozen representation encodes motion rather than static scene appearance. We report the relative setting for these extended variants; absolute-action results are not included in this table. Unless otherwise stated, all retrained variants use the same data, budget, and probe protocol as the main table. The latent-composition rows are probe-only: they reuse the same released checkpoint and only change the slice of the latent passed to the probe.

Table 10 Extended ablations on the LARY benchmark (Nie et al., 2026) (Relative Action, MSE, lower is better). All retrained variants use the same data, budget, and probe protocol as the main table. Avg. is the arithmetic mean of the displayed per-dataset MSE values, rounded to two decimal places.

| Method | Avg. | CALVIN | VLABench | RoboCOIN | AgiBotWorld-B |
| --- | --- | --- | --- | --- | --- |
| Training &amp; Objective Variants |  |  |  |  |  |
| RynnLAM w/ 2D flow | 0.56 | 0.44 | 0.44 | 0.69 | 0.67 |
| RynnLAM w/o contrastive feat. | 0.57 | 0.44 | 0.45 | 0.70 | 0.68 |
| RynnLAM w/o feat. prediction | 0.63 | 0.49 | 0.50 | 0.78 | 0.74 |
| RynnLAM w/ z<sup>m</sup> from latent tokens | 0.79 | 0.64 | 0.66 | 0.94 | 0.92 |
| RynnLAM | 0.38 | 0.40 | 0.05 | 0.42 | 0.64 |
| Latent Composition (probe-only) |  |  |  |  |  |
| z<sup>a</sup> only (64d) | 0.93 | 0.82 | 0.96 | 0.97 | 0.95 |
| z<sup>c</sup> only (32d) | 0.95 | 0.88 | 0.97 | 0.98 | 0.96 |
| z<sup>m</sup> only (512d) | 0.54 | 0.42 | 0.42 | 0.66 | 0.65 |

Tab. 10 reports the results. The first block studies training and objective variants. **RynnLAM w/ 2D flow** replaces the 3D scene-flow decoder with a 2D appearance-flow decoder; together with the w/o 3D flow variant in the main table, it compares no-flow, 2D-flow, and 3D-flow supervision. The full 3D-flow model has lower average error than either variant; adding 2D-flow supervision does not improve over the no-flow variant, so an additional motion-prediction objective alone does not explain the gain. **RynnLAM w/o contrastive feat.** removes the anti-shortcut contrastive feature objective and falls back to the plain motion-weighted cosine loss, testing the utility of contrasting warped features against unchanged source features. **RynnLAM w/o feat. prediction** disables the future feature-prediction branch entirely, measuring how much of the representation quality is driven by next-frame feature prediction versus the latent-specific objectives. **RynnLAM w/** $z ^ { m }$ **from** latent tokens compresses $z ^ { m }$ from the latent tokens instead of the raw target-frame features, removing the direct feature-space pathway into the bottleneck; it degrades the average MSE substantially, supporting the direct target-feature pathway for preserving information useful to the control-regression probe.

The second block probes $z ^ { a } ,   z ^ { c } ,$ and $z ^ { m }$ separately using the same checkpoint. Motion tokens yield the lowest average MSE among the individual components (0.54), but remain worse than the full descriptor (0.38). Thus $z ^ { m }$ carries substantial control-relevant information, while the combined descriptor gives the best average result. The global action and camera latents alone achieve 0.93 and 0.95, respectively, and are less predictive under this probe. These slice comparisons establish differences in probe utility; they do not by themselves prove a complete semantic disentanglement of the representation.

Together, these studies support the complete objective and the direct target-feature pathway for control regression. The component probes additionally show that the full descriptor is more predictive than any individual slice. These findings explain the choice of descriptor used for policy pretraining, while downstream evaluations separately assess its utility after adaptation to native robot controls.

## F Real-World Data Collection Details

Operator motion tracking provides the control interface for collecting the demonstrations. For policy post-training, recorded robot observations and native actions are aligned in time, and action chunks are expressed

<!-- page 29 of 29 -->

Table 11 Statistics of the real-world dataset used for RynnVLA-Latent training.

| Task Name | Description | Episodes |
| --- | --- | --- |
| Plate-Placement | Take the plate from the shelf and place it to the left of the oven. | 500 |
| Steak-Cooking | Flip the steak in the pan to fry it. | 800 |
| Plate-Wiping | Take the dirty plate from the shelf, wipe it with a cloth, and return it to the same shelf. | 500 |
| Table-Cleaning | Collect all scattered items on the table into the box, then place the box onto the shelf. | 850 |
| Steak-Serving | Carry the steak plate horizontally to the designated dining spot and place it gently on the placemat. | 600 |
| Plate-Retrieval | Collect the dirty plate and the knife and fork from the dining table, and return them to their original positions. | 600 |
| Total |  | 3,850 |

in the target robot’s control space. Operator skeletons are not the latent-action targets used during policy pretraining: those targets are extracted from video by RynnLAM.

In total, we collected 3,850 high-quality demonstration episodes. The corpus covers all six evaluated tasks; these counts describe the collected demonstrations across the two platforms. The specific tasks and their respective data distributions are detailed in Tab. 11.