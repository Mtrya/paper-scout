# Xiaomi-Robotics-U0: Unified Embodied Synthesis
with World Foundation Model

**Authors:** Xiaomi Robotics

**arXiv:** 2607.11643


## 1
Introduction

Figure 1
:
Embodied and general capabilities of
Xiaomi-Robotics-U0
. The rectangle corresponds to the initial observations for the same embodiment, the pairwise transfer sample, and the keyframes within a video for the embodied capabilities. All frames are referenced and generated images.
Figure 2
:
Overview of all tasks in
Xiaomi-Robotics-U0
, Gray: inputs; Orange: outputs.
Recent foundation image and video generation models
[
podell2024sdxl
,
labs2025flux1kontextflowmatching
,
openai_gptimage_2025
,
blattmann2023stable
,
cui2025emu3
,
xie2025show
,
wu2025janus
,
chen2025janus
,
team2024chameleon
]
have made remarkable progress in semantic understanding, controllable generation, and visual reasoning through training with data on the Internet. Large-scale generative models are now capable of synthesizing highly realistic images and videos from various multimodal inputs, demonstrating impressive generalization far beyond the distribution of their training data. Such capabilities make foundation generative models an attractive starting point for embodied intelligence
[
yuan2026fast
,
ye2026world
,
shang2026roboscape
,
long2026scaling
]
, where robots are required to reason about complex environments and imagine future interactions before acting.
However, embodied generation
[
long2025survey
,
liu2025aligning
,
li2025comprehensive
]
introduces challenges that differ fundamentally from conventional image and video synthesis. Unlike natural image generation, embodied scenarios require strict multi-view consistency, accurate geometric and physical coherence across cameras, explicit robot embodiment constraints, and temporally consistent interaction dynamics. The generated observations must remain compatible with robot kinematics, camera calibration, and downstream manipulation policies rather than merely appearing visually realistic. Consequently, directly applying existing foundation image or video generation models to embodied scenarios often leads to inconsistent geometry, implausible robot states, and poor compatibility with robot control.
Recent embodied world models
[
agarwal2026cosmos
,
zhang2026qwen
,
shang2026roboscape
]
attempt to bridge this gap by continually adapting pre-trained foundation models using robot trajectories
[
Neill2023open_x_embodiment
,
tian2026interndata
,
yin2026geniesim30
,
jiang2025galaxea
,
walke2023bridgedata
,
brohan2022rt
]
or egocentric manipulation videos
[
akhtyamov2025egowalk
,
grauman2022ego4d
]
. While these approaches have demonstrated encouraging progress, their adaptation paradigm remains fundamentally different from that of foundation model pre-training. Instead of preserving the diverse image and video generation tasks that originally endow foundation models with strong semantic understanding and controllable generation, post-training is typically performed solely on robot-specific datasets. Since embodied datasets are considerably smaller, less diverse and highly repetitive compared with internet-scale visual corpora
[
gadre2023datacomp
,
schuhmann2022laion
,
kakaobrain2022coyo-700m
]
, such adaptation inevitably weakens the generalization capability inherited from foundation models and limits the diversity, controllability, and scalability of the embodied generation.
In this work, we present
Xiaomi-Robotics-U0
, a Unified Embodied Synthesis Model that unifies foundation generation and embodied generation within a single training paradigm. As shown in Figure
1
, rather than specializing a foundation model into a robot-specific generator,
Xiaomi-Robotics-U0
reformulates embodied synthesis as a natural extension of foundation image and video generation. Starting from a world foundation model,
Xiaomi-Robotics-U0
is continually trained on both general-domain and embodied datasets under a unified autoregressive objective, jointly learning text-to-image generation, image editing, embodied scene generation, embodied transfer, and embodied video generation.
Figure
2
exhibits the above-mentioned capabilities of
Xiaomi-Robotics-U0
.
This unified formulation preserves the rich semantic knowledge and controllable generation capability of foundation models while effectively adapting them to robot-centric multi-view reasoning and interaction modeling.
Beyond single-step generation,
Xiaomi-Robotics-U0
further extends the embodied synthesis to sequential world modeling. By jointly learning interleaved embodied sequences and manipulation videos across multiple temporal resolutions, the model captures both long-horizon task progression and fine-grained interaction dynamics. More importantly, this enables embodied generation to evolve from static scene synthesis into a scalable trajectory generation engine: generated embodied scenes can be directly rolled out into temporally coherent manipulation videos, providing abundant synthetic trajectories for downstream policy learning. We believe that this capability represents an important step toward agentic embodied data generation, where foundation world models continuously generate increasingly diverse robot experiences through autonomous interaction rollout.
Extensive experiments demonstrate that
Xiaomi-Robotics-U0
establishes a new state of the art across both single-step and sequential embodied generation tasks. It is the first unified embodied synthesis model capable of high-quality multi-view embodied scene generation, structured controllable embodied transfer, and zero-shot embodied video generation within a single autoregressive framework.
Xiaomi-Robotics-U0
outperforms GPT-Image-2.0 in human evaluation on embodied scene generation and embodied transfer, achieves state-of-the-art performance on the World Arena benchmark, and significantly improves the out-of-distribution robustness of downstream robot policies using generated data. These results suggest that world foundation models can serve not only as embodied world models, but also as scalable data engines for embodied intelligence.
Our contributions are summarized as follows:
•
We introduce
Xiaomi-Robotics-U0
, the first unified embodied synthesis model that seamlessly integrates foundation image generation, image editing, embodied scene generation, embodied transfer, and embodied video generation within a single autoregressive framework. By jointly optimizing general-domain and embodied generation tasks under a unified continual training paradigm,
Xiaomi-Robotics-U0
effectively transfers the generalization capability of foundation world models to embodied generation while preserving their original visual generation ability.
•
We propose the first unified framework for
multi-view embodied generation
, including embodied scene generation and controllable embodied transfer. By jointly training with large-scale text-to-image and image editing data,
Xiaomi-Robotics-U0
achieves state-of-the-art performance with superior multi-view consistency, fine-grained controllability, and instruction following, outperforming GPT-Image-2.0 in human evaluations. Furthermore, we introduce a structured control formulation that disentangles workspace, background, foreground irrelevant objects, target objects, and lighting into independent control dimensions, enabling scalable embodied video augmentation while preserving geometric consistency and interaction dynamics. The generated data substantially improves the robustness of downstream robot policies on challenging real-world manipulation tasks.
•
We extend embodied generation from static scene synthesis to
sequential embodied world modeling
through interleaved subtask-subgoal learning and multi-FPS embodied video generation.
Xiaomi-Robotics-U0
achieves state-of-the-art performance on the World Arena benchmark and demonstrates strong zero-shot multi-view embodied video generation capability. By enabling coherent long-horizon rollout from synthesized embodied scenes,
Xiaomi-Robotics-U0
transforms foundation world models into scalable embodied data engines, providing a practical path toward agentic generation of diverse embodied interaction trajectories for future policy learning.

## 2
Xiaomi-Robotics-U0

Our goal is to develop a unified embodied world model that supports diverse generation and prediction tasks under a single autoregressive framework. Given a multi-modal context
𝒞
=
{
x
1
,
x
2
,
⋯
,
x
n
}
,
\mathcal{C}=\{x_{1},x_{2},\cdots,x_{n}\},
(1)
where each element can be a text token, an image token, or a robot-related control token, the model learns the joint distribution over the subsequent multi-modal sequence
P
​
(
𝒴
|
𝒞
)
=
∏
t
=
1
T
P
​
(
y
t
|
y
<
t
,
𝒞
)
,
P(\mathcal{Y}|\mathcal{C})=\prod_{t=1}^{T}P(y_{t}|y_{<t},\mathcal{C}),
(2)
using standard next-token prediction (NTP).
To support embodied intelligence,
Xiaomi-Robotics-U0
is trained on several complementary data formats with different temporal structures. Specifically, we categorize all training samples into two groups.
Single-step samples
describe a one-shot mapping between inputs and outputs without explicit temporal evolution. These samples include:
(i)
Text-to-Image (T2I)
, where the model generates an image conditioned on textual descriptions;
(ii)
Any-to-Image (X2I)
, where one to three reference images together with text are used for image editing and generation;
(iii)
Embodied Scene Generation
, where the model generates the initial multi-view robot observations conditioned on robot embodiment and scene descriptions; and
(iv)
Embodied Transfer
, where the model predicts transferred multi-view observations given the current embodied observations and a target scene description.
Sequential samples
model long-horizon interactions and future evolution. We consider two representative formats:
(i) image-text interleaved subtask-subgoal sequences, where each subtask description is followed by the corresponding multi-view observations after completion; and
(ii) embodied manipulation videos collected at multiple frame rates, enabling the model to capture both sparse task-level transitions and dense interaction dynamics.
Despite their different semantics, all tasks are represented as unified multi-modal token sequences and optimized under the same autoregressive objective, allowing knowledge to be naturally shared across image generation, scene understanding, embodied reasoning, and future prediction.
Figure 3
:
Overview structure of
Xiaomi-Robotics-U0
. The model is trained end-to-end at scale with a unified next-token prediction objective. An efficient inference extension is provided along.
Figure
3
illustrates the overall architecture of
Xiaomi-Robotics-U0
. We adopt the open-sourced EMU3.5
[
cui2025emu3
]
as the initialization model, which is built upon the Qwen-3-32B decoder-only Transformer and has demonstrated strong capabilities in image generation and image-text interleaved modeling. Images are first tokenized by the IBQ Tokenizer
[
shi2025scalable
]
with a spatial compression ratio of
16
×
16
16\times 16
. By extending the original Qwen vocabulary with the IBQ codebook, all modalities are represented within a unified discrete vocabulary and modeled using next-token prediction.
Compared with conventional embodied world models that introduce task-specific prediction heads,
Xiaomi-Robotics-U0
formulates all generation tasks—including image generation, embodied scene synthesis, scene transfer, sequential subtask prediction, and video generation—as a single autoregressive sequence modeling problem. This unified formulation enables efficient multi-task co-training while preserving the strong visual generation capabilities inherited from the large-scale pre-trained model.
To improve inference efficiency, we further adopt an inference-accelerating adaptation inspired by FlashAR
[
zhou2026flasharefficientposttrainingacceleration
]
. Specifically, an additional vertical prediction head is introduced to decode image tokens in an anti-diagonal order, allowing multiple visual tokens to be generated simultaneously. Combined with optimized KV-cache management and continuous batching provided by vLLM
[
kwon2023efficient
]
,
Xiaomi-Robotics-U0
achieves up to
82.9
×
82.9\times
faster image generation at 1024
×
\times
1024 resolution compared with the original NTP pattern, while preserving most generation quality.
Figure 4
:
The illustration of the constructed single-step generation dataset. Except for the embodied transfer and the embodied scene generation data, we additionally introduce Text-to-Image(T2I) and Any-to-Image(X2I) data for co-training. Concretely speaking, we add camera-control and time-control data to improve the spatial and temporal understanding capability. For structure and geometry perception, we construct skeleton, depth, edge, and semantic mask datasets with advanced computer vision tools. To maintain the ability of the pre-trained model, we introduce general T2I and X2I datasets for co-training.
To endow
Xiaomi-Robotics-U0
with broad visual understanding and generation capabilities, we assemble a diverse training corpus spanning six distinct domains, each contributing complementary visual and semantic priors.
•
General image-text
data provides foundational visual-semantic alignment. We include ShareGPT4V
[
chen2024sharegpt4v
]
and similar large-scale vision-language benchmarks to establish a strong baseline for text-conditioned generation.
•
Embodied manipulation
data bridges the gap between passive visual understanding and active physical interaction. This domain combines both real-world and simulated sources. On the real side, we leverage AgiBotWorld-Beta
[
bu2025agibot
]
, Open X-Embodiment
[
Neill2023open_x_embodiment
]
, and a proprietary multi-view robot manipulation dataset collected in house, which we refer to as MiBot. On the simulation side, we incorporate data from RoboTwin2.0
[
chen2025robotwin
]
, GenieSim
[
yin2026geniesim30
]
, and InternData-A1
[
tian2026interndata
]
, enabling coverage of scenarios that are difficult or dangerous to collect physically.
•
Autonomous driving
data captures structured outdoor scenes with complex spatial layouts and dynamic objects. We source this from Cosmos-Drive-Dreams
[
ren2025cosmosdrive
]
, which provides high-quality driving scenarios with diverse weather, lighting, and traffic conditions.
•
Egocentric
data, drawn from EgoWalk
[
akhtyamov2025egowalk
]
, supplies first-person perspective recordings that reflect natural human viewpoint and motion patterns, enriching the model’s understanding of human-centric visual experience.
•
3D reconstruction
data offers dense multi-view correspondences and accurate geometry. We utilize ScanNet++ v2
[
yeshwanth2023scannet++
]
and DL3DV-10K
[
ling2024dl3dv
]
, both of which provide high-quality indoor and outdoor 3D scene captures.
•
Game
data, sourced from Open-P2P
[
yue2026openp2p
]
, introduces stylized and procedurally generated environments that expand the diversity of visual distributions beyond real-world captures.
Raw data collected at scale inevitably contains artifacts that can impair generation quality. Inspired by Abot Physical World
[
chen2026abot
]
, we implement an autonomous data filtering pipeline including motion blur detection, image quality detection, and gripper-state alignment.
To sum up, we collect 9.5 million samples for single-step generation, which corresponds to 56.4 billion tokens. For sequential generation, we select 2.6 million video clips, which corresponds to 49.6 billion tokens.
Most raw samples lack fine-grained text annotations and only carry coarse task-level descriptions, which cannot support precise text-driven embodied generation. We thus construct a unified annotation pipeline to standardize supervision signals across heterogeneous data.
We leverage Qwen3-VL-235B
[
bai2025qwen3
]
, a powerful vision-language model (VLM), to generate dense, structured annotations for generic frames, multi-source video sequences, and robotic manipulation trajectories. The pipeline consists of four decoupled branches with distinct annotation objectives:
•
General Scene Annotation
All video data spanning general visual corpora and embodied robot sequences are processed without domain restrictions. Each video clip undergoes temporal downsampling to select high-clarity keyframes. The VLM produces holistic captions describing overall scene layouts, object interactions, and cross-object spatial relationships for every sampled frame.
•
Structured Embodied Scene Annotation
Annotation is only performed on embodied manipulation data. Decomposed subtask outputs from trajectory processing guide scene partitioning into five orthogonal semantic dimensions: workspace, task objects, irrelevant objects, lighting, and background. Separate descriptions are generated for each dimension according to task relevance, supporting compositional reasoning and fine-grained controllable generation during inference.
•
Geometric Modality Extraction
Temporally consistent inverse depth maps are extracted for all frames and videos via Video Depth Anything
[
chen2025videodepthanything
]
to supply geometric conditioning signals for cross-domain embodied transfer tasks. Brighter pixels correspond to closer surfaces and darker pixels to distant regions within this depth representation, allowing the model to synthesize photorealistic RGB outputs conditioned on geometric priors.
•
Trajectory-level Subtask Decomposition
As illustrated in Figure
5
, HDBSCAN clustering
[
McInnes2017hdbscanHD
]
segments robot trajectories using end-effector poses and gripper signals to identify motion and grasp transition boundaries. Keyframes with dense physical interactions are sampled at these boundaries and sent to VLM to generate temporally aligned subtask plans with frame boundaries.
Transition frames with rich physical interactions are processed with scene annotation to produce training samples for embodied scene generation and embodied transfer tasks.
To support the three training objectives of
Xiaomi-Robotics-U0
, we reorganize all annotated data into task-specific formats:
•
Embodied Transfer.
Each training sample consists of a system instruction, a scene description, and multi-view depth maps as input, with multi-view RGB images as the supervision target. This task trains the model to hallucinate photorealistic appearance from geometric cues.
•
Scene Generation.
Each sample pairs a system instruction with a scene description, supervised by multi-view RGB images. This task learns text-to-multi-view-scene synthesis without geometric priors.
•
Video Generation.
Each sample comprises a system instruction, a task instruction, and an initial observation frame, supervised by a sequence of 15–25 RGB frames. For datasets that provide robot action labels, a sequence of robot mask images rendered from the robot action and URDF is additionally appended after the initial observation frame, serving as action guidance. This task learns to predict future visual states conditioned on task semantics and initial context.
Figure 5
:
Overview of the unified data labeling pipeline.
The upper part illustrates scene description generation, where each sample is annotated with structured semantic attributes.
The lower part shows trajectory-level subtask decomposition, including action-based clip segmentation, key-frame sampling, and subtask generation.
This pipeline provides consistent scene, geometry, and action-level supervision across heterogeneous robotic datasets.
Our training corpus contains heterogeneous multi-domain data, which easily leads to domain bias and overfitting toward dominant data distributions. We adopt sample-level reweighting conditioned on data domains and generation task types (multi-view generation, scene editing, single-view generation), with the core target of maintaining full coverage of robotic manipulation scenarios.
Specifically, we first eliminate invalid embodied episodes, including sequences with corrupted visuals, misaligned timestamps, missing proprioceptive or action signals, failed execution records, and static trajectories lacking robot-object interactions. All remaining samples are categorized via hierarchical annotations derived from subtask and structured scene labeling. Grouping criteria cover task goals, scene contexts, target objects, spatial relations, subtask phases, and atomic manipulation skills.
Such fine-grained grouping supports more precise balancing than simple dataset-level weighting: redundant samples sharing identical embodiment-task-object combinations are downsampled, whereas clips with novel visuals, object layouts, manipulation skills or temporal stages are prioritized. Higher sampling weights are assigned to long-tail categories, including rare objects, seldom-used manipulation skills, and core interaction phases (grasping, releasing, placing, and object state transitions). The resulting training distribution eliminates redundant repeated demonstrations while retaining diverse visual scenes and comprehensive manipulation skill coverage.
To efficiently adapt the pre-trained image generation model into a unified embodied world model while preserving its original visual generation capability, we adapt two continued training setups:
single-step training
and
sequential training
.
This setup focuses on aligning the model with diverse one-step generation tasks while maintaining the original image generation capability. We jointly train four complementary datasets, including Text-to-Image (T2I), Any-to-Image (X2I), Embodied Scene Generation, and Embodied Transfer.
The T2I and X2I datasets continuously reinforce the general visual generation and image editing capabilities inherited from the pre-trained model, preventing catastrophic forgetting during embodied adaptation. Meanwhile, Embodied Scene Generation teaches the model to synthesize physically plausible initial robot observations conditioned on robot embodiments and scene descriptions, while Embodied Transfer enables cross-scene multi-view generation that preserves robot interaction states and camera geometry. Co-training these tasks effectively aligns the visual generation prior with embodied scene understanding under a unified autoregressive objective.
The sequential training extends the model to long-horizon embodied reasoning and interaction modeling.
We first introduce image-text interleaved subtask-subgoal sequences, where each language-described subtask is paired with the corresponding multi-view observations after execution. This data format teaches the model to reason over task decomposition while predicting future embodied observations, further aligning the sequence modeling capability inherited from the base model.
To enhance dynamic interaction modeling, we additionally incorporate large-scale embodied manipulation videos collected from InternData-A1
[
tian2026interndata
]
, GenieSim
[
yin2026geniesim30
]
, AgiBotWorld-Beta
[
bu2025agibot
]
, RoboCOIN
[
wu2025robocoin
]
, and RoboMIND
[
wu2024robomind
]
. Unlike conventional low-frame-rate trajectory datasets, we construct multi-frame-rate video sequences with FPS values of 1, 3, and 5. Sparse sequences encourage long-horizon planning and future prediction, whereas denser sequences enable the model to capture fine-grained manipulation dynamics and continuous interaction processes. Training across multiple temporal resolutions significantly improves the model’s ability to model both high-level task progression and low-level embodied dynamics within a unified autoregressive framework.
To further accelerate inference, we introduce a FlashAR
+
extension to
Xiaomi-Robotics-U0
. The extension keeps the conditional modeling of reference images and other input signals unchanged, while applying parallel autoregressive generation only to the final target image region. This design is intended to preserve the multi-source conditioning capability of
Xiaomi-Robotics-U0
and reduce the serial decoding cost during X2I inference.
FlashAR
[
zhou2026flasharefficientposttrainingacceleration
]
proposes a post-training and inference acceleration paradigm for autoregressive text-to-image generation. Its core idea is to preserve the generation capability of the original model while introducing extra vertical prediction, so that image tokens can be generated in parallel along anti-diagonal groups. This reduces the number of serial decoding steps in the image region. Inspired by FlashAR, we perform FlashAR
+
post-training on top of
Xiaomi-Robotics-U0
, extending it from single-image T2I generation to broader X2I scenarios, and use the resulting model for X2I inference acceleration in
Xiaomi-Robotics-U0
. Figure
6(a)
demonstrates the decoding steps of the FlashAR
+
extension.
In
Xiaomi-Robotics-U0
, X2I covers tasks such as T2I, I2I, and multi-reference image generation. The input conditions may include text instructions and one or more reference images.
(a)
Diagonal step schedule.
(b)
Prefix-conditioned step-causal mask.
Figure 6
:
Xiaomi-Robotics-U0
FlashAR
+
extension for X2I generation. Target image tokens are grouped by diagonal steps, while all input conditions are kept in the prefix and the target image is generated with a step-causal attention mask.
Compared with standard autoregressive generation,
Xiaomi-Robotics-U0
FlashAR
+
Extension introduces the key changes summarized in Table
1
.
Table 1:
Comparison between standard autoregressive generation and the
Xiaomi-Robotics-U0
FlashAR
+
Extension.
Aspect
Standard AR
Xiaomi-Robotics-U0
FlashAR
+
Extension
Effect
Target image generation
Serial token-by-token generation in raster order
Parallel generation by anti-diagonal steps in the target image
Reduces serial decoding steps
Target image attention
Standard one-dimensional causal attention
The conditional prefix is visible; the target image uses two-dimensional step-causal attention internally
Prevents information leakage
Prediction heads
A single AR prediction head
Horizontal head, vertical head, and gated fusion
Supports anti-diagonal generation
Directional auxiliary loss
Standard supervision
Introduces H/V loss constraints
Constrains both directional heads
During training, the FlashAR
+
extension identifies image blocks during data parsing and defines the target image. All preceding content, including text instructions, reference images, control conditions, and the target image header, is treated as the conditional prefix. Parallel autoregressive losses are computed only over the target image region, while reference images serve only as a conditional context. This prevents reference images from being incorrectly supervised as generation targets and ensures that the model learns to generate the target image from multi-source conditions.
The attention constraint is also restricted to the target image region, as shown in figure
6(b)
. The conditional prefix preserves standard autoregressive visibility. Target image tokens can attend to the entire conditional prefix, but within the target image they can only attend to tokens from earlier anti-diagonal steps. For a target grid position
(
r
,
c
)
(r,c)
, its parallel step is defined as
r
+
c
r+c
; tokens within the same step are mutually invisible, which prevents information leakage during parallel generation. Reference images, text tokens, and control tokens are not involved in the anti-diagonal mask construction inside the target image, preserving the semantic integrity of X2I conditions.
This constraint can be expressed as a visibility function within the target image. For a target-image query token
q
=
(
r
q
,
c
q
)
q=(r_{q},c_{q})
and key token
k
=
(
r
k
,
c
k
)
k=(r_{k},c_{k})
,
M
​
(
q
,
k
)
=
1
​
[
s
​
(
r
k
,
c
k
)
<
s
​
(
r
q
,
c
q
)
]
.
M(q,k)={1}\left[s(r_{k},c_{k})<s(r_{q},c_{q})\right].
Xiaomi-Robotics-U0
FlashAR
+
Extension introduces a key adjustment to directional auxiliary supervision. We match the horizontal auxiliary loss weight to the vertical auxiliary loss weight:
ℒ
=
ℒ
fuse
+
0.05
​
ℒ
h
+
0.05
​
ℒ
v
+
0.2
​
ℒ
distill
.
\mathcal{L}=\mathcal{L}_{\mathrm{fuse}}+0.05\mathcal{L}_{h}+0.05\mathcal{L}_{v}+0.2\mathcal{L}_{\mathrm{distill}}.
Here,
ℒ
fuse
\mathcal{L}_{\mathrm{fuse}}
is the target-image prediction loss after H/V gated fusion,
ℒ
h
\mathcal{L}_{h}
and
ℒ
v
\mathcal{L}_{v}
supervise the horizontal and vertical heads respectively, and
ℒ
distill
\mathcal{L}_{\mathrm{distill}}
is the AR distillation term. For a target position
(
r
,
c
)
(r,c)
, let
z
h
​
(
r
,
c
)
z_{h}(r,c)
and
z
v
​
(
r
,
c
)
z_{v}(r,c)
denote the logits from the horizontal and vertical heads, and let
g
​
(
r
,
c
)
∈
[
0
,
1
]
g(r,c)\in[0,1]
denote the gate value. The fused logits are
z
fuse
​
(
r
,
c
)
=
g
​
(
r
,
c
)
​
z
h
​
(
r
,
c
)
+
(
1
−
g
​
(
r
,
c
)
)
​
z
v
​
(
r
,
c
)
.
z_{\mathrm{fuse}}(r,c)=g(r,c)z_{h}(r,c)\;+\;\left(1-g(r,c)\right)z_{v}(r,c).
The corresponding fused loss is
ℒ
fuse
=
−
∑
(
r
,
c
)
∈
𝐘
log
⁡
softmax
​
(
z
fuse
​
(
r
,
c
)
)
y
r
,
c
.
\mathcal{L}_{\mathrm{fuse}}=-\sum_{(r,c)\in\mathbf{Y}}\log\mathrm{softmax}\left(z_{\mathrm{fuse}}(r,c)\right)_{y_{r,c}}.
The directional auxiliary losses
ℒ
h
\mathcal{L}_{h}
and
ℒ
v
\mathcal{L}_{v}
use the same target tokens to supervise
z
h
z_{h}
and
z
v
z_{v}
, respectively. The H head also needs to adapt to the target-region distribution. If
ℒ
h
=
0
\mathcal{L}_{h}=0
, the H head may fail to converge sufficiently, and the H/V gate may collapse toward the H branch in later training stages. When the gate collapses to an under-trained H head, the fused prediction degenerates into an unstable single-direction prediction, and further training can lead to degraded generation quality. Adding
0.05
​
ℒ
h
0.05\mathcal{L}_{h}
explicitly supervises the H head and stabilizes H/V fusion.
During inference,
Xiaomi-Robotics-U0
first encodes the text instruction and reference images as a conditional prefix, followed by the target image header. The model then performs anti-diagonal generation only on the target image grid: positions are generated step by step in increasing order of
r
+
c
r+c
, and all positions within the same step are sampled in parallel. Boundary positions use the available single-direction prediction, while interior positions use H/V gated fusion.
This procedure preserves the constraints imposed by multi-source conditions while extending image-region parallel decoding to
Xiaomi-Robotics-U0
FlashAR Extension inference. As a result, the model maintains Any-to-Image conditional generation capability while reducing the serial generation cost in the target image region.
Figure 7
:
Qualitative and efficiency comparison between AR, FlashAR
+
, and FlashAR
+
with vLLM.
FlashAR
+
preserves high visual quality on Embodied Transfer and Embodied Scene Generation while substantially reducing generation latency.
To further improve the image-generation efficiency of
Xiaomi-Robotics-U0
, we integrate vLLM on top of FlashAR
+
. The implementation keeps the FlashAR
+
decoding rule unchanged: vLLM handles the conditional prefix, batched execution, and paged KV-cache management, while a diagonal scheduler emits the visual tokens of each anti-diagonal step and synchronizes the paired conditional and unconditional requests used for classifier-free guidance.
As shown in Figure
7
, the accelerated path preserves high generation quality on embodied tasks. On T2I generation at
1024
×
1024
1024\times 1024
resolution with a single H20 GPU, standard AR takes 450.77 seconds per image. FlashAR
+
reduces this latency to 16.56 seconds per image, achieving a
27.22
×
27.22\times
speedup over AR, and FlashAR
+
with vLLM further reduces the latency to 5.44 seconds per image with
max_num_seq=28
, achieving an additional
3.04
×
3.04\times
speedup over FlashAR
+
.

### 2.1
Problem Formulation

Our goal is to develop a unified embodied world model that supports diverse generation and prediction tasks under a single autoregressive framework. Given a multi-modal context
𝒞
=
{
x
1
,
x
2
,
⋯
,
x
n
}
,
\mathcal{C}=\{x_{1},x_{2},\cdots,x_{n}\},
(1)
where each element can be a text token, an image token, or a robot-related control token, the model learns the joint distribution over the subsequent multi-modal sequence
P
​
(
𝒴
|
𝒞
)
=
∏
t
=
1
T
P
​
(
y
t
|
y
<
t
,
𝒞
)
,
P(\mathcal{Y}|\mathcal{C})=\prod_{t=1}^{T}P(y_{t}|y_{<t},\mathcal{C}),
(2)
using standard next-token prediction (NTP).
To support embodied intelligence,
Xiaomi-Robotics-U0
is trained on several complementary data formats with different temporal structures. Specifically, we categorize all training samples into two groups.
Single-step samples
describe a one-shot mapping between inputs and outputs without explicit temporal evolution. These samples include:
(i)
Text-to-Image (T2I)
, where the model generates an image conditioned on textual descriptions;
(ii)
Any-to-Image (X2I)
, where one to three reference images together with text are used for image editing and generation;
(iii)
Embodied Scene Generation
, where the model generates the initial multi-view robot observations conditioned on robot embodiment and scene descriptions; and
(iv)
Embodied Transfer
, where the model predicts transferred multi-view observations given the current embodied observations and a target scene description.
Sequential samples
model long-horizon interactions and future evolution. We consider two representative formats:
(i) image-text interleaved subtask-subgoal sequences, where each subtask description is followed by the corresponding multi-view observations after completion; and
(ii) embodied manipulation videos collected at multiple frame rates, enabling the model to capture both sparse task-level transitions and dense interaction dynamics.
Despite their different semantics, all tasks are represented as unified multi-modal token sequences and optimized under the same autoregressive objective, allowing knowledge to be naturally shared across image generation, scene understanding, embodied reasoning, and future prediction.
Figure 3
:
Overview structure of
Xiaomi-Robotics-U0
. The model is trained end-to-end at scale with a unified next-token prediction objective. An efficient inference extension is provided along.

### 2.2
Model Architecture

Figure
3
illustrates the overall architecture of
Xiaomi-Robotics-U0
. We adopt the open-sourced EMU3.5
[
cui2025emu3
]
as the initialization model, which is built upon the Qwen-3-32B decoder-only Transformer and has demonstrated strong capabilities in image generation and image-text interleaved modeling. Images are first tokenized by the IBQ Tokenizer
[
shi2025scalable
]
with a spatial compression ratio of
16
×
16
16\times 16
. By extending the original Qwen vocabulary with the IBQ codebook, all modalities are represented within a unified discrete vocabulary and modeled using next-token prediction.
Compared with conventional embodied world models that introduce task-specific prediction heads,
Xiaomi-Robotics-U0
formulates all generation tasks—including image generation, embodied scene synthesis, scene transfer, sequential subtask prediction, and video generation—as a single autoregressive sequence modeling problem. This unified formulation enables efficient multi-task co-training while preserving the strong visual generation capabilities inherited from the large-scale pre-trained model.
To improve inference efficiency, we further adopt an inference-accelerating adaptation inspired by FlashAR
[
zhou2026flasharefficientposttrainingacceleration
]
. Specifically, an additional vertical prediction head is introduced to decode image tokens in an anti-diagonal order, allowing multiple visual tokens to be generated simultaneously. Combined with optimized KV-cache management and continuous batching provided by vLLM
[
kwon2023efficient
]
,
Xiaomi-Robotics-U0
achieves up to
82.9
×
82.9\times
faster image generation at 1024
×
\times
1024 resolution compared with the original NTP pattern, while preserving most generation quality.

### 2.3
Dataset

Figure 4
:
The illustration of the constructed single-step generation dataset. Except for the embodied transfer and the embodied scene generation data, we additionally introduce Text-to-Image(T2I) and Any-to-Image(X2I) data for co-training. Concretely speaking, we add camera-control and time-control data to improve the spatial and temporal understanding capability. For structure and geometry perception, we construct skeleton, depth, edge, and semantic mask datasets with advanced computer vision tools. To maintain the ability of the pre-trained model, we introduce general T2I and X2I datasets for co-training.
To endow
Xiaomi-Robotics-U0
with broad visual understanding and generation capabilities, we assemble a diverse training corpus spanning six distinct domains, each contributing complementary visual and semantic priors.
•
General image-text
data provides foundational visual-semantic alignment. We include ShareGPT4V
[
chen2024sharegpt4v
]
and similar large-scale vision-language benchmarks to establish a strong baseline for text-conditioned generation.
•
Embodied manipulation
data bridges the gap between passive visual understanding and active physical interaction. This domain combines both real-world and simulated sources. On the real side, we leverage AgiBotWorld-Beta
[
bu2025agibot
]
, Open X-Embodiment
[
Neill2023open_x_embodiment
]
, and a proprietary multi-view robot manipulation dataset collected in house, which we refer to as MiBot. On the simulation side, we incorporate data from RoboTwin2.0
[
chen2025robotwin
]
, GenieSim
[
yin2026geniesim30
]
, and InternData-A1
[
tian2026interndata
]
, enabling coverage of scenarios that are difficult or dangerous to collect physically.
•
Autonomous driving
data captures structured outdoor scenes with complex spatial layouts and dynamic objects. We source this from Cosmos-Drive-Dreams
[
ren2025cosmosdrive
]
, which provides high-quality driving scenarios with diverse weather, lighting, and traffic conditions.
•
Egocentric
data, drawn from EgoWalk
[
akhtyamov2025egowalk
]
, supplies first-person perspective recordings that reflect natural human viewpoint and motion patterns, enriching the model’s understanding of human-centric visual experience.
•
3D reconstruction
data offers dense multi-view correspondences and accurate geometry. We utilize ScanNet++ v2
[
yeshwanth2023scannet++
]
and DL3DV-10K
[
ling2024dl3dv
]
, both of which provide high-quality indoor and outdoor 3D scene captures.
•
Game
data, sourced from Open-P2P
[
yue2026openp2p
]
, introduces stylized and procedurally generated environments that expand the diversity of visual distributions beyond real-world captures.
Raw data collected at scale inevitably contains artifacts that can impair generation quality. Inspired by Abot Physical World
[
chen2026abot
]
, we implement an autonomous data filtering pipeline including motion blur detection, image quality detection, and gripper-state alignment.
To sum up, we collect 9.5 million samples for single-step generation, which corresponds to 56.4 billion tokens. For sequential generation, we select 2.6 million video clips, which corresponds to 49.6 billion tokens.
Most raw samples lack fine-grained text annotations and only carry coarse task-level descriptions, which cannot support precise text-driven embodied generation. We thus construct a unified annotation pipeline to standardize supervision signals across heterogeneous data.
We leverage Qwen3-VL-235B
[
bai2025qwen3
]
, a powerful vision-language model (VLM), to generate dense, structured annotations for generic frames, multi-source video sequences, and robotic manipulation trajectories. The pipeline consists of four decoupled branches with distinct annotation objectives:
•
General Scene Annotation
All video data spanning general visual corpora and embodied robot sequences are processed without domain restrictions. Each video clip undergoes temporal downsampling to select high-clarity keyframes. The VLM produces holistic captions describing overall scene layouts, object interactions, and cross-object spatial relationships for every sampled frame.
•
Structured Embodied Scene Annotation
Annotation is only performed on embodied manipulation data. Decomposed subtask outputs from trajectory processing guide scene partitioning into five orthogonal semantic dimensions: workspace, task objects, irrelevant objects, lighting, and background. Separate descriptions are generated for each dimension according to task relevance, supporting compositional reasoning and fine-grained controllable generation during inference.
•
Geometric Modality Extraction
Temporally consistent inverse depth maps are extracted for all frames and videos via Video Depth Anything
[
chen2025videodepthanything
]
to supply geometric conditioning signals for cross-domain embodied transfer tasks. Brighter pixels correspond to closer surfaces and darker pixels to distant regions within this depth representation, allowing the model to synthesize photorealistic RGB outputs conditioned on geometric priors.
•
Trajectory-level Subtask Decomposition
As illustrated in Figure
5
, HDBSCAN clustering
[
McInnes2017hdbscanHD
]
segments robot trajectories using end-effector poses and gripper signals to identify motion and grasp transition boundaries. Keyframes with dense physical interactions are sampled at these boundaries and sent to VLM to generate temporally aligned subtask plans with frame boundaries.
Transition frames with rich physical interactions are processed with scene annotation to produce training samples for embodied scene generation and embodied transfer tasks.
To support the three training objectives of
Xiaomi-Robotics-U0
, we reorganize all annotated data into task-specific formats:
•
Embodied Transfer.
Each training sample consists of a system instruction, a scene description, and multi-view depth maps as input, with multi-view RGB images as the supervision target. This task trains the model to hallucinate photorealistic appearance from geometric cues.
•
Scene Generation.
Each sample pairs a system instruction with a scene description, supervised by multi-view RGB images. This task learns text-to-multi-view-scene synthesis without geometric priors.
•
Video Generation.
Each sample comprises a system instruction, a task instruction, and an initial observation frame, supervised by a sequence of 15–25 RGB frames. For datasets that provide robot action labels, a sequence of robot mask images rendered from the robot action and URDF is additionally appended after the initial observation frame, serving as action guidance. This task learns to predict future visual states conditioned on task semantics and initial context.
Figure 5
:
Overview of the unified data labeling pipeline.
The upper part illustrates scene description generation, where each sample is annotated with structured semantic attributes.
The lower part shows trajectory-level subtask decomposition, including action-based clip segmentation, key-frame sampling, and subtask generation.
This pipeline provides consistent scene, geometry, and action-level supervision across heterogeneous robotic datasets.
Our training corpus contains heterogeneous multi-domain data, which easily leads to domain bias and overfitting toward dominant data distributions. We adopt sample-level reweighting conditioned on data domains and generation task types (multi-view generation, scene editing, single-view generation), with the core target of maintaining full coverage of robotic manipulation scenarios.
Specifically, we first eliminate invalid embodied episodes, including sequences with corrupted visuals, misaligned timestamps, missing proprioceptive or action signals, failed execution records, and static trajectories lacking robot-object interactions. All remaining samples are categorized via hierarchical annotations derived from subtask and structured scene labeling. Grouping criteria cover task goals, scene contexts, target objects, spatial relations, subtask phases, and atomic manipulation skills.
Such fine-grained grouping supports more precise balancing than simple dataset-level weighting: redundant samples sharing identical embodiment-task-object combinations are downsampled, whereas clips with novel visuals, object layouts, manipulation skills or temporal stages are prioritized. Higher sampling weights are assigned to long-tail categories, including rare objects, seldom-used manipulation skills, and core interaction phases (grasping, releasing, placing, and object state transitions). The resulting training distribution eliminates redundant repeated demonstrations while retaining diverse visual scenes and comprehensive manipulation skill coverage.

#### 2.3.1
Dataset Curation

To endow
Xiaomi-Robotics-U0
with broad visual understanding and generation capabilities, we assemble a diverse training corpus spanning six distinct domains, each contributing complementary visual and semantic priors.
•
General image-text
data provides foundational visual-semantic alignment. We include ShareGPT4V
[
chen2024sharegpt4v
]
and similar large-scale vision-language benchmarks to establish a strong baseline for text-conditioned generation.
•
Embodied manipulation
data bridges the gap between passive visual understanding and active physical interaction. This domain combines both real-world and simulated sources. On the real side, we leverage AgiBotWorld-Beta
[
bu2025agibot
]
, Open X-Embodiment
[
Neill2023open_x_embodiment
]
, and a proprietary multi-view robot manipulation dataset collected in house, which we refer to as MiBot. On the simulation side, we incorporate data from RoboTwin2.0
[
chen2025robotwin
]
, GenieSim
[
yin2026geniesim30
]
, and InternData-A1
[
tian2026interndata
]
, enabling coverage of scenarios that are difficult or dangerous to collect physically.
•
Autonomous driving
data captures structured outdoor scenes with complex spatial layouts and dynamic objects. We source this from Cosmos-Drive-Dreams
[
ren2025cosmosdrive
]
, which provides high-quality driving scenarios with diverse weather, lighting, and traffic conditions.
•
Egocentric
data, drawn from EgoWalk
[
akhtyamov2025egowalk
]
, supplies first-person perspective recordings that reflect natural human viewpoint and motion patterns, enriching the model’s understanding of human-centric visual experience.
•
3D reconstruction
data offers dense multi-view correspondences and accurate geometry. We utilize ScanNet++ v2
[
yeshwanth2023scannet++
]
and DL3DV-10K
[
ling2024dl3dv
]
, both of which provide high-quality indoor and outdoor 3D scene captures.
•
Game
data, sourced from Open-P2P
[
yue2026openp2p
]
, introduces stylized and procedurally generated environments that expand the diversity of visual distributions beyond real-world captures.
Raw data collected at scale inevitably contains artifacts that can impair generation quality. Inspired by Abot Physical World
[
chen2026abot
]
, we implement an autonomous data filtering pipeline including motion blur detection, image quality detection, and gripper-state alignment.
To sum up, we collect 9.5 million samples for single-step generation, which corresponds to 56.4 billion tokens. For sequential generation, we select 2.6 million video clips, which corresponds to 49.6 billion tokens.

##### Data Collection

To endow
Xiaomi-Robotics-U0
with broad visual understanding and generation capabilities, we assemble a diverse training corpus spanning six distinct domains, each contributing complementary visual and semantic priors.
•
General image-text
data provides foundational visual-semantic alignment. We include ShareGPT4V
[
chen2024sharegpt4v
]
and similar large-scale vision-language benchmarks to establish a strong baseline for text-conditioned generation.
•
Embodied manipulation
data bridges the gap between passive visual understanding and active physical interaction. This domain combines both real-world and simulated sources. On the real side, we leverage AgiBotWorld-Beta
[
bu2025agibot
]
, Open X-Embodiment
[
Neill2023open_x_embodiment
]
, and a proprietary multi-view robot manipulation dataset collected in house, which we refer to as MiBot. On the simulation side, we incorporate data from RoboTwin2.0
[
chen2025robotwin
]
, GenieSim
[
yin2026geniesim30
]
, and InternData-A1
[
tian2026interndata
]
, enabling coverage of scenarios that are difficult or dangerous to collect physically.
•
Autonomous driving
data captures structured outdoor scenes with complex spatial layouts and dynamic objects. We source this from Cosmos-Drive-Dreams
[
ren2025cosmosdrive
]
, which provides high-quality driving scenarios with diverse weather, lighting, and traffic conditions.
•
Egocentric
data, drawn from EgoWalk
[
akhtyamov2025egowalk
]
, supplies first-person perspective recordings that reflect natural human viewpoint and motion patterns, enriching the model’s understanding of human-centric visual experience.
•
3D reconstruction
data offers dense multi-view correspondences and accurate geometry. We utilize ScanNet++ v2
[
yeshwanth2023scannet++
]
and DL3DV-10K
[
ling2024dl3dv
]
, both of which provide high-quality indoor and outdoor 3D scene captures.
•
Game
data, sourced from Open-P2P
[
yue2026openp2p
]
, introduces stylized and procedurally generated environments that expand the diversity of visual distributions beyond real-world captures.

##### Data Filtering

Raw data collected at scale inevitably contains artifacts that can impair generation quality. Inspired by Abot Physical World
[
chen2026abot
]
, we implement an autonomous data filtering pipeline including motion blur detection, image quality detection, and gripper-state alignment.
To sum up, we collect 9.5 million samples for single-step generation, which corresponds to 56.4 billion tokens. For sequential generation, we select 2.6 million video clips, which corresponds to 49.6 billion tokens.

#### 2.3.2
Dataset Labeling

Most raw samples lack fine-grained text annotations and only carry coarse task-level descriptions, which cannot support precise text-driven embodied generation. We thus construct a unified annotation pipeline to standardize supervision signals across heterogeneous data.
We leverage Qwen3-VL-235B
[
bai2025qwen3
]
, a powerful vision-language model (VLM), to generate dense, structured annotations for generic frames, multi-source video sequences, and robotic manipulation trajectories. The pipeline consists of four decoupled branches with distinct annotation objectives:
•
General Scene Annotation
All video data spanning general visual corpora and embodied robot sequences are processed without domain restrictions. Each video clip undergoes temporal downsampling to select high-clarity keyframes. The VLM produces holistic captions describing overall scene layouts, object interactions, and cross-object spatial relationships for every sampled frame.
•
Structured Embodied Scene Annotation
Annotation is only performed on embodied manipulation data. Decomposed subtask outputs from trajectory processing guide scene partitioning into five orthogonal semantic dimensions: workspace, task objects, irrelevant objects, lighting, and background. Separate descriptions are generated for each dimension according to task relevance, supporting compositional reasoning and fine-grained controllable generation during inference.
•
Geometric Modality Extraction
Temporally consistent inverse depth maps are extracted for all frames and videos via Video Depth Anything
[
chen2025videodepthanything
]
to supply geometric conditioning signals for cross-domain embodied transfer tasks. Brighter pixels correspond to closer surfaces and darker pixels to distant regions within this depth representation, allowing the model to synthesize photorealistic RGB outputs conditioned on geometric priors.
•
Trajectory-level Subtask Decomposition
As illustrated in Figure
5
, HDBSCAN clustering
[
McInnes2017hdbscanHD
]
segments robot trajectories using end-effector poses and gripper signals to identify motion and grasp transition boundaries. Keyframes with dense physical interactions are sampled at these boundaries and sent to VLM to generate temporally aligned subtask plans with frame boundaries.
Transition frames with rich physical interactions are processed with scene annotation to produce training samples for embodied scene generation and embodied transfer tasks.
To support the three training objectives of
Xiaomi-Robotics-U0
, we reorganize all annotated data into task-specific formats:
•
Embodied Transfer.
Each training sample consists of a system instruction, a scene description, and multi-view depth maps as input, with multi-view RGB images as the supervision target. This task trains the model to hallucinate photorealistic appearance from geometric cues.
•
Scene Generation.
Each sample pairs a system instruction with a scene description, supervised by multi-view RGB images. This task learns text-to-multi-view-scene synthesis without geometric priors.
•
Video Generation.
Each sample comprises a system instruction, a task instruction, and an initial observation frame, supervised by a sequence of 15–25 RGB frames. For datasets that provide robot action labels, a sequence of robot mask images rendered from the robot action and URDF is additionally appended after the initial observation frame, serving as action guidance. This task learns to predict future visual states conditioned on task semantics and initial context.
Figure 5
:
Overview of the unified data labeling pipeline.
The upper part illustrates scene description generation, where each sample is annotated with structured semantic attributes.
The lower part shows trajectory-level subtask decomposition, including action-based clip segmentation, key-frame sampling, and subtask generation.
This pipeline provides consistent scene, geometry, and action-level supervision across heterogeneous robotic datasets.
Our training corpus contains heterogeneous multi-domain data, which easily leads to domain bias and overfitting toward dominant data distributions. We adopt sample-level reweighting conditioned on data domains and generation task types (multi-view generation, scene editing, single-view generation), with the core target of maintaining full coverage of robotic manipulation scenarios.
Specifically, we first eliminate invalid embodied episodes, including sequences with corrupted visuals, misaligned timestamps, missing proprioceptive or action signals, failed execution records, and static trajectories lacking robot-object interactions. All remaining samples are categorized via hierarchical annotations derived from subtask and structured scene labeling. Grouping criteria cover task goals, scene contexts, target objects, spatial relations, subtask phases, and atomic manipulation skills.
Such fine-grained grouping supports more precise balancing than simple dataset-level weighting: redundant samples sharing identical embodiment-task-object combinations are downsampled, whereas clips with novel visuals, object layouts, manipulation skills or temporal stages are prioritized. Higher sampling weights are assigned to long-tail categories, including rare objects, seldom-used manipulation skills, and core interaction phases (grasping, releasing, placing, and object state transitions). The resulting training distribution eliminates redundant repeated demonstrations while retaining diverse visual scenes and comprehensive manipulation skill coverage.

##### Unified Annotation Pipeline

Most raw samples lack fine-grained text annotations and only carry coarse task-level descriptions, which cannot support precise text-driven embodied generation. We thus construct a unified annotation pipeline to standardize supervision signals across heterogeneous data.

##### Unified Annotation Pipeline

We leverage Qwen3-VL-235B
[
bai2025qwen3
]
, a powerful vision-language model (VLM), to generate dense, structured annotations for generic frames, multi-source video sequences, and robotic manipulation trajectories. The pipeline consists of four decoupled branches with distinct annotation objectives:
•
General Scene Annotation
All video data spanning general visual corpora and embodied robot sequences are processed without domain restrictions. Each video clip undergoes temporal downsampling to select high-clarity keyframes. The VLM produces holistic captions describing overall scene layouts, object interactions, and cross-object spatial relationships for every sampled frame.
•
Structured Embodied Scene Annotation
Annotation is only performed on embodied manipulation data. Decomposed subtask outputs from trajectory processing guide scene partitioning into five orthogonal semantic dimensions: workspace, task objects, irrelevant objects, lighting, and background. Separate descriptions are generated for each dimension according to task relevance, supporting compositional reasoning and fine-grained controllable generation during inference.
•
Geometric Modality Extraction
Temporally consistent inverse depth maps are extracted for all frames and videos via Video Depth Anything
[
chen2025videodepthanything
]
to supply geometric conditioning signals for cross-domain embodied transfer tasks. Brighter pixels correspond to closer surfaces and darker pixels to distant regions within this depth representation, allowing the model to synthesize photorealistic RGB outputs conditioned on geometric priors.
•
Trajectory-level Subtask Decomposition
As illustrated in Figure
5
, HDBSCAN clustering
[
McInnes2017hdbscanHD
]
segments robot trajectories using end-effector poses and gripper signals to identify motion and grasp transition boundaries. Keyframes with dense physical interactions are sampled at these boundaries and sent to VLM to generate temporally aligned subtask plans with frame boundaries.
Transition frames with rich physical interactions are processed with scene annotation to produce training samples for embodied scene generation and embodied transfer tasks.

##### Task-Oriented Data Organization

To support the three training objectives of
Xiaomi-Robotics-U0
, we reorganize all annotated data into task-specific formats:
•
Embodied Transfer.
Each training sample consists of a system instruction, a scene description, and multi-view depth maps as input, with multi-view RGB images as the supervision target. This task trains the model to hallucinate photorealistic appearance from geometric cues.
•
Scene Generation.
Each sample pairs a system instruction with a scene description, supervised by multi-view RGB images. This task learns text-to-multi-view-scene synthesis without geometric priors.
•
Video Generation.
Each sample comprises a system instruction, a task instruction, and an initial observation frame, supervised by a sequence of 15–25 RGB frames. For datasets that provide robot action labels, a sequence of robot mask images rendered from the robot action and URDF is additionally appended after the initial observation frame, serving as action guidance. This task learns to predict future visual states conditioned on task semantics and initial context.
Figure 5
:
Overview of the unified data labeling pipeline.
The upper part illustrates scene description generation, where each sample is annotated with structured semantic attributes.
The lower part shows trajectory-level subtask decomposition, including action-based clip segmentation, key-frame sampling, and subtask generation.
This pipeline provides consistent scene, geometry, and action-level supervision across heterogeneous robotic datasets.

##### Data Balancing

Our training corpus contains heterogeneous multi-domain data, which easily leads to domain bias and overfitting toward dominant data distributions. We adopt sample-level reweighting conditioned on data domains and generation task types (multi-view generation, scene editing, single-view generation), with the core target of maintaining full coverage of robotic manipulation scenarios.
Specifically, we first eliminate invalid embodied episodes, including sequences with corrupted visuals, misaligned timestamps, missing proprioceptive or action signals, failed execution records, and static trajectories lacking robot-object interactions. All remaining samples are categorized via hierarchical annotations derived from subtask and structured scene labeling. Grouping criteria cover task goals, scene contexts, target objects, spatial relations, subtask phases, and atomic manipulation skills.
Such fine-grained grouping supports more precise balancing than simple dataset-level weighting: redundant samples sharing identical embodiment-task-object combinations are downsampled, whereas clips with novel visuals, object layouts, manipulation skills or temporal stages are prioritized. Higher sampling weights are assigned to long-tail categories, including rare objects, seldom-used manipulation skills, and core interaction phases (grasping, releasing, placing, and object state transitions). The resulting training distribution eliminates redundant repeated demonstrations while retaining diverse visual scenes and comprehensive manipulation skill coverage.

### 2.4
Training

To efficiently adapt the pre-trained image generation model into a unified embodied world model while preserving its original visual generation capability, we adapt two continued training setups:
single-step training
and
sequential training
.
This setup focuses on aligning the model with diverse one-step generation tasks while maintaining the original image generation capability. We jointly train four complementary datasets, including Text-to-Image (T2I), Any-to-Image (X2I), Embodied Scene Generation, and Embodied Transfer.
The T2I and X2I datasets continuously reinforce the general visual generation and image editing capabilities inherited from the pre-trained model, preventing catastrophic forgetting during embodied adaptation. Meanwhile, Embodied Scene Generation teaches the model to synthesize physically plausible initial robot observations conditioned on robot embodiments and scene descriptions, while Embodied Transfer enables cross-scene multi-view generation that preserves robot interaction states and camera geometry. Co-training these tasks effectively aligns the visual generation prior with embodied scene understanding under a unified autoregressive objective.
The sequential training extends the model to long-horizon embodied reasoning and interaction modeling.
We first introduce image-text interleaved subtask-subgoal sequences, where each language-described subtask is paired with the corresponding multi-view observations after execution. This data format teaches the model to reason over task decomposition while predicting future embodied observations, further aligning the sequence modeling capability inherited from the base model.
To enhance dynamic interaction modeling, we additionally incorporate large-scale embodied manipulation videos collected from InternData-A1
[
tian2026interndata
]
, GenieSim
[
yin2026geniesim30
]
, AgiBotWorld-Beta
[
bu2025agibot
]
, RoboCOIN
[
wu2025robocoin
]
, and RoboMIND
[
wu2024robomind
]
. Unlike conventional low-frame-rate trajectory datasets, we construct multi-frame-rate video sequences with FPS values of 1, 3, and 5. Sparse sequences encourage long-horizon planning and future prediction, whereas denser sequences enable the model to capture fine-grained manipulation dynamics and continuous interaction processes. Training across multiple temporal resolutions significantly improves the model’s ability to model both high-level task progression and low-level embodied dynamics within a unified autoregressive framework.

##### Single-step training

This setup focuses on aligning the model with diverse one-step generation tasks while maintaining the original image generation capability. We jointly train four complementary datasets, including Text-to-Image (T2I), Any-to-Image (X2I), Embodied Scene Generation, and Embodied Transfer.
The T2I and X2I datasets continuously reinforce the general visual generation and image editing capabilities inherited from the pre-trained model, preventing catastrophic forgetting during embodied adaptation. Meanwhile, Embodied Scene Generation teaches the model to synthesize physically plausible initial robot observations conditioned on robot embodiments and scene descriptions, while Embodied Transfer enables cross-scene multi-view generation that preserves robot interaction states and camera geometry. Co-training these tasks effectively aligns the visual generation prior with embodied scene understanding under a unified autoregressive objective.

##### Sequential training

The sequential training extends the model to long-horizon embodied reasoning and interaction modeling.
We first introduce image-text interleaved subtask-subgoal sequences, where each language-described subtask is paired with the corresponding multi-view observations after execution. This data format teaches the model to reason over task decomposition while predicting future embodied observations, further aligning the sequence modeling capability inherited from the base model.
To enhance dynamic interaction modeling, we additionally incorporate large-scale embodied manipulation videos collected from InternData-A1
[
tian2026interndata
]
, GenieSim
[
yin2026geniesim30
]
, AgiBotWorld-Beta
[
bu2025agibot
]
, RoboCOIN
[
wu2025robocoin
]
, and RoboMIND
[
wu2024robomind
]
. Unlike conventional low-frame-rate trajectory datasets, we construct multi-frame-rate video sequences with FPS values of 1, 3, and 5. Sparse sequences encourage long-horizon planning and future prediction, whereas denser sequences enable the model to capture fine-grained manipulation dynamics and continuous interaction processes. Training across multiple temporal resolutions significantly improves the model’s ability to model both high-level task progression and low-level embodied dynamics within a unified autoregressive framework.

### 2.5
Inference

To further accelerate inference, we introduce a FlashAR
+
extension to
Xiaomi-Robotics-U0
. The extension keeps the conditional modeling of reference images and other input signals unchanged, while applying parallel autoregressive generation only to the final target image region. This design is intended to preserve the multi-source conditioning capability of
Xiaomi-Robotics-U0
and reduce the serial decoding cost during X2I inference.
FlashAR
[
zhou2026flasharefficientposttrainingacceleration
]
proposes a post-training and inference acceleration paradigm for autoregressive text-to-image generation. Its core idea is to preserve the generation capability of the original model while introducing extra vertical prediction, so that image tokens can be generated in parallel along anti-diagonal groups. This reduces the number of serial decoding steps in the image region. Inspired by FlashAR, we perform FlashAR
+
post-training on top of
Xiaomi-Robotics-U0
, extending it from single-image T2I generation to broader X2I scenarios, and use the resulting model for X2I inference acceleration in
Xiaomi-Robotics-U0
. Figure
6(a)
demonstrates the decoding steps of the FlashAR
+
extension.
In
Xiaomi-Robotics-U0
, X2I covers tasks such as T2I, I2I, and multi-reference image generation. The input conditions may include text instructions and one or more reference images.
(a)
Diagonal step schedule.
(b)
Prefix-conditioned step-causal mask.
Figure 6
:
Xiaomi-Robotics-U0
FlashAR
+
extension for X2I generation. Target image tokens are grouped by diagonal steps, while all input conditions are kept in the prefix and the target image is generated with a step-causal attention mask.
Compared with standard autoregressive generation,
Xiaomi-Robotics-U0
FlashAR
+
Extension introduces the key changes summarized in Table
1
.
Table 1:
Comparison between standard autoregressive generation and the
Xiaomi-Robotics-U0
FlashAR
+
Extension.
Aspect
Standard AR
Xiaomi-Robotics-U0
FlashAR
+
Extension
Effect
Target image generation
Serial token-by-token generation in raster order
Parallel generation by anti-diagonal steps in the target image
Reduces serial decoding steps
Target image attention
Standard one-dimensional causal attention
The conditional prefix is visible; the target image uses two-dimensional step-causal attention internally
Prevents information leakage
Prediction heads
A single AR prediction head
Horizontal head, vertical head, and gated fusion
Supports anti-diagonal generation
Directional auxiliary loss
Standard supervision
Introduces H/V loss constraints
Constrains both directional heads
During training, the FlashAR
+
extension identifies image blocks during data parsing and defines the target image. All preceding content, including text instructions, reference images, control conditions, and the target image header, is treated as the conditional prefix. Parallel autoregressive losses are computed only over the target image region, while reference images serve only as a conditional context. This prevents reference images from being incorrectly supervised as generation targets and ensures that the model learns to generate the target image from multi-source conditions.
The attention constraint is also restricted to the target image region, as shown in figure
6(b)
. The conditional prefix preserves standard autoregressive visibility. Target image tokens can attend to the entire conditional prefix, but within the target image they can only attend to tokens from earlier anti-diagonal steps. For a target grid position
(
r
,
c
)
(r,c)
, its parallel step is defined as
r
+
c
r+c
; tokens within the same step are mutually invisible, which prevents information leakage during parallel generation. Reference images, text tokens, and control tokens are not involved in the anti-diagonal mask construction inside the target image, preserving the semantic integrity of X2I conditions.
This constraint can be expressed as a visibility function within the target image. For a target-image query token
q
=
(
r
q
,
c
q
)
q=(r_{q},c_{q})
and key token
k
=
(
r
k
,
c
k
)
k=(r_{k},c_{k})
,
M
​
(
q
,
k
)
=
1
​
[
s
​
(
r
k
,
c
k
)
<
s
​
(
r
q
,
c
q
)
]
.
M(q,k)={1}\left[s(r_{k},c_{k})<s(r_{q},c_{q})\right].
Xiaomi-Robotics-U0
FlashAR
+
Extension introduces a key adjustment to directional auxiliary supervision. We match the horizontal auxiliary loss weight to the vertical auxiliary loss weight:
ℒ
=
ℒ
fuse
+
0.05
​
ℒ
h
+
0.05
​
ℒ
v
+
0.2
​
ℒ
distill
.
\mathcal{L}=\mathcal{L}_{\mathrm{fuse}}+0.05\mathcal{L}_{h}+0.05\mathcal{L}_{v}+0.2\mathcal{L}_{\mathrm{distill}}.
Here,
ℒ
fuse
\mathcal{L}_{\mathrm{fuse}}
is the target-image prediction loss after H/V gated fusion,
ℒ
h
\mathcal{L}_{h}
and
ℒ
v
\mathcal{L}_{v}
supervise the horizontal and vertical heads respectively, and
ℒ
distill
\mathcal{L}_{\mathrm{distill}}
is the AR distillation term. For a target position
(
r
,
c
)
(r,c)
, let
z
h
​
(
r
,
c
)
z_{h}(r,c)
and
z
v
​
(
r
,
c
)
z_{v}(r,c)
denote the logits from the horizontal and vertical heads, and let
g
​
(
r
,
c
)
∈
[
0
,
1
]
g(r,c)\in[0,1]
denote the gate value. The fused logits are
z
fuse
​
(
r
,
c
)
=
g
​
(
r
,
c
)
​
z
h
​
(
r
,
c
)
+
(
1
−
g
​
(
r
,
c
)
)
​
z
v
​
(
r
,
c
)
.
z_{\mathrm{fuse}}(r,c)=g(r,c)z_{h}(r,c)\;+\;\left(1-g(r,c)\right)z_{v}(r,c).
The corresponding fused loss is
ℒ
fuse
=
−
∑
(
r
,
c
)
∈
𝐘
log
⁡
softmax
​
(
z
fuse
​
(
r
,
c
)
)
y
r
,
c
.
\mathcal{L}_{\mathrm{fuse}}=-\sum_{(r,c)\in\mathbf{Y}}\log\mathrm{softmax}\left(z_{\mathrm{fuse}}(r,c)\right)_{y_{r,c}}.
The directional auxiliary losses
ℒ
h
\mathcal{L}_{h}
and
ℒ
v
\mathcal{L}_{v}
use the same target tokens to supervise
z
h
z_{h}
and
z
v
z_{v}
, respectively. The H head also needs to adapt to the target-region distribution. If
ℒ
h
=
0
\mathcal{L}_{h}=0
, the H head may fail to converge sufficiently, and the H/V gate may collapse toward the H branch in later training stages. When the gate collapses to an under-trained H head, the fused prediction degenerates into an unstable single-direction prediction, and further training can lead to degraded generation quality. Adding
0.05
​
ℒ
h
0.05\mathcal{L}_{h}
explicitly supervises the H head and stabilizes H/V fusion.
During inference,
Xiaomi-Robotics-U0
first encodes the text instruction and reference images as a conditional prefix, followed by the target image header. The model then performs anti-diagonal generation only on the target image grid: positions are generated step by step in increasing order of
r
+
c
r+c
, and all positions within the same step are sampled in parallel. Boundary positions use the available single-direction prediction, while interior positions use H/V gated fusion.
This procedure preserves the constraints imposed by multi-source conditions while extending image-region parallel decoding to
Xiaomi-Robotics-U0
FlashAR Extension inference. As a result, the model maintains Any-to-Image conditional generation capability while reducing the serial generation cost in the target image region.
Figure 7
:
Qualitative and efficiency comparison between AR, FlashAR
+
, and FlashAR
+
with vLLM.
FlashAR
+
preserves high visual quality on Embodied Transfer and Embodied Scene Generation while substantially reducing generation latency.
To further improve the image-generation efficiency of
Xiaomi-Robotics-U0
, we integrate vLLM on top of FlashAR
+
. The implementation keeps the FlashAR
+
decoding rule unchanged: vLLM handles the conditional prefix, batched execution, and paged KV-cache management, while a diagonal scheduler emits the visual tokens of each anti-diagonal step and synchronizes the paired conditional and unconditional requests used for classifier-free guidance.
As shown in Figure
7
, the accelerated path preserves high generation quality on embodied tasks. On T2I generation at
1024
×
1024
1024\times 1024
resolution with a single H20 GPU, standard AR takes 450.77 seconds per image. FlashAR
+
reduces this latency to 16.56 seconds per image, achieving a
27.22
×
27.22\times
speedup over AR, and FlashAR
+
with vLLM further reduces the latency to 5.44 seconds per image with
max_num_seq=28
, achieving an additional
3.04
×
3.04\times
speedup over FlashAR
+
.

#### 2.5.1
FlashAR
+
Extension

FlashAR
[
zhou2026flasharefficientposttrainingacceleration
]
proposes a post-training and inference acceleration paradigm for autoregressive text-to-image generation. Its core idea is to preserve the generation capability of the original model while introducing extra vertical prediction, so that image tokens can be generated in parallel along anti-diagonal groups. This reduces the number of serial decoding steps in the image region. Inspired by FlashAR, we perform FlashAR
+
post-training on top of
Xiaomi-Robotics-U0
, extending it from single-image T2I generation to broader X2I scenarios, and use the resulting model for X2I inference acceleration in
Xiaomi-Robotics-U0
. Figure
6(a)
demonstrates the decoding steps of the FlashAR
+
extension.
In
Xiaomi-Robotics-U0
, X2I covers tasks such as T2I, I2I, and multi-reference image generation. The input conditions may include text instructions and one or more reference images.
(a)
Diagonal step schedule.
(b)
Prefix-conditioned step-causal mask.
Figure 6
:
Xiaomi-Robotics-U0
FlashAR
+
extension for X2I generation. Target image tokens are grouped by diagonal steps, while all input conditions are kept in the prefix and the target image is generated with a step-causal attention mask.
Compared with standard autoregressive generation,
Xiaomi-Robotics-U0
FlashAR
+
Extension introduces the key changes summarized in Table
1
.
Table 1:
Comparison between standard autoregressive generation and the
Xiaomi-Robotics-U0
FlashAR
+
Extension.
Aspect
Standard AR
Xiaomi-Robotics-U0
FlashAR
+
Extension
Effect
Target image generation
Serial token-by-token generation in raster order
Parallel generation by anti-diagonal steps in the target image
Reduces serial decoding steps
Target image attention
Standard one-dimensional causal attention
The conditional prefix is visible; the target image uses two-dimensional step-causal attention internally
Prevents information leakage
Prediction heads
A single AR prediction head
Horizontal head, vertical head, and gated fusion
Supports anti-diagonal generation
Directional auxiliary loss
Standard supervision
Introduces H/V loss constraints
Constrains both directional heads
During training, the FlashAR
+
extension identifies image blocks during data parsing and defines the target image. All preceding content, including text instructions, reference images, control conditions, and the target image header, is treated as the conditional prefix. Parallel autoregressive losses are computed only over the target image region, while reference images serve only as a conditional context. This prevents reference images from being incorrectly supervised as generation targets and ensures that the model learns to generate the target image from multi-source conditions.
The attention constraint is also restricted to the target image region, as shown in figure
6(b)
. The conditional prefix preserves standard autoregressive visibility. Target image tokens can attend to the entire conditional prefix, but within the target image they can only attend to tokens from earlier anti-diagonal steps. For a target grid position
(
r
,
c
)
(r,c)
, its parallel step is defined as
r
+
c
r+c
; tokens within the same step are mutually invisible, which prevents information leakage during parallel generation. Reference images, text tokens, and control tokens are not involved in the anti-diagonal mask construction inside the target image, preserving the semantic integrity of X2I conditions.
This constraint can be expressed as a visibility function within the target image. For a target-image query token
q
=
(
r
q
,
c
q
)
q=(r_{q},c_{q})
and key token
k
=
(
r
k
,
c
k
)
k=(r_{k},c_{k})
,
M
​
(
q
,
k
)
=
1
​
[
s
​
(
r
k
,
c
k
)
<
s
​
(
r
q
,
c
q
)
]
.
M(q,k)={1}\left[s(r_{k},c_{k})<s(r_{q},c_{q})\right].
Xiaomi-Robotics-U0
FlashAR
+
Extension introduces a key adjustment to directional auxiliary supervision. We match the horizontal auxiliary loss weight to the vertical auxiliary loss weight:
ℒ
=
ℒ
fuse
+
0.05
​
ℒ
h
+
0.05
​
ℒ
v
+
0.2
​
ℒ
distill
.
\mathcal{L}=\mathcal{L}_{\mathrm{fuse}}+0.05\mathcal{L}_{h}+0.05\mathcal{L}_{v}+0.2\mathcal{L}_{\mathrm{distill}}.
Here,
ℒ
fuse
\mathcal{L}_{\mathrm{fuse}}
is the target-image prediction loss after H/V gated fusion,
ℒ
h
\mathcal{L}_{h}
and
ℒ
v
\mathcal{L}_{v}
supervise the horizontal and vertical heads respectively, and
ℒ
distill
\mathcal{L}_{\mathrm{distill}}
is the AR distillation term. For a target position
(
r
,
c
)
(r,c)
, let
z
h
​
(
r
,
c
)
z_{h}(r,c)
and
z
v
​
(
r
,
c
)
z_{v}(r,c)
denote the logits from the horizontal and vertical heads, and let
g
​
(
r
,
c
)
∈
[
0
,
1
]
g(r,c)\in[0,1]
denote the gate value. The fused logits are
z
fuse
​
(
r
,
c
)
=
g
​
(
r
,
c
)
​
z
h
​
(
r
,
c
)
+
(
1
−
g
​
(
r
,
c
)
)
​
z
v
​
(
r
,
c
)
.
z_{\mathrm{fuse}}(r,c)=g(r,c)z_{h}(r,c)\;+\;\left(1-g(r,c)\right)z_{v}(r,c).
The corresponding fused loss is
ℒ
fuse
=
−
∑
(
r
,
c
)
∈
𝐘
log
⁡
softmax
​
(
z
fuse
​
(
r
,
c
)
)
y
r
,
c
.
\mathcal{L}_{\mathrm{fuse}}=-\sum_{(r,c)\in\mathbf{Y}}\log\mathrm{softmax}\left(z_{\mathrm{fuse}}(r,c)\right)_{y_{r,c}}.
The directional auxiliary losses
ℒ
h
\mathcal{L}_{h}
and
ℒ
v
\mathcal{L}_{v}
use the same target tokens to supervise
z
h
z_{h}
and
z
v
z_{v}
, respectively. The H head also needs to adapt to the target-region distribution. If
ℒ
h
=
0
\mathcal{L}_{h}=0
, the H head may fail to converge sufficiently, and the H/V gate may collapse toward the H branch in later training stages. When the gate collapses to an under-trained H head, the fused prediction degenerates into an unstable single-direction prediction, and further training can lead to degraded generation quality. Adding
0.05
​
ℒ
h
0.05\mathcal{L}_{h}
explicitly supervises the H head and stabilizes H/V fusion.
During inference,
Xiaomi-Robotics-U0
first encodes the text instruction and reference images as a conditional prefix, followed by the target image header. The model then performs anti-diagonal generation only on the target image grid: positions are generated step by step in increasing order of
r
+
c
r+c
, and all positions within the same step are sampled in parallel. Boundary positions use the available single-direction prediction, while interior positions use H/V gated fusion.
This procedure preserves the constraints imposed by multi-source conditions while extending image-region parallel decoding to
Xiaomi-Robotics-U0
FlashAR Extension inference. As a result, the model maintains Any-to-Image conditional generation capability while reducing the serial generation cost in the target image region.
Figure 7
:
Qualitative and efficiency comparison between AR, FlashAR
+
, and FlashAR
+
with vLLM.
FlashAR
+
preserves high visual quality on Embodied Transfer and Embodied Scene Generation while substantially reducing generation latency.

##### FlashAR baseline

FlashAR
[
zhou2026flasharefficientposttrainingacceleration
]
proposes a post-training and inference acceleration paradigm for autoregressive text-to-image generation. Its core idea is to preserve the generation capability of the original model while introducing extra vertical prediction, so that image tokens can be generated in parallel along anti-diagonal groups. This reduces the number of serial decoding steps in the image region. Inspired by FlashAR, we perform FlashAR
+
post-training on top of
Xiaomi-Robotics-U0
, extending it from single-image T2I generation to broader X2I scenarios, and use the resulting model for X2I inference acceleration in
Xiaomi-Robotics-U0
. Figure
6(a)
demonstrates the decoding steps of the FlashAR
+
extension.
In
Xiaomi-Robotics-U0
, X2I covers tasks such as T2I, I2I, and multi-reference image generation. The input conditions may include text instructions and one or more reference images.
(a)
Diagonal step schedule.
(b)
Prefix-conditioned step-causal mask.
Figure 6
:
Xiaomi-Robotics-U0
FlashAR
+
extension for X2I generation. Target image tokens are grouped by diagonal steps, while all input conditions are kept in the prefix and the target image is generated with a step-causal attention mask.

##### FlashAR
+
mask and attention

Compared with standard autoregressive generation,
Xiaomi-Robotics-U0
FlashAR
+
Extension introduces the key changes summarized in Table
1
.
Table 1:
Comparison between standard autoregressive generation and the
Xiaomi-Robotics-U0
FlashAR
+
Extension.
Aspect
Standard AR
Xiaomi-Robotics-U0
FlashAR
+
Extension
Effect
Target image generation
Serial token-by-token generation in raster order
Parallel generation by anti-diagonal steps in the target image
Reduces serial decoding steps
Target image attention
Standard one-dimensional causal attention
The conditional prefix is visible; the target image uses two-dimensional step-causal attention internally
Prevents information leakage
Prediction heads
A single AR prediction head
Horizontal head, vertical head, and gated fusion
Supports anti-diagonal generation
Directional auxiliary loss
Standard supervision
Introduces H/V loss constraints
Constrains both directional heads
During training, the FlashAR
+
extension identifies image blocks during data parsing and defines the target image. All preceding content, including text instructions, reference images, control conditions, and the target image header, is treated as the conditional prefix. Parallel autoregressive losses are computed only over the target image region, while reference images serve only as a conditional context. This prevents reference images from being incorrectly supervised as generation targets and ensures that the model learns to generate the target image from multi-source conditions.
The attention constraint is also restricted to the target image region, as shown in figure
6(b)
. The conditional prefix preserves standard autoregressive visibility. Target image tokens can attend to the entire conditional prefix, but within the target image they can only attend to tokens from earlier anti-diagonal steps. For a target grid position
(
r
,
c
)
(r,c)
, its parallel step is defined as
r
+
c
r+c
; tokens within the same step are mutually invisible, which prevents information leakage during parallel generation. Reference images, text tokens, and control tokens are not involved in the anti-diagonal mask construction inside the target image, preserving the semantic integrity of X2I conditions.
This constraint can be expressed as a visibility function within the target image. For a target-image query token
q
=
(
r
q
,
c
q
)
q=(r_{q},c_{q})
and key token
k
=
(
r
k
,
c
k
)
k=(r_{k},c_{k})
,
M
​
(
q
,
k
)
=
1
​
[
s
​
(
r
k
,
c
k
)
<
s
​
(
r
q
,
c
q
)
]
.
M(q,k)={1}\left[s(r_{k},c_{k})<s(r_{q},c_{q})\right].

##### FlashAR
+
loss

Xiaomi-Robotics-U0
FlashAR
+
Extension introduces a key adjustment to directional auxiliary supervision. We match the horizontal auxiliary loss weight to the vertical auxiliary loss weight:
ℒ
=
ℒ
fuse
+
0.05
​
ℒ
h
+
0.05
​
ℒ
v
+
0.2
​
ℒ
distill
.
\mathcal{L}=\mathcal{L}_{\mathrm{fuse}}+0.05\mathcal{L}_{h}+0.05\mathcal{L}_{v}+0.2\mathcal{L}_{\mathrm{distill}}.
Here,
ℒ
fuse
\mathcal{L}_{\mathrm{fuse}}
is the target-image prediction loss after H/V gated fusion,
ℒ
h
\mathcal{L}_{h}
and
ℒ
v
\mathcal{L}_{v}
supervise the horizontal and vertical heads respectively, and
ℒ
distill
\mathcal{L}_{\mathrm{distill}}
is the AR distillation term. For a target position
(
r
,
c
)
(r,c)
, let
z
h
​
(
r
,
c
)
z_{h}(r,c)
and
z
v
​
(
r
,
c
)
z_{v}(r,c)
denote the logits from the horizontal and vertical heads, and let
g
​
(
r
,
c
)
∈
[
0
,
1
]
g(r,c)\in[0,1]
denote the gate value. The fused logits are
z
fuse
​
(
r
,
c
)
=
g
​
(
r
,
c
)
​
z
h
​
(
r
,
c
)
+
(
1
−
g
​
(
r
,
c
)
)
​
z
v
​
(
r
,
c
)
.
z_{\mathrm{fuse}}(r,c)=g(r,c)z_{h}(r,c)\;+\;\left(1-g(r,c)\right)z_{v}(r,c).
The corresponding fused loss is
ℒ
fuse
=
−
∑
(
r
,
c
)
∈
𝐘
log
⁡
softmax
​
(
z
fuse
​
(
r
,
c
)
)
y
r
,
c
.
\mathcal{L}_{\mathrm{fuse}}=-\sum_{(r,c)\in\mathbf{Y}}\log\mathrm{softmax}\left(z_{\mathrm{fuse}}(r,c)\right)_{y_{r,c}}.
The directional auxiliary losses
ℒ
h
\mathcal{L}_{h}
and
ℒ
v
\mathcal{L}_{v}
use the same target tokens to supervise
z
h
z_{h}
and
z
v
z_{v}
, respectively. The H head also needs to adapt to the target-region distribution. If
ℒ
h
=
0
\mathcal{L}_{h}=0
, the H head may fail to converge sufficiently, and the H/V gate may collapse toward the H branch in later training stages. When the gate collapses to an under-trained H head, the fused prediction degenerates into an unstable single-direction prediction, and further training can lead to degraded generation quality. Adding
0.05
​
ℒ
h
0.05\mathcal{L}_{h}
explicitly supervises the H head and stabilizes H/V fusion.

##### Xiaomi-Robotics-U0
FlashAR
+
inference

During inference,
Xiaomi-Robotics-U0
first encodes the text instruction and reference images as a conditional prefix, followed by the target image header. The model then performs anti-diagonal generation only on the target image grid: positions are generated step by step in increasing order of
r
+
c
r+c
, and all positions within the same step are sampled in parallel. Boundary positions use the available single-direction prediction, while interior positions use H/V gated fusion.
This procedure preserves the constraints imposed by multi-source conditions while extending image-region parallel decoding to
Xiaomi-Robotics-U0
FlashAR Extension inference. As a result, the model maintains Any-to-Image conditional generation capability while reducing the serial generation cost in the target image region.
Figure 7
:
Qualitative and efficiency comparison between AR, FlashAR
+
, and FlashAR
+
with vLLM.
FlashAR
+
preserves high visual quality on Embodied Transfer and Embodied Scene Generation while substantially reducing generation latency.

#### 2.5.2
vLLM Integration

To further improve the image-generation efficiency of
Xiaomi-Robotics-U0
, we integrate vLLM on top of FlashAR
+
. The implementation keeps the FlashAR
+
decoding rule unchanged: vLLM handles the conditional prefix, batched execution, and paged KV-cache management, while a diagonal scheduler emits the visual tokens of each anti-diagonal step and synchronizes the paired conditional and unconditional requests used for classifier-free guidance.
As shown in Figure
7
, the accelerated path preserves high generation quality on embodied tasks. On T2I generation at
1024
×
1024
1024\times 1024
resolution with a single H20 GPU, standard AR takes 450.77 seconds per image. FlashAR
+
reduces this latency to 16.56 seconds per image, achieving a
27.22
×
27.22\times
speedup over AR, and FlashAR
+
with vLLM further reduces the latency to 5.44 seconds per image with
max_num_seq=28
, achieving an additional
3.04
×
3.04\times
speedup over FlashAR
+
.

## 3
Experiments

Our model unifies embodied transfer, embodied scene generation, and embodied video generation while retaining general image editing and generation capabilities. This section presents benchmark and qualitative results for our embodied transfer and scene generation tasks, which outperform GPT-Image-2.0 in human evaluation. Specifically, our embodied transfer method substantially enhances robot trajectory diversity while preserving geometric consistency, boosting the success rate of
π
0.5
\pi_{0.5}
from 36.9% to 63.2%. We further validate the model’s general image generation capacity via evaluations on the Geneval and ImageEdit benchmarks, confirming that our method retains dominant general image generation performance. For embodied video generation, our model achieves state-of-the-art results on the World Arena benchmark. It also demonstrates strong generalization to self-synthesized initial scenes, enabling a closed-loop data engine for infinite video generation conditioned on arbitrary scene descriptions and robotic instructions.
To evaluate embodied transfer, we construct a benchmark of 300 samples, each consisting of a textual scene description together with a set of multi-view depth maps; the model is required to synthesize multi-view-consistent RGB images of the described robot manipulation scene.
Each sample additionally provides the multi-view RGB reference images from the original dataset. These references do not fully correspond to the (edited) scene description, but they capture the structural information beyond depth and are used to compute the benchmark metrics.
Since the depth maps are drawn from the same domain as the training set, we deliberately edit a subset of the scene descriptions so that they call for content not present in the training data, thereby probing the model’s general-purpose generation ability rather than mere reconstruction.
According to how much the edited description deviates from the original scene, we split the benchmark into 150
Easy
samples (descriptions close to the underlying scene) and 150
Hard
samples (descriptions requiring substantial novel content).
We compare against GPT-Image-2, a strong image generation model.
We assess generation quality from three complementary aspects, all computed against the reference views.
For
depth consistency
, we extract monocular depth from both the generated image and the reference with a shared estimator and report SI-RMSE (scale-invariant RMSE,
↓
\downarrow
),
δ
1
\delta_{1}
(the fraction of pixels with relative depth error below
1.25
1.25
,
↑
\uparrow
), and AbsRel (absolute relative error,
↓
\downarrow
); these measure how faithfully the generated image respects the 3D geometry imposed by the input depth.
For
structural fidelity
, we compute Canny edge F1, Precision, and Recall (
↑
\uparrow
), capturing whether the generated layout preserves the fine-grained contours of the scene.
For
semantic grounding
, we run open-vocabulary detection and segmentation and report Segmentation mIoU and Recall (
↑
\uparrow
), reflecting whether the described objects are correctly synthesized and localized.
As summarized in Table
2
,
Xiaomi-Robotics-U0
outperforms GPT-Image-2 by a large margin across all metrics on both the Easy and Hard splits, indicating markedly better adherence to the input geometry and stronger semantic alignment with the scene description.
Table 2
:
Objective metrics on the embodied transfer benchmark. Values in parentheses denote the change of
Xiaomi-Robotics-U0
relative to GPT-Image-2 on the same split (signed toward the actual value; a lower value is better for
↓
\downarrow
metrics and a higher value for
↑
\uparrow
metrics).
Metric
GPT-Image-2
Xiaomi-Robotics-U0
(Ours)
Easy
Hard
Easy
Hard
Depth SI-RMSE (
↓
\downarrow
)
0.4007
0.3444
0.1407 (
−
-
0.260
)
0.1435 (
−
-
0.201
)
Depth
δ
1
\delta_{1}
(
↑
\uparrow
)
0.5307
0.5340
0.8017 (
+
+
0.271
)
0.7846 (
+
+
0.251
)
Depth AbsRel (
↓
\downarrow
)
0.4350
0.4045
0.1647 (
−
-
0.270
)
0.1702 (
−
-
0.234
)
Canny F1 (
↑
\uparrow
)
0.0519
0.0507
0.2066 (
+
+
0.155
)
0.1815 (
+
+
0.131
)
Canny Precision (
↑
\uparrow
)
0.0560
0.0536
0.2037 (
+
+
0.148
)
0.1770 (
+
+
0.123
)
Canny Recall (
↑
\uparrow
)
0.0572
0.0582
0.2290 (
+
+
0.172
)
0.2176 (
+
+
0.159
)
Segmentation mIoU (
↑
\uparrow
)
0.4105
0.4098
0.7621 (
+
+
0.352
)
0.7436 (
+
+
0.334
)
Segmentation Recall (
↑
\uparrow
)
0.6208
0.6524
0.8136 (
+
+
0.193
)
0.8032 (
+
+
0.151
)
Figure 8
:
Qualitative results on embodied transfer. For each example we show the scene description, the original multi-view RGB images from which the depth conditions are extracted, and the images generated by
Xiaomi-Robotics-U0
. Starting from the original images, we estimate their multi-view depth maps and edit the scene description (edited text highlighted); the depth maps together with the edited description are then fed to
Xiaomi-Robotics-U0
, which synthesizes multi-view-consistent RGB images that adhere to both the edited text and the input geometry.
Figure
8
presents qualitative examples of embodied transfer.
For each example we display the scene description, the original multi-view RGB images from the source dataset, and the images generated by
Xiaomi-Robotics-U0
.
Our pipeline first estimates the multi-view depth maps from the original images, then edits the scene description—with the edited spans highlighted—to call for content that departs from the original scene; the extracted depth maps and the edited description are jointly fed to the model for generation.
Across diverse scenes, and given a scene description in either natural-language or structured form,
Xiaomi-Robotics-U0
faithfully follows both the edited textual description and the multi-view depth conditions, correctly rendering the described objects, lighting, and background while respecting the spatial geometry of each view.
Crucially, the generated views remain mutually consistent in object identity, appearance, and arrangement, demonstrating that our unified model produces coherent multi-view scenes rather than independently plausible but conflicting images.
Figure 9
:
Qualitative comparison between GPT-Image-2 and
Xiaomi-Robotics-U0
on the embodied transfer benchmark. GPT-Image-2 generates plausible, well-aligned images but violates the input depth and produces multi-view-inconsistent results, whereas
Xiaomi-Robotics-U0
adheres to the geometry of every view and keeps the views mutually consistent.
We further compare
Xiaomi-Robotics-U0
with GPT-Image-2 in Figure
9
, on three samples from the source datasets, each showing the scene description, the input multi-view depth maps, and the two models’ outputs.
GPT-Image-2 produces visually appealing images with strong text–image alignment: the described objects, materials, lighting, and background are all rendered plausibly, and in isolation each view looks convincing.
However, it exhibits a poor understanding of the input geometry and of cross-view relationships.
Its outputs frequently violate the depth conditions—objects are placed at inconsistent distances, surfaces are warped, and the overall spatial layout drifts away from the imposed geometry—which is reflected in its high SI-RMSE and low
δ
1
\delta_{1}
in Table
2
.
Moreover, the views generated for the same scene are largely independent: object identity, pose, and arrangement fail to remain consistent across viewpoints, so the three images do not correspond to a single coherent 3D scene.
In contrast,
Xiaomi-Robotics-U0
faithfully respects the depth of every view while keeping the multi-view outputs mutually consistent, indicating that it not only matches the strong text-following ability of GPT-Image-2 but also genuinely grounds generation in the input geometry.
Beyond benchmark evaluation, we leverage the structured text-to-image capability of
Xiaomi-Robotics-U0
to augment the demonstration trajectories collected for four real-world tasks (pack box, fold towel, pack phone, and pack earphone), aiming to endow downstream policies with better generalization to novel scenes.
For each trajectory, we prompt a VLM (Qwen3-VL) to generate, conditioned on the current depth maps and the original scene description, 1000 diverse alternatives for every dimension except the task objects, encouraging maximal diversity across background, workspace, and lighting.
We then randomly combine these per-dimension descriptions and feed them to
Xiaomi-Robotics-U0
for generation.
As shown in Figure
10
and Figure
11
, the resulting scenes exhibit rich diversity, and none of these compositions appear in the training set, demonstrating that
Xiaomi-Robotics-U0
can substantially expand the visual coverage of the collected trajectories in a zero-shot manner.
Figure 10
:
The zero-shot augmented keyframes of the multi-view manipulation trajectories with the embodied transfer capability of
Xiaomi-Robotics-U0
. We generate 1000 descriptions for the background, workspace, and lighting condition with Qwen3-VL and replace the original condition randomly. The left panel is the original keyframes, and the right corresponds to the transferred.
Figure 11
:
More results of the zero-shot augmented keyframes of the multi-view manipulation trajectories with the embodied transfer capability of
Xiaomi-Robotics-U0
.
Figure 12
:
Qualitative results on embodied scene generation.
Figure 13
:
Qualitative comparison between GPT-Image-2 and
Xiaomi-Robotics-U0
on the embodied scence generation benchmark.
Figure 14
:
Pairwise evaluation results of embodied transfer and embodied scene generation.
To evaluate embodied scene generation, we construct a dedicated benchmark using LLM-generated scene prompts.
The benchmark contains 400 samples split equally into two difficulty levels: 200 Easy samples for conventional indoor manipulation scenes, and 200 Hard samples for unconstrained open-domain scenes.
Each prompt specifies a robot arm type and a structured scene description covering workspace layout, object configuration, lighting, and background.
We compare
Xiaomi-Robotics-U0
against GPT-Image-2 as the baseline.
To ensure a fair comparison, multi-view reference images corresponding to each target robot arm type are provided to GPT-Image-2 as visual context.
Quantifying cross-view geometric consistency remains an open challenge, as no automated metric reliably evaluates global 3D coherence across viewpoints.
We thus adopt a human pairwise preference protocol for quantitative assessment.
Annotators are presented with paired outputs from
Xiaomi-Robotics-U0
and GPT-Image-2 in randomly shuffled left-right orders to mitigate positional preference bias. Annotators select the superior result based on two criteria: (1) multi-view geometric consistency, measuring unified 3D layouts with invariant object positions, scales, and occlusion across views;
(2) textual instruction following, measuring alignment with the workspace, objects, lighting, and background in prompts.
Figure
14
reports the overall win rate.
Xiaomi-Robotics-U0
outperforms GPT-Image-2 in most comparisons, achieving clear advantages in multi-view consistency while maintaining comparable instruction-following performance.
Qualitative comparisons in Figure
13
show that GPT-Image-2 produces high-quality images but suffers from severe cross-view geometric inconsistencies, with conflicting object positions, scales, and layouts across cameras. By contrast,
Xiaomi-Robotics-U0
generates geometrically coherent multi-view observations with consistent spatial layouts, while retaining photorealistic quality. This verifies that
Xiaomi-Robotics-U0
simultaneously achieves high visual fidelity and robust cross-view consistency.
Figure
12
presents qualitative results covering four robot arm types.
Xiaomi-Robotics-U0
generates photorealistic and geometrically consistent multi-view initial scenes for diverse scene prompts, demonstrating strong open-ended generalization to arbitrary textual descriptions. This shows that our model extends conventional single-view text-to-image synthesis to multi-view robotic observation generation by learning intrinsic geometric relations between arm configurations and camera projection. Notably, among the four evaluated robot arm types, the AGIBOT G2 is trained exclusively on simulated data, yet
Xiaomi-Robotics-U0
can still produce photorealistic, real-world-aligned observations for this specific arm type. This validates the model’s promising sim-to-real generalization potential via learned real-world visual priors.
Figure 15
:
Stage-level progress visualization for each task. Each row corresponds to one task, and each column shows the head view at the completion of a progress milestone.
For
Store Earphones
, the eight milestones run from grasping the case, through storing both earbuds, to closing the lid and placing the case. For
Fold Towel
, the milestones cover lifting and flattening the towel, two sequential folds, and placing the folded towel. For
Pack Box
, they cover four pick-and-place steps followed by closing the box lid.
We conduct the real-world evaluation to investigate whether the proposed world model improves downstream policy learning. Specifically, we use the
Xiaomi-Robotics-U0
model to augment the expert demonstrations, and compare the VLA policies trained on the original and augmented data across a suite of real-world tasks.
The benchmark consists of three real-world tabletop manipulation tasks:
Store Earphones
,
Fold Towel
, and
Pack Box
, encompassing diverse challenges such as dexterous manipulation, deformable object handling, and multi-object interaction. Each task is defined by a language instruction and an ordered set of progress milestones.
Figure
15
visualizes the milestone states for each task, providing a concrete reference for the progress metric defined later in this section.
Figure 16
:
Task layouts for all three tasks.
The base group varies the object layout for each task.
For
Store Earphones
, different layouts vary the positions of the case and the earbuds.
For
Fold Towel
, the towel is initialized in a randomized configuration, which is clearer in the left-wrist view shown as an inset at the top-left corner.
For
Pack Box
, a new layout is generated by randomly placing the four objects on the table.
The interference group uses the same layout randomization but additionally introduces substantial changes in tablecloth and lighting.
Figure 17
:
Comparison of task completion progress between Original and
Xiaomi-Robotics-U0
-Aug, split by evaluation group.
Xiaomi-Robotics-U0
-Aug is on par with Original under in-distribution conditions (base group) but achieves substantially higher progress under held-out backgrounds and lighting (interference group), leading to a clear overall advantage.
For each task in the suite above, we independently post-train a
π
0.5
\pi_{0.5}
[
intelligence2025pi_0_5
]
policy starting from the
pi05_base
checkpoint, using its official PyTorch implementation.
All training settings are kept identical across compared policies; the only variable is the supervised fine-tuning data mixture.
For each task, we collect approximately 40 hours of real-world demonstrations on the robot platform described below, forming the task-specific clean data.
To construct augmentation data, we apply the embodied scene generation pipeline (Section
3.2
) to the clean demonstrations, producing approximately 40 hours of style-transferred episodes per task that vary visual appearance—backgrounds, lighting, and textures—while preserving the original robot states and action labels.
We compare two resulting per-task policies:
Original
is post-trained exclusively on the task-specific clean data;
Xiaomi-Robotics-U0
-Aug
is post-trained on the mixture of clean data and its corresponding style-transferred augmentation.
No inference-time world-model module is used during robot execution; the world model influences the policy solely through the post-training data mixture.
To evaluate robustness beyond the original demonstration distribution while keeping the comparison controlled, we evaluate each policy under two groups of real-world conditions: a base group and an interference group.
Each group contains three object-layout variants per task, and both policies are evaluated on an identical condition schedule.
In the base group, the three layouts differ only in object placement; the background uses tablecloths seen in the demonstration data, and the scene is evaluated under nominal laboratory illumination.
In the interference group, the layouts again vary object placement, but the background is replaced by held-out tablecloths absent from the demonstration data—often with more complex patterns and different textures or surface reflectance. The interference group further introduces held-out illumination, including low light, colored light, and dynamic patterned lighting such as disco-ball effects. The full set of evaluation layouts for every task and group is shown in Figure
16
.
For every policy, task, test group, and layout, we run three independent real-robot trials.
This yields
2
×
3
×
3
=
18
2\times 3\times 3=18
trials (two groups
×
\times
three layouts
×
\times
three trials) per policy-task pair.
We deploy all trained policies on a bimanual ARX robot platform with two 6-DoF arms, each equipped with a 1-DoF parallel-jaw gripper.
At each policy query, the observation consists of the current proprioceptive arm state and three RGB views: one head view from the head camera and two wrist views from the left and right wrist cameras. For
Store Earphones
and
Fold Towel
, these views are captured with Intel RealSense D405 cameras; for
Pack Box
, we use fisheye cameras in the same viewpoint configuration to cover the larger packing workspace. The fisheye images are undistorted before use, both as policy inputs and as inputs to the
Xiaomi-Robotics-U0
augmentation.
During real-robot evaluation, the policy is running on a separate inference server and communicates with the robot control process over a WebSocket connection on the local-area network.
At 1 Hz, the robot transmits the current arm state and the three RGB views to the server, which returns an action chunk that is executed until the next query.
The observation stream, communication stack, and action-chunk execution interface are identical across all compared policies.
We use task completion progress as our sole evaluation metric. Rather than scoring only full success, progress assigns partial credit based on the ordered milestones a rollout completes, making it better suited to long-horizon manipulation, where a policy may reliably solve early subgoals yet fail at a later stage. Specifically, if a task has
K
K
ordered milestones and the rollout completes the first
ℓ
\ell
of them, its progress is
ℓ
/
K
∈
[
0
,
1
]
\ell/K\in[0,1]
, where
1
1
denotes full success. For task
t
t
and group
g
∈
{
base
,
interference
}
g\in\{\text{base},\text{interference}\}
, the reported progress of the tested policy is defined as:
Prog
​
(
t
,
g
)
=
1
N
​
∑
i
=
1
N
ℓ
i
K
t
,
\mathrm{Prog}(t,g)=\frac{1}{N}\sum_{i=1}^{N}\frac{\ell_{i}}{K_{t}},
where
N
N
is the number of rollouts for each task and group (three layouts
×
\times
three trials in our setting). A higher progress score for
Xiaomi-Robotics-U0
-Aug indicates that the world model provides useful post-training variation for improving real-world robustness across both fully and partially completed executions.
Figure
17
reports the task completion progress of both policies.
In the base group, where test conditions closely resemble the demonstration distribution,
Xiaomi-Robotics-U0
-Aug performs comparably to Original, with a slight decrease on some tasks—an expected consequence of sharing training capacity with augmented data that does not add new information for in-distribution scenes.
In the interference group, however,
Xiaomi-Robotics-U0
-Aug consistently outperforms Original across all tasks, indicating that style-transferred augmentation induces visual invariance to novel backgrounds and lighting unseen in the clean demonstrations.
Overall, the robustness gains under distribution shift outweigh the marginal in-distribution trade-off, which is especially valuable for real-world deployment, where scene conditions cannot be tightly controlled.
To illustrate this gap qualitatively, Figure
18
presents representative
Xiaomi-Robotics-U0
-Aug rollouts under interference conditions for all three tasks.
Despite challenging perturbations—held-out tablecloths, colored lighting, and projected patterns—the
Xiaomi-Robotics-U0
-Aug policy still executes the intended manipulation sequences:
it stores both earbuds and closes the case in
Store Earphones
, completes the full folding sequence in
Fold Towel
, and packs all four objects before closing the lid in
Pack Box
.
Figure
19
further shows the typical failure modes that arise in our evaluation:
(1) failing to grasp the earbud under saturated lighting;
(2) misidentifying the orientation of the earphone case;
(3) failing to locate the towel edges against a visually similar tablecloth;
(4) unfolding an already-folded towel due to an incomplete grasp of the cloth edge;
(5)(6) attempting to grasp a projected light spot instead of the target object.
Under heavily corrupted visual observations, both policies occasionally exhibit these unexpected behaviors; however,
Xiaomi-Robotics-U0
-Aug is more likely to act correctly despite the perturbations or self-correct after a brief deviation.
The Original policy, by contrast, tends to become permanently stuck on visual artifacts or commit unrecoverable errors at a higher rate.
Figure 18
:
Xiaomi-Robotics-U0
-Aug rollouts under interference scenarios for
Store Earphones
,
Fold Towel
, and
Pack Box
(top to bottom). Within each task, the three rows are the head view, left wrist view, and right wrist view, with frames sampled across the trajectory.
Figure 19
:
Typical failure cases across tasks: (1)–(2):
Store Earphones
, (3)–(4):
Fold Towel
, (5)–(6):
Pack Box
.
Figure 20
:
Qualitative video generation results of
Xiaomi-Robotics-U0
.
A key capability of an Embodied World Model is serving as a data engine that generates physically plausible and visually high-quality videos from a single observation and a corresponding instruction. To evaluate this video generation capability for embodiment tasks, we benchmark
Xiaomi-Robotics-U0
on WorldArena
[
shang2026worldarena
]
. Given an initial frame along with a language instruction and robot action, WorldArena assesses the generated video across 15 metrics spanning six dimensions: visual quality, motion quality, content consistency, physics adherence, 3D accuracy, and controllability. We formulate this task as processing interleaved vision-language inputs and producing multi-frame interleaved outputs with empty language tokens. The initial frame paired with the language instruction serves as input, along with a sequence of robot mask images rendered from the robot action, camera parameters, and URDF as action guidance. Quantitative results are reported in Table
3
and Figure
21
, qualitative visualizations are shown in Figure
22
(rows 1–5).
Xiaomi-Robotics-U0
achieves the best overall EWMScore (73.64), ranking first among over 100 submitted models. In terms of controllability,
Xiaomi-Robotics-U0
attains the highest Instruction Following (93.86) and ranks competitively on Semantic Alignment, demonstrating its strong ability to translate language instructions into faithful visual outcomes. For physics adherence,
Xiaomi-Robotics-U0
leads on Interaction Quality (87.30), indicating that it captures physically plausible object interactions.
Xiaomi-Robotics-U0
also achieves the best Perspectivity (98.84) and Motion Smoothness (95.51), reflecting accurate 3D viewpoint geometry and temporally stable motion. Overall,
Xiaomi-Robotics-U0
strikes the best balance across all evaluation dimensions, confirming its effectiveness as a data engine for embodiment tasks.
Beyond the simulated WorldArena benchmark, we further leverage the scene generation capability of
Xiaomi-Robotics-U0
to produce novel initial frames and subsequently generate videos conditioned on language instructions. Representative examples are shown in Figure
20
. In the first row, the robot arm correctly interacts with the cup—grasping it, lifting it, and placing it stably on the table, demonstrating accurate contact modeling. In the second row, beyond the correct tabletop manipulation, the model also renders dynamic background changes by simulating a person walking toward the table to observe the robot’s operation, indicating awareness of scene-level dynamics beyond the immediate task. In the third row, the model faithfully simulates the physics of a deformable object (a sheet of paper): the paper bends realistically when picked up and set down, and exhibits a slight lag during lateral movement due to its low mass, reflecting an understanding of material-dependent physical properties. More visualizations are provided in Figure
22
.
Figure 21
:
WorldArena leaderboard (July 13, 2026). UNIS is the anonymous evaluation codename for
Xiaomi-Robotics-U0
.
Table 3
:
WorldArena dimension-level scores and overall EWMScore. The best score in each column is
bolded
.
Xiaomi-Robotics-U0
SisyphusWorld
BWM-Fast
SACWM
DexWorldEngine
Visual
Quality
Image Quality
53.94
45.57
51.22
45.83
51.44
Aesthetic Quality
40.79
38.38
40.15
37.30
40.96
JEPA Similarity
90.60
96.58
97.87
96.26
86.71
Motion
Quality
Dynamic Degree
73.70
76.27
69.58
70.35
69.16
Flow Score
86.02
99.82
75.11
95.79
74.31
Motion Smoothness
95.51
91.42
94.33
90.60
94.27
Content
Consistency
Subject Consistency
79.05
82.11
81.42
82.17
81.46
Background Consistency
86.44
86.52
90.17
87.90
88.85
Photometric Consistency
2.13
5.69
3.08
6.50
4.42
Physics
Adherence
Interaction Quality
87.30
71.98
79.88
72.28
81.04
Trajectory Accuracy
41.89
44.58
44.89
46.95
51.41
3D
Accuracy
Depth Accuracy
85.25
94.85
86.41
94.99
86.19
Perspectivity
98.84
90.34
97.14
92.34
97.58
Controll-
ability
Instruction Following
93.86
82.68
90.22
81.66
92.62
Semantic Alignment
89.35
89.18
89.15
89.16
89.45
EWMScore
(
↑
\uparrow
)
73.64
73.06
72.71
72.67
72.66
We evaluate the autoregressive
Xiaomi-Robotics-U0
image model on general text-to-image generation and image editing benchmarks. The goal of this evaluation is not to claim that
Xiaomi-Robotics-U0
is a dedicated open-domain image generator, but to quantify how much general image generation and editing ability is retained after adapting the model toward embodied scene synthesis. We use GenEval
[
ghosh2023geneval
]
for compositional T2I evaluation and ImgEdit
[
ye2026imgedit
]
Basic for single-turn image editing.
Table 4
:
Text-to-image and image-editing benchmark comparison with category-level breakdowns. GenEval scores are reported on a 0–1 scale, and ImgEdit scores use a 1–5 judge scale.
Model
Overall
Single object
Two object
Counting
Colors
Position
Attribute binding
Qwen-Image
0.87
0.99
0.92
0.89
0.88
0.76
0.77
Xiaomi-Robotics-U0
-AR
0.74
0.99
0.92
0.49
0.88
0.60
0.53
Model
Overall
Add
Adjust
Extract
Replace
Remove
Background
Style
Hybrid
Action
Qwen-Image
4.27
4.38
4.16
3.43
4.66
4.14
4.38
4.81
3.82
4.69
Xiaomi-Robotics-U0
-AR
4.08
4.21
4.15
3.28
4.35
4.21
4.03
4.72
3.27
4.53
As shown in Table
4
,
Xiaomi-Robotics-U0
preserves strong general T2I and image-editing capabilities after embodied post-training. It performs competitively on most appearance-oriented generation and local editing tasks. Performance drops are largely confined to highly complex scenarios that require multi-constraint composition, precise spatial grounding, or symbolic binding.
Table
4
demonstrates that
Xiaomi-Robotics-U0
retains robust general T2I and image-editing capabilities after embodied post-training. While it slightly lags behind Qwen-Image in highly complex cases—such as multi-constraint composition or precise spatial alignment—
Xiaomi-Robotics-U0
maintains highly comparable performance on common image generation and local editing instructions without catastrophic forgetting.

### 3.1
Embodied Transfer

To evaluate embodied transfer, we construct a benchmark of 300 samples, each consisting of a textual scene description together with a set of multi-view depth maps; the model is required to synthesize multi-view-consistent RGB images of the described robot manipulation scene.
Each sample additionally provides the multi-view RGB reference images from the original dataset. These references do not fully correspond to the (edited) scene description, but they capture the structural information beyond depth and are used to compute the benchmark metrics.
Since the depth maps are drawn from the same domain as the training set, we deliberately edit a subset of the scene descriptions so that they call for content not present in the training data, thereby probing the model’s general-purpose generation ability rather than mere reconstruction.
According to how much the edited description deviates from the original scene, we split the benchmark into 150
Easy
samples (descriptions close to the underlying scene) and 150
Hard
samples (descriptions requiring substantial novel content).
We compare against GPT-Image-2, a strong image generation model.
We assess generation quality from three complementary aspects, all computed against the reference views.
For
depth consistency
, we extract monocular depth from both the generated image and the reference with a shared estimator and report SI-RMSE (scale-invariant RMSE,
↓
\downarrow
),
δ
1
\delta_{1}
(the fraction of pixels with relative depth error below
1.25
1.25
,
↑
\uparrow
), and AbsRel (absolute relative error,
↓
\downarrow
); these measure how faithfully the generated image respects the 3D geometry imposed by the input depth.
For
structural fidelity
, we compute Canny edge F1, Precision, and Recall (
↑
\uparrow
), capturing whether the generated layout preserves the fine-grained contours of the scene.
For
semantic grounding
, we run open-vocabulary detection and segmentation and report Segmentation mIoU and Recall (
↑
\uparrow
), reflecting whether the described objects are correctly synthesized and localized.
As summarized in Table
2
,
Xiaomi-Robotics-U0
outperforms GPT-Image-2 by a large margin across all metrics on both the Easy and Hard splits, indicating markedly better adherence to the input geometry and stronger semantic alignment with the scene description.
Table 2
:
Objective metrics on the embodied transfer benchmark. Values in parentheses denote the change of
Xiaomi-Robotics-U0
relative to GPT-Image-2 on the same split (signed toward the actual value; a lower value is better for
↓
\downarrow
metrics and a higher value for
↑
\uparrow
metrics).
Metric
GPT-Image-2
Xiaomi-Robotics-U0
(Ours)
Easy
Hard
Easy
Hard
Depth SI-RMSE (
↓
\downarrow
)
0.4007
0.3444
0.1407 (
−
-
0.260
)
0.1435 (
−
-
0.201
)
Depth
δ
1
\delta_{1}
(
↑
\uparrow
)
0.5307
0.5340
0.8017 (
+
+
0.271
)
0.7846 (
+
+
0.251
)
Depth AbsRel (
↓
\downarrow
)
0.4350
0.4045
0.1647 (
−
-
0.270
)
0.1702 (
−
-
0.234
)
Canny F1 (
↑
\uparrow
)
0.0519
0.0507
0.2066 (
+
+
0.155
)
0.1815 (
+
+
0.131
)
Canny Precision (
↑
\uparrow
)
0.0560
0.0536
0.2037 (
+
+
0.148
)
0.1770 (
+
+
0.123
)
Canny Recall (
↑
\uparrow
)
0.0572
0.0582
0.2290 (
+
+
0.172
)
0.2176 (
+
+
0.159
)
Segmentation mIoU (
↑
\uparrow
)
0.4105
0.4098
0.7621 (
+
+
0.352
)
0.7436 (
+
+
0.334
)
Segmentation Recall (
↑
\uparrow
)
0.6208
0.6524
0.8136 (
+
+
0.193
)
0.8032 (
+
+
0.151
)
Figure 8
:
Qualitative results on embodied transfer. For each example we show the scene description, the original multi-view RGB images from which the depth conditions are extracted, and the images generated by
Xiaomi-Robotics-U0
. Starting from the original images, we estimate their multi-view depth maps and edit the scene description (edited text highlighted); the depth maps together with the edited description are then fed to
Xiaomi-Robotics-U0
, which synthesizes multi-view-consistent RGB images that adhere to both the edited text and the input geometry.
Figure
8
presents qualitative examples of embodied transfer.
For each example we display the scene description, the original multi-view RGB images from the source dataset, and the images generated by
Xiaomi-Robotics-U0
.
Our pipeline first estimates the multi-view depth maps from the original images, then edits the scene description—with the edited spans highlighted—to call for content that departs from the original scene; the extracted depth maps and the edited description are jointly fed to the model for generation.
Across diverse scenes, and given a scene description in either natural-language or structured form,
Xiaomi-Robotics-U0
faithfully follows both the edited textual description and the multi-view depth conditions, correctly rendering the described objects, lighting, and background while respecting the spatial geometry of each view.
Crucially, the generated views remain mutually consistent in object identity, appearance, and arrangement, demonstrating that our unified model produces coherent multi-view scenes rather than independently plausible but conflicting images.
Figure 9
:
Qualitative comparison between GPT-Image-2 and
Xiaomi-Robotics-U0
on the embodied transfer benchmark. GPT-Image-2 generates plausible, well-aligned images but violates the input depth and produces multi-view-inconsistent results, whereas
Xiaomi-Robotics-U0
adheres to the geometry of every view and keeps the views mutually consistent.
We further compare
Xiaomi-Robotics-U0
with GPT-Image-2 in Figure
9
, on three samples from the source datasets, each showing the scene description, the input multi-view depth maps, and the two models’ outputs.
GPT-Image-2 produces visually appealing images with strong text–image alignment: the described objects, materials, lighting, and background are all rendered plausibly, and in isolation each view looks convincing.
However, it exhibits a poor understanding of the input geometry and of cross-view relationships.
Its outputs frequently violate the depth conditions—objects are placed at inconsistent distances, surfaces are warped, and the overall spatial layout drifts away from the imposed geometry—which is reflected in its high SI-RMSE and low
δ
1
\delta_{1}
in Table
2
.
Moreover, the views generated for the same scene are largely independent: object identity, pose, and arrangement fail to remain consistent across viewpoints, so the three images do not correspond to a single coherent 3D scene.
In contrast,
Xiaomi-Robotics-U0
faithfully respects the depth of every view while keeping the multi-view outputs mutually consistent, indicating that it not only matches the strong text-following ability of GPT-Image-2 but also genuinely grounds generation in the input geometry.
Beyond benchmark evaluation, we leverage the structured text-to-image capability of
Xiaomi-Robotics-U0
to augment the demonstration trajectories collected for four real-world tasks (pack box, fold towel, pack phone, and pack earphone), aiming to endow downstream policies with better generalization to novel scenes.
For each trajectory, we prompt a VLM (Qwen3-VL) to generate, conditioned on the current depth maps and the original scene description, 1000 diverse alternatives for every dimension except the task objects, encouraging maximal diversity across background, workspace, and lighting.
We then randomly combine these per-dimension descriptions and feed them to
Xiaomi-Robotics-U0
for generation.
As shown in Figure
10
and Figure
11
, the resulting scenes exhibit rich diversity, and none of these compositions appear in the training set, demonstrating that
Xiaomi-Robotics-U0
can substantially expand the visual coverage of the collected trajectories in a zero-shot manner.
Figure 10
:
The zero-shot augmented keyframes of the multi-view manipulation trajectories with the embodied transfer capability of
Xiaomi-Robotics-U0
. We generate 1000 descriptions for the background, workspace, and lighting condition with Qwen3-VL and replace the original condition randomly. The left panel is the original keyframes, and the right corresponds to the transferred.
Figure 11
:
More results of the zero-shot augmented keyframes of the multi-view manipulation trajectories with the embodied transfer capability of
Xiaomi-Robotics-U0
.

##### Benchmark and protocol

To evaluate embodied transfer, we construct a benchmark of 300 samples, each consisting of a textual scene description together with a set of multi-view depth maps; the model is required to synthesize multi-view-consistent RGB images of the described robot manipulation scene.
Each sample additionally provides the multi-view RGB reference images from the original dataset. These references do not fully correspond to the (edited) scene description, but they capture the structural information beyond depth and are used to compute the benchmark metrics.
Since the depth maps are drawn from the same domain as the training set, we deliberately edit a subset of the scene descriptions so that they call for content not present in the training data, thereby probing the model’s general-purpose generation ability rather than mere reconstruction.
According to how much the edited description deviates from the original scene, we split the benchmark into 150
Easy
samples (descriptions close to the underlying scene) and 150
Hard
samples (descriptions requiring substantial novel content).
We compare against GPT-Image-2, a strong image generation model.

##### Metrics

We assess generation quality from three complementary aspects, all computed against the reference views.
For
depth consistency
, we extract monocular depth from both the generated image and the reference with a shared estimator and report SI-RMSE (scale-invariant RMSE,
↓
\downarrow
),
δ
1
\delta_{1}
(the fraction of pixels with relative depth error below
1.25
1.25
,
↑
\uparrow
), and AbsRel (absolute relative error,
↓
\downarrow
); these measure how faithfully the generated image respects the 3D geometry imposed by the input depth.
For
structural fidelity
, we compute Canny edge F1, Precision, and Recall (
↑
\uparrow
), capturing whether the generated layout preserves the fine-grained contours of the scene.
For
semantic grounding
, we run open-vocabulary detection and segmentation and report Segmentation mIoU and Recall (
↑
\uparrow
), reflecting whether the described objects are correctly synthesized and localized.
As summarized in Table
2
,
Xiaomi-Robotics-U0
outperforms GPT-Image-2 by a large margin across all metrics on both the Easy and Hard splits, indicating markedly better adherence to the input geometry and stronger semantic alignment with the scene description.
Table 2
:
Objective metrics on the embodied transfer benchmark. Values in parentheses denote the change of
Xiaomi-Robotics-U0
relative to GPT-Image-2 on the same split (signed toward the actual value; a lower value is better for
↓
\downarrow
metrics and a higher value for
↑
\uparrow
metrics).
Metric
GPT-Image-2
Xiaomi-Robotics-U0
(Ours)
Easy
Hard
Easy
Hard
Depth SI-RMSE (
↓
\downarrow
)
0.4007
0.3444
0.1407 (
−
-
0.260
)
0.1435 (
−
-
0.201
)
Depth
δ
1
\delta_{1}
(
↑
\uparrow
)
0.5307
0.5340
0.8017 (
+
+
0.271
)
0.7846 (
+
+
0.251
)
Depth AbsRel (
↓
\downarrow
)
0.4350
0.4045
0.1647 (
−
-
0.270
)
0.1702 (
−
-
0.234
)
Canny F1 (
↑
\uparrow
)
0.0519
0.0507
0.2066 (
+
+
0.155
)
0.1815 (
+
+
0.131
)
Canny Precision (
↑
\uparrow
)
0.0560
0.0536
0.2037 (
+
+
0.148
)
0.1770 (
+
+
0.123
)
Canny Recall (
↑
\uparrow
)
0.0572
0.0582
0.2290 (
+
+
0.172
)
0.2176 (
+
+
0.159
)
Segmentation mIoU (
↑
\uparrow
)
0.4105
0.4098
0.7621 (
+
+
0.352
)
0.7436 (
+
+
0.334
)
Segmentation Recall (
↑
\uparrow
)
0.6208
0.6524
0.8136 (
+
+
0.193
)
0.8032 (
+
+
0.151
)
Figure 8
:
Qualitative results on embodied transfer. For each example we show the scene description, the original multi-view RGB images from which the depth conditions are extracted, and the images generated by
Xiaomi-Robotics-U0
. Starting from the original images, we estimate their multi-view depth maps and edit the scene description (edited text highlighted); the depth maps together with the edited description are then fed to
Xiaomi-Robotics-U0
, which synthesizes multi-view-consistent RGB images that adhere to both the edited text and the input geometry.

##### Qualitative results

Figure
8
presents qualitative examples of embodied transfer.
For each example we display the scene description, the original multi-view RGB images from the source dataset, and the images generated by
Xiaomi-Robotics-U0
.
Our pipeline first estimates the multi-view depth maps from the original images, then edits the scene description—with the edited spans highlighted—to call for content that departs from the original scene; the extracted depth maps and the edited description are jointly fed to the model for generation.
Across diverse scenes, and given a scene description in either natural-language or structured form,
Xiaomi-Robotics-U0
faithfully follows both the edited textual description and the multi-view depth conditions, correctly rendering the described objects, lighting, and background while respecting the spatial geometry of each view.
Crucially, the generated views remain mutually consistent in object identity, appearance, and arrangement, demonstrating that our unified model produces coherent multi-view scenes rather than independently plausible but conflicting images.
Figure 9
:
Qualitative comparison between GPT-Image-2 and
Xiaomi-Robotics-U0
on the embodied transfer benchmark. GPT-Image-2 generates plausible, well-aligned images but violates the input depth and produces multi-view-inconsistent results, whereas
Xiaomi-Robotics-U0
adheres to the geometry of every view and keeps the views mutually consistent.
We further compare
Xiaomi-Robotics-U0
with GPT-Image-2 in Figure
9
, on three samples from the source datasets, each showing the scene description, the input multi-view depth maps, and the two models’ outputs.
GPT-Image-2 produces visually appealing images with strong text–image alignment: the described objects, materials, lighting, and background are all rendered plausibly, and in isolation each view looks convincing.
However, it exhibits a poor understanding of the input geometry and of cross-view relationships.
Its outputs frequently violate the depth conditions—objects are placed at inconsistent distances, surfaces are warped, and the overall spatial layout drifts away from the imposed geometry—which is reflected in its high SI-RMSE and low
δ
1
\delta_{1}
in Table
2
.
Moreover, the views generated for the same scene are largely independent: object identity, pose, and arrangement fail to remain consistent across viewpoints, so the three images do not correspond to a single coherent 3D scene.
In contrast,
Xiaomi-Robotics-U0
faithfully respects the depth of every view while keeping the multi-view outputs mutually consistent, indicating that it not only matches the strong text-following ability of GPT-Image-2 but also genuinely grounds generation in the input geometry.

##### Zero-shot data augmentation

Beyond benchmark evaluation, we leverage the structured text-to-image capability of
Xiaomi-Robotics-U0
to augment the demonstration trajectories collected for four real-world tasks (pack box, fold towel, pack phone, and pack earphone), aiming to endow downstream policies with better generalization to novel scenes.
For each trajectory, we prompt a VLM (Qwen3-VL) to generate, conditioned on the current depth maps and the original scene description, 1000 diverse alternatives for every dimension except the task objects, encouraging maximal diversity across background, workspace, and lighting.
We then randomly combine these per-dimension descriptions and feed them to
Xiaomi-Robotics-U0
for generation.
As shown in Figure
10
and Figure
11
, the resulting scenes exhibit rich diversity, and none of these compositions appear in the training set, demonstrating that
Xiaomi-Robotics-U0
can substantially expand the visual coverage of the collected trajectories in a zero-shot manner.
Figure 10
:
The zero-shot augmented keyframes of the multi-view manipulation trajectories with the embodied transfer capability of
Xiaomi-Robotics-U0
. We generate 1000 descriptions for the background, workspace, and lighting condition with Qwen3-VL and replace the original condition randomly. The left panel is the original keyframes, and the right corresponds to the transferred.
Figure 11
:
More results of the zero-shot augmented keyframes of the multi-view manipulation trajectories with the embodied transfer capability of
Xiaomi-Robotics-U0
.

### 3.2
Embodied Scene Generation

Figure 12
:
Qualitative results on embodied scene generation.
Figure 13
:
Qualitative comparison between GPT-Image-2 and
Xiaomi-Robotics-U0
on the embodied scence generation benchmark.
Figure 14
:
Pairwise evaluation results of embodied transfer and embodied scene generation.
To evaluate embodied scene generation, we construct a dedicated benchmark using LLM-generated scene prompts.
The benchmark contains 400 samples split equally into two difficulty levels: 200 Easy samples for conventional indoor manipulation scenes, and 200 Hard samples for unconstrained open-domain scenes.
Each prompt specifies a robot arm type and a structured scene description covering workspace layout, object configuration, lighting, and background.
We compare
Xiaomi-Robotics-U0
against GPT-Image-2 as the baseline.
To ensure a fair comparison, multi-view reference images corresponding to each target robot arm type are provided to GPT-Image-2 as visual context.
Quantifying cross-view geometric consistency remains an open challenge, as no automated metric reliably evaluates global 3D coherence across viewpoints.
We thus adopt a human pairwise preference protocol for quantitative assessment.
Annotators are presented with paired outputs from
Xiaomi-Robotics-U0
and GPT-Image-2 in randomly shuffled left-right orders to mitigate positional preference bias. Annotators select the superior result based on two criteria: (1) multi-view geometric consistency, measuring unified 3D layouts with invariant object positions, scales, and occlusion across views;
(2) textual instruction following, measuring alignment with the workspace, objects, lighting, and background in prompts.
Figure
14
reports the overall win rate.
Xiaomi-Robotics-U0
outperforms GPT-Image-2 in most comparisons, achieving clear advantages in multi-view consistency while maintaining comparable instruction-following performance.
Qualitative comparisons in Figure
13
show that GPT-Image-2 produces high-quality images but suffers from severe cross-view geometric inconsistencies, with conflicting object positions, scales, and layouts across cameras. By contrast,
Xiaomi-Robotics-U0
generates geometrically coherent multi-view observations with consistent spatial layouts, while retaining photorealistic quality. This verifies that
Xiaomi-Robotics-U0
simultaneously achieves high visual fidelity and robust cross-view consistency.
Figure
12
presents qualitative results covering four robot arm types.
Xiaomi-Robotics-U0
generates photorealistic and geometrically consistent multi-view initial scenes for diverse scene prompts, demonstrating strong open-ended generalization to arbitrary textual descriptions. This shows that our model extends conventional single-view text-to-image synthesis to multi-view robotic observation generation by learning intrinsic geometric relations between arm configurations and camera projection. Notably, among the four evaluated robot arm types, the AGIBOT G2 is trained exclusively on simulated data, yet
Xiaomi-Robotics-U0
can still produce photorealistic, real-world-aligned observations for this specific arm type. This validates the model’s promising sim-to-real generalization potential via learned real-world visual priors.
Figure 15
:
Stage-level progress visualization for each task. Each row corresponds to one task, and each column shows the head view at the completion of a progress milestone.
For
Store Earphones
, the eight milestones run from grasping the case, through storing both earbuds, to closing the lid and placing the case. For
Fold Towel
, the milestones cover lifting and flattening the towel, two sequential folds, and placing the folded towel. For
Pack Box
, they cover four pick-and-place steps followed by closing the box lid.

##### Benchmark and Protocol

To evaluate embodied scene generation, we construct a dedicated benchmark using LLM-generated scene prompts.
The benchmark contains 400 samples split equally into two difficulty levels: 200 Easy samples for conventional indoor manipulation scenes, and 200 Hard samples for unconstrained open-domain scenes.
Each prompt specifies a robot arm type and a structured scene description covering workspace layout, object configuration, lighting, and background.
We compare
Xiaomi-Robotics-U0
against GPT-Image-2 as the baseline.
To ensure a fair comparison, multi-view reference images corresponding to each target robot arm type are provided to GPT-Image-2 as visual context.

##### Human-Based Pairwise Evaluation

Quantifying cross-view geometric consistency remains an open challenge, as no automated metric reliably evaluates global 3D coherence across viewpoints.
We thus adopt a human pairwise preference protocol for quantitative assessment.
Annotators are presented with paired outputs from
Xiaomi-Robotics-U0
and GPT-Image-2 in randomly shuffled left-right orders to mitigate positional preference bias. Annotators select the superior result based on two criteria: (1) multi-view geometric consistency, measuring unified 3D layouts with invariant object positions, scales, and occlusion across views;
(2) textual instruction following, measuring alignment with the workspace, objects, lighting, and background in prompts.
Figure
14
reports the overall win rate.
Xiaomi-Robotics-U0
outperforms GPT-Image-2 in most comparisons, achieving clear advantages in multi-view consistency while maintaining comparable instruction-following performance.
Qualitative comparisons in Figure
13
show that GPT-Image-2 produces high-quality images but suffers from severe cross-view geometric inconsistencies, with conflicting object positions, scales, and layouts across cameras. By contrast,
Xiaomi-Robotics-U0
generates geometrically coherent multi-view observations with consistent spatial layouts, while retaining photorealistic quality. This verifies that
Xiaomi-Robotics-U0
simultaneously achieves high visual fidelity and robust cross-view consistency.

##### Qualitative Results

Figure
12
presents qualitative results covering four robot arm types.
Xiaomi-Robotics-U0
generates photorealistic and geometrically consistent multi-view initial scenes for diverse scene prompts, demonstrating strong open-ended generalization to arbitrary textual descriptions. This shows that our model extends conventional single-view text-to-image synthesis to multi-view robotic observation generation by learning intrinsic geometric relations between arm configurations and camera projection. Notably, among the four evaluated robot arm types, the AGIBOT G2 is trained exclusively on simulated data, yet
Xiaomi-Robotics-U0
can still produce photorealistic, real-world-aligned observations for this specific arm type. This validates the model’s promising sim-to-real generalization potential via learned real-world visual priors.
Figure 15
:
Stage-level progress visualization for each task. Each row corresponds to one task, and each column shows the head view at the completion of a progress milestone.
For
Store Earphones
, the eight milestones run from grasping the case, through storing both earbuds, to closing the lid and placing the case. For
Fold Towel
, the milestones cover lifting and flattening the towel, two sequential folds, and placing the folded towel. For
Pack Box
, they cover four pick-and-place steps followed by closing the box lid.

### 3.3
Real World Experiments

We conduct the real-world evaluation to investigate whether the proposed world model improves downstream policy learning. Specifically, we use the
Xiaomi-Robotics-U0
model to augment the expert demonstrations, and compare the VLA policies trained on the original and augmented data across a suite of real-world tasks.
The benchmark consists of three real-world tabletop manipulation tasks:
Store Earphones
,
Fold Towel
, and
Pack Box
, encompassing diverse challenges such as dexterous manipulation, deformable object handling, and multi-object interaction. Each task is defined by a language instruction and an ordered set of progress milestones.
Figure
15
visualizes the milestone states for each task, providing a concrete reference for the progress metric defined later in this section.
Figure 16
:
Task layouts for all three tasks.
The base group varies the object layout for each task.
For
Store Earphones
, different layouts vary the positions of the case and the earbuds.
For
Fold Towel
, the towel is initialized in a randomized configuration, which is clearer in the left-wrist view shown as an inset at the top-left corner.
For
Pack Box
, a new layout is generated by randomly placing the four objects on the table.
The interference group uses the same layout randomization but additionally introduces substantial changes in tablecloth and lighting.
Figure 17
:
Comparison of task completion progress between Original and
Xiaomi-Robotics-U0
-Aug, split by evaluation group.
Xiaomi-Robotics-U0
-Aug is on par with Original under in-distribution conditions (base group) but achieves substantially higher progress under held-out backgrounds and lighting (interference group), leading to a clear overall advantage.
For each task in the suite above, we independently post-train a
π
0.5
\pi_{0.5}
[
intelligence2025pi_0_5
]
policy starting from the
pi05_base
checkpoint, using its official PyTorch implementation.
All training settings are kept identical across compared policies; the only variable is the supervised fine-tuning data mixture.
For each task, we collect approximately 40 hours of real-world demonstrations on the robot platform described below, forming the task-specific clean data.
To construct augmentation data, we apply the embodied scene generation pipeline (Section
3.2
) to the clean demonstrations, producing approximately 40 hours of style-transferred episodes per task that vary visual appearance—backgrounds, lighting, and textures—while preserving the original robot states and action labels.
We compare two resulting per-task policies:
Original
is post-trained exclusively on the task-specific clean data;
Xiaomi-Robotics-U0
-Aug
is post-trained on the mixture of clean data and its corresponding style-transferred augmentation.
No inference-time world-model module is used during robot execution; the world model influences the policy solely through the post-training data mixture.
To evaluate robustness beyond the original demonstration distribution while keeping the comparison controlled, we evaluate each policy under two groups of real-world conditions: a base group and an interference group.
Each group contains three object-layout variants per task, and both policies are evaluated on an identical condition schedule.
In the base group, the three layouts differ only in object placement; the background uses tablecloths seen in the demonstration data, and the scene is evaluated under nominal laboratory illumination.
In the interference group, the layouts again vary object placement, but the background is replaced by held-out tablecloths absent from the demonstration data—often with more complex patterns and different textures or surface reflectance. The interference group further introduces held-out illumination, including low light, colored light, and dynamic patterned lighting such as disco-ball effects. The full set of evaluation layouts for every task and group is shown in Figure
16
.
For every policy, task, test group, and layout, we run three independent real-robot trials.
This yields
2
×
3
×
3
=
18
2\times 3\times 3=18
trials (two groups
×
\times
three layouts
×
\times
three trials) per policy-task pair.
We deploy all trained policies on a bimanual ARX robot platform with two 6-DoF arms, each equipped with a 1-DoF parallel-jaw gripper.
At each policy query, the observation consists of the current proprioceptive arm state and three RGB views: one head view from the head camera and two wrist views from the left and right wrist cameras. For
Store Earphones
and
Fold Towel
, these views are captured with Intel RealSense D405 cameras; for
Pack Box
, we use fisheye cameras in the same viewpoint configuration to cover the larger packing workspace. The fisheye images are undistorted before use, both as policy inputs and as inputs to the
Xiaomi-Robotics-U0
augmentation.
During real-robot evaluation, the policy is running on a separate inference server and communicates with the robot control process over a WebSocket connection on the local-area network.
At 1 Hz, the robot transmits the current arm state and the three RGB views to the server, which returns an action chunk that is executed until the next query.
The observation stream, communication stack, and action-chunk execution interface are identical across all compared policies.
We use task completion progress as our sole evaluation metric. Rather than scoring only full success, progress assigns partial credit based on the ordered milestones a rollout completes, making it better suited to long-horizon manipulation, where a policy may reliably solve early subgoals yet fail at a later stage. Specifically, if a task has
K
K
ordered milestones and the rollout completes the first
ℓ
\ell
of them, its progress is
ℓ
/
K
∈
[
0
,
1
]
\ell/K\in[0,1]
, where
1
1
denotes full success. For task
t
t
and group
g
∈
{
base
,
interference
}
g\in\{\text{base},\text{interference}\}
, the reported progress of the tested policy is defined as:
Prog
​
(
t
,
g
)
=
1
N
​
∑
i
=
1
N
ℓ
i
K
t
,
\mathrm{Prog}(t,g)=\frac{1}{N}\sum_{i=1}^{N}\frac{\ell_{i}}{K_{t}},
where
N
N
is the number of rollouts for each task and group (three layouts
×
\times
three trials in our setting). A higher progress score for
Xiaomi-Robotics-U0
-Aug indicates that the world model provides useful post-training variation for improving real-world robustness across both fully and partially completed executions.
Figure
17
reports the task completion progress of both policies.
In the base group, where test conditions closely resemble the demonstration distribution,
Xiaomi-Robotics-U0
-Aug performs comparably to Original, with a slight decrease on some tasks—an expected consequence of sharing training capacity with augmented data that does not add new information for in-distribution scenes.
In the interference group, however,
Xiaomi-Robotics-U0
-Aug consistently outperforms Original across all tasks, indicating that style-transferred augmentation induces visual invariance to novel backgrounds and lighting unseen in the clean demonstrations.
Overall, the robustness gains under distribution shift outweigh the marginal in-distribution trade-off, which is especially valuable for real-world deployment, where scene conditions cannot be tightly controlled.
To illustrate this gap qualitatively, Figure
18
presents representative
Xiaomi-Robotics-U0
-Aug rollouts under interference conditions for all three tasks.
Despite challenging perturbations—held-out tablecloths, colored lighting, and projected patterns—the
Xiaomi-Robotics-U0
-Aug policy still executes the intended manipulation sequences:
it stores both earbuds and closes the case in
Store Earphones
, completes the full folding sequence in
Fold Towel
, and packs all four objects before closing the lid in
Pack Box
.
Figure
19
further shows the typical failure modes that arise in our evaluation:
(1) failing to grasp the earbud under saturated lighting;
(2) misidentifying the orientation of the earphone case;
(3) failing to locate the towel edges against a visually similar tablecloth;
(4) unfolding an already-folded towel due to an incomplete grasp of the cloth edge;
(5)(6) attempting to grasp a projected light spot instead of the target object.
Under heavily corrupted visual observations, both policies occasionally exhibit these unexpected behaviors; however,
Xiaomi-Robotics-U0
-Aug is more likely to act correctly despite the perturbations or self-correct after a brief deviation.
The Original policy, by contrast, tends to become permanently stuck on visual artifacts or commit unrecoverable errors at a higher rate.
Figure 18
:
Xiaomi-Robotics-U0
-Aug rollouts under interference scenarios for
Store Earphones
,
Fold Towel
, and
Pack Box
(top to bottom). Within each task, the three rows are the head view, left wrist view, and right wrist view, with frames sampled across the trajectory.
Figure 19
:
Typical failure cases across tasks: (1)–(2):
Store Earphones
, (3)–(4):
Fold Towel
, (5)–(6):
Pack Box
.
Figure 20
:
Qualitative video generation results of
Xiaomi-Robotics-U0
.

##### Task suite

The benchmark consists of three real-world tabletop manipulation tasks:
Store Earphones
,
Fold Towel
, and
Pack Box
, encompassing diverse challenges such as dexterous manipulation, deformable object handling, and multi-object interaction. Each task is defined by a language instruction and an ordered set of progress milestones.
Figure
15
visualizes the milestone states for each task, providing a concrete reference for the progress metric defined later in this section.
Figure 16
:
Task layouts for all three tasks.
The base group varies the object layout for each task.
For
Store Earphones
, different layouts vary the positions of the case and the earbuds.
For
Fold Towel
, the towel is initialized in a randomized configuration, which is clearer in the left-wrist view shown as an inset at the top-left corner.
For
Pack Box
, a new layout is generated by randomly placing the four objects on the table.
The interference group uses the same layout randomization but additionally introduces substantial changes in tablecloth and lighting.
Figure 17
:
Comparison of task completion progress between Original and
Xiaomi-Robotics-U0
-Aug, split by evaluation group.
Xiaomi-Robotics-U0
-Aug is on par with Original under in-distribution conditions (base group) but achieves substantially higher progress under held-out backgrounds and lighting (interference group), leading to a clear overall advantage.

##### Policy training and data

For each task in the suite above, we independently post-train a
π
0.5
\pi_{0.5}
[
intelligence2025pi_0_5
]
policy starting from the
pi05_base
checkpoint, using its official PyTorch implementation.
All training settings are kept identical across compared policies; the only variable is the supervised fine-tuning data mixture.
For each task, we collect approximately 40 hours of real-world demonstrations on the robot platform described below, forming the task-specific clean data.
To construct augmentation data, we apply the embodied scene generation pipeline (Section
3.2
) to the clean demonstrations, producing approximately 40 hours of style-transferred episodes per task that vary visual appearance—backgrounds, lighting, and textures—while preserving the original robot states and action labels.
We compare two resulting per-task policies:
Original
is post-trained exclusively on the task-specific clean data;
Xiaomi-Robotics-U0
-Aug
is post-trained on the mixture of clean data and its corresponding style-transferred augmentation.
No inference-time world-model module is used during robot execution; the world model influences the policy solely through the post-training data mixture.

##### Evaluation schedule

To evaluate robustness beyond the original demonstration distribution while keeping the comparison controlled, we evaluate each policy under two groups of real-world conditions: a base group and an interference group.
Each group contains three object-layout variants per task, and both policies are evaluated on an identical condition schedule.
In the base group, the three layouts differ only in object placement; the background uses tablecloths seen in the demonstration data, and the scene is evaluated under nominal laboratory illumination.
In the interference group, the layouts again vary object placement, but the background is replaced by held-out tablecloths absent from the demonstration data—often with more complex patterns and different textures or surface reflectance. The interference group further introduces held-out illumination, including low light, colored light, and dynamic patterned lighting such as disco-ball effects. The full set of evaluation layouts for every task and group is shown in Figure
16
.
For every policy, task, test group, and layout, we run three independent real-robot trials.
This yields
2
×
3
×
3
=
18
2\times 3\times 3=18
trials (two groups
×
\times
three layouts
×
\times
three trials) per policy-task pair.

##### Robot setup

We deploy all trained policies on a bimanual ARX robot platform with two 6-DoF arms, each equipped with a 1-DoF parallel-jaw gripper.
At each policy query, the observation consists of the current proprioceptive arm state and three RGB views: one head view from the head camera and two wrist views from the left and right wrist cameras. For
Store Earphones
and
Fold Towel
, these views are captured with Intel RealSense D405 cameras; for
Pack Box
, we use fisheye cameras in the same viewpoint configuration to cover the larger packing workspace. The fisheye images are undistorted before use, both as policy inputs and as inputs to the
Xiaomi-Robotics-U0
augmentation.
During real-robot evaluation, the policy is running on a separate inference server and communicates with the robot control process over a WebSocket connection on the local-area network.
At 1 Hz, the robot transmits the current arm state and the three RGB views to the server, which returns an action chunk that is executed until the next query.
The observation stream, communication stack, and action-chunk execution interface are identical across all compared policies.

##### Metric

We use task completion progress as our sole evaluation metric. Rather than scoring only full success, progress assigns partial credit based on the ordered milestones a rollout completes, making it better suited to long-horizon manipulation, where a policy may reliably solve early subgoals yet fail at a later stage. Specifically, if a task has
K
K
ordered milestones and the rollout completes the first
ℓ
\ell
of them, its progress is
ℓ
/
K
∈
[
0
,
1
]
\ell/K\in[0,1]
, where
1
1
denotes full success. For task
t
t
and group
g
∈
{
base
,
interference
}
g\in\{\text{base},\text{interference}\}
, the reported progress of the tested policy is defined as:
Prog
​
(
t
,
g
)
=
1
N
​
∑
i
=
1
N
ℓ
i
K
t
,
\mathrm{Prog}(t,g)=\frac{1}{N}\sum_{i=1}^{N}\frac{\ell_{i}}{K_{t}},
where
N
N
is the number of rollouts for each task and group (three layouts
×
\times
three trials in our setting). A higher progress score for
Xiaomi-Robotics-U0
-Aug indicates that the world model provides useful post-training variation for improving real-world robustness across both fully and partially completed executions.

##### Experiment results

Figure
17
reports the task completion progress of both policies.
In the base group, where test conditions closely resemble the demonstration distribution,
Xiaomi-Robotics-U0
-Aug performs comparably to Original, with a slight decrease on some tasks—an expected consequence of sharing training capacity with augmented data that does not add new information for in-distribution scenes.
In the interference group, however,
Xiaomi-Robotics-U0
-Aug consistently outperforms Original across all tasks, indicating that style-transferred augmentation induces visual invariance to novel backgrounds and lighting unseen in the clean demonstrations.
Overall, the robustness gains under distribution shift outweigh the marginal in-distribution trade-off, which is especially valuable for real-world deployment, where scene conditions cannot be tightly controlled.
To illustrate this gap qualitatively, Figure
18
presents representative
Xiaomi-Robotics-U0
-Aug rollouts under interference conditions for all three tasks.
Despite challenging perturbations—held-out tablecloths, colored lighting, and projected patterns—the
Xiaomi-Robotics-U0
-Aug policy still executes the intended manipulation sequences:
it stores both earbuds and closes the case in
Store Earphones
, completes the full folding sequence in
Fold Towel
, and packs all four objects before closing the lid in
Pack Box
.
Figure
19
further shows the typical failure modes that arise in our evaluation:
(1) failing to grasp the earbud under saturated lighting;
(2) misidentifying the orientation of the earphone case;
(3) failing to locate the towel edges against a visually similar tablecloth;
(4) unfolding an already-folded towel due to an incomplete grasp of the cloth edge;
(5)(6) attempting to grasp a projected light spot instead of the target object.
Under heavily corrupted visual observations, both policies occasionally exhibit these unexpected behaviors; however,
Xiaomi-Robotics-U0
-Aug is more likely to act correctly despite the perturbations or self-correct after a brief deviation.
The Original policy, by contrast, tends to become permanently stuck on visual artifacts or commit unrecoverable errors at a higher rate.
Figure 18
:
Xiaomi-Robotics-U0
-Aug rollouts under interference scenarios for
Store Earphones
,
Fold Towel
, and
Pack Box
(top to bottom). Within each task, the three rows are the head view, left wrist view, and right wrist view, with frames sampled across the trajectory.
Figure 19
:
Typical failure cases across tasks: (1)–(2):
Store Earphones
, (3)–(4):
Fold Towel
, (5)–(6):
Pack Box
.
Figure 20
:
Qualitative video generation results of
Xiaomi-Robotics-U0
.

### 3.4
Video Generation

A key capability of an Embodied World Model is serving as a data engine that generates physically plausible and visually high-quality videos from a single observation and a corresponding instruction. To evaluate this video generation capability for embodiment tasks, we benchmark
Xiaomi-Robotics-U0
on WorldArena
[
shang2026worldarena
]
. Given an initial frame along with a language instruction and robot action, WorldArena assesses the generated video across 15 metrics spanning six dimensions: visual quality, motion quality, content consistency, physics adherence, 3D accuracy, and controllability. We formulate this task as processing interleaved vision-language inputs and producing multi-frame interleaved outputs with empty language tokens. The initial frame paired with the language instruction serves as input, along with a sequence of robot mask images rendered from the robot action, camera parameters, and URDF as action guidance. Quantitative results are reported in Table
3
and Figure
21
, qualitative visualizations are shown in Figure
22
(rows 1–5).
Xiaomi-Robotics-U0
achieves the best overall EWMScore (73.64), ranking first among over 100 submitted models. In terms of controllability,
Xiaomi-Robotics-U0
attains the highest Instruction Following (93.86) and ranks competitively on Semantic Alignment, demonstrating its strong ability to translate language instructions into faithful visual outcomes. For physics adherence,
Xiaomi-Robotics-U0
leads on Interaction Quality (87.30), indicating that it captures physically plausible object interactions.
Xiaomi-Robotics-U0
also achieves the best Perspectivity (98.84) and Motion Smoothness (95.51), reflecting accurate 3D viewpoint geometry and temporally stable motion. Overall,
Xiaomi-Robotics-U0
strikes the best balance across all evaluation dimensions, confirming its effectiveness as a data engine for embodiment tasks.
Beyond the simulated WorldArena benchmark, we further leverage the scene generation capability of
Xiaomi-Robotics-U0
to produce novel initial frames and subsequently generate videos conditioned on language instructions. Representative examples are shown in Figure
20
. In the first row, the robot arm correctly interacts with the cup—grasping it, lifting it, and placing it stably on the table, demonstrating accurate contact modeling. In the second row, beyond the correct tabletop manipulation, the model also renders dynamic background changes by simulating a person walking toward the table to observe the robot’s operation, indicating awareness of scene-level dynamics beyond the immediate task. In the third row, the model faithfully simulates the physics of a deformable object (a sheet of paper): the paper bends realistically when picked up and set down, and exhibits a slight lag during lateral movement due to its low mass, reflecting an understanding of material-dependent physical properties. More visualizations are provided in Figure
22
.
Figure 21
:
WorldArena leaderboard (July 13, 2026). UNIS is the anonymous evaluation codename for
Xiaomi-Robotics-U0
.
Table 3
:
WorldArena dimension-level scores and overall EWMScore. The best score in each column is
bolded
.
Xiaomi-Robotics-U0
SisyphusWorld
BWM-Fast
SACWM
DexWorldEngine
Visual
Quality
Image Quality
53.94
45.57
51.22
45.83
51.44
Aesthetic Quality
40.79
38.38
40.15
37.30
40.96
JEPA Similarity
90.60
96.58
97.87
96.26
86.71
Motion
Quality
Dynamic Degree
73.70
76.27
69.58
70.35
69.16
Flow Score
86.02
99.82
75.11
95.79
74.31
Motion Smoothness
95.51
91.42
94.33
90.60
94.27
Content
Consistency
Subject Consistency
79.05
82.11
81.42
82.17
81.46
Background Consistency
86.44
86.52
90.17
87.90
88.85
Photometric Consistency
2.13
5.69
3.08
6.50
4.42
Physics
Adherence
Interaction Quality
87.30
71.98
79.88
72.28
81.04
Trajectory Accuracy
41.89
44.58
44.89
46.95
51.41
3D
Accuracy
Depth Accuracy
85.25
94.85
86.41
94.99
86.19
Perspectivity
98.84
90.34
97.14
92.34
97.58
Controll-
ability
Instruction Following
93.86
82.68
90.22
81.66
92.62
Semantic Alignment
89.35
89.18
89.15
89.16
89.45
EWMScore
(
↑
\uparrow
)
73.64
73.06
72.71
72.67
72.66

##### WorldArena

A key capability of an Embodied World Model is serving as a data engine that generates physically plausible and visually high-quality videos from a single observation and a corresponding instruction. To evaluate this video generation capability for embodiment tasks, we benchmark
Xiaomi-Robotics-U0
on WorldArena
[
shang2026worldarena
]
. Given an initial frame along with a language instruction and robot action, WorldArena assesses the generated video across 15 metrics spanning six dimensions: visual quality, motion quality, content consistency, physics adherence, 3D accuracy, and controllability. We formulate this task as processing interleaved vision-language inputs and producing multi-frame interleaved outputs with empty language tokens. The initial frame paired with the language instruction serves as input, along with a sequence of robot mask images rendered from the robot action, camera parameters, and URDF as action guidance. Quantitative results are reported in Table
3
and Figure
21
, qualitative visualizations are shown in Figure
22
(rows 1–5).
Xiaomi-Robotics-U0
achieves the best overall EWMScore (73.64), ranking first among over 100 submitted models. In terms of controllability,
Xiaomi-Robotics-U0
attains the highest Instruction Following (93.86) and ranks competitively on Semantic Alignment, demonstrating its strong ability to translate language instructions into faithful visual outcomes. For physics adherence,
Xiaomi-Robotics-U0
leads on Interaction Quality (87.30), indicating that it captures physically plausible object interactions.
Xiaomi-Robotics-U0
also achieves the best Perspectivity (98.84) and Motion Smoothness (95.51), reflecting accurate 3D viewpoint geometry and temporally stable motion. Overall,
Xiaomi-Robotics-U0
strikes the best balance across all evaluation dimensions, confirming its effectiveness as a data engine for embodiment tasks.

##### Embodiment Video Generation

Beyond the simulated WorldArena benchmark, we further leverage the scene generation capability of
Xiaomi-Robotics-U0
to produce novel initial frames and subsequently generate videos conditioned on language instructions. Representative examples are shown in Figure
20
. In the first row, the robot arm correctly interacts with the cup—grasping it, lifting it, and placing it stably on the table, demonstrating accurate contact modeling. In the second row, beyond the correct tabletop manipulation, the model also renders dynamic background changes by simulating a person walking toward the table to observe the robot’s operation, indicating awareness of scene-level dynamics beyond the immediate task. In the third row, the model faithfully simulates the physics of a deformable object (a sheet of paper): the paper bends realistically when picked up and set down, and exhibits a slight lag during lateral movement due to its low mass, reflecting an understanding of material-dependent physical properties. More visualizations are provided in Figure
22
.
Figure 21
:
WorldArena leaderboard (July 13, 2026). UNIS is the anonymous evaluation codename for
Xiaomi-Robotics-U0
.
Table 3
:
WorldArena dimension-level scores and overall EWMScore. The best score in each column is
bolded
.
Xiaomi-Robotics-U0
SisyphusWorld
BWM-Fast
SACWM
DexWorldEngine
Visual
Quality
Image Quality
53.94
45.57
51.22
45.83
51.44
Aesthetic Quality
40.79
38.38
40.15
37.30
40.96
JEPA Similarity
90.60
96.58
97.87
96.26
86.71
Motion
Quality
Dynamic Degree
73.70
76.27
69.58
70.35
69.16
Flow Score
86.02
99.82
75.11
95.79
74.31
Motion Smoothness
95.51
91.42
94.33
90.60
94.27
Content
Consistency
Subject Consistency
79.05
82.11
81.42
82.17
81.46
Background Consistency
86.44
86.52
90.17
87.90
88.85
Photometric Consistency
2.13
5.69
3.08
6.50
4.42
Physics
Adherence
Interaction Quality
87.30
71.98
79.88
72.28
81.04
Trajectory Accuracy
41.89
44.58
44.89
46.95
51.41
3D
Accuracy
Depth Accuracy
85.25
94.85
86.41
94.99
86.19
Perspectivity
98.84
90.34
97.14
92.34
97.58
Controll-
ability
Instruction Following
93.86
82.68
90.22
81.66
92.62
Semantic Alignment
89.35
89.18
89.15
89.16
89.45
EWMScore
(
↑
\uparrow
)
73.64
73.06
72.71
72.67
72.66

### 3.5
Text2Image & Image Edit Benchmarks

We evaluate the autoregressive
Xiaomi-Robotics-U0
image model on general text-to-image generation and image editing benchmarks. The goal of this evaluation is not to claim that
Xiaomi-Robotics-U0
is a dedicated open-domain image generator, but to quantify how much general image generation and editing ability is retained after adapting the model toward embodied scene synthesis. We use GenEval
[
ghosh2023geneval
]
for compositional T2I evaluation and ImgEdit
[
ye2026imgedit
]
Basic for single-turn image editing.
Table 4
:
Text-to-image and image-editing benchmark comparison with category-level breakdowns. GenEval scores are reported on a 0–1 scale, and ImgEdit scores use a 1–5 judge scale.
Model
Overall
Single object
Two object
Counting
Colors
Position
Attribute binding
Qwen-Image
0.87
0.99
0.92
0.89
0.88
0.76
0.77
Xiaomi-Robotics-U0
-AR
0.74
0.99
0.92
0.49
0.88
0.60
0.53
Model
Overall
Add
Adjust
Extract
Replace
Remove
Background
Style
Hybrid
Action
Qwen-Image
4.27
4.38
4.16
3.43
4.66
4.14
4.38
4.81
3.82
4.69
Xiaomi-Robotics-U0
-AR
4.08
4.21
4.15
3.28
4.35
4.21
4.03
4.72
3.27
4.53
As shown in Table
4
,
Xiaomi-Robotics-U0
preserves strong general T2I and image-editing capabilities after embodied post-training. It performs competitively on most appearance-oriented generation and local editing tasks. Performance drops are largely confined to highly complex scenarios that require multi-constraint composition, precise spatial grounding, or symbolic binding.
Table
4
demonstrates that
Xiaomi-Robotics-U0
retains robust general T2I and image-editing capabilities after embodied post-training. While it slightly lags behind Qwen-Image in highly complex cases—such as multi-constraint composition or precise spatial alignment—
Xiaomi-Robotics-U0
maintains highly comparable performance on common image generation and local editing instructions without catastrophic forgetting.

## 4
Related Work

Recent years have witnessed rapid advances in foundation generative models, evolving from image synthesis to unified multimodal world modeling. Large-scale image generation and multimodal generation models, including Stable Diffusion
[
blattmann2023stable
]
, FLUX
[
labs2025flux1kontextflowmatching
]
, GPT-Image-2.0
[
openai_gptimage_2025
]
, Show-o
[
xie2025show
]
, Janus
[
wu2025janus
]
, Janus-Pro
[
chen2025janus
]
, Chameleon
[
team2024chameleon
]
, BAGEL
[
deng2025emerging
]
, and EMU3.5
[
cui2025emu3
]
, have demonstrated remarkable capabilities in semantic understanding, instruction following, compositional reasoning, and controllable visual generation. More recently, foundation video and world models, such as Genie
[
bruce2024genie
]
, Genie2
[
parkerholder2024genie2
]
, Sora
[
openai2024sora
]
, MovieGen
[
polyak2024movie
]
, Wan
[
wan2025wan
]
, and Cosmos
[
agarwal2026cosmos
]
, further extend generation from static images to long-horizon videos and interactive world simulation, enabling coherent prediction of future observations under a unified autoregressive or diffusion framework. Despite their impressive generalization ability, these models are designed for generic visual content and do not explicitly model the geometric constraints, robot embodiments, calibrated camera systems, or physical interaction dynamics required by embodied manipulation.
To bridge this gap, recent embodied world models introduce robot-centric world modeling by adapting foundation models to manipulation trajectories and robot videos. Representative approaches include Dreamer Series
[
hafner2019dream
]
, World Action Models (WAMs)
[
yuan2026fast
,
ye2026gigaworld
,
li2026causal
,
team2026motubrain
]
, DreamZero
[
ye2026world
]
, and Qwen-RobotWorld
[
zhang2026qwen
]
, which learn predictive models of future observations for planning, simulation, or policy learning. While these methods significantly advance embodied prediction, they are typically optimized using robot-only datasets whose scale and diversity remain substantially smaller than those used for foundation generative models. Consequently, continual adaptation often sacrifices part of the semantic richness, controllability, and visual diversity inherited from foundation pre-training. In contrast, our work treats embodied generation as a natural extension of foundation image and video generation. Rather than specializing a foundation model solely on robot trajectories, we jointly optimize general-domain generation and embodied synthesis within a unified autoregressive framework, preserving the original generation capability while acquiring robot-centric geometric reasoning and interaction modeling.
Recent advances in controllable image generation and image editing have substantially improved semantic manipulation through instruction-guided synthesis and reference-conditioned editing. Representative approaches, including InstructPix2Pix
[
brooks2023instructpix2pix
]
, AnyDoor
[
chen2024anydoor
]
, OmniGen
[
xiao2025omnigen
]
, GPT-Image-2.0
[
openai_gptimage_2025
]
, and FLUX Kontext
[
labs2025flux1kontextflowmatching
]
, enable high-quality editing with strong instruction following and semantic consistency. However, these methods are primarily designed for natural images and generally assume single-view observations, making them difficult to directly apply to embodied manipulation, where multiple calibrated cameras, robot embodiments, and interaction states must remain geometrically consistent.
Meanwhile, multi-view generation has attracted increasing attention in image synthesis and 3D content creation. Representative methods, including MVDream
[
shi2024mvdream
]
, SyncDreamer
[
liu2024syncdreamer
]
, Wonder3D
[
long2024wonder3d
]
, Era3D
[
li2024era3d
]
, Zero123++
[
shi2023zero123++
]
, and related multi-view diffusion models
[
gao2024cat3d
,
tang2024mvdiffusion++
]
, explicitly enforce cross-view consistency to facilitate high-quality 3D reconstruction and asset generation. Nevertheless, these methods mainly focus on object-centric generation and do not explicitly consider robot embodiments, calibrated manipulation cameras, or interaction-aware scene synthesis.
Our work bridges these two research directions by unifying controllable image editing, multi-view embodied scene generation, and embodied transfer within a single autoregressive framework. Furthermore, we introduce a structured control formulation that disentangles workspace layout, background appearance, foreground irrelevant objects, target objects, and lighting into independent control dimensions, enabling significantly richer embodied scene variations while preserving robot geometry, embodiment constraints, and multi-view consistency.
Video generation and embodied world simulation have recently become increasingly important for scalable robot learning. Foundation video models such as MovieGen
[
polyak2024movie
]
, Wan
[
wan2025wan
]
, Cosmos
[
agarwal2026cosmos
]
, Genie
[
bruce2024genie
]
, and Qwen-RobotWorld
[
zhang2026qwen
]
demonstrate impressive capabilities in long-horizon video synthesis and future observation prediction. Meanwhile, large-scale robot datasets and simulation platforms, including Open X-Embodiment
[
o2024open
]
, RoboCasa
[
nasiriany2024robocasa
]
, ManiSkill
[
mu2021maniskill
]
, BridgeData V2
[
walke2023bridgedata
]
, and DROID
[
khazatsky2024droid
]
, have substantially accelerated the development of data-driven robot learning. More recently, embodied video datasets and synthetic data pipelines, such as RoboMIND
[
wu2024robomind
]
, RoboCoin
[
wu2025robocoin
]
, GenieSim
[
yin2026geniesim30
]
, InternA1
[
tian2026interndata
]
, and AgiBotWorld
[
bu2025agibot
]
, further explore large-scale embodied video generation and robot data augmentation.
Despite these advances, existing approaches typically synthesize isolated images, simulation assets, or short robot videos independently, without jointly modeling controllable scene generation, embodied transfer, and long-horizon trajectory rollout. In contrast, our work unifies embodied scene generation, embodied transfer, interleaved subtask-subgoal prediction, and embodied video generation within a single autoregressive model. By jointly learning embodied videos at multiple temporal resolutions, our model captures both long-horizon task progression and fine-grained interaction dynamics, transforming foundation world models into scalable embodied data engines for downstream policy learning and future agentic embodied systems.

### 4.1
Foundation Generative Models and Embodied World Models

Recent years have witnessed rapid advances in foundation generative models, evolving from image synthesis to unified multimodal world modeling. Large-scale image generation and multimodal generation models, including Stable Diffusion
[
blattmann2023stable
]
, FLUX
[
labs2025flux1kontextflowmatching
]
, GPT-Image-2.0
[
openai_gptimage_2025
]
, Show-o
[
xie2025show
]
, Janus
[
wu2025janus
]
, Janus-Pro
[
chen2025janus
]
, Chameleon
[
team2024chameleon
]
, BAGEL
[
deng2025emerging
]
, and EMU3.5
[
cui2025emu3
]
, have demonstrated remarkable capabilities in semantic understanding, instruction following, compositional reasoning, and controllable visual generation. More recently, foundation video and world models, such as Genie
[
bruce2024genie
]
, Genie2
[
parkerholder2024genie2
]
, Sora
[
openai2024sora
]
, MovieGen
[
polyak2024movie
]
, Wan
[
wan2025wan
]
, and Cosmos
[
agarwal2026cosmos
]
, further extend generation from static images to long-horizon videos and interactive world simulation, enabling coherent prediction of future observations under a unified autoregressive or diffusion framework. Despite their impressive generalization ability, these models are designed for generic visual content and do not explicitly model the geometric constraints, robot embodiments, calibrated camera systems, or physical interaction dynamics required by embodied manipulation.
To bridge this gap, recent embodied world models introduce robot-centric world modeling by adapting foundation models to manipulation trajectories and robot videos. Representative approaches include Dreamer Series
[
hafner2019dream
]
, World Action Models (WAMs)
[
yuan2026fast
,
ye2026gigaworld
,
li2026causal
,
team2026motubrain
]
, DreamZero
[
ye2026world
]
, and Qwen-RobotWorld
[
zhang2026qwen
]
, which learn predictive models of future observations for planning, simulation, or policy learning. While these methods significantly advance embodied prediction, they are typically optimized using robot-only datasets whose scale and diversity remain substantially smaller than those used for foundation generative models. Consequently, continual adaptation often sacrifices part of the semantic richness, controllability, and visual diversity inherited from foundation pre-training. In contrast, our work treats embodied generation as a natural extension of foundation image and video generation. Rather than specializing a foundation model solely on robot trajectories, we jointly optimize general-domain generation and embodied synthesis within a unified autoregressive framework, preserving the original generation capability while acquiring robot-centric geometric reasoning and interaction modeling.

### 4.2
Controllable and Multi-view Embodied Generation

Recent advances in controllable image generation and image editing have substantially improved semantic manipulation through instruction-guided synthesis and reference-conditioned editing. Representative approaches, including InstructPix2Pix
[
brooks2023instructpix2pix
]
, AnyDoor
[
chen2024anydoor
]
, OmniGen
[
xiao2025omnigen
]
, GPT-Image-2.0
[
openai_gptimage_2025
]
, and FLUX Kontext
[
labs2025flux1kontextflowmatching
]
, enable high-quality editing with strong instruction following and semantic consistency. However, these methods are primarily designed for natural images and generally assume single-view observations, making them difficult to directly apply to embodied manipulation, where multiple calibrated cameras, robot embodiments, and interaction states must remain geometrically consistent.
Meanwhile, multi-view generation has attracted increasing attention in image synthesis and 3D content creation. Representative methods, including MVDream
[
shi2024mvdream
]
, SyncDreamer
[
liu2024syncdreamer
]
, Wonder3D
[
long2024wonder3d
]
, Era3D
[
li2024era3d
]
, Zero123++
[
shi2023zero123++
]
, and related multi-view diffusion models
[
gao2024cat3d
,
tang2024mvdiffusion++
]
, explicitly enforce cross-view consistency to facilitate high-quality 3D reconstruction and asset generation. Nevertheless, these methods mainly focus on object-centric generation and do not explicitly consider robot embodiments, calibrated manipulation cameras, or interaction-aware scene synthesis.
Our work bridges these two research directions by unifying controllable image editing, multi-view embodied scene generation, and embodied transfer within a single autoregressive framework. Furthermore, we introduce a structured control formulation that disentangles workspace layout, background appearance, foreground irrelevant objects, target objects, and lighting into independent control dimensions, enabling significantly richer embodied scene variations while preserving robot geometry, embodiment constraints, and multi-view consistency.

### 4.3
Embodied Video Generation and Synthetic Data Engines

Video generation and embodied world simulation have recently become increasingly important for scalable robot learning. Foundation video models such as MovieGen
[
polyak2024movie
]
, Wan
[
wan2025wan
]
, Cosmos
[
agarwal2026cosmos
]
, Genie
[
bruce2024genie
]
, and Qwen-RobotWorld
[
zhang2026qwen
]
demonstrate impressive capabilities in long-horizon video synthesis and future observation prediction. Meanwhile, large-scale robot datasets and simulation platforms, including Open X-Embodiment
[
o2024open
]
, RoboCasa
[
nasiriany2024robocasa
]
, ManiSkill
[
mu2021maniskill
]
, BridgeData V2
[
walke2023bridgedata
]
, and DROID
[
khazatsky2024droid
]
, have substantially accelerated the development of data-driven robot learning. More recently, embodied video datasets and synthetic data pipelines, such as RoboMIND
[
wu2024robomind
]
, RoboCoin
[
wu2025robocoin
]
, GenieSim
[
yin2026geniesim30
]
, InternA1
[
tian2026interndata
]
, and AgiBotWorld
[
bu2025agibot
]
, further explore large-scale embodied video generation and robot data augmentation.
Despite these advances, existing approaches typically synthesize isolated images, simulation assets, or short robot videos independently, without jointly modeling controllable scene generation, embodied transfer, and long-horizon trajectory rollout. In contrast, our work unifies embodied scene generation, embodied transfer, interleaved subtask-subgoal prediction, and embodied video generation within a single autoregressive model. By jointly learning embodied videos at multiple temporal resolutions, our model captures both long-horizon task progression and fine-grained interaction dynamics, transforming foundation world models into scalable embodied data engines for downstream policy learning and future agentic embodied systems.

## 5
Conclusions

In this report, we presented
Xiaomi-Robotics-U0
, a unified embodied synthesis model built upon a world foundation model through lightweight continual training on general-domain and embodied datasets.
Xiaomi-Robotics-U0
unifies text-to-image generation, image editing, multi-view embodied scene generation, embodied transfer, and embodied video generation within a single autoregressive framework, successfully transferring the rich visual knowledge of foundation models into embodied intelligence. To the best of our knowledge,
Xiaomi-Robotics-U0
is the first unified embodied synthesis model supporting both high-quality multi-view embodied generation and sequential embodied video generation, as well as the first to introduce a structured control formulation for scalable embodied data generation. Extensive experiments demonstrate state-of-the-art performance across both single-step and sequential embodied generation tasks, outperforming GPT-Image-2.0 in human evaluation, achieving state-of-the-art results on the World Arena benchmark, and substantially improving the robustness of downstream robot policies through generated data. We hope this work provides a practical step toward scalable agentic data engines for embodied intelligence.
Despite the promising results,
Xiaomi-Robotics-U0
has several limitations. First, embodied transfer still relies on depth estimation as an intermediate representation, which may introduce occasional artifacts and limit fine-grained control over object textures and local appearances. Future work will investigate embodied multi-view image editing directly on raw observations. Second, embodied scene generation and video generation are performed separately, making long-horizon rollout susceptible to accumulated errors; jointly generating scenes and subsequent videos is a promising direction. Finally, the current 32K context window limits long-duration video modeling, and extending the model to longer contexts will enable minute-level embodied interaction generation.

## Contributions

∗
Core contributors, alphabetically listed by first name
∗∗
External contributor
Project Lead
•
Xinghang Li
Data
:
•
Jun Guo
∗
•
Qiwei Li
∗
•
Hang Lai
∗
•
Long Qian
∗
•
Yueze Wang
∗
•
Nan Sun
∗
•
Hongyu Yan
∗
•
Jingen Qu
•
Jiaxi Song
•
Jiahang Cao
•
Hanye Zhao
•
Heyun Wang
•
Yunhong Wang
•
Jiaxi Song
•
Caoyu Xia
•
Jack Zhao
Training
:
•
Jun Guo
∗
•
Qiwei Li
∗
•
Long Qian
∗
•
Yueze Wang
∗
•
Jingen Qu
•
Jiaxi Song
•
Nan Sun
•
Hongyu Yan
Deployment & Experiments
:
•
Jiahang Cao
∗
•
Jun Guo
∗
•
Hang Lai
∗
•
Qiwei Li
∗
•
Long Qian
∗
•
Jiaxi Song
∗
•
Hongyu Yan
∗
•
Hanye Zhao
∗
•
Futeng Liu
•
Wanli Peng
Inference Speedup
:
•
Xi Chen
∗
•
Jingen Qu
∗
Infrastructure
:
•
Xinghang Li
∗
•
Jingen Qu
•
Xi Chen
Senior Lead
:
•
Jason Li
∗
•
Diyun Xiang
•
Hangjun Ye
•
Heng Qu
•
Huaping Liu
∗∗

## Acknowledgment

We would like to express our sincere appreciation to the broader team for their tremendous support, including those not listed above: Li Jiang, Zhiyin Lai, Xiaoke Xilinjueluo, Qingyi Li, Qi Liu, Yayun Liu, Jun Xia, Feng Qiu, Donghao Wang, Yan Hou, Dong Wang, Quanyun Zhou, Liangliang He, Jiaxin Liu, Kang Zhou, Rui Cai, Shuoxue Bi, Yingchao Zhou, Kun Ma, Yiwei Zhou
Figure 22
:
Qualitative video generation results of
Xiaomi-Robotics-U0
. Rows 1–5: WorldArena benchmark samples conditioned on language instructions and initial observations. Rows 6–10: videos generated from
Xiaomi-Robotics-U0
produced scene images. Each row shows the initial frame and language instructions followed by predicted future frames.
Figure 23
:
Qualitative results of embodied transfer. We extract depths and modify descriptions from the original scene to generate the new scene.
Figure 24
:
Embodied scene generation results.
Figure 25
:
Text-to-image results of
Xiaomi-Robotics-U0
.
Figure 26
:
Image editing results of
Xiaomi-Robotics-U0
.

## References

