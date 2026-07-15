# EgoSteer: A Full-Stack System Towards Steerable Dexterous Manipulation from Egocentric Videos

**Authors:** Yifan Zhong
1,2*
,
Zhang Chen
1,2*
,
Tianrui Guan
1,2*
,
Fanlian Zeng
2,3*
,
Yuyao Ye
1,2
,
Tianjia He
2
,
Ka Nam Lui
1,2
,
Jiayi Li
1,2
,
Tingrui Zhang
1,2
,
Ruilin Yan
1,2
,
Xinhao Ji
1,2
,
Guangyu Zhao
1,2
,
Wenjie Lou
1,2
,
Jiayuan Zhang
1,2
,
Yuanpei Chen
1,2
†
{\dagger}
,
Yaodong Yang
1,2
†
{\dagger}
1
Institute for AI, PKU,
2
PKU-PsiBot Joint Lab,
3
UPenn

**arXiv:** 2607.09701


## 1
Introduction

A central goal of general-purpose embodied intelligence is to enable robots to perform diverse manipulation tasks from open-ended human intent. Despite rapid progress in embodied foundation models
[
5
,
22
,
21
,
33
,
32
,
66
,
35
,
36
,
37
,
39
,
28
,
57
,
8
,
69
]
, most systems still require task-specific fine-tuning, while the few that follow free-form language are largely limited to grippers
[
20
,
60
]
. Dual-dexterous-hand robots provide a more expressive embodiment, with greater actuation capacity and fine-grained interaction potential for general-purpose manipulation. Yet on this more capable but more challenging platform, steerable dexterous manipulation remains largely unrealized.
The key bottleneck lies in data and system scalability. While language-guided manipulation demands large-scale, high-quality data, collecting such demonstrations directly on dexterous robots is exceptionally difficult, particularly for a specific embodiment. Egocentric human videos
[
19
,
43
]
offer a scalable alternative, as human hand manipulations contain rich interaction knowledge and are spontaneously generated at a massive scale. However, raw egocentric videos are noisy and lack reliable language and action annotations. Without systematic curation, these unstructured videos provide unstable supervision and can degrade downstream robot policies. Even with high-quality human data, the system must employ high-capacity models trained with effective objectives on scalable infrastructure, and ground the learned priors to the target robot. Failing to address any of these co-dependent components prevents the realization of general language-following manipulation.
In this paper, we close this gap by proposing and open-sourcing a full-stack system for steerable dexterous manipulation. Our system begins with
EgoSmith
, an egocentric data pipeline that curates in-the-wild egocentric videos into clean, fully-annotated training data. By integrating pre-filtering, 4D motion estimation, language labeling, and post-filtering, EgoSmith aligns RGB-D images, world-space hand trajectories, camera parameters, and textual instructions, achieving a
9
×
9\times
throughput increase and more precise, comprehensive annotations than prior open-source SOTA
[
64
]
. Using EgoSmith, we curate a
9.6
9.6
K-hour pre-training corpus across
12
12
egocentric datasets. To ground these human-hand interaction priors onto physical embodiments, we design a unified robot stack for teleoperation, model inference, and human-in-the-loop correction. By mapping the operator’s relative motions onto the intervened robot states, this stack enables seamless expert intervention for efficient DAgger
[
48
]
refinement from arbitrary deployment states, effectively correcting policy failures. Using this framework, we collect
187
187
hours of high-quality teleoperation data across
193
193
semantically-diverse tasks. Finally, we introduce
EgoSteer
, a novel world-model-enhanced VLA trained on an optimized infrastructure. By integrating a world-model expert that predicts action-induced future states in the DINOv3
[
49
]
latent space, EgoSteer enhances the VLM backbone’s action imagination and modality alignment, enabling steerable and fine-grained manipulation. To facilitate human-robot transfer, EgoSteer employs a unified action space of wrist poses and fingertip keypoints
[
9
,
14
]
, coupled with training-time RTC
[
6
]
to eliminate real-world execution pauses.
Empirically, through large-scale egocentric pre-training, diverse real-robot post-training, and efficient DAgger refinement, EgoSteer robustly follows free-form instructions to execute over
40
40
tasks with a
75
%
75\%
average success rate, exhibiting fine-grained dexterity, failure recovery, and generalization. Furthermore, systematic evaluations confirm the significance of each component, including egocentric pre-training data scale and quality, the world-model objective, training-time RTC, and DAgger refinement, enabling EgoSteer to consistently outperform strong baselines such as
π
0.5
\pi_{0.5}
[
22
]
and Being-H0.5
[
36
]
. Additionally, the pre-trained manipulation priors also enable few-shot adaptation to challenging long-horizon tasks, such as box folding and cake unboxing, across multiple embodiments, achieving a 75+% success rate. Conversely, our from-scratch baseline and sample-efficient imitation learning methods, including Diffusion Policy
[
11
]
and IMLE
[
45
]
, fail entirely, illustrating the inherent difficulty of these tasks and the efficacy of our curated pre-training priors. To summarize, our contributions are five-fold:
•
EgoSmith
, an egocentric data pipeline that curates a
9.6
9.6
K-hour fully-annotated pre-training corpus across 12 datasets, achieving a 9
×
\times
throughput and better accuracy over prior SOTA.
•
A unified robot stack
integrating teleoperation, inference, and seamless human-in-the-loop DAgger correction, that collects 187 hours of data across 193 dexterous tasks.
•
EgoSteer
, a world-model-enhanced VLA alongside an optimized training infrastructure.
•
Extensive real-robot evaluations
demonstrating that EgoSteer robustly performs free-form language-following across 40 tasks and few-shot adapts to complex long-horizon tasks.
•
Open-source release
of our complete system, datasets, and model checkpoints.

## 2
Related Work

Generalist robot policies.
Generalist manipulation policies based on foundation models have recently emerged towards general-purpose manipulation
[
22
,
67
,
51
,
25
,
40
]
. However, early efforts remain largely confined to simple tasks and rely heavily on single-task fine-tuning, struggling to follow free-form language instructions. Although recent works
[
20
,
60
]
have made breakthroughs in generalization, they rely heavily on massive real-robot multi-task datasets, or requires intensive computation and complex optimization to sustain real-time control and are limited to grippers.
Scaling with egocentric human videos.
Egocentric videos
[
19
,
43
,
16
,
2
,
54
,
63
,
34
,
4
,
1
]
(approximately
116
116
K hours) and data processing tools
[
64
,
29
]
offer a scalable source for learning dexterous manipulation. Existing policies
[
66
,
35
,
36
,
9
,
14
,
59
]
leverage these videos through large-scale pre-training or cross-embodiment co-training. By aligning human hands and robot action spaces through diverse approaches, these methods effectively transfer human priors to the robot domain. Notably, EgoScale
[
66
]
reveals a log-linear scaling law in pre-training and introduces a mid-training stage to align human and robot space. Nevertheless, converting massive human videos into training signals remains highly inefficient, and current policies still struggle with steerable control.
Human-in-the-loop post-training.
Human-in-the-loop post-training is key to elevating the performance ceiling and resolving out-of-distribution failures with high sample efficiency. Recent paradigms leverage online reinforcement learning with human copilots
[
38
]
, compliant residual feedback for contact-rich tasks
[
58
]
, or hand-arm intervention frameworks for dexterous VLAs
[
17
,
30
]
. However, these approaches still struggle with real-time, high-frequency corrections in high-DoF joint spaces and require prohibitive trajectory labeling labor.

## 3
EgoSmith: Curating Egocentric Videos into Grounded Dexterous Priors

While egocentric data are rich in fine-grained interactions and highly scalable for general embodied learning, they are typically monocular RGB videos suffering from camera jitter, frequent occlusions, and a lack of annotations. This section presents
EgoSmith
(
Figure
2
), an efficient automated pipeline that transforms massive raw videos into fully-annotated training samples to enable effective learning.
To achieve raw data cleaning, labeling, and quality control, EgoSmith employs a four-stage pipeline. The
first
stage,
pre-filtering
, uses simple yet effective heuristics to discard locomotion segments and hand misidentifications that degrade downstream 4D motion estimation. Specifically, we filter out active displacement by computing average optical flow over a
128
128
-point grid, leveraging its strong correlation with human locomotion in egocentric videos. We then eliminate frames with severe occlusions or bystander interference by applying geometric criteria to the hand counts, scales, and coordinates detected by YOLO
[
46
,
42
]
, preserving only clearly visible egocentric manipulations.
Building upon the state-of-the-art method HaWoR
[
64
]
, the
second
stage,
4D motion estimation
, reconstructs camera extrinsics, depth, and world-space hand trajectories. Since HaWoR lacks depth estimation and relies on DROID-SLAM
[
52
]
for tracking, which is computationally expensive and subject to drift under rapid head movements and drastic scene changes, we propose an improved, more robust and efficient scheme. We leverage DPVO
[
53
]
for more stable,
metric-free
camera tracking and keyframe depth estimation, and Any4D
[
24
]
for frame-wise,
metric-scale
depth prediction. Aligning their scale ratio recovers more accurate metric-scale camera trajectories, which we use to transform camera-frame hand motions into more consistent world-space trajectories. By leveraging DPVO, which is significantly faster than DROID-SLAM, and optimizing I/O and batching, the pipeline achieves a
9
×
9\times
throughput speedup over HaWoR, facilitating large-scale processing.
Figure 2:
Overview of EgoSmith
. Integrating pre-filtering, 4D motion estimation, language labeling, and post-filtering, EgoSmith efficiently curates in-the-wild egocentric videos into clean and annotated training samples.
The
third
stage,
language labeling
, performs multi-granularity language annotation, which is crucial for enabling free-form instruction following. We first employ Qwen3.5-VL-Plus
[
44
]
to filter out segments lacking meaningful hand-object manipulation, discarding an additional
3.5
%
3.5\%
of clips that passed the heuristic rules but lacked active operations. For the remaining clips, the model generates coarse-to-fine, five-level language instructions. This hierarchical annotation simultaneously provides task-level semantic grounding and action-level spatiotemporal grounding, enabling downstream models
to learn and respond to instructions across varying levels of abstraction.
The
fourth
stage,
post-filtering
, performs multi-level quality control on the generated data. First, at the episode level, we compute camera translation distributions to discard outliers, while applying hard rotation thresholds to drop episodes with excessive head motions. Second, at the chunk level, we transform wrist poses into its middle camera frame, project finger keypoints into frame-wise wrist frames, and discard spatial outliers across wrist and finger coordinates. Finally, at the frame level, we compute frame-to-frame deltas of camera, wrist, and finger motions, filtering out abrupt jumps with hard thresholds. This coarse-to-fine filtering systematically eliminates unreliable clips caused by action jumps, inaccurate metric scales, or head tracking drift, ensuring high corpus quality.
With EgoSmith, we curate a large-scale egocentric pre-training corpus across 12 raw datasets
[
34
,
26
,
63
,
54
,
15
,
4
,
12
,
16
,
43
,
19
,
2
,
1
]
. To filter out highly repetitive videos, we subsample Egocentric-10K
[
2
]
and Egocentric-100K
[
1
]
. For Ego4D
[
16
]
and EPIC-KITCHENS
[
12
]
, we apply EgoSmith to their respective VITRA
[
29
]
subsets. Ultimately, this pipeline yields a fully-annotated egocentric dataset comprising
9.60
​
K
9.60\text{K}
hours,
2.09
​
M
2.09\text{M}
episodes, and
1.04
​
B
1.04\text{B}
frames.

## 4
A Unified Robot Stack for Teleoperation and DAgger Post-Training

While EgoSmith’s curated egocentric data provides rich manipulation priors, direct transfer to real robots is prevented by the embodiment gap across visual, dynamics, and kinematic degrees of freedom, necessitating real-robot teleoperation to ground these priors onto the target embodiment. This section presents the
Unified Robot Stack
(
Figure
3
), which shares low-level control and dynamics to simultaneously support teleoperation, policy inference, and human-in-the-loop correction.
For teleoperation, a pair of PsiBot SynGlove-Air gloves and Vive Trackers capture the operator’s
S
​
E
​
(
3
)
SE(3)
wrist poses and hand joint angles, which respectively drive two robotic arms through inverse kinematics (IK) computed via
mink
[
62
]
and two 6-DoF robotic hands via joint mapping. During policy inference, the trained policy publishes the wrist pose trajectory in the camera frame and the hand keypoints in the wrist frame. These actions share the same arm and hand FK/IK and control nodes with teleoperation, ensuring identical execution dynamics across training and inference.
Figure 3:
Overview of the Robot Stack
. It unifiedly supports teleoperation, policy inference, and human-in-the-loop correction. A relative motion mapping scheme is employed to facilitate seamless transitions during interventions, and the bottom row illustrates the two robotic embodiments utilized in our experiments.
The primary challenge in enabling human-in-the-loop intervention is preventing sudden state jumps at the handover boundary to ensure a smooth transition. To address this, we propose a
relative motion mapping
scheme. When the operator signals intervention by pressing a foot pedal at step
t
t
, the system records the robot end-effector poses
𝐓
t
R
,
i
∈
S
​
E
​
(
3
)
\mathbf{T}^{\text{R},i}_{t}\in SE(3)
, human wrist poses
𝐓
t
H
,
i
∈
S
​
E
​
(
3
)
\mathbf{T}^{\text{H},i}_{t}\in SE(3)
, robot hand joint states
𝐪
t
R
,
i
∈
ℝ
6
\mathbf{q}^{\text{R},i}_{t}\in\mathbb{R}^{6}
, and glove states
𝐪
t
H
,
i
∈
ℝ
6
\mathbf{q}^{\text{H},i}_{t}\in\mathbb{R}^{6}
for each arm/hand index
i
∈
{
1
,
2
}
i\in\{1,2\}
. Subsequently, at any
t
′
≥
t
t^{\prime}\geq t
, the operator’s relative motions,
Δ
​
𝐓
t
→
t
′
H
,
i
=
(
𝐓
t
H
,
i
)
−
1
​
𝐓
t
′
H
,
i
\Delta\mathbf{T}_{t\rightarrow t^{\prime}}^{\text{H},i}=(\mathbf{T}^{\text{H},i}_{t})^{-1}\mathbf{T}^{\text{H},i}_{t^{\prime}}
,
Δ
​
𝐪
t
→
t
′
H
,
i
=
𝐪
t
′
H
,
i
−
𝐪
t
H
,
i
\Delta\mathbf{q}^{\text{H},i}_{t\rightarrow t^{\prime}}=\mathbf{q}^{\text{H},i}_{t^{\prime}}-\mathbf{q}^{\text{H},i}_{t}
, are mapped to the robot, computing the commanded end-effector poses
𝐓
t
′
R
,
i
\mathbf{T}^{\text{R},i}_{t^{\prime}}
and hand joint states
𝐪
t
′
R
,
i
\mathbf{q}^{\text{R},i}_{t^{\prime}}
as
𝐓
t
′
R
,
i
=
𝐓
t
R
,
i
​
Δ
​
𝐓
t
→
t
′
H
,
i
\mathbf{T}^{\text{R},i}_{t^{\prime}}=\mathbf{T}^{\text{R},i}_{t}\Delta\mathbf{T}_{t\rightarrow t^{\prime}}^{\text{H},i}
and
𝐪
t
′
R
,
i
=
𝐪
t
R
,
i
+
Δ
​
𝐪
t
→
t
′
H
,
i
\mathbf{q}^{\text{R},i}_{t^{\prime}}=\mathbf{q}^{\text{R},i}_{t}+\Delta\mathbf{q}^{\text{H},i}_{t\rightarrow t^{\prime}}
.
This formulation allows the operator to smoothly take over control by simply mimicking the robot’s motion. After correcting failures, the operator hands control back to the policy via another pedal press, resuming inference. Only these intervention segments are utilized for subsequent training. This design achieves a handover success rate exceeding
85
%
85\%
, enabling efficient collection of corrective demonstrations.
With unified robot stack, we construct a
187
187
-hour robot dataset across
193
193
tabletop tasks spanning seven categories: Pick-and-Place(PnP)-Easy/Medium/Hard, non-prehensile, reorient, bimanual, and contact-rich operations. These comprise
56
56
common tasks covering most core manipulation primitives, alongside
137
137
long-tail tasks to facilitate human-to-robot transfer and grounding. Multi-level language annotations are generated using Qwen3-VL-Flash
[
3
]
, followed by manual verification. To cover diverse primitives and establish modal alignment, data collection follows a free-form protocol where environments are cluttered, and tablecloths, object instances, and initial configurations are randomized without pre-defined trajectories. This emphasis on natural, human-like execution yields substantial rollout diversity, fostering policy robustness and multi-task generalization.

## 5
EgoSteer: A World-Model-Enhanced VLA for Steerable Dexterity

To effectively learn language-guided manipulation from human data, teleoperation, and corrections, we propose
EgoSteer
, a flow-based VLA model enhanced by a world-model objective, shown in
Figure
4
. To ensure robust vision-language understanding while modeling multimodal continuous actions, EgoSteer pairs a Qwen3-VL 2B backbone
[
3
]
with a DiT-based
[
41
]
action expert, which jointly attends to itself and backbone to generate action chunks via flow-matching
[
5
]
. To facilitate human-robot transfer, we design a unified data format and state-action space across both domains. An episode
τ
\tau
of length
N
N
is represented as
τ
=
{
l
,
𝐊
,
(
𝐈
t
,
𝐃
t
,
𝐓
t
w
​
2
​
c
,
𝐬
t
w
,
𝐚
t
w
)
t
=
0
N
−
1
}
\tau=\{l,\mathbf{K},(\mathbf{I}_{t},\mathbf{D}_{t},\mathbf{T}^{w2c}_{t},\mathbf{s}^{w}_{t},\mathbf{a}^{w}_{t})^{N-1}_{t=0}\}
, where
l
l
is the instruction,
𝐊
∈
ℝ
3
×
3
\mathbf{K}\in\mathbb{R}^{3\times 3}
is the camera intrinsics,
t
t
is the timestep,
𝐈
t
∈
ℝ
H
×
W
×
3
\mathbf{I}_{t}\in\mathbb{R}^{H\times W\times 3}
and
𝐃
t
∈
ℝ
H
×
W
×
1
\mathbf{D}_{t}\in\mathbb{R}^{H\times W\times 1}
are the RGB and depth images,
𝐓
t
w
​
2
​
c
∈
S
​
E
​
(
3
)
\mathbf{T}^{w2c}_{t}\in SE(3)
is the world-to-camera extrinsics, and bimanual world-frame states and actions
𝐬
t
w
,
𝐚
t
w
∈
ℝ
48
\mathbf{s}^{w}_{t},\mathbf{a}^{w}_{t}\in\mathbb{R}^{48}
comprise the 3D wrist translation, 6D wrist rotation, and 15D fingertip keypoints of both hands. Since depth is unused in our model, the training sample at each timestep
t
t
becomes
{
l
,
𝐊
,
𝐈
t
−
k
+
1
:
t
,
𝐬
t
−
k
+
1
:
t
c
t
,
𝐚
t
:
t
+
h
−
1
c
t
}
\{l,\mathbf{K},\mathbf{I}_{t-k+1:t},\mathbf{s}^{c_{t}}_{t-k+1:t},\mathbf{a}^{c_{t}}_{t:t+h-1}\}
, where
k
k
and
h
h
denote the history and prediction lengths, and
𝐬
t
−
k
+
1
:
t
c
t
\mathbf{s}^{c_{t}}_{t-k+1:t}
is the state history transformed into the current camera frame
c
t
c_{t}
via
𝐓
t
w
​
2
​
c
\mathbf{T}^{w2c}_{t}
. The relative action chunk
𝐚
t
:
t
+
h
−
1
c
t
\mathbf{a}^{c_{t}}_{t:t+h-1}
is computed in
c
t
c_{t}
relative to
𝐬
t
c
t
\mathbf{s}^{c_{t}}_{t}
, where wrist motions are relative
S
​
E
​
(
3
)
SE(3)
transforms and finger movements are coordinate displacements. For simplicity, we omit the
c
t
c_{t}
superscript for
𝐬
\mathbf{s}
and
𝐚
\mathbf{a}
hereafter. Furthermore, to avoid execution pauses during real-robot inference, we implement training-time Real-Time Chunking (RTC)
[
6
]
in the action expert. Specifically, we feed a clean action prefix
𝐚
pre
=
𝐚
t
:
t
+
d
−
1
\mathbf{a}_{\text{pre}}=\mathbf{a}_{t:t+d-1}
of randomly sampled delay
d
d
as ground truth and train the expert solely to denoise the subsequent actions
𝐚
~
suf
=
𝐚
~
t
+
d
:
t
+
h
−
1
\tilde{\mathbf{a}}_{\text{suf}}=\tilde{\mathbf{a}}_{t+d:t+h-1}
. During deployment, the robot executes the reserved prefix
𝐚
pre
\mathbf{a}_{\text{pre}}
during asynchronous VLA inference, transitioning seamlessly to the new chunk
𝐚
suf
\mathbf{a}_{\text{suf}}
without execution gaps. Denote our model by
π
\pi
, we train it using Conditional Flow Matching (CFM)
[
31
]
by regressing the linear velocity field of the target suffix
𝐚
suf
\mathbf{a}_{\text{suf}}
conditioned on the context
𝐂
t
=
{
l
,
𝐊
,
𝐈
t
−
k
+
1
:
t
,
𝐬
t
−
k
+
1
:
t
,
𝐚
pre
}
\mathbf{C}_{t}=\{l,\mathbf{K},\mathbf{I}_{t-k+1:t},\mathbf{s}_{t-k+1:t},\mathbf{a}_{\text{pre}}\}
with
ℒ
CFM
​
(
π
)
=
𝔼
t
,
η
,
ϵ
​
[
‖
π
​
(
𝐚
~
suf
,
η
,
𝐂
t
)
−
(
𝐚
suf
−
ϵ
)
‖
2
]
\mathcal{L}_{\text{CFM}}(\pi)=\mathbb{E}_{t,\eta,\boldsymbol{\epsilon}}\left[\|\pi(\tilde{\mathbf{a}}_{\text{suf}},\eta,\mathbf{C}_{t})-(\mathbf{a}_{\text{suf}}-\boldsymbol{\epsilon})\|^{2}\right]
,
where
η
∈
[
0
,
1
]
\eta\in[0,1]
,
ϵ
∼
𝒩
​
(
𝟎
,
𝐈
)
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
, and
𝐚
~
suf
=
(
1
−
η
)
​
ϵ
+
η
​
𝐚
suf
\tilde{\mathbf{a}}_{\text{suf}}=(1-\eta)\boldsymbol{\epsilon}+\eta\mathbf{a}_{\text{suf}}
. To expand the effective batch size and improve loss gradient, we sample four random
η
\eta
per sample.
While VLA excels at vision-language understanding, lacking future imagination limits its action generation accuracy
[
60
]
. To address this, we introduce a world-model expert to predict action-induced future DINOv3 features
[
49
]
. The expert takes the ground-truth
𝐚
t
:
t
+
h
−
1
\mathbf{a}_{t:t+h-1}
, the relative camera motion
Δ
​
𝐓
=
𝐓
t
w
​
2
​
c
​
(
𝐓
t
+
h
−
1
w
​
2
​
c
)
−
1
\Delta\mathbf{T}=\mathbf{T}^{w2c}_{t}(\mathbf{T}^{w2c}_{t+h-1})^{-1}
, and learnable query tokens
𝐳
0
:
L
𝐳
−
1
\mathbf{z}_{0:L_{\mathbf{z}}-1}
of length
L
𝐳
L_{\mathbf{z}}
as inputs to jointly attend to themselves and the backbone. The upsampled output of
𝐳
\mathbf{z}
is supervised against future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
DINOv3 features via regression loss, which provides a more direct and stable supervision signal than generative loss. For robot setups with an additional chest camera, the backbone inputs both cameras’ image histories and intrinsics, while the expert receives their relative motions and regresses both future DINOv3 features by adding distinct camera embeddings to
𝐳
\mathbf{z}
. To focus optimization on the backbone’s representation, this module comprises only four Transformer layers attending to the backbone layers at regular intervals, guiding the gradient to primarily shape the backbone. Crucially, the expert is discarded during inference, avoiding computational overhead.
Figure 4:
Overview of EgoSteer
, a world-model-enhanced VLA model for steerable dexterity. A shared Qwen3-VL backbone extracts KV cache representations from multi-modal inputs. The action expert jointly attends to itself and the backbone to generate action chunks via flow-matching, integrating training-time RTC to eliminate execution pauses. The training-only world model expert predicts future DINOv3 features to improve action accuracy with zero inference overhead.
To enable efficient training, we develop an optimized infrastructure. We use Hybrid Sharded Data Parallel (HSDP)
[
65
]
to scale batch size and overlap computation with communication, while incorporating mixed-precision training. To enhance GPU utilization, we leverage
torch.compile
for kernel fusion and integrate FlexAttention
[
13
]
to optimize attention. To mitigate I/O bottlenecks, WebDataset is employed for sequential streaming instead of random reads, drastically reducing I/O pressure while maintaining training randomness via shuffle buffers, random sample dropping, and randomized shard reading. This pipeline achieves a
44.5
%
44.5\%
Model FLOPs utilization (MFU) and a throughput of
97
97
samples/s on an
8
8
-A800 node, scaling near-linearly to
128
128
GPUs.

## 6
Experiments

We conduct extensive experiments to answer five core research questions:
Q1
. How well does EgoSteer follow free-form instructions to complete various tasks?
Q2
. Does DAgger efficiently and effectively improve performance?
Q3
. How does the pre-training scale affect downstream performance, and how does EgoSteer compare with other VLA baselines?
Q4
. Are egocentric pre-training data quality, the world-model objective, and training-time RTC essential to strong performance?
Q5
. Can large-scale egocentric pre-training enable few-shot adaptation to complex long-horizon tasks?
Setup.
EgoSteer is pre-trained on the
9.6
9.6
K-hour egocentric dataset at
384
×
384
384\times 384
resolution and post-trained on the
187
187
-hour real-robot dataset using head and chest cameras at
640
×
480
640\times 480
resolution. Next, three DAgger iterations are conducted, collecting
3.7
3.7
K trajectories across
56
56
tasks, yielding
8.3
8.3
hours of correction data to refine the policy. Finally, the policy is evaluated across
32
32
seen tasks,
4
4
compositional generalization tasks, and
4
4
unseen tasks. Compositional tasks recombine seen primitives into novel sequences, while unseen tasks feature completely novel action semantics. Each task is tested over
10
10
randomized trials under free-form instructions to measure success rates.
Results.
As shown in
Figure
5
, EgoSteer achieves 80+% success rates on 22 tasks and an overall average of
75
%
75\%
. Crucially, in cluttered, randomized layouts, the policy strictly adheres to language instructions regarding target objects, hand selections, and specific actions to execute correct tasks, even for fine-grained manipulation of flat and small objects. Furthermore, EgoSteer exhibits robust failure recovery, executing multiple retries if a previous step fails. It also achieves average success rates of
65
%
65\%
on compositional generalization and
62
%
62\%
on unseen tasks, respectively, confirming that our full-stack system endows EgoSteer with robust, steerable dexterity that covers most common tasks while generalizing effectively to novel environments and tasks.
Figure 5:
Steerable manipulation performance of EgoSteer across 40 tasks spanning 7 categories. It robustly follows free-form language instructions to achieve an overall success rate of 75%, demonstrating generalization.
Setup.
The model fine-tuned solely on the teleoperation data in
Section
6.1
is denoted as EgoSteer-FT, whereas the model refined through three DAgger iterations is referred to as EgoSteer-DG. These models are compared on four dexterous and failure-prone seen tasks, such as “
place phone on stand
”. For each task,
10
10
evaluation trials are conducted using the same settings as in
Section
6.1
.
Results.
As shown in
Table
1(a)
, after DAgger iterations totaling
8.3
8.3
hours, the average success rate increases from
22.5
%
22.5\%
to
62.5
%
62.5\%
. This efficacy stems from the targeted collection of corrective demonstrations addressing deployment failures, achieving a performance leap with minimal data. The refined policy not only exhibits robust failure recovery but also adaptively adjusts actions at critical manipulation bottlenecks. Crucially, these recovery and adjustment capabilities generalize to novel tasks, substantially improving the overall robustness of the DAgger-trained policy.
Figure 6:
Scaling behavior of pre-training loss and downstream real-robot post-training performance.
Setup.
The EgoSteer models pre-trained on
3
3
K,
6
6
K, and
9.6
9.6
K hours of egocentric data, alongside a non-pretrained baseline trained from scratch, are post-trained on the real-robot dataset. These models, denoted as EgoSteer-0/3/6/9.6K, are evaluated across
10
10
tasks. Additionally, the baselines
π
0.5
\pi_{0.5}
[
22
]
and Being-H0.5
[
36
]
are post-trained on our real-robot dataset and compared across
10
10
easier tasks.
Results.
As shown by the pre-training loss curves of EgoSteer-3K/6K/9.6K in
Figure
6
a and the real-robot success and progress rates in
Figure
6
b, scaling pre-training data drives training loss to lower convergence values while improving real-world execution performance. With expanding pre-training data, the policy exhibits the emergence of failure recovery, enhanced instruction-following, and improved action accuracy, indicating that the model successfully acquires physical common sense for error adjustment and language-guided manipulation priors from increasingly larger datasets curated by EgoSmith. These results reveal that scaling egocentric pre-training is highly beneficial for downstream bimanual manipulation, validating the quality of EgoSmith’s annotations, EgoSteer’s learning capacity, and the stability of our training infrastructure.
The comparison of EgoSteer-9.6K with the baselines is presented in
Table
1(b)
, where EgoSteer consistently outperforms both. Specifically, both baselines suffer from inconsistent action representations between pre- and post-training phases, utilize a smaller resolution, and lack deployment optimizations. Consequently, although they can handle basic PnP actions, they exhibit weak instruction-following, limited generalization, and imprecise execution. These performance gains highlight the critical advantage of our unified, full-stack system.
Setup.
The model EgoSteer-1K is pre-trained on 1K hours of egocentric data and post-trained on the real-robot dataset. It is compared against three ablated variants across
10
10
seen tasks: first,
No WM-objective
, which omits the world-model expert during both pre-training and post-training; second,
No training-RTC
, which disables training-time RTC during training and inference; and third,
Noisy data
, which utilizes noisy egocentric data unfiltered by EgoSmith pre-filtering and post-filtering.
Results.
As shown in
Table
1(c)
, removing any core component leads to a substantial performance decline, validating the necessity of each module. Specifically, the
No WM-objective
variant exhibits a significant reduction in fine-grained manipulation accuracy, confirming that enhancing the backbone’s action imagination via the world model is critical for precise action generation. The
No training-RTC
variant introduces severe action pauses and disrupts execution dynamics, causing contact-rich tasks to fail entirely due to continuous jitter. Finally, the
Noisy data
variant fails to converge effectively, leading to severe degradation in both instruction-following and manipulation precision. These ablation results strongly validate the efficacy of our unified full-stack system.
Method
Avg.
EgoSteer-FT
22.5%
EgoSteer-DG
62.5%
(a)
DAgger ablation.
Method
Avg.
π
0.5
\pi_{0.5}
[
22
]
22%
Being-H0.5
[
36
]
39%
Ours
74%
(b)
Baseline comparison.
Method
Avg.
No WM-objective
31%
No training-RTC
39%
Noisy data
33%
Ours
44%
(c)
Training ablation.
Method
Box-Folding
Cake-Unboxing
DP
[
11
]
0%
0%
IMLE
[
45
]
0%
0%
Ours (scratch)
0%
0%
Ours
75%
83%
(d)
Few-shot adaptation.
Table 1:
Extensive experiments to validate the significance of system components and compare with baselines.
Setup.
We few-shot fine-tune the pre-trained EgoSteer-9.6K on two challenging long-horizon tasks. These include
18
18
-step
40
40
-second “
box folding
” on RealMan using
120
120
demonstrations, and
9
9
-step
1
1
-minute “
cake unboxing
” on AgiBot-G1 using
200
200
demonstrations. The adapted policy is compared against strong imitation learning baselines, namely DP
[
11
]
and IMLE
[
45
]
, alongside our non-pretrained ablation, across
24
24
real-world trials per task under randomized object configurations.
Results.
As shown in
Table
1(d)
, despite the long-horizon and contact-rich nature of these tasks, and limited demonstrations, EgoSteer-9.6K achieves 75+% success while adapting robustly to spatial randomization. Conversely, the complete failure of DP, IMLE, and our from-scratch variant highlights the difficulty of these tasks, thereby validating that our
9.6
9.6
K-hour pre-training provides robust dexterous priors that can be few-shot adapted to novel embodiments and complex tasks.

### 6.1
Steerable Multi-Task Manipulation and Generalization

Setup.
EgoSteer is pre-trained on the
9.6
9.6
K-hour egocentric dataset at
384
×
384
384\times 384
resolution and post-trained on the
187
187
-hour real-robot dataset using head and chest cameras at
640
×
480
640\times 480
resolution. Next, three DAgger iterations are conducted, collecting
3.7
3.7
K trajectories across
56
56
tasks, yielding
8.3
8.3
hours of correction data to refine the policy. Finally, the policy is evaluated across
32
32
seen tasks,
4
4
compositional generalization tasks, and
4
4
unseen tasks. Compositional tasks recombine seen primitives into novel sequences, while unseen tasks feature completely novel action semantics. Each task is tested over
10
10
randomized trials under free-form instructions to measure success rates.
Results.
As shown in
Figure
5
, EgoSteer achieves 80+% success rates on 22 tasks and an overall average of
75
%
75\%
. Crucially, in cluttered, randomized layouts, the policy strictly adheres to language instructions regarding target objects, hand selections, and specific actions to execute correct tasks, even for fine-grained manipulation of flat and small objects. Furthermore, EgoSteer exhibits robust failure recovery, executing multiple retries if a previous step fails. It also achieves average success rates of
65
%
65\%
on compositional generalization and
62
%
62\%
on unseen tasks, respectively, confirming that our full-stack system endows EgoSteer with robust, steerable dexterity that covers most common tasks while generalizing effectively to novel environments and tasks.
Figure 5:
Steerable manipulation performance of EgoSteer across 40 tasks spanning 7 categories. It robustly follows free-form language instructions to achieve an overall success rate of 75%, demonstrating generalization.

### 6.2
Efficacy of DAgger Post-Training

Setup.
The model fine-tuned solely on the teleoperation data in
Section
6.1
is denoted as EgoSteer-FT, whereas the model refined through three DAgger iterations is referred to as EgoSteer-DG. These models are compared on four dexterous and failure-prone seen tasks, such as “
place phone on stand
”. For each task,
10
10
evaluation trials are conducted using the same settings as in
Section
6.1
.
Results.
As shown in
Table
1(a)
, after DAgger iterations totaling
8.3
8.3
hours, the average success rate increases from
22.5
%
22.5\%
to
62.5
%
62.5\%
. This efficacy stems from the targeted collection of corrective demonstrations addressing deployment failures, achieving a performance leap with minimal data. The refined policy not only exhibits robust failure recovery but also adaptively adjusts actions at critical manipulation bottlenecks. Crucially, these recovery and adjustment capabilities generalize to novel tasks, substantially improving the overall robustness of the DAgger-trained policy.

### 6.3
Scaling of Pre-Training and Baseline Comparisons

Figure 6:
Scaling behavior of pre-training loss and downstream real-robot post-training performance.
Setup.
The EgoSteer models pre-trained on
3
3
K,
6
6
K, and
9.6
9.6
K hours of egocentric data, alongside a non-pretrained baseline trained from scratch, are post-trained on the real-robot dataset. These models, denoted as EgoSteer-0/3/6/9.6K, are evaluated across
10
10
tasks. Additionally, the baselines
π
0.5
\pi_{0.5}
[
22
]
and Being-H0.5
[
36
]
are post-trained on our real-robot dataset and compared across
10
10
easier tasks.
Results.
As shown by the pre-training loss curves of EgoSteer-3K/6K/9.6K in
Figure
6
a and the real-robot success and progress rates in
Figure
6
b, scaling pre-training data drives training loss to lower convergence values while improving real-world execution performance. With expanding pre-training data, the policy exhibits the emergence of failure recovery, enhanced instruction-following, and improved action accuracy, indicating that the model successfully acquires physical common sense for error adjustment and language-guided manipulation priors from increasingly larger datasets curated by EgoSmith. These results reveal that scaling egocentric pre-training is highly beneficial for downstream bimanual manipulation, validating the quality of EgoSmith’s annotations, EgoSteer’s learning capacity, and the stability of our training infrastructure.
The comparison of EgoSteer-9.6K with the baselines is presented in
Table
1(b)
, where EgoSteer consistently outperforms both. Specifically, both baselines suffer from inconsistent action representations between pre- and post-training phases, utilize a smaller resolution, and lack deployment optimizations. Consequently, although they can handle basic PnP actions, they exhibit weak instruction-following, limited generalization, and imprecise execution. These performance gains highlight the critical advantage of our unified, full-stack system.

### 6.4
Ablation Studies

Setup.
The model EgoSteer-1K is pre-trained on 1K hours of egocentric data and post-trained on the real-robot dataset. It is compared against three ablated variants across
10
10
seen tasks: first,
No WM-objective
, which omits the world-model expert during both pre-training and post-training; second,
No training-RTC
, which disables training-time RTC during training and inference; and third,
Noisy data
, which utilizes noisy egocentric data unfiltered by EgoSmith pre-filtering and post-filtering.
Results.
As shown in
Table
1(c)
, removing any core component leads to a substantial performance decline, validating the necessity of each module. Specifically, the
No WM-objective
variant exhibits a significant reduction in fine-grained manipulation accuracy, confirming that enhancing the backbone’s action imagination via the world model is critical for precise action generation. The
No training-RTC
variant introduces severe action pauses and disrupts execution dynamics, causing contact-rich tasks to fail entirely due to continuous jitter. Finally, the
Noisy data
variant fails to converge effectively, leading to severe degradation in both instruction-following and manipulation precision. These ablation results strongly validate the efficacy of our unified full-stack system.
Method
Avg.
EgoSteer-FT
22.5%
EgoSteer-DG
62.5%
(a)
DAgger ablation.
Method
Avg.
π
0.5
\pi_{0.5}
[
22
]
22%
Being-H0.5
[
36
]
39%
Ours
74%
(b)
Baseline comparison.
Method
Avg.
No WM-objective
31%
No training-RTC
39%
Noisy data
33%
Ours
44%
(c)
Training ablation.
Method
Box-Folding
Cake-Unboxing
DP
[
11
]
0%
0%
IMLE
[
45
]
0%
0%
Ours (scratch)
0%
0%
Ours
75%
83%
(d)
Few-shot adaptation.
Table 1:
Extensive experiments to validate the significance of system components and compare with baselines.

### 6.5
Few-Shot Adaptation to Complex Long-Horizon Tasks

Setup.
We few-shot fine-tune the pre-trained EgoSteer-9.6K on two challenging long-horizon tasks. These include
18
18
-step
40
40
-second “
box folding
” on RealMan using
120
120
demonstrations, and
9
9
-step
1
1
-minute “
cake unboxing
” on AgiBot-G1 using
200
200
demonstrations. The adapted policy is compared against strong imitation learning baselines, namely DP
[
11
]
and IMLE
[
45
]
, alongside our non-pretrained ablation, across
24
24
real-world trials per task under randomized object configurations.
Results.
As shown in
Table
1(d)
, despite the long-horizon and contact-rich nature of these tasks, and limited demonstrations, EgoSteer-9.6K achieves 75+% success while adapting robustly to spatial randomization. Conversely, the complete failure of DP, IMLE, and our from-scratch variant highlights the difficulty of these tasks, thereby validating that our
9.6
9.6
K-hour pre-training provides robust dexterous priors that can be few-shot adapted to novel embodiments and complex tasks.

## 7
Limitations & Conclusion

This paper presents a full-stack system for steerable dexterous manipulation by integrating EgoSmith, an efficient egocentric video curation pipeline; a unified robot stack for teleoperation, inference, and correction; and EgoSteer, a world-model-enhanced VLA trained on optimized infrastructure. EgoSteer demonstrates robust steerability across 40+ semantically diverse tasks, exhibiting dexterity, failure recovery, and generalization, while achieving few-shot adaptation to long-horizon tasks on multiple embodiments. These results substantiate the efficacy of our full-stack system. Despite these achievements, key limitations remain: first, robotic DoF limitations prevent transferring highly dexterous human knowledge, restricting intricate operations; second, the lack of tactile feedback across datasets, model, and embodiment limits contact-rich performance; and third, the pre-training scale can be expanded to capture broader priors and facilitate unseen task generalization. Addressing these challenges remains a key focus of our future research.
We sincerely thank Chengdong Ma, Wenxi Xu, and Shaoyang Guo for their generous help. We also extend our gratitude to our colleagues at PsiBot, including but not limited to Xiaojie Chai, Jianxin Du, Lin Huang, Ruochong Li, Haoyi Su, Tang Li, Yunlong Wang, Hongze Yu, Chaoyang Liu, and Hui Zhang, for their valuable support and helpful discussions.

#### Acknowledgments

We sincerely thank Chengdong Ma, Wenxi Xu, and Shaoyang Guo for their generous help. We also extend our gratitude to our colleagues at PsiBot, including but not limited to Xiaojie Chai, Jianxin Du, Lin Huang, Ruochong Li, Haoyi Su, Tang Li, Yunlong Wang, Hongze Yu, Chaoyang Liu, and Hui Zhang, for their valuable support and helpful discussions.

## References

[1]
B. AI
(2025)
Egocentric-100k
.
Hugging Face Datasets
.
External Links:
Link
Cited by:
Figure 7
,
Figure 7
,
§A.1.1
,
§2
,
§3
.
[2]
B. AI
(2025)
Egocentric-10k
.
Hugging Face Datasets
.
External Links:
Link
Cited by:
§A.1.1
,
§2
,
§3
.
[3]
S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu
(2025)
Qwen3-vl technical report
.
arXiv preprint arXiv:2511.21631
.
Cited by:
§B.1.4
,
§4
,
§5
.
[4]
P. Banerjee, S. Shkodrani, P. Moulon, S. Hampali, S. Han, F. Zhang, L. Zhang, J. Fountain, E. Miller, S. Basol,
et al.
(2025)
Hot3d: hand and object tracking in 3d from egocentric multi-view videos
.
In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
,
pp. 7061–7071
.
Cited by:
Figure 7
,
§2
,
§3
.
[5]
K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky
(2026)
π
0
\pi_{0}
: A vision-language-action flow model for general robot control
.
External Links:
2410.24164
,
Link
Cited by:
§C.2
,
§1
,
§5
.
[6]
K. Black, A. Z. Ren, M. Equi, and S. Levine
(2025)
Training-time action conditioning for efficient real-time chunking
.
arXiv preprint arXiv:2512.05964
.
Cited by:
§1
,
§5
.
[7]
J. Bouguet
et al.
(2001)
Pyramidal implementation of the affine lucas kanade feature tracker description of the algorithm
.
Intel corporation
5
(
1-10
),
pp. 4
.
Cited by:
§A.1.1
.
[8]
A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich
(2023)
RT-1: robotics transformer for real-world control at scale
.
External Links:
2212.06817
,
Link
Cited by:
§1
.
[9]
X. Cai, R. Qiu, G. Chen, L. Wei, I. Liu, T. Huang, X. Cheng, and X. Wang
(2025)
In-n-on: scaling egocentric manipulation with in-the-wild and on-task data
.
arXiv preprint arXiv:2511.15704
.
Cited by:
§1
,
§2
.
[10]
K. Chen, S. Xie, Z. Ma, P. R. Sanketi, and K. Goldberg
(2025)
Robo2vlm: visual question answering from large-scale in-the-wild robot manipulation datasets
.
arXiv preprint arXiv:2505.15517
.
Cited by:
Table 3
,
3rd item
.
[11]
C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song
(2025)
Diffusion policy: visuomotor policy learning via action diffusion
.
The International Journal of Robotics Research
44
(
10-11
),
pp. 1684–1704
.
Cited by:
§1
,
§6.5
,
1(d)
.
[12]
D. Damen, H. Doughty, G. M. Farinella, S. Fidler, A. Furnari, E. Kazakos, D. Moltisanti, J. Munro, T. Perrett, W. Price,
et al.
(2018)
Scaling egocentric vision: the epic-kitchens dataset
.
In
Proceedings of the European conference on computer vision (ECCV)
,
pp. 720–736
.
Cited by:
Figure 7
,
§3
.
[13]
J. Dong, B. Feng, D. Guessous, Y. Liang, and H. He
(2024)
Flex attention: a programming model for generating optimized attention kernels
.
arXiv preprint arXiv:2412.05496
2
(
3
),
pp. 4
.
Cited by:
§5
.
[14]
Y. Fu, N. Chen, J. Zhao, S. Shan, G. Yao, P. Wang, Z. Wang, and S. Zhang
(2025)
METIS: multi-source egocentric training for integrated dexterous vision-language-action model
.
arXiv preprint arXiv:2511.17366
.
Cited by:
§1
,
§2
.
[15]
G. Garcia-Hernando, S. Yuan, S. Baek, and T. Kim
(2018)
First-person hand action benchmark with rgb-d videos and 3d hand pose annotations
.
In
Proceedings of the IEEE conference on computer vision and pattern recognition
,
pp. 409–419
.
Cited by:
Figure 7
,
§3
.
[16]
K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, J. Hamburger, H. Jiang, M. Liu, X. Liu,
et al.
(2022)
Ego4d: around the world in 3,000 hours of egocentric video
.
In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition
,
pp. 18995–19012
.
Cited by:
Figure 7
,
§2
,
§3
.
[17]
Y. Han, Z. Chen, Y. Zhao, C. Xu, Y. Shao, Y. Peng, Y. Mu, and W. Lian
(2026)
DexHiL: a human-in-the-loop framework for vision-language-action model post-training in dexterous manipulation
.
External Links:
2603.09121
,
Link
Cited by:
§2
.
[18]
R. Hartley and A. Zisserman
(2003)
Multiple view geometry in computer vision
.
Cambridge university press
.
Cited by:
§A.1.1
.
[19]
R. Hoque, P. Huang, D. J. Yoon, M. Sivapurapu, and J. Zhang
(2025)
Egodex: learning dexterous manipulation from large-scale egocentric video
.
arXiv preprint arXiv:2505.11709
.
Cited by:
Figure 7
,
§1
,
§2
,
§3
.
[20]
P. Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, G. Bokinsky, S. Cao, T. Charbonnier,
et al.
(2026)
π
0.7
\pi_{0.7}
: a steerable generalist robotic foundation model with emergent capabilities
.
arXiv preprint arXiv:2604.15483
.
Cited by:
§1
,
§2
.
[21]
P. Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo,
et al.
(2025)
π
0.6
∗
\pi^{*}_{0.6}
: a vla that learns from experience
.
arXiv preprint arXiv:2511.14759
.
Cited by:
§1
.
[22]
P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky
(2025)
π
0.5
\pi_{0.5}
: A vision-language-action model with open-world generalization
.
External Links:
2504.16054
,
Link
Cited by:
§1
,
§1
,
§2
,
§6.3
,
1(b)
.
[23]
Y. Ji, H. Tan, J. Shi, X. Hao, Y. Zhang, H. Zhang, P. Wang, M. Zhao, Y. Mu, P. An,
et al.
(2025)
Robobrain: a unified brain model for robotic manipulation from abstract to concrete
.
In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
,
pp. 1724–1734
.
Cited by:
Table 3
,
3rd item
.
[24]
J. Karhade, N. Keetha, Y. Zhang, T. Gupta, A. Sharma, S. Scherer, and D. Ramanan
(2025)
Any4D: unified feed-forward metric 4d reconstruction
.
arXiv preprint arXiv:2512.10935
.
Cited by:
§A.1.2
,
§3
.
[25]
M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn
(2024)
OpenVLA: an open-source vision-language-action model
.
External Links:
2406.09246
,
Link
Cited by:
§2
.
[26]
T. Kwon, B. Tekin, J. Stühmer, F. Bogo, and M. Pollefeys
(2021)
H2o: two hands manipulating objects for first person interaction recognition
.
In
Proceedings of the IEEE/CVF international conference on computer vision
,
pp. 10138–10148
.
Cited by:
Figure 7
,
§A.1.2
,
§3
.
[27]
H. Li, Z. Wang, Z. Ding, S. Yang, Y. Chen, Y. Tian, X. Hu, T. Wang, D. Lin, F. Zhao,
et al.
(2026)
Robointer: a holistic intermediate representation suite towards robotic manipulation
.
arXiv preprint arXiv:2602.09973
.
Cited by:
Table 3
,
3rd item
.
[28]
L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu, Y. Shen, and Y. Xu
(2026-01)
Causal world modeling for robot control
.
CoRR
abs/2601.21998
.
External Links:
Link
Cited by:
§1
.
[29]
Q. Li, Y. Deng, Y. Liang, L. Luo, L. Zhou, C. Yao, L. Zeng, Z. Feng, H. Liang, S. Xu,
et al.
(2025)
Scalable vision-language-action model pretraining for robotic manipulation with real-life human activity videos
.
arXiv preprint arXiv:2510.21571
.
Cited by:
§2
,
§3
.
[30]
Z. Li, L. Huang, W. Xu, Z. Zhu, N. Lin, X. Ma, X. Sheng, and R. Wen
(2026)
Hand-in-the-loop: improving vla policies for dexterous manipulation via seamless hand-arm intervention
.
External Links:
2605.15157
,
Link
Cited by:
§2
.
[31]
Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le
(2023)
Flow matching for generative modeling
.
In
The Eleventh International Conference on Learning Representations
,
Cited by:
§5
.
[32]
S. Liu, B. Li, K. Ma, L. Wu, H. Tan, X. Ouyang, H. Su, and J. Zhu
(2026)
RDT2: exploring the scaling limit of umi data towards zero-shot cross-embodiment generalization
.
arXiv preprint arXiv:2602.03310
.
Cited by:
§1
.
[33]
S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu
(2025)
Rdt-1b: a diffusion foundation model for bimanual manipulation
.
In
International Conference on Learning Representations
,
Vol.
2025
,
pp. 29982–30009
.
Cited by:
§1
.
[34]
Y. Liu, H. Yang, X. Si, L. Liu, Z. Li, Y. Zhang, Y. Liu, and L. Yi
(2024)
Taco: benchmarking generalizable bimanual tool-action-object understanding
.
In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
,
pp. 21740–21751
.
Cited by:
Figure 7
,
§A.1.2
,
§2
,
§3
.
[35]
H. Luo, Y. Feng, W. Zhang, S. Zheng, Y. Wang, H. Yuan, J. Liu, C. Xu, Q. Jin, and Z. Lu
(2025)
Being-h0: vision-language-action pretraining from large-scale human videos
.
External Links:
2507.15597
,
Link
Cited by:
§1
,
§2
.
[36]
H. Luo, Y. Wang, W. Zhang, S. Zheng, Z. Xi, C. Xu, H. Xu, H. Yuan, C. Zhang, Y. Wang, Y. Feng, and Z. Lu
(2026)
Being-h0.5: scaling human-centric robot learning for cross-embodiment generalization
.
External Links:
2601.12993
,
Link
Cited by:
§1
,
§1
,
§2
,
§6.3
,
1(b)
.
[37]
H. Luo, W. Zhang, Y. Feng, S. Zheng, H. Xu, C. Xu, Z. Xi, Y. Fu, and Z. Lu
(2026)
Being-h0. 7: a latent world-action model from egocentric videos
.
arXiv preprint arXiv:2605.00078
.
Cited by:
§1
.
[38]
J. Luo, C. Xu, J. Wu, and S. Levine
(2025)
Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning
.
Science Robotics
10
(
105
),
pp. eads5033
.
Cited by:
§2
.
[39]
J. Lyu, K. Liu, X. Zhang, H. Liao, Y. Feng, W. Zhu, T. Shen, J. Chen, J. Zhang, Y. Dong,
et al.
(2026)
Lda-1b: scaling latent dynamics action model via universal embodied data ingestion
.
arXiv preprint arXiv:2602.12215
.
Cited by:
§1
.
[40]
NVIDIA, :, J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. ”. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. Llontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu
(2025)
GR00T n1: an open foundation model for generalist humanoid robots
.
External Links:
2503.14734
,
Link
Cited by:
§2
.
[41]
W. Peebles and S. Xie
(2023)
Scalable diffusion models with transformers
.
In
Proceedings of the IEEE/CVF international conference on computer vision
,
pp. 4195–4205
.
Cited by:
§5
.
[42]
R. A. Potamias, J. Zhang, J. Deng, and S. Zafeiriou
(2025)
Wilor: end-to-end 3d hand localization and reconstruction in-the-wild
.
In
Proceedings of the Computer Vision and Pattern Recognition Conference
,
pp. 12242–12254
.
Cited by:
§A.1.1
,
§3
.
[43]
R. Punamiya, S. Kareer, Z. Liu, J. Citron, R. Qiu, X. Cai, A. Gavryushin, J. Chen, D. Liconti, L. Y. Zhu,
et al.
(2026)
EgoVerse: an egocentric human dataset for robot learning from around the world
.
arXiv preprint arXiv:2604.07607
.
Cited by:
Figure 7
,
§A.1.2
,
§1
,
§2
,
§3
.
[44]
Qwen Team
(2026-02)
Qwen3.5: towards native multimodal agents
.
External Links:
Link
Cited by:
§A.1.3
,
§3
.
[45]
K. Rana, R. Lee, D. Pershouse, and N. Suenderhauf
(2025)
Imle policy: fast and sample efficient visuomotor policy learning via implicit maximum likelihood estimation
.
arXiv preprint arXiv:2502.12371
.
Cited by:
§1
,
§6.5
,
1(d)
.
[46]
J. Redmon and A. Farhadi
(2018)
YOLOv3: an incremental improvement
.
CoRR
abs/1804.02767
.
External Links:
Link
,
1804.02767
Cited by:
§A.1.1
,
§3
.
[47]
J. Romero, D. Tzionas, and M. J. Black
(2017-11)
Embodied hands: modeling and capturing hands and bodies together
.
ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)
36
(
6
).
Cited by:
§A.1.2
.
[48]
S. Ross, G. Gordon, and D. Bagnell
(2011)
A reduction of imitation learning and structured prediction to no-regret online learning
.
In
Proceedings of the fourteenth international conference on artificial intelligence and statistics
,
pp. 627–635
.
Cited by:
§1
.
[49]
O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa,
et al.
(2025)
Dinov3
.
arXiv preprint arXiv:2508.10104
.
Cited by:
§C.2
,
§1
,
§5
.
[50]
Y. Tang, L. Zhang, S. Zhang, Y. Zhao, and X. Hao
(2025)
Roboafford: a dataset and benchmark for enhancing object and spatial affordance learning in robot manipulation
.
In
Proceedings of the 33rd ACM International Conference on Multimedia
,
pp. 12706–12713
.
Cited by:
Table 3
,
4th item
.
[51]
O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, Y. L. Tan, L. Y. Chen, P. Sanketi, Q. Vuong, T. Xiao, D. Sadigh, C. Finn, and S. Levine
(2024)
Octo: an open-source generalist robot policy
.
External Links:
2405.12213
,
Link
Cited by:
§2
.
[52]
Z. Teed and J. Deng
(2021)
Droid-slam: deep visual slam for monocular, stereo, and rgb-d cameras
.
Advances in neural information processing systems
34
,
pp. 16558–16569
.
Cited by:
§A.1.2
,
§3
.
[53]
Z. Teed, L. Lipson, and J. Deng
(2023)
Deep patch visual odometry
.
Advances in Neural Information Processing Systems
36
,
pp. 39033–39051
.
Cited by:
§A.1.2
,
§3
.
[54]
X. Wang, T. Kwon, M. Rad, B. Pan, I. Chakraborty, S. Andrist, D. Bohus, A. Feniello, B. Tekin, F. V. Frujeri,
et al.
(2023)
Holoassist: an egocentric human interaction dataset for interactive ai assistants in the real world
.
In
Proceedings of the IEEE/CVF International Conference on Computer Vision
,
pp. 20270–20281
.
Cited by:
Figure 7
,
§2
,
§3
.
[55]
B. Wen, W. Yang, J. Kautz, and S. Birchfield
(2024)
Foundationpose: unified 6d pose estimation and tracking of novel objects
.
In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition
,
pp. 17868–17879
.
Cited by:
§B.1.3
.
[56]
L. Wiedmann, O. Zohar, A. Mahla, X. Wang, R. Li, T. Frere, L. von Werra, A. R. Gosthipaty, and A. Marafioti
(2025)
Finevision: open data is all you need
.
arXiv preprint arXiv:2510.17269
.
Cited by:
Table 3
,
1st item
.
[57]
W. Wu, F. Lu, Y. Wang, S. Yang, S. Liu, F. Wang, Q. Zhu, H. Sun, Y. Wang, S. Ma,
et al.
(2026)
A pragmatic vla foundation model
.
arXiv preprint arXiv:2601.18692
.
Cited by:
§1
.
[58]
X. Xu, Y. Hou, Z. Liu, and S. Song
(2026)
Compliant residual dagger: improving real-world contact-rich manipulation with human corrections
.
Advances in Neural Information Processing Systems
38
,
pp. 139559–139581
.
Cited by:
§2
.
[59]
R. Yang, Q. Yu, Y. Wu, R. Yan, B. Li, A. Cheng, X. Zou, Y. Fang, X. Cheng, R. Qiu, H. Yin, S. Liu, S. Han, Y. Lu, and X. Wang
(2025)
EgoVLA: learning vision-language-action models from egocentric human videos
.
External Links:
2507.12440
,
Link
Cited by:
§2
.
[60]
S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, A. Malik, K. Lee, W. Liang, N. Ranawaka, J. Gu, Y. Xu, G. Wang, F. Hu, A. Narayan, J. Bjorck, J. Wang, G. Kim, D. Niu, R. Zheng, Y. Xie, J. Wu, Q. Wang, R. Julian, D. Xu, Y. Du, Y. Chebotar, S. Reed, J. Kautz, Y. Zhu, L. ”. Fan, and J. Jang
(2026)
World action models are zero-shot policies
.
External Links:
2602.15922
,
Link
Cited by:
§1
,
§2
,
§5
.
[61]
W. Yuan, J. Duan, V. Blukis, W. Pumacay, R. Krishna, A. Murali, A. Mousavian, and D. Fox
(2024)
Robopoint: a vision-language model for spatial affordance prediction for robotics
.
arXiv preprint arXiv:2406.10721
.
Cited by:
Table 3
,
2nd item
.
[62]
K. Zakka
(2026-02)
Mink: Python inverse kinematics based on MuJoCo
.
External Links:
Link
Cited by:
§4
.
[63]
X. Zhan, L. Yang, Y. Zhao, K. Mao, H. Xu, Z. Lin, K. Li, and C. Lu
(2024)
Oakink2: a dataset of bimanual hands-object manipulation in complex task completion
.
In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
,
pp. 445–456
.
Cited by:
Figure 7
,
§A.1.2
,
§2
,
§3
.
[64]
J. Zhang, J. Deng, C. Ma, and R. A. Potamias
(2025)
Hawor: world-space hand motion reconstruction from egocentric videos
.
In
Proceedings of the Computer Vision and Pattern Recognition Conference
,
pp. 1805–1815
.
Cited by:
§A.1.2
,
§A.1.2
,
Table 2
,
§1
,
§2
,
§3
.
[65]
Y. Zhao, A. Gu, R. Varma, L. Luo, C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer,
et al.
(2023)
Pytorch fsdp: experiences on scaling fully sharded data parallel
.
arXiv preprint arXiv:2304.11277
.
Cited by:
§5
.
[66]
R. Zheng, D. Niu, Y. Xie, J. Wang, M. Xu, Y. Jiang, F. Castañeda, F. Hu, Y. L. Tan, L. Fu, T. Darrell, F. Huang, Y. Zhu, D. Xu, and L. Fan
(2026)
EgoScale: scaling dexterous manipulation with diverse egocentric human data
.
External Links:
2602.16710
,
Link
Cited by:
§1
,
§2
.
[67]
Y. Zhong, F. Bai, S. Cai, X. Huang, Z. Chen, X. Zhang, Y. Wang, S. Guo, T. Guan, K. N. Lui, Z. Qi, Y. Liang, Y. Chen, and Y. Yang
(2025)
A survey on vision-language-action models: an action tokenization perspective
.
External Links:
2507.01925
,
Link
Cited by:
§2
.
[68]
E. Zhou, J. An, C. Chi, Y. Han, S. Rong, C. Zhang, P. Wang, Z. Wang, T. Huang, L. Sheng,
et al.
(2026)
Roborefer: towards spatial referring with reasoning in vision-language models for robotics
.
Advances in Neural Information Processing Systems
38
,
pp. 28404–28481
.
Cited by:
Table 3
,
2nd item
.
[69]
B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid,
et al.
(2023)
Rt-2: vision-language-action models transfer web knowledge to robotic control
.
In
Conference on Robot Learning
,
pp. 2165–2183
.
Cited by:
§1
.

## Appendix A
Details of EgoSmith

This section presents the implementation details of EgoSmith (
Section
A.1
) and provides detailed statistics of the curated
9.6
​
K
9.6\text{K}
-hour pre-training corpus derived from 12 egocentric human video datasets (
Section
A.2
).
The pre-filtering stage employs frame-wise heuristics to rapidly discard low-quality segments, such as those containing locomotion, excessive head movement, hand absence, occlusion, or others’ hand interference. These five scenarios are handled by two specialized gates: a camera gate for motion-related issues, and a hand gate for visibility anomalies. A contiguous segment is pruned only if it contains at least three consecutive invalid frames; isolated failures are retained as they exert negligible impact on subsequent reconstruction.
The camera gate estimates ego-motion using sparse optical flow. For each frame, a 128-point grid is tracked back to their positions 15 frames earlier via pyramidal Lucas–Kanade
[
7
]
. We fit a similarity transform to these correspondences using RANSAC
[
18
]
; the frame passes this gate if the translation of the estimated transform is within
10
%
10\%
of the image’s longer dimension.
The hand gate uses YOLO
[
46
,
42
]
to detect hands, retaining a bounding box as valid only if it satisfies three criteria: (a) confidence
≥
0.30
\geq 0.30
to reject false positives; (b) area within
[
2
%
,
50
%
]
[2\%,50\%]
of the image, where the
50
%
50\%
upper bound excludes hands abnormally close to the lens, and the
2
%
2\%
lower bound is calibrated on Egocentric-10K/100K
[
2
,
1
]
subsets, where we manually labeled small bounding boxes to find a threshold that filters out most other people’s hands while retaining the operator’s; and (c) spatially intersecting with the lower-central region (normalized
[
0.075
,
0.925
]
[0.075,0.925]
horizontally,
[
0.075
,
1.0
]
[0.075,1.0]
vertically). The gate requires
≥
2
\geq 2
valid detections per frame, which naturally filters out hand absence, occlusion, or cases where only other people’s hands are present, while preserving clear bimanual manipulation.
Together, the two gates yield video segments characterized by stable camera motion and clearly visible bimanual interactions.
In stage 2, we reconstruct hand motions in a unified, metric world-space coordinate system. Within this pipeline, we adopt the ViT module from HaWoR
[
64
]
as an off-the-shelf camera-frame hand reconstructor. Specifically, hands are detected and cropped across frames, and the ViT processes these inputs in temporal windows to regress the frame-wise MANO
[
47
]
pose parameters
𝜽
t
∈
ℝ
51
\boldsymbol{\theta}_{t}\in\mathbb{R}^{51}
, shape parameters
𝜷
t
∈
ℝ
10
\boldsymbol{\beta}_{t}\in\mathbb{R}^{10}
, and the camera-relative root translation
𝐭
t
∈
ℝ
3
\mathbf{t}_{t}\in\mathbb{R}^{3}
. Although this camera-space hand reconstruction is highly reliable, placing these reconstructions in world space with accurate physical dimensions requires a robust, metric camera trajectory. This is a primary limitation of the original HaWoR pipeline, as its dependency on DROID-SLAM
[
52
]
can accumulate drift under egocentric conditions with rapid head movements or textureless environments. To address this, we design a robust pipeline to recover a metric, temporally consistent camera trajectory in world space, onto which the hand reconstructions are subsequently mapped.
We replace DROID-SLAM with DPVO
[
53
]
to estimate the camera trajectory. DPVO is more robust in long-range egocentric scenarios and incurs much lower computational cost, outputting up-to-scale camera poses
𝐓
^
t
=
(
𝐑
t
,
𝐩
^
t
)
∈
S
​
E
​
(
3
)
\hat{\mathbf{T}}_{t}=(\mathbf{R}_{t},\hat{\mathbf{p}}_{t})\in SE(3)
along with focal length, where the hat notation
⋅
^
\hat{\cdot}
denotes up-to-scale quantities. We then anchor the trajectory to the physical scale using metric depth estimates from Any4D
[
24
]
as a reference. To ensure temporal coherence across the entire sequence, we perform a cross-chunk alignment on the local Any4D depth windows, yielding a temporally consistent metric depth sequence.
To recover the metric scale factor
s
s
for DPVO trajectory,
we compute the median ratio of the aligned Any4D depth
𝐃
t
Any4D
\mathbf{D}^{\mathrm{Any4D}}_{t}
to the DPVO depth
𝐃
^
t
DPVO
\hat{\mathbf{D}}^{\mathrm{DPVO}}_{t}
over the background pixels across all frames:
s
=
median
t
,
(
u
,
v
)
∈
ℬ
t
⁡
𝐃
t
Any4D
​
(
u
,
v
)
𝐃
^
t
DPVO
​
(
u
,
v
)
,
s=\operatorname{median}_{t,(u,v)\in\mathcal{B}_{t}}\frac{\mathbf{D}^{\mathrm{Any4D}}_{t}(u,v)}{\hat{{\mathbf{D}}}^{\mathrm{DPVO}}_{t}(u,v)},
(1)
where
ℬ
t
\mathcal{B}_{t}
is the valid background region in frame
t
t
, defined by excluding the hand regions projected from our reconstructed 3D hand mesh. We then calibrate the entire trajectory via
𝐩
t
=
s
​
𝐩
^
t
\mathbf{p}_{t}=s\hat{\mathbf{p}}_{t}
, yielding a metric camera trajectory.
Given the metric camera poses, we transform the camera-space hand vertices and joints, denoted generally as 3D coordinates
𝐱
t
cam
∈
ℝ
3
\mathbf{x}^{\mathrm{cam}}_{t}\in\mathbb{R}^{3}
, into world space via
𝐱
t
world
=
𝐑
t
⊤
​
(
𝐱
t
cam
−
𝐩
t
)
\mathbf{x}^{\mathrm{world}}_{t}=\mathbf{R}_{t}^{\top}(\mathbf{x}^{\mathrm{cam}}_{t}-\mathbf{p}_{t})
. The output for each video segment consists of the world-space bimanual states and actions, frame-wise camera intrinsics and extrinsics
(
𝐊
t
,
𝐓
t
)
(\mathbf{K}_{t},\mathbf{T}_{t})
, MANO parameters, and the Any4D metric scene depth.
Concerning efficiency, stage 2 represents the primary computational bottleneck of our entire pipeline, with the majority of the overhead stemming from camera trajectory estimation. This highlights another advantage of EgoSmith: DPVO that we use is far more lightweight than the dense DROID-SLAM, substantially reducing this major cost. Building on this, we further optimize the throughput of this stage through parallelized batching and asynchronous I/O pipelining.
While the original HaWoR processes only a single 16-frame temporal window at a time, we group multiple windows into a single batch for parallel forward passes. Furthermore, we overlap CPU-based frame decoding and cropping with GPU-based model inference to prevent GPU idling. Benchmarks on an 8
×
\times
A800 server using 8 video segments of 2K frames each show that our pipeline achieves an overall speedup of approximately 9
×
\times
compared to HaWoR.
Table 2:
Benchmark results of 4D motion estimation.
Method
RPE
↓
\downarrow
ATE
↓
\downarrow
WA-MPJPE
↓
\downarrow
W-MPJPE
↓
\downarrow
HaWoR
[
64
]
5.17
9.44
38.7
106.9
EgoSmith
(Ours)
2.42
7.60
25.9
86.0
We further quantitatively benchmark the
accuracy
of our 4D motion estimation pipeline against HaWoR
[
64
]
on high-quality annotated subsets of TACO
[
34
]
, H2O
[
26
]
, OakInk-v2
[
63
]
, and EgoVerse
[
43
]
. To jointly assess
camera-trajectory and hand-motion accuracy, we adopt four complementary metrics (all in mm):
•
Relative Pose Error (RPE)
: Quantifies local tracking drift and frame-to-frame jitter over a fixed temporal interval. Because it is computed without any global alignment, this metric is highly sensitive to metric-scale inaccuracies.
•
Absolute Trajectory Error (ATE)
: Assesses the overall camera trajectory shape and long-term drift. The estimated trajectory is aligned to the ground truth via a global
Sim
​
(
3
)
\mathrm{Sim}(3)
transform before evaluation, making this metric insensitive to absolute scale errors.
•
World-Aligned Mean Per Joint Position Error (WA-MPJPE)
: Measures hand joint errors while accounting for global hand placement. The hand joints are aligned to the ground truth using a single
Sim
​
(
3
)
\mathrm{Sim}(3)
transform over each
100
100
-frame segment, rather than performing per-frame local alignment.
•
World Mean Per Joint Position Error (W-MPJPE)
: Serves as the strictest metric for world-space hand tracking. It rigidly aligns only the first frame of each 100-frame segment via an SE(3) transform, thereby heavily penalizing absolute scale errors, temporal drift, and orientation misalignment across subsequent frames.
As shown in
Table
2
, EgoSmith substantially outperforms HaWoR across all four metrics. Our pipeline reduces RPE by over
50
%
50\%
, from 5.17 to 2.42 mm, and lowers ATE from
9.44
9.44
to
7.60
7.60
mm, indicating that the DPVO-based trajectory estimation yields superior local consistency and structural robustness. EgoSmith further improves WA-MPJPE from
38.7
38.7
to
25.9
25.9
mm and W-MPJPE from
106.9
106.9
to
86.0
86.0
mm, showing that our Any4D-based metric scaling, cross-window scale alignment, and global re-anchoring effectively mitigate scale
distortion and long-term drift, ensuring physically plausible world-space hand tracking over extended sequences.
Below, we present the prompt template designed for Qwen3.5-VL-Plus
[
44
]
to generate multi-granularity language annotations for egocentric human videos.
Egocentric Human Video Language Labeling Prompt
In this stage, we perform quality control on the reconstructed outputs to filter out reconstruction anomalies and problematic segments, ensuring overall data quality. This evaluation is conducted from coarse to fine across three granularities: entire episodes, chunk windows, and adjacent frames.
Episode-level checks assess the overall camera motion. We compute the statistics of camera extrinsics, both translation and rotation, for each episode and compare them against the distribution of other episodes within the same dataset, discarding those that deviate significantly. Because reasonable camera motion magnitudes vary across datasets due to different devices, scenes, and manipulation styles, we employ a dataset-specific IQR criterion rather than a universal threshold. Specifically, an episode is classified as an outlier if its statistics fall outside the range
[
Q
1
−
2.5
​
IQR
,
Q
3
+
2.5
​
IQR
]
[Q_{1}-2.5\mathrm{IQR},Q_{3}+2.5\mathrm{IQR}]
. This step filters out camera tracking drift, as well as segments dominated by walking or looking around instead of manipulation.
Chunk-level checks evaluate whether hands fall within physically reasonable spatial boundaries in a standardized egocentric coordinate frame. Directly comparing absolute hand coordinates is problematic, since they are coupled with camera and body movements. Specifically, within a sliding window spanning approximately the past 5 seconds and the future 30 frames, we transform all hand states and actions into the current camera frame. Under this canonical system, wrist positions are defined relative to the camera, and finger joints relative to the wrist. Within this coordinate system, we evaluate both distributional outliers and absolute physical limits. First, outliers in wrist translation/rotation and finger positions are identified using the same IQR criterion based on the respective dataset’s distribution. Second, we enforce a universal physical ceiling of 1.5 meters on each coordinate axis for the hands, as a human hand cannot physically reach further than this distance from the head. If any sliding chunk window within an episode violates either the IQR outlier threshold or the 1.5-meter physical limit, the entire episode is discarded.
Frame-level checks identify sudden jumps between adjacent frames. We compute the frame-to-frame changes in camera translation and rotation, wrist translation and rotation, and finger translation. Unlike the previous levels, we do not rely on dataset-specific distributions here, as the physical speed of human hands and heads has a universal limit. Any movement exceeding this limit is attributed to reconstruction artifacts rather than valid motion. We therefore apply fixed physical thresholds: camera translation
≤
0.20
​
m/frame
\leq 0.20\,\text{m/frame}
, wrist and finger translation
≤
0.30
​
m/frame
\leq 0.30\,\text{m/frame}
, camera rotation
≤
28
∘
/
frame
\leq 28^{\circ}/\text{frame}
, and wrist rotation
≤
41
∘
/
frame
\leq 41^{\circ}/\text{frame}
. An episode is discarded if any of its frames violate these thresholds.
Collectively, these three levels of checks filter out problematic segments caused by head tracking drift, inaccurate motion reconstruction, and motion discontinuities.
The final pre-training corpus is constructed by utilizing EgoSmith to process 12 raw egocentric datasets, ultimately yielding a standardized,
9.6
​
K
9.6\text{K}
-hour dataset. This curated corpus comprises world-space bimanual states and actions, frame-wise camera intrinsics and extrinsics, metric scene depth, and coarse-to-fine language annotations. In this section, we detail the scale, source composition, and semantic diversity of this dataset. The primary objective is not merely scaling up dataset volume; instead, the core advantage lies in data quality. Every sample is richly annotated and filtered through a rigorous quality-control pipeline, guaranteeing high-quality, modality-aligned manipulation knowledge to assist the model in learning steerable dexterous manipulation. The dataset further offers broad coverage of manipulation tasks, objects, and multi-granularity language descriptions.
Scale and Source Composition.
Figure
7
presents the duration contribution and proportion of each source dataset, while also indicating whether each annotation is natively provided or reconstructed by our pipeline. The source datasets are highly complementary, covering a wide range of devices, scenes, and manipulation styles. Although a small number of large-scale collections dominate the total duration, the breadth of sources contributes substantial diversity.
Task and Semantic Diversity.
To characterize the manipulation span of our dataset, we extract (verb, object) tuples from the L1 verb–object annotations and analyze the distributions of action verbs and manipulated objects. The dataset covers 8969 distinct object nouns and 623 action verbs (
Figures
7
and
7
).
Figure
7
illustrates the top 50 most frequent “verb + object” atomic tasks. Common tasks concentrate on fundamental manipulation skills, aligning with the natural distribution of daily hand–object interactions. At the same time, the distribution exhibits a prominent long tail: a large number of low-frequency yet semantically diverse tasks and objects provides broad coverage, preventing the dataset from being biased toward a few dominant actions. This combination of common fundamental skills, long-tail task diversity, and multi-granularity language annotations provides downstream models with abundant training samples, broad task coverage, and rich dexterous manipulation knowledge.
Dataset
Hours
Percentage (%)
Episodes
Hand
Depth
Camera
Language
Egocentric-100K
[
1
]
8
,
049
8,049
83.8
83.8
1
,
795
,
731
1{,}795{,}731
✓
✓
✓
✓
EgoVerse
[
43
]
690
690
7.2
7.2
35
,
175
35{,}175
✓
✓
EgoDex
[
19
]
370
370
3.9
3.9
147
,
588
147{,}588
✓
✓
Egocentric-10K
[
1
]
288
288
3.0
3.0
194
,
915
194{,}915
✓
✓
✓
✓
Ego4D
[
16
]
138
138
1.4
1.4
74
,
505
74{,}505
✓
✓
✓
✓
Epic-Kitchens
[
12
]
49
49
0.5
0.5
26
,
454
26{,}454
✓
✓
✓
✓
HoloAssist
[
54
]
11.5
11.5
0.1
0.1
11
,
426
11{,}426
✓
✓
HOT3D
[
4
]
4.5
4.5
0.05
0.05
1
,
105
1{,}105
✓
TACO
[
34
]
3.0
3.0
0.03
0.03
1
,
558
1{,}558
✓
OakInk-v2
[
63
]
1.7
1.7
0.02
0.02
891
891
✓
H2O
[
26
]
1.0
1.0
0.01
0.01
935
935
FPHA
[
15
]
0.5
0.5
0.01
0.01
578
578
✓
✓
✓
Total
𝟗
,
𝟔𝟎𝟔
\mathbf{9,606}
100
100
𝟐
,
𝟐𝟗𝟎
,
𝟖𝟔𝟏
\mathbf{2{,}290{,}861}
(a)
Dataset composition and annotation sources
(b)
Word cloud of manipulated objects
(c)
Word cloud of action verbs
(d)
Top-
50
50
“verb + object” atomic tasks
Figure 7:
Curated dataset statistics.
(a) Source composition of the
12
12
egocentric datasets, detailing duration, percentage, episode count, and per-annotation origin. Checkmarks denote annotations generated by our pipeline, while blank entries indicate natively provided annotations. For FPHA, the checkmark under
Hand
specifically denotes the additionally annotated second hand. (b)–(d) Task and semantic diversity: word clouds of the manipulated objects and action verbs, and the most frequent verb–object atomic tasks, exhibiting long-tailed coverage over diverse manipulation skills.
This section details the implementation of the Robot Stack (
Section
B.1
), alongside the collection protocols and statistics of the
187
187
-hour real-robot dataset (
Section
B.2
).
The two physical embodiments utilized in our work are illustrated in
Figure
3
. The primary platform, referred to as the RealMan embodiment, consists of two 7-DoF RealMan RM75-6F robotic arms and two 6-DoF Ruiyan RY-H2 dexterous hands. It is equipped with one head-mounted and one chest-mounted Intel RealSense D455 camera, providing dual egocentric viewpoints to ensure a comprehensive visual field. The AgiBot G1 embodiment is adapted from an AgiBot G1 humanoid robot by replacing its default end-effectors with two 6-DoF Ruiyan RY-H2 hands. It features one head-mounted Intel RealSense D455 camera and two wrist cameras, the latter of which are unused. During experiments, the robot’s neck, waist, and mobile base are kept fixed. Both platforms conduct tabletop manipulation on an operational table. The RealMan platform serves as the primary embodiment, on which the entire
187
187
-hour dataset was collected. All experiments are conducted on this setup, with the sole exception of the few-shot adaptation experiment performed on the AgiBot G1.
For both platforms, the human wrist tracker, arm kinematic solvers, and joint control ROS 2 nodes operate at
100
100
Hz, while the glove, hand solvers, and control ROS 2 nodes run at
80
80
Hz. This high-frequency loop ensures near-zero latency, enabling intuitive bimanual teleoperation and the collection of fine-grained manipulation demonstrations. The cameras capture frames at
30
30
Hz. Data is recorded at these native frequencies and subsequently resampled to
30
30
Hz for training.
Robot data collection aims to ground the human manipulation priors learned from large-scale pre-training onto the target physical embodiment. To preserve and transfer this pre-trained knowledge, minimizing the domain gap between robot and human data is essential. Because the robotic palm is slightly longer than a human palm, we translate the robot’s wrist coordinate frame axially forward. This alignment ensures that the scale from the wrist to the fingertips matches human anatomical proportions. When converting raw robot data into training data, this coordinate transformation is applied alongside hand-eye calibration to map hand actions into the camera coordinate frame. Correspondingly, during real-time policy inference, the control stack executes the inverse transformation and hand-eye mapping to project predicted actions back to the robot’s physical coordinate space.
In practice, hand-eye calibration is prone to drift due to mechanical maintenance, wear, or collisions, with such discrepancies sometimes only identified
post
-acquisition. To ensure data quality, we introduce an offline verification and automated recalibration pipeline. RGB-D images from the camera are projected into a 3D point cloud, and the robot’s 3D mesh is rendered in the same space based on the hand-eye calibration, utilizing their spatial overlap for validation. Upon detecting misalignment, our pipeline runs FoundationPose
[
55
]
on the point cloud to estimate the robot hand’s 6D pose based on its URDF. Through FK, the calibration matrix is back-calculated for each frame. Averaging these matrices across frames and removing outliers robustly recovers the calibration parameters, preventing data degradation.
Below, we present the prompt template designed for Qwen3-VL-Flash
[
3
]
to generate multi-granularity language annotations for teleoperated demonstrations.
Robot Data Language Labeling Prompt
Although pre-training on egocentric human videos equips the model with rich dexterous manipulation priors, direct physical deployment is prevented by the embodiment gap across visual appearance, dynamics, and kinematics. Furthermore, because these pre-training labels are generated by automated depth and hand pose estimation models, their accuracy is constrained by current model capabilities, potentially introducing systematic biases into the pre-trained policy. Therefore, collecting real-robot teleoperation data aims to correct and ground these priors onto the target embodiment with high sample efficiency. Additionally, due to kinematic constraints on robotic degrees of freedom, the robot cannot perfectly replicate all fine-grained human hand movements. To achieve steerable manipulation, we must collect a highly diverse range of free-form tasks to maximize the coverage of basic manipulation primitives, guiding the model to efficiently transfer cross-domain knowledge and fostering robust compositional generalization and multi-task instruction following.
Based on these considerations, within the kinematic limits of RealMan’s degrees of freedom, we design
193
193
semantically distinct dexterous manipulation tasks. For each task, approximately
300
300
randomized, diverse demonstration trajectories, totaling around
1
1
hour, are collected under cluttered scenarios, constructing a high-quality real-robot dataset of
187
187
hours and
55
​
K
55\text{K}
trajectories.
These
193
193
tasks are classified into two major categories:
•
Common Tasks
(
56
56
tasks): Everyday manipulations that are readily achievable within the current robot hardware configuration and sensor limits, yielding high teleoperation success rates.
•
Long-Tail Tasks
(
137
137
tasks): Infrequent and physically challenging manipulations, such as contact-sensitive operations lacking tactile feedback, with lower collection success rates, primarily designed to ensure comprehensive semantic coverage of the dexterous manipulation space.
Furthermore, based on motion characteristics and physical interactions, all tasks are categorized into seven classes:
•
PnP-Easy
: Single-step tabletop pick-and-place where objects are easily graspable and the placement space is open.
•
PnP-Medium
: Non-planar or 3D spatial pick-and-place that requires precise operations related to containers, demanding higher control accuracy and spatial perception, such as “
put tennis ball into ball holder
”.
•
PnP-Hard
: Multi-step or high-precision pick-and-place sequences, such as “
stack paper cups
”.
•
Non-prehensile
: Actions not involving traditional finger grasping, including pushing, pulling, and pressing.
•
Reorient
: Operations involving rotation and reorientation, such as “
pour water
” or “
flip paper cups
”.
•
Bimanual
: Bimanual tasks requiring high synchronization and spatial coordination between arms and hands, such as “
plug cable into charger
”.
•
Contact-rich
: Operations involving frequent and complex physical contact interactions, requiring physical understanding, such as “
wipe whiteboard
”.
Throughout data collection, strict requirements are enforced on the randomness, diversity, and quality of each trajectory. Specifically, the tabletop features cluttered, unstructured scenes rather than pre-arranged, simplified environments, avoiding task identification from visual inputs alone. This design forces the model to deeply align language instructions with physical actions, encouraging it to learn general task semantics and execution goals rather than memorizing demonstration trajectories. Furthermore, given the highly randomized object configurations, operators are instructed to perform teleoperation in a natural, human-like manner. Consequently, different demonstrations of the same task exhibit substantial variations in execution trajectories, thereby covering a broader distribution of the mapping from human priors to the robot’s action space.
As illustrated in
Figure
8
, we analyze the
187
187
-hour real-robot dataset across several key dimensions, including the verb and noun word clouds with their top 30 frequency distributions, the task duration breakdown across seven manipulation categories, and the detailed duration statistics for the
56
56
common and
137
137
long-tail tasks. These statistics reveal a diverse vocabulary of actions and manipulated objects, alongside a balanced distribution across both manipulation categories and task durations. Encompassing rich semantic concepts and manipulation primitives, this dataset effectively grounds pre-trained human manipulation priors onto the RealMan embodiment. Additionally,
Figure
9
showcases the dual-view image sequences and language annotations of a teleoperation trajectory, while
Figure
10
visualizes sample tasks across each category.
(a)
Word cloud of nouns in annotation
(b)
Top 30 noun frequencies
(c)
Word cloud of verbs in annotation
(d)
Top 30 verb frequencies
(e)
Task duration by seven categories
(f)
Common task duration statistics
(g)
Long-tail task duration statistics
Figure 8:
Teleoperation dataset statistics. (1)
Figure
8
-
Figure
8
: Word clouds and top-30 frequency distributions of nouns and verbs in language annotations. (2)
Figure
8
: Task duration breakdown by seven manipulation categories (PnP-Easy/Medium/Hard, Non-prehensile, Reorient, Bimanual, Contact-rich). (3)
Figure
8
and
Figure
8
: Detailed duration statistics for common tasks (56 tasks) and long-tail tasks (137 tasks).
Figure 9:
Dual-view image sequence and three-level language annotations of a representative trajectory
. The frames illustrate a cluttered, randomized setup where the operator executes the task in a natural, human-like manner. The hierarchical language annotations describe the manipulation process from coarse to fine, assisting the model in aligning free-form instructions with physical actions.
Figure 10:
Representative task examples across seven manipulation categories
. These diverse tasks are executed in cluttered, randomized scenarios using natural, human-like manipulations, covering a wide range of action semantics and manipulation primitives to facilitate the efficient grounding of pre-trained egocentric human priors onto the physical robot.
Dataset
Samples
Captioning
Visual Question
Answering
Multiple-Choice
Question
Pointing
Bounding Box
Affordance
Trajectory
Spatial
Planning
FineVision
[
56
]
3.5
​
M
3.5\text{M}
✓
✓
✓
✓
RefSpatial
[
68
]
2.5
​
M
2.5\text{M}
✓
✓
✓
✓
RoboInter-VQA
[
27
]
1.6
​
M
1.6\text{M}
✓
✓
✓
✓
✓
✓
✓
RoboPoint
[
61
]
1.3
​
M
1.3\text{M}
✓
✓
✓
✓
RoboAfford
[
50
]
765
​
K
765\text{K}
✓
✓
✓
✓
Robo2VLM
[
10
]
678
​
K
678\text{K}
✓
✓
ShareRobot
[
23
]
13
​
K
13\text{K}
✓
✓
✓
✓
✓
Table 3:
VLM datasets used for EgoSteer co-training
. This table details the sample sizes and the multi-modal and embodied knowledge domains covered by each dataset. This co-training mixture integrates general vision-language knowledge with interaction understanding and spatial-geometric priors, preserving the model’s inherent world knowledge while facilitating its ability to follow free-form instructions to generalize across diverse manipulation tasks.
Figure 11:
Statistical distribution of samples across VLM co-training datasets.
This section first introduces the VLM datasets utilized for co-training with VLA data,
i.e.
egocentric human videos and real-robot data, to preserve EgoSteer’s vision-language knowledge and ensure generalization (
Section
C.1
). We then present implementation details of EgoSteer (
Section
C.2
).
To preserve general vision-language reasoning capabilities while simultaneously cultivating robust robotic task comprehension, a
10.4
​
M
10.4\text{M}
-sample VLM co-training mixture is curated across seven datasets, ranging from open-world perception to embodied interaction grounding, to co-train EgoSteer. This mixture comprises four categories of data:
•
General VLM Pre-Training
: Incorporates FineVision
[
56
]
to prevent the catastrophic forgetting of open-world semantic concepts and preserve general visual-language reasoning.
•
Spatial Grounding
: Utilizes RefSpatial
[
68
]
and RoboPoint
[
61
]
for multi-step spatial referring, 2D visual grounding, and precise coordinate localization.
•
Embodied QA
: Integrates RoboInter-VQA
[
27
]
, Robo2VLM
[
10
]
, and ShareRobot
[
23
]
to support embodied question answering and temporal reasoning. This preserves the model’s capabilities in high-level task planning and causal scene understanding.
•
Affordance Perception
: Adopts RoboAfford
[
50
]
for fine-grained manipulation affordance prediction and spatial interaction grounding.
The sample distribution of these datasets is illustrated in
Figure
11
, with their covered multimodal, embodied knowledge domains detailed in
Table
3
. We standardize these datasets to comply with the conversational input format of Qwen3-VL. Specifically, we normalize 2D bounding box and point coordinates to Qwen3-VL’s native
[
0
,
1000
]
[0,1000]
scale and adopt its standard representation formats. To ensure training stability, we only utilize single-image samples and exclude those with context lengths exceeding our maximum budget. By co-training on this VLM mixture, EgoSteer preserves open-world vision-language understanding and reasoning while enhancing its comprehension of robotics-specific tasks, thereby assisting the policy in following open-ended instructions and generalizing to novel manipulation scenarios.
EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.
The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.
The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.
The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.
Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.
To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.
Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.
This section provides supplementary details for the experimental evaluations presented in
Section
6
, detailing the model training hyperparameters and per-task success rates. In particular,
Table
4
lists the hyperparameter configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
Table
5
presents the per-task success rates on the four evaluation tasks in
Section
6.2
to demonstrate the performance gains from DAgger refinement. For the scaling analysis in
Section
6.3
,
Table
6
lists the training hyperparameters of EgoSteer-3K/6K/9.6K pre-trained models, their respective post-training runs, and the baseline trained from scratch, with their corresponding task-specific success rates reported in
Table
7
. Additionally,
Table
8
compares the detailed success rates of our method against the Being-H0.5 and
π
0.5
\pi_{0.5}
baselines across ten tasks.
Table
10
specifies the training configurations of the ablation experiment from
Section
6.4
, including the EgoSteer-1K model and its three ablated variants,
No WM-objective
,
No training-RTC
, and
Noisy data
. Their respective performance comparison is provided in
Table
9
. Finally,
Table
11
specifies the hyperparameters for the few-shot fine-tuning of our pre-trained EgoSteer-9.6K on the two challenging, long-horizon tasks evaluated in
Section
6.5
, namely
Box-Folding
and
Cake-Unboxing
.
Hyperparameter
Pre-Training
Post-Training
Camera setup
Head
Head & Chest
Resolution
384
×
384
384\times 384
640
×
480
640\times 480
GPUs
128 A800
96 A800
Gradient accumulation
2
1
Global batch size
4608
384
Training steps
175K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Freeze-VLM steps
5000
0
Warmup steps
2000
2000
Training time
164 h
29 h
Table 4:
Training configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
During pre-training, the VLM backbone is frozen for the first 5,000 steps, during which AE and WM are warmed up for 2,000 steps; once it is unfrozen, VLM is warmed up for 2,000 steps.
Task
EgoSteer-DG
EgoSteer-FT
Stack tableware
80.0%
50.0%
Close laptop
70.0%
10.0%
Place phone on stand
50.0%
0.0%
Flip cup
50.0%
30.0%
Average
62.5%
22.5%
Table 5:
Success rates across four highly dexterous and failure-prone manipulation tasks, comparing EgoSteer-DG against EgoSteer-FT in
Section
6.2
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Camera setup
Head
Head
Head
Head & Chest
Head & Chest
Head & Chest
Head & Chest
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
GPUs
64 A800
64 A800
128 A800
64 A800
32 A800
64 A800
64 A800
Gradient accumulation
2
4
2
2
4
2
2
Global batch size
2304
4608
4608
512
512
512
512
Training steps
100K
100K
160K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 6:
Training configurations for the pre-training scaling study in
Section
6.3
.
Pre-Training Data (Hours)
Task
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Grasp object
80%
80%
70%
100%
Hand over object
70%
80%
80%
100%
Place items into container
40%
80%
90%
100%
Point at object
20%
40%
60%
70%
Place toy chick into slot
30%
0%
20%
40%
Pull out tissue
60%
20%
50%
80%
Push ball into box
0%
0%
10%
20%
Put garbage into trash bin
0%
60%
30%
30%
Stack paper cups
0%
20%
10%
40%
Stack tableware
0%
20%
10%
20%
Average
30%
40%
43%
60%
Table 7:
Per-task success rates for the pre-training scaling study in
Section
6.3
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Task
Ours
Being-H0.5
π
0.5
\pi_{0.5}
Grasp object
100%
80%
80%
Hand over object
100%
60%
20%
Place items into container
100%
50%
0%
Pour items out of box
50%
30%
0%
Place bread on tray
70%
80%
40%
Pull out tissue
80%
10%
30%
Place item at specific orientation
30%
30%
30%
Attach eraser to whiteboard
90%
50%
20%
Put garbage into trash bin
30%
0%
0%
Put tennis ball into bucket
90%
0%
0%
Average
74%
39%
22%
Table 8:
Per-task comparison between EgoSteer-9.6K and two VLA baselines, Being-H0.5 and
π
0.5
\pi_{0.5}
, in
Section
6.3
. All methods are post-trained on our real-robot dataset and evaluated on the same 10 tasks. Each task uses 10 randomized trials. Bold values highlight the best performance.
Task
Ours
No WM-objective
No training-RTC
Noisy data
Grasp object
60%
40%
30%
60%
Hand over object
40%
20%
30%
40%
Place items into container
50%
30%
50%
70%
Pour items out of box
60%
50%
10%
0%
Place bread on tray
70%
10%
60%
50%
Pull out tissue
20%
30%
60%
10%
Place item at specific orientation
20%
10%
10%
20%
Attach eraser to whiteboard
50%
80%
50%
40%
Put garbage into trash bin
30%
0%
0%
0%
Put tennis ball into bucket
40%
40%
90%
40%
Average
44%
31%
39%
33%
Table 9:
Per-task success rates for the ablation study in
Section
6.4
. EgoSteer-1K is compared with ablated variants. Each task uses 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
Ours
No WM-objective
No training-RTC
Noisy data
Ours
No WM-objective
No training-RTC
Noisy data
Camera setup
Head
Head
Head
Head
Head
Head
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
Global batch size
1152
1152
1152
1152
1152
1152
1152
1152
Training steps
30K
80K
30K
20K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 10:
Training hyperparameters for the ablation study in
Section
6.4
. All variants are pre-trained on 1K hours of egocentric data and post-trained on the same real-robot dataset, without DAgger refinement. Pre-training steps are selected at the lowest evaluation
L
1
L_{1}
loss for each pre-training run.
Hyperparameter
Box-Folding
Cake-Unboxing
Pre-training checkpoint
EgoSteer-9.6K at 155K steps
Camera setup
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
8 A800
8 A800
Gradient accumulation
1
1
Global batch size
144
144
Fine-tuning steps
44K
12K
Learning rate (VLM / AE / WM)
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Demonstrations
120
229
Table 11:
Few-shot fine-tuning hyperparameters of EgoSteer-9.6K for the two long-horizon dexterous tasks in
Section
6.5
.

### A.1
Implementation Details

The pre-filtering stage employs frame-wise heuristics to rapidly discard low-quality segments, such as those containing locomotion, excessive head movement, hand absence, occlusion, or others’ hand interference. These five scenarios are handled by two specialized gates: a camera gate for motion-related issues, and a hand gate for visibility anomalies. A contiguous segment is pruned only if it contains at least three consecutive invalid frames; isolated failures are retained as they exert negligible impact on subsequent reconstruction.
The camera gate estimates ego-motion using sparse optical flow. For each frame, a 128-point grid is tracked back to their positions 15 frames earlier via pyramidal Lucas–Kanade
[
7
]
. We fit a similarity transform to these correspondences using RANSAC
[
18
]
; the frame passes this gate if the translation of the estimated transform is within
10
%
10\%
of the image’s longer dimension.
The hand gate uses YOLO
[
46
,
42
]
to detect hands, retaining a bounding box as valid only if it satisfies three criteria: (a) confidence
≥
0.30
\geq 0.30
to reject false positives; (b) area within
[
2
%
,
50
%
]
[2\%,50\%]
of the image, where the
50
%
50\%
upper bound excludes hands abnormally close to the lens, and the
2
%
2\%
lower bound is calibrated on Egocentric-10K/100K
[
2
,
1
]
subsets, where we manually labeled small bounding boxes to find a threshold that filters out most other people’s hands while retaining the operator’s; and (c) spatially intersecting with the lower-central region (normalized
[
0.075
,
0.925
]
[0.075,0.925]
horizontally,
[
0.075
,
1.0
]
[0.075,1.0]
vertically). The gate requires
≥
2
\geq 2
valid detections per frame, which naturally filters out hand absence, occlusion, or cases where only other people’s hands are present, while preserving clear bimanual manipulation.
Together, the two gates yield video segments characterized by stable camera motion and clearly visible bimanual interactions.
In stage 2, we reconstruct hand motions in a unified, metric world-space coordinate system. Within this pipeline, we adopt the ViT module from HaWoR
[
64
]
as an off-the-shelf camera-frame hand reconstructor. Specifically, hands are detected and cropped across frames, and the ViT processes these inputs in temporal windows to regress the frame-wise MANO
[
47
]
pose parameters
𝜽
t
∈
ℝ
51
\boldsymbol{\theta}_{t}\in\mathbb{R}^{51}
, shape parameters
𝜷
t
∈
ℝ
10
\boldsymbol{\beta}_{t}\in\mathbb{R}^{10}
, and the camera-relative root translation
𝐭
t
∈
ℝ
3
\mathbf{t}_{t}\in\mathbb{R}^{3}
. Although this camera-space hand reconstruction is highly reliable, placing these reconstructions in world space with accurate physical dimensions requires a robust, metric camera trajectory. This is a primary limitation of the original HaWoR pipeline, as its dependency on DROID-SLAM
[
52
]
can accumulate drift under egocentric conditions with rapid head movements or textureless environments. To address this, we design a robust pipeline to recover a metric, temporally consistent camera trajectory in world space, onto which the hand reconstructions are subsequently mapped.
We replace DROID-SLAM with DPVO
[
53
]
to estimate the camera trajectory. DPVO is more robust in long-range egocentric scenarios and incurs much lower computational cost, outputting up-to-scale camera poses
𝐓
^
t
=
(
𝐑
t
,
𝐩
^
t
)
∈
S
​
E
​
(
3
)
\hat{\mathbf{T}}_{t}=(\mathbf{R}_{t},\hat{\mathbf{p}}_{t})\in SE(3)
along with focal length, where the hat notation
⋅
^
\hat{\cdot}
denotes up-to-scale quantities. We then anchor the trajectory to the physical scale using metric depth estimates from Any4D
[
24
]
as a reference. To ensure temporal coherence across the entire sequence, we perform a cross-chunk alignment on the local Any4D depth windows, yielding a temporally consistent metric depth sequence.
To recover the metric scale factor
s
s
for DPVO trajectory,
we compute the median ratio of the aligned Any4D depth
𝐃
t
Any4D
\mathbf{D}^{\mathrm{Any4D}}_{t}
to the DPVO depth
𝐃
^
t
DPVO
\hat{\mathbf{D}}^{\mathrm{DPVO}}_{t}
over the background pixels across all frames:
s
=
median
t
,
(
u
,
v
)
∈
ℬ
t
⁡
𝐃
t
Any4D
​
(
u
,
v
)
𝐃
^
t
DPVO
​
(
u
,
v
)
,
s=\operatorname{median}_{t,(u,v)\in\mathcal{B}_{t}}\frac{\mathbf{D}^{\mathrm{Any4D}}_{t}(u,v)}{\hat{{\mathbf{D}}}^{\mathrm{DPVO}}_{t}(u,v)},
(1)
where
ℬ
t
\mathcal{B}_{t}
is the valid background region in frame
t
t
, defined by excluding the hand regions projected from our reconstructed 3D hand mesh. We then calibrate the entire trajectory via
𝐩
t
=
s
​
𝐩
^
t
\mathbf{p}_{t}=s\hat{\mathbf{p}}_{t}
, yielding a metric camera trajectory.
Given the metric camera poses, we transform the camera-space hand vertices and joints, denoted generally as 3D coordinates
𝐱
t
cam
∈
ℝ
3
\mathbf{x}^{\mathrm{cam}}_{t}\in\mathbb{R}^{3}
, into world space via
𝐱
t
world
=
𝐑
t
⊤
​
(
𝐱
t
cam
−
𝐩
t
)
\mathbf{x}^{\mathrm{world}}_{t}=\mathbf{R}_{t}^{\top}(\mathbf{x}^{\mathrm{cam}}_{t}-\mathbf{p}_{t})
. The output for each video segment consists of the world-space bimanual states and actions, frame-wise camera intrinsics and extrinsics
(
𝐊
t
,
𝐓
t
)
(\mathbf{K}_{t},\mathbf{T}_{t})
, MANO parameters, and the Any4D metric scene depth.
Concerning efficiency, stage 2 represents the primary computational bottleneck of our entire pipeline, with the majority of the overhead stemming from camera trajectory estimation. This highlights another advantage of EgoSmith: DPVO that we use is far more lightweight than the dense DROID-SLAM, substantially reducing this major cost. Building on this, we further optimize the throughput of this stage through parallelized batching and asynchronous I/O pipelining.
While the original HaWoR processes only a single 16-frame temporal window at a time, we group multiple windows into a single batch for parallel forward passes. Furthermore, we overlap CPU-based frame decoding and cropping with GPU-based model inference to prevent GPU idling. Benchmarks on an 8
×
\times
A800 server using 8 video segments of 2K frames each show that our pipeline achieves an overall speedup of approximately 9
×
\times
compared to HaWoR.
Table 2:
Benchmark results of 4D motion estimation.
Method
RPE
↓
\downarrow
ATE
↓
\downarrow
WA-MPJPE
↓
\downarrow
W-MPJPE
↓
\downarrow
HaWoR
[
64
]
5.17
9.44
38.7
106.9
EgoSmith
(Ours)
2.42
7.60
25.9
86.0
We further quantitatively benchmark the
accuracy
of our 4D motion estimation pipeline against HaWoR
[
64
]
on high-quality annotated subsets of TACO
[
34
]
, H2O
[
26
]
, OakInk-v2
[
63
]
, and EgoVerse
[
43
]
. To jointly assess
camera-trajectory and hand-motion accuracy, we adopt four complementary metrics (all in mm):
•
Relative Pose Error (RPE)
: Quantifies local tracking drift and frame-to-frame jitter over a fixed temporal interval. Because it is computed without any global alignment, this metric is highly sensitive to metric-scale inaccuracies.
•
Absolute Trajectory Error (ATE)
: Assesses the overall camera trajectory shape and long-term drift. The estimated trajectory is aligned to the ground truth via a global
Sim
​
(
3
)
\mathrm{Sim}(3)
transform before evaluation, making this metric insensitive to absolute scale errors.
•
World-Aligned Mean Per Joint Position Error (WA-MPJPE)
: Measures hand joint errors while accounting for global hand placement. The hand joints are aligned to the ground truth using a single
Sim
​
(
3
)
\mathrm{Sim}(3)
transform over each
100
100
-frame segment, rather than performing per-frame local alignment.
•
World Mean Per Joint Position Error (W-MPJPE)
: Serves as the strictest metric for world-space hand tracking. It rigidly aligns only the first frame of each 100-frame segment via an SE(3) transform, thereby heavily penalizing absolute scale errors, temporal drift, and orientation misalignment across subsequent frames.
As shown in
Table
2
, EgoSmith substantially outperforms HaWoR across all four metrics. Our pipeline reduces RPE by over
50
%
50\%
, from 5.17 to 2.42 mm, and lowers ATE from
9.44
9.44
to
7.60
7.60
mm, indicating that the DPVO-based trajectory estimation yields superior local consistency and structural robustness. EgoSmith further improves WA-MPJPE from
38.7
38.7
to
25.9
25.9
mm and W-MPJPE from
106.9
106.9
to
86.0
86.0
mm, showing that our Any4D-based metric scaling, cross-window scale alignment, and global re-anchoring effectively mitigate scale
distortion and long-term drift, ensuring physically plausible world-space hand tracking over extended sequences.
Below, we present the prompt template designed for Qwen3.5-VL-Plus
[
44
]
to generate multi-granularity language annotations for egocentric human videos.
Egocentric Human Video Language Labeling Prompt
In this stage, we perform quality control on the reconstructed outputs to filter out reconstruction anomalies and problematic segments, ensuring overall data quality. This evaluation is conducted from coarse to fine across three granularities: entire episodes, chunk windows, and adjacent frames.
Episode-level checks assess the overall camera motion. We compute the statistics of camera extrinsics, both translation and rotation, for each episode and compare them against the distribution of other episodes within the same dataset, discarding those that deviate significantly. Because reasonable camera motion magnitudes vary across datasets due to different devices, scenes, and manipulation styles, we employ a dataset-specific IQR criterion rather than a universal threshold. Specifically, an episode is classified as an outlier if its statistics fall outside the range
[
Q
1
−
2.5
​
IQR
,
Q
3
+
2.5
​
IQR
]
[Q_{1}-2.5\mathrm{IQR},Q_{3}+2.5\mathrm{IQR}]
. This step filters out camera tracking drift, as well as segments dominated by walking or looking around instead of manipulation.
Chunk-level checks evaluate whether hands fall within physically reasonable spatial boundaries in a standardized egocentric coordinate frame. Directly comparing absolute hand coordinates is problematic, since they are coupled with camera and body movements. Specifically, within a sliding window spanning approximately the past 5 seconds and the future 30 frames, we transform all hand states and actions into the current camera frame. Under this canonical system, wrist positions are defined relative to the camera, and finger joints relative to the wrist. Within this coordinate system, we evaluate both distributional outliers and absolute physical limits. First, outliers in wrist translation/rotation and finger positions are identified using the same IQR criterion based on the respective dataset’s distribution. Second, we enforce a universal physical ceiling of 1.5 meters on each coordinate axis for the hands, as a human hand cannot physically reach further than this distance from the head. If any sliding chunk window within an episode violates either the IQR outlier threshold or the 1.5-meter physical limit, the entire episode is discarded.
Frame-level checks identify sudden jumps between adjacent frames. We compute the frame-to-frame changes in camera translation and rotation, wrist translation and rotation, and finger translation. Unlike the previous levels, we do not rely on dataset-specific distributions here, as the physical speed of human hands and heads has a universal limit. Any movement exceeding this limit is attributed to reconstruction artifacts rather than valid motion. We therefore apply fixed physical thresholds: camera translation
≤
0.20
​
m/frame
\leq 0.20\,\text{m/frame}
, wrist and finger translation
≤
0.30
​
m/frame
\leq 0.30\,\text{m/frame}
, camera rotation
≤
28
∘
/
frame
\leq 28^{\circ}/\text{frame}
, and wrist rotation
≤
41
∘
/
frame
\leq 41^{\circ}/\text{frame}
. An episode is discarded if any of its frames violate these thresholds.
Collectively, these three levels of checks filter out problematic segments caused by head tracking drift, inaccurate motion reconstruction, and motion discontinuities.
The final pre-training corpus is constructed by utilizing EgoSmith to process 12 raw egocentric datasets, ultimately yielding a standardized,
9.6
​
K
9.6\text{K}
-hour dataset. This curated corpus comprises world-space bimanual states and actions, frame-wise camera intrinsics and extrinsics, metric scene depth, and coarse-to-fine language annotations. In this section, we detail the scale, source composition, and semantic diversity of this dataset. The primary objective is not merely scaling up dataset volume; instead, the core advantage lies in data quality. Every sample is richly annotated and filtered through a rigorous quality-control pipeline, guaranteeing high-quality, modality-aligned manipulation knowledge to assist the model in learning steerable dexterous manipulation. The dataset further offers broad coverage of manipulation tasks, objects, and multi-granularity language descriptions.
Scale and Source Composition.
Figure
7
presents the duration contribution and proportion of each source dataset, while also indicating whether each annotation is natively provided or reconstructed by our pipeline. The source datasets are highly complementary, covering a wide range of devices, scenes, and manipulation styles. Although a small number of large-scale collections dominate the total duration, the breadth of sources contributes substantial diversity.
Task and Semantic Diversity.
To characterize the manipulation span of our dataset, we extract (verb, object) tuples from the L1 verb–object annotations and analyze the distributions of action verbs and manipulated objects. The dataset covers 8969 distinct object nouns and 623 action verbs (
Figures
7
and
7
).
Figure
7
illustrates the top 50 most frequent “verb + object” atomic tasks. Common tasks concentrate on fundamental manipulation skills, aligning with the natural distribution of daily hand–object interactions. At the same time, the distribution exhibits a prominent long tail: a large number of low-frequency yet semantically diverse tasks and objects provides broad coverage, preventing the dataset from being biased toward a few dominant actions. This combination of common fundamental skills, long-tail task diversity, and multi-granularity language annotations provides downstream models with abundant training samples, broad task coverage, and rich dexterous manipulation knowledge.
Dataset
Hours
Percentage (%)
Episodes
Hand
Depth
Camera
Language
Egocentric-100K
[
1
]
8
,
049
8,049
83.8
83.8
1
,
795
,
731
1{,}795{,}731
✓
✓
✓
✓
EgoVerse
[
43
]
690
690
7.2
7.2
35
,
175
35{,}175
✓
✓
EgoDex
[
19
]
370
370
3.9
3.9
147
,
588
147{,}588
✓
✓
Egocentric-10K
[
1
]
288
288
3.0
3.0
194
,
915
194{,}915
✓
✓
✓
✓
Ego4D
[
16
]
138
138
1.4
1.4
74
,
505
74{,}505
✓
✓
✓
✓
Epic-Kitchens
[
12
]
49
49
0.5
0.5
26
,
454
26{,}454
✓
✓
✓
✓
HoloAssist
[
54
]
11.5
11.5
0.1
0.1
11
,
426
11{,}426
✓
✓
HOT3D
[
4
]
4.5
4.5
0.05
0.05
1
,
105
1{,}105
✓
TACO
[
34
]
3.0
3.0
0.03
0.03
1
,
558
1{,}558
✓
OakInk-v2
[
63
]
1.7
1.7
0.02
0.02
891
891
✓
H2O
[
26
]
1.0
1.0
0.01
0.01
935
935
FPHA
[
15
]
0.5
0.5
0.01
0.01
578
578
✓
✓
✓
Total
𝟗
,
𝟔𝟎𝟔
\mathbf{9,606}
100
100
𝟐
,
𝟐𝟗𝟎
,
𝟖𝟔𝟏
\mathbf{2{,}290{,}861}
(a)
Dataset composition and annotation sources
(b)
Word cloud of manipulated objects
(c)
Word cloud of action verbs
(d)
Top-
50
50
“verb + object” atomic tasks
Figure 7:
Curated dataset statistics.
(a) Source composition of the
12
12
egocentric datasets, detailing duration, percentage, episode count, and per-annotation origin. Checkmarks denote annotations generated by our pipeline, while blank entries indicate natively provided annotations. For FPHA, the checkmark under
Hand
specifically denotes the additionally annotated second hand. (b)–(d) Task and semantic diversity: word clouds of the manipulated objects and action verbs, and the most frequent verb–object atomic tasks, exhibiting long-tailed coverage over diverse manipulation skills.
This section details the implementation of the Robot Stack (
Section
B.1
), alongside the collection protocols and statistics of the
187
187
-hour real-robot dataset (
Section
B.2
).
The two physical embodiments utilized in our work are illustrated in
Figure
3
. The primary platform, referred to as the RealMan embodiment, consists of two 7-DoF RealMan RM75-6F robotic arms and two 6-DoF Ruiyan RY-H2 dexterous hands. It is equipped with one head-mounted and one chest-mounted Intel RealSense D455 camera, providing dual egocentric viewpoints to ensure a comprehensive visual field. The AgiBot G1 embodiment is adapted from an AgiBot G1 humanoid robot by replacing its default end-effectors with two 6-DoF Ruiyan RY-H2 hands. It features one head-mounted Intel RealSense D455 camera and two wrist cameras, the latter of which are unused. During experiments, the robot’s neck, waist, and mobile base are kept fixed. Both platforms conduct tabletop manipulation on an operational table. The RealMan platform serves as the primary embodiment, on which the entire
187
187
-hour dataset was collected. All experiments are conducted on this setup, with the sole exception of the few-shot adaptation experiment performed on the AgiBot G1.
For both platforms, the human wrist tracker, arm kinematic solvers, and joint control ROS 2 nodes operate at
100
100
Hz, while the glove, hand solvers, and control ROS 2 nodes run at
80
80
Hz. This high-frequency loop ensures near-zero latency, enabling intuitive bimanual teleoperation and the collection of fine-grained manipulation demonstrations. The cameras capture frames at
30
30
Hz. Data is recorded at these native frequencies and subsequently resampled to
30
30
Hz for training.
Robot data collection aims to ground the human manipulation priors learned from large-scale pre-training onto the target physical embodiment. To preserve and transfer this pre-trained knowledge, minimizing the domain gap between robot and human data is essential. Because the robotic palm is slightly longer than a human palm, we translate the robot’s wrist coordinate frame axially forward. This alignment ensures that the scale from the wrist to the fingertips matches human anatomical proportions. When converting raw robot data into training data, this coordinate transformation is applied alongside hand-eye calibration to map hand actions into the camera coordinate frame. Correspondingly, during real-time policy inference, the control stack executes the inverse transformation and hand-eye mapping to project predicted actions back to the robot’s physical coordinate space.
In practice, hand-eye calibration is prone to drift due to mechanical maintenance, wear, or collisions, with such discrepancies sometimes only identified
post
-acquisition. To ensure data quality, we introduce an offline verification and automated recalibration pipeline. RGB-D images from the camera are projected into a 3D point cloud, and the robot’s 3D mesh is rendered in the same space based on the hand-eye calibration, utilizing their spatial overlap for validation. Upon detecting misalignment, our pipeline runs FoundationPose
[
55
]
on the point cloud to estimate the robot hand’s 6D pose based on its URDF. Through FK, the calibration matrix is back-calculated for each frame. Averaging these matrices across frames and removing outliers robustly recovers the calibration parameters, preventing data degradation.
Below, we present the prompt template designed for Qwen3-VL-Flash
[
3
]
to generate multi-granularity language annotations for teleoperated demonstrations.
Robot Data Language Labeling Prompt
Although pre-training on egocentric human videos equips the model with rich dexterous manipulation priors, direct physical deployment is prevented by the embodiment gap across visual appearance, dynamics, and kinematics. Furthermore, because these pre-training labels are generated by automated depth and hand pose estimation models, their accuracy is constrained by current model capabilities, potentially introducing systematic biases into the pre-trained policy. Therefore, collecting real-robot teleoperation data aims to correct and ground these priors onto the target embodiment with high sample efficiency. Additionally, due to kinematic constraints on robotic degrees of freedom, the robot cannot perfectly replicate all fine-grained human hand movements. To achieve steerable manipulation, we must collect a highly diverse range of free-form tasks to maximize the coverage of basic manipulation primitives, guiding the model to efficiently transfer cross-domain knowledge and fostering robust compositional generalization and multi-task instruction following.
Based on these considerations, within the kinematic limits of RealMan’s degrees of freedom, we design
193
193
semantically distinct dexterous manipulation tasks. For each task, approximately
300
300
randomized, diverse demonstration trajectories, totaling around
1
1
hour, are collected under cluttered scenarios, constructing a high-quality real-robot dataset of
187
187
hours and
55
​
K
55\text{K}
trajectories.
These
193
193
tasks are classified into two major categories:
•
Common Tasks
(
56
56
tasks): Everyday manipulations that are readily achievable within the current robot hardware configuration and sensor limits, yielding high teleoperation success rates.
•
Long-Tail Tasks
(
137
137
tasks): Infrequent and physically challenging manipulations, such as contact-sensitive operations lacking tactile feedback, with lower collection success rates, primarily designed to ensure comprehensive semantic coverage of the dexterous manipulation space.
Furthermore, based on motion characteristics and physical interactions, all tasks are categorized into seven classes:
•
PnP-Easy
: Single-step tabletop pick-and-place where objects are easily graspable and the placement space is open.
•
PnP-Medium
: Non-planar or 3D spatial pick-and-place that requires precise operations related to containers, demanding higher control accuracy and spatial perception, such as “
put tennis ball into ball holder
”.
•
PnP-Hard
: Multi-step or high-precision pick-and-place sequences, such as “
stack paper cups
”.
•
Non-prehensile
: Actions not involving traditional finger grasping, including pushing, pulling, and pressing.
•
Reorient
: Operations involving rotation and reorientation, such as “
pour water
” or “
flip paper cups
”.
•
Bimanual
: Bimanual tasks requiring high synchronization and spatial coordination between arms and hands, such as “
plug cable into charger
”.
•
Contact-rich
: Operations involving frequent and complex physical contact interactions, requiring physical understanding, such as “
wipe whiteboard
”.
Throughout data collection, strict requirements are enforced on the randomness, diversity, and quality of each trajectory. Specifically, the tabletop features cluttered, unstructured scenes rather than pre-arranged, simplified environments, avoiding task identification from visual inputs alone. This design forces the model to deeply align language instructions with physical actions, encouraging it to learn general task semantics and execution goals rather than memorizing demonstration trajectories. Furthermore, given the highly randomized object configurations, operators are instructed to perform teleoperation in a natural, human-like manner. Consequently, different demonstrations of the same task exhibit substantial variations in execution trajectories, thereby covering a broader distribution of the mapping from human priors to the robot’s action space.
As illustrated in
Figure
8
, we analyze the
187
187
-hour real-robot dataset across several key dimensions, including the verb and noun word clouds with their top 30 frequency distributions, the task duration breakdown across seven manipulation categories, and the detailed duration statistics for the
56
56
common and
137
137
long-tail tasks. These statistics reveal a diverse vocabulary of actions and manipulated objects, alongside a balanced distribution across both manipulation categories and task durations. Encompassing rich semantic concepts and manipulation primitives, this dataset effectively grounds pre-trained human manipulation priors onto the RealMan embodiment. Additionally,
Figure
9
showcases the dual-view image sequences and language annotations of a teleoperation trajectory, while
Figure
10
visualizes sample tasks across each category.
(a)
Word cloud of nouns in annotation
(b)
Top 30 noun frequencies
(c)
Word cloud of verbs in annotation
(d)
Top 30 verb frequencies
(e)
Task duration by seven categories
(f)
Common task duration statistics
(g)
Long-tail task duration statistics
Figure 8:
Teleoperation dataset statistics. (1)
Figure
8
-
Figure
8
: Word clouds and top-30 frequency distributions of nouns and verbs in language annotations. (2)
Figure
8
: Task duration breakdown by seven manipulation categories (PnP-Easy/Medium/Hard, Non-prehensile, Reorient, Bimanual, Contact-rich). (3)
Figure
8
and
Figure
8
: Detailed duration statistics for common tasks (56 tasks) and long-tail tasks (137 tasks).
Figure 9:
Dual-view image sequence and three-level language annotations of a representative trajectory
. The frames illustrate a cluttered, randomized setup where the operator executes the task in a natural, human-like manner. The hierarchical language annotations describe the manipulation process from coarse to fine, assisting the model in aligning free-form instructions with physical actions.
Figure 10:
Representative task examples across seven manipulation categories
. These diverse tasks are executed in cluttered, randomized scenarios using natural, human-like manipulations, covering a wide range of action semantics and manipulation primitives to facilitate the efficient grounding of pre-trained egocentric human priors onto the physical robot.
Dataset
Samples
Captioning
Visual Question
Answering
Multiple-Choice
Question
Pointing
Bounding Box
Affordance
Trajectory
Spatial
Planning
FineVision
[
56
]
3.5
​
M
3.5\text{M}
✓
✓
✓
✓
RefSpatial
[
68
]
2.5
​
M
2.5\text{M}
✓
✓
✓
✓
RoboInter-VQA
[
27
]
1.6
​
M
1.6\text{M}
✓
✓
✓
✓
✓
✓
✓
RoboPoint
[
61
]
1.3
​
M
1.3\text{M}
✓
✓
✓
✓
RoboAfford
[
50
]
765
​
K
765\text{K}
✓
✓
✓
✓
Robo2VLM
[
10
]
678
​
K
678\text{K}
✓
✓
ShareRobot
[
23
]
13
​
K
13\text{K}
✓
✓
✓
✓
✓
Table 3:
VLM datasets used for EgoSteer co-training
. This table details the sample sizes and the multi-modal and embodied knowledge domains covered by each dataset. This co-training mixture integrates general vision-language knowledge with interaction understanding and spatial-geometric priors, preserving the model’s inherent world knowledge while facilitating its ability to follow free-form instructions to generalize across diverse manipulation tasks.
Figure 11:
Statistical distribution of samples across VLM co-training datasets.
This section first introduces the VLM datasets utilized for co-training with VLA data,
i.e.
egocentric human videos and real-robot data, to preserve EgoSteer’s vision-language knowledge and ensure generalization (
Section
C.1
). We then present implementation details of EgoSteer (
Section
C.2
).
To preserve general vision-language reasoning capabilities while simultaneously cultivating robust robotic task comprehension, a
10.4
​
M
10.4\text{M}
-sample VLM co-training mixture is curated across seven datasets, ranging from open-world perception to embodied interaction grounding, to co-train EgoSteer. This mixture comprises four categories of data:
•
General VLM Pre-Training
: Incorporates FineVision
[
56
]
to prevent the catastrophic forgetting of open-world semantic concepts and preserve general visual-language reasoning.
•
Spatial Grounding
: Utilizes RefSpatial
[
68
]
and RoboPoint
[
61
]
for multi-step spatial referring, 2D visual grounding, and precise coordinate localization.
•
Embodied QA
: Integrates RoboInter-VQA
[
27
]
, Robo2VLM
[
10
]
, and ShareRobot
[
23
]
to support embodied question answering and temporal reasoning. This preserves the model’s capabilities in high-level task planning and causal scene understanding.
•
Affordance Perception
: Adopts RoboAfford
[
50
]
for fine-grained manipulation affordance prediction and spatial interaction grounding.
The sample distribution of these datasets is illustrated in
Figure
11
, with their covered multimodal, embodied knowledge domains detailed in
Table
3
. We standardize these datasets to comply with the conversational input format of Qwen3-VL. Specifically, we normalize 2D bounding box and point coordinates to Qwen3-VL’s native
[
0
,
1000
]
[0,1000]
scale and adopt its standard representation formats. To ensure training stability, we only utilize single-image samples and exclude those with context lengths exceeding our maximum budget. By co-training on this VLM mixture, EgoSteer preserves open-world vision-language understanding and reasoning while enhancing its comprehension of robotics-specific tasks, thereby assisting the policy in following open-ended instructions and generalizing to novel manipulation scenarios.
EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.
The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.
The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.
The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.
Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.
To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.
Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.
This section provides supplementary details for the experimental evaluations presented in
Section
6
, detailing the model training hyperparameters and per-task success rates. In particular,
Table
4
lists the hyperparameter configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
Table
5
presents the per-task success rates on the four evaluation tasks in
Section
6.2
to demonstrate the performance gains from DAgger refinement. For the scaling analysis in
Section
6.3
,
Table
6
lists the training hyperparameters of EgoSteer-3K/6K/9.6K pre-trained models, their respective post-training runs, and the baseline trained from scratch, with their corresponding task-specific success rates reported in
Table
7
. Additionally,
Table
8
compares the detailed success rates of our method against the Being-H0.5 and
π
0.5
\pi_{0.5}
baselines across ten tasks.
Table
10
specifies the training configurations of the ablation experiment from
Section
6.4
, including the EgoSteer-1K model and its three ablated variants,
No WM-objective
,
No training-RTC
, and
Noisy data
. Their respective performance comparison is provided in
Table
9
. Finally,
Table
11
specifies the hyperparameters for the few-shot fine-tuning of our pre-trained EgoSteer-9.6K on the two challenging, long-horizon tasks evaluated in
Section
6.5
, namely
Box-Folding
and
Cake-Unboxing
.
Hyperparameter
Pre-Training
Post-Training
Camera setup
Head
Head & Chest
Resolution
384
×
384
384\times 384
640
×
480
640\times 480
GPUs
128 A800
96 A800
Gradient accumulation
2
1
Global batch size
4608
384
Training steps
175K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Freeze-VLM steps
5000
0
Warmup steps
2000
2000
Training time
164 h
29 h
Table 4:
Training configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
During pre-training, the VLM backbone is frozen for the first 5,000 steps, during which AE and WM are warmed up for 2,000 steps; once it is unfrozen, VLM is warmed up for 2,000 steps.
Task
EgoSteer-DG
EgoSteer-FT
Stack tableware
80.0%
50.0%
Close laptop
70.0%
10.0%
Place phone on stand
50.0%
0.0%
Flip cup
50.0%
30.0%
Average
62.5%
22.5%
Table 5:
Success rates across four highly dexterous and failure-prone manipulation tasks, comparing EgoSteer-DG against EgoSteer-FT in
Section
6.2
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Camera setup
Head
Head
Head
Head & Chest
Head & Chest
Head & Chest
Head & Chest
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
GPUs
64 A800
64 A800
128 A800
64 A800
32 A800
64 A800
64 A800
Gradient accumulation
2
4
2
2
4
2
2
Global batch size
2304
4608
4608
512
512
512
512
Training steps
100K
100K
160K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 6:
Training configurations for the pre-training scaling study in
Section
6.3
.
Pre-Training Data (Hours)
Task
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Grasp object
80%
80%
70%
100%
Hand over object
70%
80%
80%
100%
Place items into container
40%
80%
90%
100%
Point at object
20%
40%
60%
70%
Place toy chick into slot
30%
0%
20%
40%
Pull out tissue
60%
20%
50%
80%
Push ball into box
0%
0%
10%
20%
Put garbage into trash bin
0%
60%
30%
30%
Stack paper cups
0%
20%
10%
40%
Stack tableware
0%
20%
10%
20%
Average
30%
40%
43%
60%
Table 7:
Per-task success rates for the pre-training scaling study in
Section
6.3
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Task
Ours
Being-H0.5
π
0.5
\pi_{0.5}
Grasp object
100%
80%
80%
Hand over object
100%
60%
20%
Place items into container
100%
50%
0%
Pour items out of box
50%
30%
0%
Place bread on tray
70%
80%
40%
Pull out tissue
80%
10%
30%
Place item at specific orientation
30%
30%
30%
Attach eraser to whiteboard
90%
50%
20%
Put garbage into trash bin
30%
0%
0%
Put tennis ball into bucket
90%
0%
0%
Average
74%
39%
22%
Table 8:
Per-task comparison between EgoSteer-9.6K and two VLA baselines, Being-H0.5 and
π
0.5
\pi_{0.5}
, in
Section
6.3
. All methods are post-trained on our real-robot dataset and evaluated on the same 10 tasks. Each task uses 10 randomized trials. Bold values highlight the best performance.
Task
Ours
No WM-objective
No training-RTC
Noisy data
Grasp object
60%
40%
30%
60%
Hand over object
40%
20%
30%
40%
Place items into container
50%
30%
50%
70%
Pour items out of box
60%
50%
10%
0%
Place bread on tray
70%
10%
60%
50%
Pull out tissue
20%
30%
60%
10%
Place item at specific orientation
20%
10%
10%
20%
Attach eraser to whiteboard
50%
80%
50%
40%
Put garbage into trash bin
30%
0%
0%
0%
Put tennis ball into bucket
40%
40%
90%
40%
Average
44%
31%
39%
33%
Table 9:
Per-task success rates for the ablation study in
Section
6.4
. EgoSteer-1K is compared with ablated variants. Each task uses 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
Ours
No WM-objective
No training-RTC
Noisy data
Ours
No WM-objective
No training-RTC
Noisy data
Camera setup
Head
Head
Head
Head
Head
Head
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
Global batch size
1152
1152
1152
1152
1152
1152
1152
1152
Training steps
30K
80K
30K
20K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 10:
Training hyperparameters for the ablation study in
Section
6.4
. All variants are pre-trained on 1K hours of egocentric data and post-trained on the same real-robot dataset, without DAgger refinement. Pre-training steps are selected at the lowest evaluation
L
1
L_{1}
loss for each pre-training run.
Hyperparameter
Box-Folding
Cake-Unboxing
Pre-training checkpoint
EgoSteer-9.6K at 155K steps
Camera setup
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
8 A800
8 A800
Gradient accumulation
1
1
Global batch size
144
144
Fine-tuning steps
44K
12K
Learning rate (VLM / AE / WM)
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Demonstrations
120
229
Table 11:
Few-shot fine-tuning hyperparameters of EgoSteer-9.6K for the two long-horizon dexterous tasks in
Section
6.5
.

#### A.1.1
Pre-Filtering Heuristics

The pre-filtering stage employs frame-wise heuristics to rapidly discard low-quality segments, such as those containing locomotion, excessive head movement, hand absence, occlusion, or others’ hand interference. These five scenarios are handled by two specialized gates: a camera gate for motion-related issues, and a hand gate for visibility anomalies. A contiguous segment is pruned only if it contains at least three consecutive invalid frames; isolated failures are retained as they exert negligible impact on subsequent reconstruction.
The camera gate estimates ego-motion using sparse optical flow. For each frame, a 128-point grid is tracked back to their positions 15 frames earlier via pyramidal Lucas–Kanade
[
7
]
. We fit a similarity transform to these correspondences using RANSAC
[
18
]
; the frame passes this gate if the translation of the estimated transform is within
10
%
10\%
of the image’s longer dimension.
The hand gate uses YOLO
[
46
,
42
]
to detect hands, retaining a bounding box as valid only if it satisfies three criteria: (a) confidence
≥
0.30
\geq 0.30
to reject false positives; (b) area within
[
2
%
,
50
%
]
[2\%,50\%]
of the image, where the
50
%
50\%
upper bound excludes hands abnormally close to the lens, and the
2
%
2\%
lower bound is calibrated on Egocentric-10K/100K
[
2
,
1
]
subsets, where we manually labeled small bounding boxes to find a threshold that filters out most other people’s hands while retaining the operator’s; and (c) spatially intersecting with the lower-central region (normalized
[
0.075
,
0.925
]
[0.075,0.925]
horizontally,
[
0.075
,
1.0
]
[0.075,1.0]
vertically). The gate requires
≥
2
\geq 2
valid detections per frame, which naturally filters out hand absence, occlusion, or cases where only other people’s hands are present, while preserving clear bimanual manipulation.
Together, the two gates yield video segments characterized by stable camera motion and clearly visible bimanual interactions.

#### A.1.2
4D Motion Estimation

In stage 2, we reconstruct hand motions in a unified, metric world-space coordinate system. Within this pipeline, we adopt the ViT module from HaWoR
[
64
]
as an off-the-shelf camera-frame hand reconstructor. Specifically, hands are detected and cropped across frames, and the ViT processes these inputs in temporal windows to regress the frame-wise MANO
[
47
]
pose parameters
𝜽
t
∈
ℝ
51
\boldsymbol{\theta}_{t}\in\mathbb{R}^{51}
, shape parameters
𝜷
t
∈
ℝ
10
\boldsymbol{\beta}_{t}\in\mathbb{R}^{10}
, and the camera-relative root translation
𝐭
t
∈
ℝ
3
\mathbf{t}_{t}\in\mathbb{R}^{3}
. Although this camera-space hand reconstruction is highly reliable, placing these reconstructions in world space with accurate physical dimensions requires a robust, metric camera trajectory. This is a primary limitation of the original HaWoR pipeline, as its dependency on DROID-SLAM
[
52
]
can accumulate drift under egocentric conditions with rapid head movements or textureless environments. To address this, we design a robust pipeline to recover a metric, temporally consistent camera trajectory in world space, onto which the hand reconstructions are subsequently mapped.
We replace DROID-SLAM with DPVO
[
53
]
to estimate the camera trajectory. DPVO is more robust in long-range egocentric scenarios and incurs much lower computational cost, outputting up-to-scale camera poses
𝐓
^
t
=
(
𝐑
t
,
𝐩
^
t
)
∈
S
​
E
​
(
3
)
\hat{\mathbf{T}}_{t}=(\mathbf{R}_{t},\hat{\mathbf{p}}_{t})\in SE(3)
along with focal length, where the hat notation
⋅
^
\hat{\cdot}
denotes up-to-scale quantities. We then anchor the trajectory to the physical scale using metric depth estimates from Any4D
[
24
]
as a reference. To ensure temporal coherence across the entire sequence, we perform a cross-chunk alignment on the local Any4D depth windows, yielding a temporally consistent metric depth sequence.
To recover the metric scale factor
s
s
for DPVO trajectory,
we compute the median ratio of the aligned Any4D depth
𝐃
t
Any4D
\mathbf{D}^{\mathrm{Any4D}}_{t}
to the DPVO depth
𝐃
^
t
DPVO
\hat{\mathbf{D}}^{\mathrm{DPVO}}_{t}
over the background pixels across all frames:
s
=
median
t
,
(
u
,
v
)
∈
ℬ
t
⁡
𝐃
t
Any4D
​
(
u
,
v
)
𝐃
^
t
DPVO
​
(
u
,
v
)
,
s=\operatorname{median}_{t,(u,v)\in\mathcal{B}_{t}}\frac{\mathbf{D}^{\mathrm{Any4D}}_{t}(u,v)}{\hat{{\mathbf{D}}}^{\mathrm{DPVO}}_{t}(u,v)},
(1)
where
ℬ
t
\mathcal{B}_{t}
is the valid background region in frame
t
t
, defined by excluding the hand regions projected from our reconstructed 3D hand mesh. We then calibrate the entire trajectory via
𝐩
t
=
s
​
𝐩
^
t
\mathbf{p}_{t}=s\hat{\mathbf{p}}_{t}
, yielding a metric camera trajectory.
Given the metric camera poses, we transform the camera-space hand vertices and joints, denoted generally as 3D coordinates
𝐱
t
cam
∈
ℝ
3
\mathbf{x}^{\mathrm{cam}}_{t}\in\mathbb{R}^{3}
, into world space via
𝐱
t
world
=
𝐑
t
⊤
​
(
𝐱
t
cam
−
𝐩
t
)
\mathbf{x}^{\mathrm{world}}_{t}=\mathbf{R}_{t}^{\top}(\mathbf{x}^{\mathrm{cam}}_{t}-\mathbf{p}_{t})
. The output for each video segment consists of the world-space bimanual states and actions, frame-wise camera intrinsics and extrinsics
(
𝐊
t
,
𝐓
t
)
(\mathbf{K}_{t},\mathbf{T}_{t})
, MANO parameters, and the Any4D metric scene depth.
Concerning efficiency, stage 2 represents the primary computational bottleneck of our entire pipeline, with the majority of the overhead stemming from camera trajectory estimation. This highlights another advantage of EgoSmith: DPVO that we use is far more lightweight than the dense DROID-SLAM, substantially reducing this major cost. Building on this, we further optimize the throughput of this stage through parallelized batching and asynchronous I/O pipelining.
While the original HaWoR processes only a single 16-frame temporal window at a time, we group multiple windows into a single batch for parallel forward passes. Furthermore, we overlap CPU-based frame decoding and cropping with GPU-based model inference to prevent GPU idling. Benchmarks on an 8
×
\times
A800 server using 8 video segments of 2K frames each show that our pipeline achieves an overall speedup of approximately 9
×
\times
compared to HaWoR.
Table 2:
Benchmark results of 4D motion estimation.
Method
RPE
↓
\downarrow
ATE
↓
\downarrow
WA-MPJPE
↓
\downarrow
W-MPJPE
↓
\downarrow
HaWoR
[
64
]
5.17
9.44
38.7
106.9
EgoSmith
(Ours)
2.42
7.60
25.9
86.0
We further quantitatively benchmark the
accuracy
of our 4D motion estimation pipeline against HaWoR
[
64
]
on high-quality annotated subsets of TACO
[
34
]
, H2O
[
26
]
, OakInk-v2
[
63
]
, and EgoVerse
[
43
]
. To jointly assess
camera-trajectory and hand-motion accuracy, we adopt four complementary metrics (all in mm):
•
Relative Pose Error (RPE)
: Quantifies local tracking drift and frame-to-frame jitter over a fixed temporal interval. Because it is computed without any global alignment, this metric is highly sensitive to metric-scale inaccuracies.
•
Absolute Trajectory Error (ATE)
: Assesses the overall camera trajectory shape and long-term drift. The estimated trajectory is aligned to the ground truth via a global
Sim
​
(
3
)
\mathrm{Sim}(3)
transform before evaluation, making this metric insensitive to absolute scale errors.
•
World-Aligned Mean Per Joint Position Error (WA-MPJPE)
: Measures hand joint errors while accounting for global hand placement. The hand joints are aligned to the ground truth using a single
Sim
​
(
3
)
\mathrm{Sim}(3)
transform over each
100
100
-frame segment, rather than performing per-frame local alignment.
•
World Mean Per Joint Position Error (W-MPJPE)
: Serves as the strictest metric for world-space hand tracking. It rigidly aligns only the first frame of each 100-frame segment via an SE(3) transform, thereby heavily penalizing absolute scale errors, temporal drift, and orientation misalignment across subsequent frames.
As shown in
Table
2
, EgoSmith substantially outperforms HaWoR across all four metrics. Our pipeline reduces RPE by over
50
%
50\%
, from 5.17 to 2.42 mm, and lowers ATE from
9.44
9.44
to
7.60
7.60
mm, indicating that the DPVO-based trajectory estimation yields superior local consistency and structural robustness. EgoSmith further improves WA-MPJPE from
38.7
38.7
to
25.9
25.9
mm and W-MPJPE from
106.9
106.9
to
86.0
86.0
mm, showing that our Any4D-based metric scaling, cross-window scale alignment, and global re-anchoring effectively mitigate scale
distortion and long-term drift, ensuring physically plausible world-space hand tracking over extended sequences.

#### A.1.3
Language Labeling Prompt

Below, we present the prompt template designed for Qwen3.5-VL-Plus
[
44
]
to generate multi-granularity language annotations for egocentric human videos.
Egocentric Human Video Language Labeling Prompt
In this stage, we perform quality control on the reconstructed outputs to filter out reconstruction anomalies and problematic segments, ensuring overall data quality. This evaluation is conducted from coarse to fine across three granularities: entire episodes, chunk windows, and adjacent frames.
Episode-level checks assess the overall camera motion. We compute the statistics of camera extrinsics, both translation and rotation, for each episode and compare them against the distribution of other episodes within the same dataset, discarding those that deviate significantly. Because reasonable camera motion magnitudes vary across datasets due to different devices, scenes, and manipulation styles, we employ a dataset-specific IQR criterion rather than a universal threshold. Specifically, an episode is classified as an outlier if its statistics fall outside the range
[
Q
1
−
2.5
​
IQR
,
Q
3
+
2.5
​
IQR
]
[Q_{1}-2.5\mathrm{IQR},Q_{3}+2.5\mathrm{IQR}]
. This step filters out camera tracking drift, as well as segments dominated by walking or looking around instead of manipulation.
Chunk-level checks evaluate whether hands fall within physically reasonable spatial boundaries in a standardized egocentric coordinate frame. Directly comparing absolute hand coordinates is problematic, since they are coupled with camera and body movements. Specifically, within a sliding window spanning approximately the past 5 seconds and the future 30 frames, we transform all hand states and actions into the current camera frame. Under this canonical system, wrist positions are defined relative to the camera, and finger joints relative to the wrist. Within this coordinate system, we evaluate both distributional outliers and absolute physical limits. First, outliers in wrist translation/rotation and finger positions are identified using the same IQR criterion based on the respective dataset’s distribution. Second, we enforce a universal physical ceiling of 1.5 meters on each coordinate axis for the hands, as a human hand cannot physically reach further than this distance from the head. If any sliding chunk window within an episode violates either the IQR outlier threshold or the 1.5-meter physical limit, the entire episode is discarded.
Frame-level checks identify sudden jumps between adjacent frames. We compute the frame-to-frame changes in camera translation and rotation, wrist translation and rotation, and finger translation. Unlike the previous levels, we do not rely on dataset-specific distributions here, as the physical speed of human hands and heads has a universal limit. Any movement exceeding this limit is attributed to reconstruction artifacts rather than valid motion. We therefore apply fixed physical thresholds: camera translation
≤
0.20
​
m/frame
\leq 0.20\,\text{m/frame}
, wrist and finger translation
≤
0.30
​
m/frame
\leq 0.30\,\text{m/frame}
, camera rotation
≤
28
∘
/
frame
\leq 28^{\circ}/\text{frame}
, and wrist rotation
≤
41
∘
/
frame
\leq 41^{\circ}/\text{frame}
. An episode is discarded if any of its frames violate these thresholds.
Collectively, these three levels of checks filter out problematic segments caused by head tracking drift, inaccurate motion reconstruction, and motion discontinuities.
The final pre-training corpus is constructed by utilizing EgoSmith to process 12 raw egocentric datasets, ultimately yielding a standardized,
9.6
​
K
9.6\text{K}
-hour dataset. This curated corpus comprises world-space bimanual states and actions, frame-wise camera intrinsics and extrinsics, metric scene depth, and coarse-to-fine language annotations. In this section, we detail the scale, source composition, and semantic diversity of this dataset. The primary objective is not merely scaling up dataset volume; instead, the core advantage lies in data quality. Every sample is richly annotated and filtered through a rigorous quality-control pipeline, guaranteeing high-quality, modality-aligned manipulation knowledge to assist the model in learning steerable dexterous manipulation. The dataset further offers broad coverage of manipulation tasks, objects, and multi-granularity language descriptions.
Scale and Source Composition.
Figure
7
presents the duration contribution and proportion of each source dataset, while also indicating whether each annotation is natively provided or reconstructed by our pipeline. The source datasets are highly complementary, covering a wide range of devices, scenes, and manipulation styles. Although a small number of large-scale collections dominate the total duration, the breadth of sources contributes substantial diversity.
Task and Semantic Diversity.
To characterize the manipulation span of our dataset, we extract (verb, object) tuples from the L1 verb–object annotations and analyze the distributions of action verbs and manipulated objects. The dataset covers 8969 distinct object nouns and 623 action verbs (
Figures
7
and
7
).
Figure
7
illustrates the top 50 most frequent “verb + object” atomic tasks. Common tasks concentrate on fundamental manipulation skills, aligning with the natural distribution of daily hand–object interactions. At the same time, the distribution exhibits a prominent long tail: a large number of low-frequency yet semantically diverse tasks and objects provides broad coverage, preventing the dataset from being biased toward a few dominant actions. This combination of common fundamental skills, long-tail task diversity, and multi-granularity language annotations provides downstream models with abundant training samples, broad task coverage, and rich dexterous manipulation knowledge.
Dataset
Hours
Percentage (%)
Episodes
Hand
Depth
Camera
Language
Egocentric-100K
[
1
]
8
,
049
8,049
83.8
83.8
1
,
795
,
731
1{,}795{,}731
✓
✓
✓
✓
EgoVerse
[
43
]
690
690
7.2
7.2
35
,
175
35{,}175
✓
✓
EgoDex
[
19
]
370
370
3.9
3.9
147
,
588
147{,}588
✓
✓
Egocentric-10K
[
1
]
288
288
3.0
3.0
194
,
915
194{,}915
✓
✓
✓
✓
Ego4D
[
16
]
138
138
1.4
1.4
74
,
505
74{,}505
✓
✓
✓
✓
Epic-Kitchens
[
12
]
49
49
0.5
0.5
26
,
454
26{,}454
✓
✓
✓
✓
HoloAssist
[
54
]
11.5
11.5
0.1
0.1
11
,
426
11{,}426
✓
✓
HOT3D
[
4
]
4.5
4.5
0.05
0.05
1
,
105
1{,}105
✓
TACO
[
34
]
3.0
3.0
0.03
0.03
1
,
558
1{,}558
✓
OakInk-v2
[
63
]
1.7
1.7
0.02
0.02
891
891
✓
H2O
[
26
]
1.0
1.0
0.01
0.01
935
935
FPHA
[
15
]
0.5
0.5
0.01
0.01
578
578
✓
✓
✓
Total
𝟗
,
𝟔𝟎𝟔
\mathbf{9,606}
100
100
𝟐
,
𝟐𝟗𝟎
,
𝟖𝟔𝟏
\mathbf{2{,}290{,}861}
(a)
Dataset composition and annotation sources
(b)
Word cloud of manipulated objects
(c)
Word cloud of action verbs
(d)
Top-
50
50
“verb + object” atomic tasks
Figure 7:
Curated dataset statistics.
(a) Source composition of the
12
12
egocentric datasets, detailing duration, percentage, episode count, and per-annotation origin. Checkmarks denote annotations generated by our pipeline, while blank entries indicate natively provided annotations. For FPHA, the checkmark under
Hand
specifically denotes the additionally annotated second hand. (b)–(d) Task and semantic diversity: word clouds of the manipulated objects and action verbs, and the most frequent verb–object atomic tasks, exhibiting long-tailed coverage over diverse manipulation skills.
This section details the implementation of the Robot Stack (
Section
B.1
), alongside the collection protocols and statistics of the
187
187
-hour real-robot dataset (
Section
B.2
).
The two physical embodiments utilized in our work are illustrated in
Figure
3
. The primary platform, referred to as the RealMan embodiment, consists of two 7-DoF RealMan RM75-6F robotic arms and two 6-DoF Ruiyan RY-H2 dexterous hands. It is equipped with one head-mounted and one chest-mounted Intel RealSense D455 camera, providing dual egocentric viewpoints to ensure a comprehensive visual field. The AgiBot G1 embodiment is adapted from an AgiBot G1 humanoid robot by replacing its default end-effectors with two 6-DoF Ruiyan RY-H2 hands. It features one head-mounted Intel RealSense D455 camera and two wrist cameras, the latter of which are unused. During experiments, the robot’s neck, waist, and mobile base are kept fixed. Both platforms conduct tabletop manipulation on an operational table. The RealMan platform serves as the primary embodiment, on which the entire
187
187
-hour dataset was collected. All experiments are conducted on this setup, with the sole exception of the few-shot adaptation experiment performed on the AgiBot G1.
For both platforms, the human wrist tracker, arm kinematic solvers, and joint control ROS 2 nodes operate at
100
100
Hz, while the glove, hand solvers, and control ROS 2 nodes run at
80
80
Hz. This high-frequency loop ensures near-zero latency, enabling intuitive bimanual teleoperation and the collection of fine-grained manipulation demonstrations. The cameras capture frames at
30
30
Hz. Data is recorded at these native frequencies and subsequently resampled to
30
30
Hz for training.
Robot data collection aims to ground the human manipulation priors learned from large-scale pre-training onto the target physical embodiment. To preserve and transfer this pre-trained knowledge, minimizing the domain gap between robot and human data is essential. Because the robotic palm is slightly longer than a human palm, we translate the robot’s wrist coordinate frame axially forward. This alignment ensures that the scale from the wrist to the fingertips matches human anatomical proportions. When converting raw robot data into training data, this coordinate transformation is applied alongside hand-eye calibration to map hand actions into the camera coordinate frame. Correspondingly, during real-time policy inference, the control stack executes the inverse transformation and hand-eye mapping to project predicted actions back to the robot’s physical coordinate space.
In practice, hand-eye calibration is prone to drift due to mechanical maintenance, wear, or collisions, with such discrepancies sometimes only identified
post
-acquisition. To ensure data quality, we introduce an offline verification and automated recalibration pipeline. RGB-D images from the camera are projected into a 3D point cloud, and the robot’s 3D mesh is rendered in the same space based on the hand-eye calibration, utilizing their spatial overlap for validation. Upon detecting misalignment, our pipeline runs FoundationPose
[
55
]
on the point cloud to estimate the robot hand’s 6D pose based on its URDF. Through FK, the calibration matrix is back-calculated for each frame. Averaging these matrices across frames and removing outliers robustly recovers the calibration parameters, preventing data degradation.
Below, we present the prompt template designed for Qwen3-VL-Flash
[
3
]
to generate multi-granularity language annotations for teleoperated demonstrations.
Robot Data Language Labeling Prompt
Although pre-training on egocentric human videos equips the model with rich dexterous manipulation priors, direct physical deployment is prevented by the embodiment gap across visual appearance, dynamics, and kinematics. Furthermore, because these pre-training labels are generated by automated depth and hand pose estimation models, their accuracy is constrained by current model capabilities, potentially introducing systematic biases into the pre-trained policy. Therefore, collecting real-robot teleoperation data aims to correct and ground these priors onto the target embodiment with high sample efficiency. Additionally, due to kinematic constraints on robotic degrees of freedom, the robot cannot perfectly replicate all fine-grained human hand movements. To achieve steerable manipulation, we must collect a highly diverse range of free-form tasks to maximize the coverage of basic manipulation primitives, guiding the model to efficiently transfer cross-domain knowledge and fostering robust compositional generalization and multi-task instruction following.
Based on these considerations, within the kinematic limits of RealMan’s degrees of freedom, we design
193
193
semantically distinct dexterous manipulation tasks. For each task, approximately
300
300
randomized, diverse demonstration trajectories, totaling around
1
1
hour, are collected under cluttered scenarios, constructing a high-quality real-robot dataset of
187
187
hours and
55
​
K
55\text{K}
trajectories.
These
193
193
tasks are classified into two major categories:
•
Common Tasks
(
56
56
tasks): Everyday manipulations that are readily achievable within the current robot hardware configuration and sensor limits, yielding high teleoperation success rates.
•
Long-Tail Tasks
(
137
137
tasks): Infrequent and physically challenging manipulations, such as contact-sensitive operations lacking tactile feedback, with lower collection success rates, primarily designed to ensure comprehensive semantic coverage of the dexterous manipulation space.
Furthermore, based on motion characteristics and physical interactions, all tasks are categorized into seven classes:
•
PnP-Easy
: Single-step tabletop pick-and-place where objects are easily graspable and the placement space is open.
•
PnP-Medium
: Non-planar or 3D spatial pick-and-place that requires precise operations related to containers, demanding higher control accuracy and spatial perception, such as “
put tennis ball into ball holder
”.
•
PnP-Hard
: Multi-step or high-precision pick-and-place sequences, such as “
stack paper cups
”.
•
Non-prehensile
: Actions not involving traditional finger grasping, including pushing, pulling, and pressing.
•
Reorient
: Operations involving rotation and reorientation, such as “
pour water
” or “
flip paper cups
”.
•
Bimanual
: Bimanual tasks requiring high synchronization and spatial coordination between arms and hands, such as “
plug cable into charger
”.
•
Contact-rich
: Operations involving frequent and complex physical contact interactions, requiring physical understanding, such as “
wipe whiteboard
”.
Throughout data collection, strict requirements are enforced on the randomness, diversity, and quality of each trajectory. Specifically, the tabletop features cluttered, unstructured scenes rather than pre-arranged, simplified environments, avoiding task identification from visual inputs alone. This design forces the model to deeply align language instructions with physical actions, encouraging it to learn general task semantics and execution goals rather than memorizing demonstration trajectories. Furthermore, given the highly randomized object configurations, operators are instructed to perform teleoperation in a natural, human-like manner. Consequently, different demonstrations of the same task exhibit substantial variations in execution trajectories, thereby covering a broader distribution of the mapping from human priors to the robot’s action space.
As illustrated in
Figure
8
, we analyze the
187
187
-hour real-robot dataset across several key dimensions, including the verb and noun word clouds with their top 30 frequency distributions, the task duration breakdown across seven manipulation categories, and the detailed duration statistics for the
56
56
common and
137
137
long-tail tasks. These statistics reveal a diverse vocabulary of actions and manipulated objects, alongside a balanced distribution across both manipulation categories and task durations. Encompassing rich semantic concepts and manipulation primitives, this dataset effectively grounds pre-trained human manipulation priors onto the RealMan embodiment. Additionally,
Figure
9
showcases the dual-view image sequences and language annotations of a teleoperation trajectory, while
Figure
10
visualizes sample tasks across each category.
(a)
Word cloud of nouns in annotation
(b)
Top 30 noun frequencies
(c)
Word cloud of verbs in annotation
(d)
Top 30 verb frequencies
(e)
Task duration by seven categories
(f)
Common task duration statistics
(g)
Long-tail task duration statistics
Figure 8:
Teleoperation dataset statistics. (1)
Figure
8
-
Figure
8
: Word clouds and top-30 frequency distributions of nouns and verbs in language annotations. (2)
Figure
8
: Task duration breakdown by seven manipulation categories (PnP-Easy/Medium/Hard, Non-prehensile, Reorient, Bimanual, Contact-rich). (3)
Figure
8
and
Figure
8
: Detailed duration statistics for common tasks (56 tasks) and long-tail tasks (137 tasks).
Figure 9:
Dual-view image sequence and three-level language annotations of a representative trajectory
. The frames illustrate a cluttered, randomized setup where the operator executes the task in a natural, human-like manner. The hierarchical language annotations describe the manipulation process from coarse to fine, assisting the model in aligning free-form instructions with physical actions.
Figure 10:
Representative task examples across seven manipulation categories
. These diverse tasks are executed in cluttered, randomized scenarios using natural, human-like manipulations, covering a wide range of action semantics and manipulation primitives to facilitate the efficient grounding of pre-trained egocentric human priors onto the physical robot.
Dataset
Samples
Captioning
Visual Question
Answering
Multiple-Choice
Question
Pointing
Bounding Box
Affordance
Trajectory
Spatial
Planning
FineVision
[
56
]
3.5
​
M
3.5\text{M}
✓
✓
✓
✓
RefSpatial
[
68
]
2.5
​
M
2.5\text{M}
✓
✓
✓
✓
RoboInter-VQA
[
27
]
1.6
​
M
1.6\text{M}
✓
✓
✓
✓
✓
✓
✓
RoboPoint
[
61
]
1.3
​
M
1.3\text{M}
✓
✓
✓
✓
RoboAfford
[
50
]
765
​
K
765\text{K}
✓
✓
✓
✓
Robo2VLM
[
10
]
678
​
K
678\text{K}
✓
✓
ShareRobot
[
23
]
13
​
K
13\text{K}
✓
✓
✓
✓
✓
Table 3:
VLM datasets used for EgoSteer co-training
. This table details the sample sizes and the multi-modal and embodied knowledge domains covered by each dataset. This co-training mixture integrates general vision-language knowledge with interaction understanding and spatial-geometric priors, preserving the model’s inherent world knowledge while facilitating its ability to follow free-form instructions to generalize across diverse manipulation tasks.
Figure 11:
Statistical distribution of samples across VLM co-training datasets.
This section first introduces the VLM datasets utilized for co-training with VLA data,
i.e.
egocentric human videos and real-robot data, to preserve EgoSteer’s vision-language knowledge and ensure generalization (
Section
C.1
). We then present implementation details of EgoSteer (
Section
C.2
).
To preserve general vision-language reasoning capabilities while simultaneously cultivating robust robotic task comprehension, a
10.4
​
M
10.4\text{M}
-sample VLM co-training mixture is curated across seven datasets, ranging from open-world perception to embodied interaction grounding, to co-train EgoSteer. This mixture comprises four categories of data:
•
General VLM Pre-Training
: Incorporates FineVision
[
56
]
to prevent the catastrophic forgetting of open-world semantic concepts and preserve general visual-language reasoning.
•
Spatial Grounding
: Utilizes RefSpatial
[
68
]
and RoboPoint
[
61
]
for multi-step spatial referring, 2D visual grounding, and precise coordinate localization.
•
Embodied QA
: Integrates RoboInter-VQA
[
27
]
, Robo2VLM
[
10
]
, and ShareRobot
[
23
]
to support embodied question answering and temporal reasoning. This preserves the model’s capabilities in high-level task planning and causal scene understanding.
•
Affordance Perception
: Adopts RoboAfford
[
50
]
for fine-grained manipulation affordance prediction and spatial interaction grounding.
The sample distribution of these datasets is illustrated in
Figure
11
, with their covered multimodal, embodied knowledge domains detailed in
Table
3
. We standardize these datasets to comply with the conversational input format of Qwen3-VL. Specifically, we normalize 2D bounding box and point coordinates to Qwen3-VL’s native
[
0
,
1000
]
[0,1000]
scale and adopt its standard representation formats. To ensure training stability, we only utilize single-image samples and exclude those with context lengths exceeding our maximum budget. By co-training on this VLM mixture, EgoSteer preserves open-world vision-language understanding and reasoning while enhancing its comprehension of robotics-specific tasks, thereby assisting the policy in following open-ended instructions and generalizing to novel manipulation scenarios.
EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.
The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.
The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.
The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.
Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.
To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.
Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.
This section provides supplementary details for the experimental evaluations presented in
Section
6
, detailing the model training hyperparameters and per-task success rates. In particular,
Table
4
lists the hyperparameter configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
Table
5
presents the per-task success rates on the four evaluation tasks in
Section
6.2
to demonstrate the performance gains from DAgger refinement. For the scaling analysis in
Section
6.3
,
Table
6
lists the training hyperparameters of EgoSteer-3K/6K/9.6K pre-trained models, their respective post-training runs, and the baseline trained from scratch, with their corresponding task-specific success rates reported in
Table
7
. Additionally,
Table
8
compares the detailed success rates of our method against the Being-H0.5 and
π
0.5
\pi_{0.5}
baselines across ten tasks.
Table
10
specifies the training configurations of the ablation experiment from
Section
6.4
, including the EgoSteer-1K model and its three ablated variants,
No WM-objective
,
No training-RTC
, and
Noisy data
. Their respective performance comparison is provided in
Table
9
. Finally,
Table
11
specifies the hyperparameters for the few-shot fine-tuning of our pre-trained EgoSteer-9.6K on the two challenging, long-horizon tasks evaluated in
Section
6.5
, namely
Box-Folding
and
Cake-Unboxing
.
Hyperparameter
Pre-Training
Post-Training
Camera setup
Head
Head & Chest
Resolution
384
×
384
384\times 384
640
×
480
640\times 480
GPUs
128 A800
96 A800
Gradient accumulation
2
1
Global batch size
4608
384
Training steps
175K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Freeze-VLM steps
5000
0
Warmup steps
2000
2000
Training time
164 h
29 h
Table 4:
Training configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
During pre-training, the VLM backbone is frozen for the first 5,000 steps, during which AE and WM are warmed up for 2,000 steps; once it is unfrozen, VLM is warmed up for 2,000 steps.
Task
EgoSteer-DG
EgoSteer-FT
Stack tableware
80.0%
50.0%
Close laptop
70.0%
10.0%
Place phone on stand
50.0%
0.0%
Flip cup
50.0%
30.0%
Average
62.5%
22.5%
Table 5:
Success rates across four highly dexterous and failure-prone manipulation tasks, comparing EgoSteer-DG against EgoSteer-FT in
Section
6.2
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Camera setup
Head
Head
Head
Head & Chest
Head & Chest
Head & Chest
Head & Chest
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
GPUs
64 A800
64 A800
128 A800
64 A800
32 A800
64 A800
64 A800
Gradient accumulation
2
4
2
2
4
2
2
Global batch size
2304
4608
4608
512
512
512
512
Training steps
100K
100K
160K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 6:
Training configurations for the pre-training scaling study in
Section
6.3
.
Pre-Training Data (Hours)
Task
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Grasp object
80%
80%
70%
100%
Hand over object
70%
80%
80%
100%
Place items into container
40%
80%
90%
100%
Point at object
20%
40%
60%
70%
Place toy chick into slot
30%
0%
20%
40%
Pull out tissue
60%
20%
50%
80%
Push ball into box
0%
0%
10%
20%
Put garbage into trash bin
0%
60%
30%
30%
Stack paper cups
0%
20%
10%
40%
Stack tableware
0%
20%
10%
20%
Average
30%
40%
43%
60%
Table 7:
Per-task success rates for the pre-training scaling study in
Section
6.3
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Task
Ours
Being-H0.5
π
0.5
\pi_{0.5}
Grasp object
100%
80%
80%
Hand over object
100%
60%
20%
Place items into container
100%
50%
0%
Pour items out of box
50%
30%
0%
Place bread on tray
70%
80%
40%
Pull out tissue
80%
10%
30%
Place item at specific orientation
30%
30%
30%
Attach eraser to whiteboard
90%
50%
20%
Put garbage into trash bin
30%
0%
0%
Put tennis ball into bucket
90%
0%
0%
Average
74%
39%
22%
Table 8:
Per-task comparison between EgoSteer-9.6K and two VLA baselines, Being-H0.5 and
π
0.5
\pi_{0.5}
, in
Section
6.3
. All methods are post-trained on our real-robot dataset and evaluated on the same 10 tasks. Each task uses 10 randomized trials. Bold values highlight the best performance.
Task
Ours
No WM-objective
No training-RTC
Noisy data
Grasp object
60%
40%
30%
60%
Hand over object
40%
20%
30%
40%
Place items into container
50%
30%
50%
70%
Pour items out of box
60%
50%
10%
0%
Place bread on tray
70%
10%
60%
50%
Pull out tissue
20%
30%
60%
10%
Place item at specific orientation
20%
10%
10%
20%
Attach eraser to whiteboard
50%
80%
50%
40%
Put garbage into trash bin
30%
0%
0%
0%
Put tennis ball into bucket
40%
40%
90%
40%
Average
44%
31%
39%
33%
Table 9:
Per-task success rates for the ablation study in
Section
6.4
. EgoSteer-1K is compared with ablated variants. Each task uses 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
Ours
No WM-objective
No training-RTC
Noisy data
Ours
No WM-objective
No training-RTC
Noisy data
Camera setup
Head
Head
Head
Head
Head
Head
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
Global batch size
1152
1152
1152
1152
1152
1152
1152
1152
Training steps
30K
80K
30K
20K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 10:
Training hyperparameters for the ablation study in
Section
6.4
. All variants are pre-trained on 1K hours of egocentric data and post-trained on the same real-robot dataset, without DAgger refinement. Pre-training steps are selected at the lowest evaluation
L
1
L_{1}
loss for each pre-training run.
Hyperparameter
Box-Folding
Cake-Unboxing
Pre-training checkpoint
EgoSteer-9.6K at 155K steps
Camera setup
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
8 A800
8 A800
Gradient accumulation
1
1
Global batch size
144
144
Fine-tuning steps
44K
12K
Learning rate (VLM / AE / WM)
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Demonstrations
120
229
Table 11:
Few-shot fine-tuning hyperparameters of EgoSteer-9.6K for the two long-horizon dexterous tasks in
Section
6.5
.

#### A.1.4
Post-Filtering Criteria

In this stage, we perform quality control on the reconstructed outputs to filter out reconstruction anomalies and problematic segments, ensuring overall data quality. This evaluation is conducted from coarse to fine across three granularities: entire episodes, chunk windows, and adjacent frames.
Episode-level checks assess the overall camera motion. We compute the statistics of camera extrinsics, both translation and rotation, for each episode and compare them against the distribution of other episodes within the same dataset, discarding those that deviate significantly. Because reasonable camera motion magnitudes vary across datasets due to different devices, scenes, and manipulation styles, we employ a dataset-specific IQR criterion rather than a universal threshold. Specifically, an episode is classified as an outlier if its statistics fall outside the range
[
Q
1
−
2.5
​
IQR
,
Q
3
+
2.5
​
IQR
]
[Q_{1}-2.5\mathrm{IQR},Q_{3}+2.5\mathrm{IQR}]
. This step filters out camera tracking drift, as well as segments dominated by walking or looking around instead of manipulation.
Chunk-level checks evaluate whether hands fall within physically reasonable spatial boundaries in a standardized egocentric coordinate frame. Directly comparing absolute hand coordinates is problematic, since they are coupled with camera and body movements. Specifically, within a sliding window spanning approximately the past 5 seconds and the future 30 frames, we transform all hand states and actions into the current camera frame. Under this canonical system, wrist positions are defined relative to the camera, and finger joints relative to the wrist. Within this coordinate system, we evaluate both distributional outliers and absolute physical limits. First, outliers in wrist translation/rotation and finger positions are identified using the same IQR criterion based on the respective dataset’s distribution. Second, we enforce a universal physical ceiling of 1.5 meters on each coordinate axis for the hands, as a human hand cannot physically reach further than this distance from the head. If any sliding chunk window within an episode violates either the IQR outlier threshold or the 1.5-meter physical limit, the entire episode is discarded.
Frame-level checks identify sudden jumps between adjacent frames. We compute the frame-to-frame changes in camera translation and rotation, wrist translation and rotation, and finger translation. Unlike the previous levels, we do not rely on dataset-specific distributions here, as the physical speed of human hands and heads has a universal limit. Any movement exceeding this limit is attributed to reconstruction artifacts rather than valid motion. We therefore apply fixed physical thresholds: camera translation
≤
0.20
​
m/frame
\leq 0.20\,\text{m/frame}
, wrist and finger translation
≤
0.30
​
m/frame
\leq 0.30\,\text{m/frame}
, camera rotation
≤
28
∘
/
frame
\leq 28^{\circ}/\text{frame}
, and wrist rotation
≤
41
∘
/
frame
\leq 41^{\circ}/\text{frame}
. An episode is discarded if any of its frames violate these thresholds.
Collectively, these three levels of checks filter out problematic segments caused by head tracking drift, inaccurate motion reconstruction, and motion discontinuities.
The final pre-training corpus is constructed by utilizing EgoSmith to process 12 raw egocentric datasets, ultimately yielding a standardized,
9.6
​
K
9.6\text{K}
-hour dataset. This curated corpus comprises world-space bimanual states and actions, frame-wise camera intrinsics and extrinsics, metric scene depth, and coarse-to-fine language annotations. In this section, we detail the scale, source composition, and semantic diversity of this dataset. The primary objective is not merely scaling up dataset volume; instead, the core advantage lies in data quality. Every sample is richly annotated and filtered through a rigorous quality-control pipeline, guaranteeing high-quality, modality-aligned manipulation knowledge to assist the model in learning steerable dexterous manipulation. The dataset further offers broad coverage of manipulation tasks, objects, and multi-granularity language descriptions.
Scale and Source Composition.
Figure
7
presents the duration contribution and proportion of each source dataset, while also indicating whether each annotation is natively provided or reconstructed by our pipeline. The source datasets are highly complementary, covering a wide range of devices, scenes, and manipulation styles. Although a small number of large-scale collections dominate the total duration, the breadth of sources contributes substantial diversity.
Task and Semantic Diversity.
To characterize the manipulation span of our dataset, we extract (verb, object) tuples from the L1 verb–object annotations and analyze the distributions of action verbs and manipulated objects. The dataset covers 8969 distinct object nouns and 623 action verbs (
Figures
7
and
7
).
Figure
7
illustrates the top 50 most frequent “verb + object” atomic tasks. Common tasks concentrate on fundamental manipulation skills, aligning with the natural distribution of daily hand–object interactions. At the same time, the distribution exhibits a prominent long tail: a large number of low-frequency yet semantically diverse tasks and objects provides broad coverage, preventing the dataset from being biased toward a few dominant actions. This combination of common fundamental skills, long-tail task diversity, and multi-granularity language annotations provides downstream models with abundant training samples, broad task coverage, and rich dexterous manipulation knowledge.
Dataset
Hours
Percentage (%)
Episodes
Hand
Depth
Camera
Language
Egocentric-100K
[
1
]
8
,
049
8,049
83.8
83.8
1
,
795
,
731
1{,}795{,}731
✓
✓
✓
✓
EgoVerse
[
43
]
690
690
7.2
7.2
35
,
175
35{,}175
✓
✓
EgoDex
[
19
]
370
370
3.9
3.9
147
,
588
147{,}588
✓
✓
Egocentric-10K
[
1
]
288
288
3.0
3.0
194
,
915
194{,}915
✓
✓
✓
✓
Ego4D
[
16
]
138
138
1.4
1.4
74
,
505
74{,}505
✓
✓
✓
✓
Epic-Kitchens
[
12
]
49
49
0.5
0.5
26
,
454
26{,}454
✓
✓
✓
✓
HoloAssist
[
54
]
11.5
11.5
0.1
0.1
11
,
426
11{,}426
✓
✓
HOT3D
[
4
]
4.5
4.5
0.05
0.05
1
,
105
1{,}105
✓
TACO
[
34
]
3.0
3.0
0.03
0.03
1
,
558
1{,}558
✓
OakInk-v2
[
63
]
1.7
1.7
0.02
0.02
891
891
✓
H2O
[
26
]
1.0
1.0
0.01
0.01
935
935
FPHA
[
15
]
0.5
0.5
0.01
0.01
578
578
✓
✓
✓
Total
𝟗
,
𝟔𝟎𝟔
\mathbf{9,606}
100
100
𝟐
,
𝟐𝟗𝟎
,
𝟖𝟔𝟏
\mathbf{2{,}290{,}861}
(a)
Dataset composition and annotation sources
(b)
Word cloud of manipulated objects
(c)
Word cloud of action verbs
(d)
Top-
50
50
“verb + object” atomic tasks
Figure 7:
Curated dataset statistics.
(a) Source composition of the
12
12
egocentric datasets, detailing duration, percentage, episode count, and per-annotation origin. Checkmarks denote annotations generated by our pipeline, while blank entries indicate natively provided annotations. For FPHA, the checkmark under
Hand
specifically denotes the additionally annotated second hand. (b)–(d) Task and semantic diversity: word clouds of the manipulated objects and action verbs, and the most frequent verb–object atomic tasks, exhibiting long-tailed coverage over diverse manipulation skills.
This section details the implementation of the Robot Stack (
Section
B.1
), alongside the collection protocols and statistics of the
187
187
-hour real-robot dataset (
Section
B.2
).
The two physical embodiments utilized in our work are illustrated in
Figure
3
. The primary platform, referred to as the RealMan embodiment, consists of two 7-DoF RealMan RM75-6F robotic arms and two 6-DoF Ruiyan RY-H2 dexterous hands. It is equipped with one head-mounted and one chest-mounted Intel RealSense D455 camera, providing dual egocentric viewpoints to ensure a comprehensive visual field. The AgiBot G1 embodiment is adapted from an AgiBot G1 humanoid robot by replacing its default end-effectors with two 6-DoF Ruiyan RY-H2 hands. It features one head-mounted Intel RealSense D455 camera and two wrist cameras, the latter of which are unused. During experiments, the robot’s neck, waist, and mobile base are kept fixed. Both platforms conduct tabletop manipulation on an operational table. The RealMan platform serves as the primary embodiment, on which the entire
187
187
-hour dataset was collected. All experiments are conducted on this setup, with the sole exception of the few-shot adaptation experiment performed on the AgiBot G1.
For both platforms, the human wrist tracker, arm kinematic solvers, and joint control ROS 2 nodes operate at
100
100
Hz, while the glove, hand solvers, and control ROS 2 nodes run at
80
80
Hz. This high-frequency loop ensures near-zero latency, enabling intuitive bimanual teleoperation and the collection of fine-grained manipulation demonstrations. The cameras capture frames at
30
30
Hz. Data is recorded at these native frequencies and subsequently resampled to
30
30
Hz for training.
Robot data collection aims to ground the human manipulation priors learned from large-scale pre-training onto the target physical embodiment. To preserve and transfer this pre-trained knowledge, minimizing the domain gap between robot and human data is essential. Because the robotic palm is slightly longer than a human palm, we translate the robot’s wrist coordinate frame axially forward. This alignment ensures that the scale from the wrist to the fingertips matches human anatomical proportions. When converting raw robot data into training data, this coordinate transformation is applied alongside hand-eye calibration to map hand actions into the camera coordinate frame. Correspondingly, during real-time policy inference, the control stack executes the inverse transformation and hand-eye mapping to project predicted actions back to the robot’s physical coordinate space.
In practice, hand-eye calibration is prone to drift due to mechanical maintenance, wear, or collisions, with such discrepancies sometimes only identified
post
-acquisition. To ensure data quality, we introduce an offline verification and automated recalibration pipeline. RGB-D images from the camera are projected into a 3D point cloud, and the robot’s 3D mesh is rendered in the same space based on the hand-eye calibration, utilizing their spatial overlap for validation. Upon detecting misalignment, our pipeline runs FoundationPose
[
55
]
on the point cloud to estimate the robot hand’s 6D pose based on its URDF. Through FK, the calibration matrix is back-calculated for each frame. Averaging these matrices across frames and removing outliers robustly recovers the calibration parameters, preventing data degradation.
Below, we present the prompt template designed for Qwen3-VL-Flash
[
3
]
to generate multi-granularity language annotations for teleoperated demonstrations.
Robot Data Language Labeling Prompt
Although pre-training on egocentric human videos equips the model with rich dexterous manipulation priors, direct physical deployment is prevented by the embodiment gap across visual appearance, dynamics, and kinematics. Furthermore, because these pre-training labels are generated by automated depth and hand pose estimation models, their accuracy is constrained by current model capabilities, potentially introducing systematic biases into the pre-trained policy. Therefore, collecting real-robot teleoperation data aims to correct and ground these priors onto the target embodiment with high sample efficiency. Additionally, due to kinematic constraints on robotic degrees of freedom, the robot cannot perfectly replicate all fine-grained human hand movements. To achieve steerable manipulation, we must collect a highly diverse range of free-form tasks to maximize the coverage of basic manipulation primitives, guiding the model to efficiently transfer cross-domain knowledge and fostering robust compositional generalization and multi-task instruction following.
Based on these considerations, within the kinematic limits of RealMan’s degrees of freedom, we design
193
193
semantically distinct dexterous manipulation tasks. For each task, approximately
300
300
randomized, diverse demonstration trajectories, totaling around
1
1
hour, are collected under cluttered scenarios, constructing a high-quality real-robot dataset of
187
187
hours and
55
​
K
55\text{K}
trajectories.
These
193
193
tasks are classified into two major categories:
•
Common Tasks
(
56
56
tasks): Everyday manipulations that are readily achievable within the current robot hardware configuration and sensor limits, yielding high teleoperation success rates.
•
Long-Tail Tasks
(
137
137
tasks): Infrequent and physically challenging manipulations, such as contact-sensitive operations lacking tactile feedback, with lower collection success rates, primarily designed to ensure comprehensive semantic coverage of the dexterous manipulation space.
Furthermore, based on motion characteristics and physical interactions, all tasks are categorized into seven classes:
•
PnP-Easy
: Single-step tabletop pick-and-place where objects are easily graspable and the placement space is open.
•
PnP-Medium
: Non-planar or 3D spatial pick-and-place that requires precise operations related to containers, demanding higher control accuracy and spatial perception, such as “
put tennis ball into ball holder
”.
•
PnP-Hard
: Multi-step or high-precision pick-and-place sequences, such as “
stack paper cups
”.
•
Non-prehensile
: Actions not involving traditional finger grasping, including pushing, pulling, and pressing.
•
Reorient
: Operations involving rotation and reorientation, such as “
pour water
” or “
flip paper cups
”.
•
Bimanual
: Bimanual tasks requiring high synchronization and spatial coordination between arms and hands, such as “
plug cable into charger
”.
•
Contact-rich
: Operations involving frequent and complex physical contact interactions, requiring physical understanding, such as “
wipe whiteboard
”.
Throughout data collection, strict requirements are enforced on the randomness, diversity, and quality of each trajectory. Specifically, the tabletop features cluttered, unstructured scenes rather than pre-arranged, simplified environments, avoiding task identification from visual inputs alone. This design forces the model to deeply align language instructions with physical actions, encouraging it to learn general task semantics and execution goals rather than memorizing demonstration trajectories. Furthermore, given the highly randomized object configurations, operators are instructed to perform teleoperation in a natural, human-like manner. Consequently, different demonstrations of the same task exhibit substantial variations in execution trajectories, thereby covering a broader distribution of the mapping from human priors to the robot’s action space.
As illustrated in
Figure
8
, we analyze the
187
187
-hour real-robot dataset across several key dimensions, including the verb and noun word clouds with their top 30 frequency distributions, the task duration breakdown across seven manipulation categories, and the detailed duration statistics for the
56
56
common and
137
137
long-tail tasks. These statistics reveal a diverse vocabulary of actions and manipulated objects, alongside a balanced distribution across both manipulation categories and task durations. Encompassing rich semantic concepts and manipulation primitives, this dataset effectively grounds pre-trained human manipulation priors onto the RealMan embodiment. Additionally,
Figure
9
showcases the dual-view image sequences and language annotations of a teleoperation trajectory, while
Figure
10
visualizes sample tasks across each category.
(a)
Word cloud of nouns in annotation
(b)
Top 30 noun frequencies
(c)
Word cloud of verbs in annotation
(d)
Top 30 verb frequencies
(e)
Task duration by seven categories
(f)
Common task duration statistics
(g)
Long-tail task duration statistics
Figure 8:
Teleoperation dataset statistics. (1)
Figure
8
-
Figure
8
: Word clouds and top-30 frequency distributions of nouns and verbs in language annotations. (2)
Figure
8
: Task duration breakdown by seven manipulation categories (PnP-Easy/Medium/Hard, Non-prehensile, Reorient, Bimanual, Contact-rich). (3)
Figure
8
and
Figure
8
: Detailed duration statistics for common tasks (56 tasks) and long-tail tasks (137 tasks).
Figure 9:
Dual-view image sequence and three-level language annotations of a representative trajectory
. The frames illustrate a cluttered, randomized setup where the operator executes the task in a natural, human-like manner. The hierarchical language annotations describe the manipulation process from coarse to fine, assisting the model in aligning free-form instructions with physical actions.
Figure 10:
Representative task examples across seven manipulation categories
. These diverse tasks are executed in cluttered, randomized scenarios using natural, human-like manipulations, covering a wide range of action semantics and manipulation primitives to facilitate the efficient grounding of pre-trained egocentric human priors onto the physical robot.
Dataset
Samples
Captioning
Visual Question
Answering
Multiple-Choice
Question
Pointing
Bounding Box
Affordance
Trajectory
Spatial
Planning
FineVision
[
56
]
3.5
​
M
3.5\text{M}
✓
✓
✓
✓
RefSpatial
[
68
]
2.5
​
M
2.5\text{M}
✓
✓
✓
✓
RoboInter-VQA
[
27
]
1.6
​
M
1.6\text{M}
✓
✓
✓
✓
✓
✓
✓
RoboPoint
[
61
]
1.3
​
M
1.3\text{M}
✓
✓
✓
✓
RoboAfford
[
50
]
765
​
K
765\text{K}
✓
✓
✓
✓
Robo2VLM
[
10
]
678
​
K
678\text{K}
✓
✓
ShareRobot
[
23
]
13
​
K
13\text{K}
✓
✓
✓
✓
✓
Table 3:
VLM datasets used for EgoSteer co-training
. This table details the sample sizes and the multi-modal and embodied knowledge domains covered by each dataset. This co-training mixture integrates general vision-language knowledge with interaction understanding and spatial-geometric priors, preserving the model’s inherent world knowledge while facilitating its ability to follow free-form instructions to generalize across diverse manipulation tasks.
Figure 11:
Statistical distribution of samples across VLM co-training datasets.
This section first introduces the VLM datasets utilized for co-training with VLA data,
i.e.
egocentric human videos and real-robot data, to preserve EgoSteer’s vision-language knowledge and ensure generalization (
Section
C.1
). We then present implementation details of EgoSteer (
Section
C.2
).
To preserve general vision-language reasoning capabilities while simultaneously cultivating robust robotic task comprehension, a
10.4
​
M
10.4\text{M}
-sample VLM co-training mixture is curated across seven datasets, ranging from open-world perception to embodied interaction grounding, to co-train EgoSteer. This mixture comprises four categories of data:
•
General VLM Pre-Training
: Incorporates FineVision
[
56
]
to prevent the catastrophic forgetting of open-world semantic concepts and preserve general visual-language reasoning.
•
Spatial Grounding
: Utilizes RefSpatial
[
68
]
and RoboPoint
[
61
]
for multi-step spatial referring, 2D visual grounding, and precise coordinate localization.
•
Embodied QA
: Integrates RoboInter-VQA
[
27
]
, Robo2VLM
[
10
]
, and ShareRobot
[
23
]
to support embodied question answering and temporal reasoning. This preserves the model’s capabilities in high-level task planning and causal scene understanding.
•
Affordance Perception
: Adopts RoboAfford
[
50
]
for fine-grained manipulation affordance prediction and spatial interaction grounding.
The sample distribution of these datasets is illustrated in
Figure
11
, with their covered multimodal, embodied knowledge domains detailed in
Table
3
. We standardize these datasets to comply with the conversational input format of Qwen3-VL. Specifically, we normalize 2D bounding box and point coordinates to Qwen3-VL’s native
[
0
,
1000
]
[0,1000]
scale and adopt its standard representation formats. To ensure training stability, we only utilize single-image samples and exclude those with context lengths exceeding our maximum budget. By co-training on this VLM mixture, EgoSteer preserves open-world vision-language understanding and reasoning while enhancing its comprehension of robotics-specific tasks, thereby assisting the policy in following open-ended instructions and generalizing to novel manipulation scenarios.
EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.
The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.
The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.
The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.
Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.
To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.
Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.
This section provides supplementary details for the experimental evaluations presented in
Section
6
, detailing the model training hyperparameters and per-task success rates. In particular,
Table
4
lists the hyperparameter configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
Table
5
presents the per-task success rates on the four evaluation tasks in
Section
6.2
to demonstrate the performance gains from DAgger refinement. For the scaling analysis in
Section
6.3
,
Table
6
lists the training hyperparameters of EgoSteer-3K/6K/9.6K pre-trained models, their respective post-training runs, and the baseline trained from scratch, with their corresponding task-specific success rates reported in
Table
7
. Additionally,
Table
8
compares the detailed success rates of our method against the Being-H0.5 and
π
0.5
\pi_{0.5}
baselines across ten tasks.
Table
10
specifies the training configurations of the ablation experiment from
Section
6.4
, including the EgoSteer-1K model and its three ablated variants,
No WM-objective
,
No training-RTC
, and
Noisy data
. Their respective performance comparison is provided in
Table
9
. Finally,
Table
11
specifies the hyperparameters for the few-shot fine-tuning of our pre-trained EgoSteer-9.6K on the two challenging, long-horizon tasks evaluated in
Section
6.5
, namely
Box-Folding
and
Cake-Unboxing
.
Hyperparameter
Pre-Training
Post-Training
Camera setup
Head
Head & Chest
Resolution
384
×
384
384\times 384
640
×
480
640\times 480
GPUs
128 A800
96 A800
Gradient accumulation
2
1
Global batch size
4608
384
Training steps
175K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Freeze-VLM steps
5000
0
Warmup steps
2000
2000
Training time
164 h
29 h
Table 4:
Training configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
During pre-training, the VLM backbone is frozen for the first 5,000 steps, during which AE and WM are warmed up for 2,000 steps; once it is unfrozen, VLM is warmed up for 2,000 steps.
Task
EgoSteer-DG
EgoSteer-FT
Stack tableware
80.0%
50.0%
Close laptop
70.0%
10.0%
Place phone on stand
50.0%
0.0%
Flip cup
50.0%
30.0%
Average
62.5%
22.5%
Table 5:
Success rates across four highly dexterous and failure-prone manipulation tasks, comparing EgoSteer-DG against EgoSteer-FT in
Section
6.2
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Camera setup
Head
Head
Head
Head & Chest
Head & Chest
Head & Chest
Head & Chest
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
GPUs
64 A800
64 A800
128 A800
64 A800
32 A800
64 A800
64 A800
Gradient accumulation
2
4
2
2
4
2
2
Global batch size
2304
4608
4608
512
512
512
512
Training steps
100K
100K
160K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 6:
Training configurations for the pre-training scaling study in
Section
6.3
.
Pre-Training Data (Hours)
Task
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Grasp object
80%
80%
70%
100%
Hand over object
70%
80%
80%
100%
Place items into container
40%
80%
90%
100%
Point at object
20%
40%
60%
70%
Place toy chick into slot
30%
0%
20%
40%
Pull out tissue
60%
20%
50%
80%
Push ball into box
0%
0%
10%
20%
Put garbage into trash bin
0%
60%
30%
30%
Stack paper cups
0%
20%
10%
40%
Stack tableware
0%
20%
10%
20%
Average
30%
40%
43%
60%
Table 7:
Per-task success rates for the pre-training scaling study in
Section
6.3
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Task
Ours
Being-H0.5
π
0.5
\pi_{0.5}
Grasp object
100%
80%
80%
Hand over object
100%
60%
20%
Place items into container
100%
50%
0%
Pour items out of box
50%
30%
0%
Place bread on tray
70%
80%
40%
Pull out tissue
80%
10%
30%
Place item at specific orientation
30%
30%
30%
Attach eraser to whiteboard
90%
50%
20%
Put garbage into trash bin
30%
0%
0%
Put tennis ball into bucket
90%
0%
0%
Average
74%
39%
22%
Table 8:
Per-task comparison between EgoSteer-9.6K and two VLA baselines, Being-H0.5 and
π
0.5
\pi_{0.5}
, in
Section
6.3
. All methods are post-trained on our real-robot dataset and evaluated on the same 10 tasks. Each task uses 10 randomized trials. Bold values highlight the best performance.
Task
Ours
No WM-objective
No training-RTC
Noisy data
Grasp object
60%
40%
30%
60%
Hand over object
40%
20%
30%
40%
Place items into container
50%
30%
50%
70%
Pour items out of box
60%
50%
10%
0%
Place bread on tray
70%
10%
60%
50%
Pull out tissue
20%
30%
60%
10%
Place item at specific orientation
20%
10%
10%
20%
Attach eraser to whiteboard
50%
80%
50%
40%
Put garbage into trash bin
30%
0%
0%
0%
Put tennis ball into bucket
40%
40%
90%
40%
Average
44%
31%
39%
33%
Table 9:
Per-task success rates for the ablation study in
Section
6.4
. EgoSteer-1K is compared with ablated variants. Each task uses 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
Ours
No WM-objective
No training-RTC
Noisy data
Ours
No WM-objective
No training-RTC
Noisy data
Camera setup
Head
Head
Head
Head
Head
Head
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
Global batch size
1152
1152
1152
1152
1152
1152
1152
1152
Training steps
30K
80K
30K
20K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 10:
Training hyperparameters for the ablation study in
Section
6.4
. All variants are pre-trained on 1K hours of egocentric data and post-trained on the same real-robot dataset, without DAgger refinement. Pre-training steps are selected at the lowest evaluation
L
1
L_{1}
loss for each pre-training run.
Hyperparameter
Box-Folding
Cake-Unboxing
Pre-training checkpoint
EgoSteer-9.6K at 155K steps
Camera setup
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
8 A800
8 A800
Gradient accumulation
1
1
Global batch size
144
144
Fine-tuning steps
44K
12K
Learning rate (VLM / AE / WM)
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Demonstrations
120
229
Table 11:
Few-shot fine-tuning hyperparameters of EgoSteer-9.6K for the two long-horizon dexterous tasks in
Section
6.5
.

### A.2
Curated Dataset Statistics

The final pre-training corpus is constructed by utilizing EgoSmith to process 12 raw egocentric datasets, ultimately yielding a standardized,
9.6
​
K
9.6\text{K}
-hour dataset. This curated corpus comprises world-space bimanual states and actions, frame-wise camera intrinsics and extrinsics, metric scene depth, and coarse-to-fine language annotations. In this section, we detail the scale, source composition, and semantic diversity of this dataset. The primary objective is not merely scaling up dataset volume; instead, the core advantage lies in data quality. Every sample is richly annotated and filtered through a rigorous quality-control pipeline, guaranteeing high-quality, modality-aligned manipulation knowledge to assist the model in learning steerable dexterous manipulation. The dataset further offers broad coverage of manipulation tasks, objects, and multi-granularity language descriptions.
Scale and Source Composition.
Figure
7
presents the duration contribution and proportion of each source dataset, while also indicating whether each annotation is natively provided or reconstructed by our pipeline. The source datasets are highly complementary, covering a wide range of devices, scenes, and manipulation styles. Although a small number of large-scale collections dominate the total duration, the breadth of sources contributes substantial diversity.
Task and Semantic Diversity.
To characterize the manipulation span of our dataset, we extract (verb, object) tuples from the L1 verb–object annotations and analyze the distributions of action verbs and manipulated objects. The dataset covers 8969 distinct object nouns and 623 action verbs (
Figures
7
and
7
).
Figure
7
illustrates the top 50 most frequent “verb + object” atomic tasks. Common tasks concentrate on fundamental manipulation skills, aligning with the natural distribution of daily hand–object interactions. At the same time, the distribution exhibits a prominent long tail: a large number of low-frequency yet semantically diverse tasks and objects provides broad coverage, preventing the dataset from being biased toward a few dominant actions. This combination of common fundamental skills, long-tail task diversity, and multi-granularity language annotations provides downstream models with abundant training samples, broad task coverage, and rich dexterous manipulation knowledge.
Dataset
Hours
Percentage (%)
Episodes
Hand
Depth
Camera
Language
Egocentric-100K
[
1
]
8
,
049
8,049
83.8
83.8
1
,
795
,
731
1{,}795{,}731
✓
✓
✓
✓
EgoVerse
[
43
]
690
690
7.2
7.2
35
,
175
35{,}175
✓
✓
EgoDex
[
19
]
370
370
3.9
3.9
147
,
588
147{,}588
✓
✓
Egocentric-10K
[
1
]
288
288
3.0
3.0
194
,
915
194{,}915
✓
✓
✓
✓
Ego4D
[
16
]
138
138
1.4
1.4
74
,
505
74{,}505
✓
✓
✓
✓
Epic-Kitchens
[
12
]
49
49
0.5
0.5
26
,
454
26{,}454
✓
✓
✓
✓
HoloAssist
[
54
]
11.5
11.5
0.1
0.1
11
,
426
11{,}426
✓
✓
HOT3D
[
4
]
4.5
4.5
0.05
0.05
1
,
105
1{,}105
✓
TACO
[
34
]
3.0
3.0
0.03
0.03
1
,
558
1{,}558
✓
OakInk-v2
[
63
]
1.7
1.7
0.02
0.02
891
891
✓
H2O
[
26
]
1.0
1.0
0.01
0.01
935
935
FPHA
[
15
]
0.5
0.5
0.01
0.01
578
578
✓
✓
✓
Total
𝟗
,
𝟔𝟎𝟔
\mathbf{9,606}
100
100
𝟐
,
𝟐𝟗𝟎
,
𝟖𝟔𝟏
\mathbf{2{,}290{,}861}
(a)
Dataset composition and annotation sources
(b)
Word cloud of manipulated objects
(c)
Word cloud of action verbs
(d)
Top-
50
50
“verb + object” atomic tasks
Figure 7:
Curated dataset statistics.
(a) Source composition of the
12
12
egocentric datasets, detailing duration, percentage, episode count, and per-annotation origin. Checkmarks denote annotations generated by our pipeline, while blank entries indicate natively provided annotations. For FPHA, the checkmark under
Hand
specifically denotes the additionally annotated second hand. (b)–(d) Task and semantic diversity: word clouds of the manipulated objects and action verbs, and the most frequent verb–object atomic tasks, exhibiting long-tailed coverage over diverse manipulation skills.
This section details the implementation of the Robot Stack (
Section
B.1
), alongside the collection protocols and statistics of the
187
187
-hour real-robot dataset (
Section
B.2
).
The two physical embodiments utilized in our work are illustrated in
Figure
3
. The primary platform, referred to as the RealMan embodiment, consists of two 7-DoF RealMan RM75-6F robotic arms and two 6-DoF Ruiyan RY-H2 dexterous hands. It is equipped with one head-mounted and one chest-mounted Intel RealSense D455 camera, providing dual egocentric viewpoints to ensure a comprehensive visual field. The AgiBot G1 embodiment is adapted from an AgiBot G1 humanoid robot by replacing its default end-effectors with two 6-DoF Ruiyan RY-H2 hands. It features one head-mounted Intel RealSense D455 camera and two wrist cameras, the latter of which are unused. During experiments, the robot’s neck, waist, and mobile base are kept fixed. Both platforms conduct tabletop manipulation on an operational table. The RealMan platform serves as the primary embodiment, on which the entire
187
187
-hour dataset was collected. All experiments are conducted on this setup, with the sole exception of the few-shot adaptation experiment performed on the AgiBot G1.
For both platforms, the human wrist tracker, arm kinematic solvers, and joint control ROS 2 nodes operate at
100
100
Hz, while the glove, hand solvers, and control ROS 2 nodes run at
80
80
Hz. This high-frequency loop ensures near-zero latency, enabling intuitive bimanual teleoperation and the collection of fine-grained manipulation demonstrations. The cameras capture frames at
30
30
Hz. Data is recorded at these native frequencies and subsequently resampled to
30
30
Hz for training.
Robot data collection aims to ground the human manipulation priors learned from large-scale pre-training onto the target physical embodiment. To preserve and transfer this pre-trained knowledge, minimizing the domain gap between robot and human data is essential. Because the robotic palm is slightly longer than a human palm, we translate the robot’s wrist coordinate frame axially forward. This alignment ensures that the scale from the wrist to the fingertips matches human anatomical proportions. When converting raw robot data into training data, this coordinate transformation is applied alongside hand-eye calibration to map hand actions into the camera coordinate frame. Correspondingly, during real-time policy inference, the control stack executes the inverse transformation and hand-eye mapping to project predicted actions back to the robot’s physical coordinate space.
In practice, hand-eye calibration is prone to drift due to mechanical maintenance, wear, or collisions, with such discrepancies sometimes only identified
post
-acquisition. To ensure data quality, we introduce an offline verification and automated recalibration pipeline. RGB-D images from the camera are projected into a 3D point cloud, and the robot’s 3D mesh is rendered in the same space based on the hand-eye calibration, utilizing their spatial overlap for validation. Upon detecting misalignment, our pipeline runs FoundationPose
[
55
]
on the point cloud to estimate the robot hand’s 6D pose based on its URDF. Through FK, the calibration matrix is back-calculated for each frame. Averaging these matrices across frames and removing outliers robustly recovers the calibration parameters, preventing data degradation.
Below, we present the prompt template designed for Qwen3-VL-Flash
[
3
]
to generate multi-granularity language annotations for teleoperated demonstrations.
Robot Data Language Labeling Prompt
Although pre-training on egocentric human videos equips the model with rich dexterous manipulation priors, direct physical deployment is prevented by the embodiment gap across visual appearance, dynamics, and kinematics. Furthermore, because these pre-training labels are generated by automated depth and hand pose estimation models, their accuracy is constrained by current model capabilities, potentially introducing systematic biases into the pre-trained policy. Therefore, collecting real-robot teleoperation data aims to correct and ground these priors onto the target embodiment with high sample efficiency. Additionally, due to kinematic constraints on robotic degrees of freedom, the robot cannot perfectly replicate all fine-grained human hand movements. To achieve steerable manipulation, we must collect a highly diverse range of free-form tasks to maximize the coverage of basic manipulation primitives, guiding the model to efficiently transfer cross-domain knowledge and fostering robust compositional generalization and multi-task instruction following.
Based on these considerations, within the kinematic limits of RealMan’s degrees of freedom, we design
193
193
semantically distinct dexterous manipulation tasks. For each task, approximately
300
300
randomized, diverse demonstration trajectories, totaling around
1
1
hour, are collected under cluttered scenarios, constructing a high-quality real-robot dataset of
187
187
hours and
55
​
K
55\text{K}
trajectories.
These
193
193
tasks are classified into two major categories:
•
Common Tasks
(
56
56
tasks): Everyday manipulations that are readily achievable within the current robot hardware configuration and sensor limits, yielding high teleoperation success rates.
•
Long-Tail Tasks
(
137
137
tasks): Infrequent and physically challenging manipulations, such as contact-sensitive operations lacking tactile feedback, with lower collection success rates, primarily designed to ensure comprehensive semantic coverage of the dexterous manipulation space.
Furthermore, based on motion characteristics and physical interactions, all tasks are categorized into seven classes:
•
PnP-Easy
: Single-step tabletop pick-and-place where objects are easily graspable and the placement space is open.
•
PnP-Medium
: Non-planar or 3D spatial pick-and-place that requires precise operations related to containers, demanding higher control accuracy and spatial perception, such as “
put tennis ball into ball holder
”.
•
PnP-Hard
: Multi-step or high-precision pick-and-place sequences, such as “
stack paper cups
”.
•
Non-prehensile
: Actions not involving traditional finger grasping, including pushing, pulling, and pressing.
•
Reorient
: Operations involving rotation and reorientation, such as “
pour water
” or “
flip paper cups
”.
•
Bimanual
: Bimanual tasks requiring high synchronization and spatial coordination between arms and hands, such as “
plug cable into charger
”.
•
Contact-rich
: Operations involving frequent and complex physical contact interactions, requiring physical understanding, such as “
wipe whiteboard
”.
Throughout data collection, strict requirements are enforced on the randomness, diversity, and quality of each trajectory. Specifically, the tabletop features cluttered, unstructured scenes rather than pre-arranged, simplified environments, avoiding task identification from visual inputs alone. This design forces the model to deeply align language instructions with physical actions, encouraging it to learn general task semantics and execution goals rather than memorizing demonstration trajectories. Furthermore, given the highly randomized object configurations, operators are instructed to perform teleoperation in a natural, human-like manner. Consequently, different demonstrations of the same task exhibit substantial variations in execution trajectories, thereby covering a broader distribution of the mapping from human priors to the robot’s action space.
As illustrated in
Figure
8
, we analyze the
187
187
-hour real-robot dataset across several key dimensions, including the verb and noun word clouds with their top 30 frequency distributions, the task duration breakdown across seven manipulation categories, and the detailed duration statistics for the
56
56
common and
137
137
long-tail tasks. These statistics reveal a diverse vocabulary of actions and manipulated objects, alongside a balanced distribution across both manipulation categories and task durations. Encompassing rich semantic concepts and manipulation primitives, this dataset effectively grounds pre-trained human manipulation priors onto the RealMan embodiment. Additionally,
Figure
9
showcases the dual-view image sequences and language annotations of a teleoperation trajectory, while
Figure
10
visualizes sample tasks across each category.
(a)
Word cloud of nouns in annotation
(b)
Top 30 noun frequencies
(c)
Word cloud of verbs in annotation
(d)
Top 30 verb frequencies
(e)
Task duration by seven categories
(f)
Common task duration statistics
(g)
Long-tail task duration statistics
Figure 8:
Teleoperation dataset statistics. (1)
Figure
8
-
Figure
8
: Word clouds and top-30 frequency distributions of nouns and verbs in language annotations. (2)
Figure
8
: Task duration breakdown by seven manipulation categories (PnP-Easy/Medium/Hard, Non-prehensile, Reorient, Bimanual, Contact-rich). (3)
Figure
8
and
Figure
8
: Detailed duration statistics for common tasks (56 tasks) and long-tail tasks (137 tasks).
Figure 9:
Dual-view image sequence and three-level language annotations of a representative trajectory
. The frames illustrate a cluttered, randomized setup where the operator executes the task in a natural, human-like manner. The hierarchical language annotations describe the manipulation process from coarse to fine, assisting the model in aligning free-form instructions with physical actions.
Figure 10:
Representative task examples across seven manipulation categories
. These diverse tasks are executed in cluttered, randomized scenarios using natural, human-like manipulations, covering a wide range of action semantics and manipulation primitives to facilitate the efficient grounding of pre-trained egocentric human priors onto the physical robot.
Dataset
Samples
Captioning
Visual Question
Answering
Multiple-Choice
Question
Pointing
Bounding Box
Affordance
Trajectory
Spatial
Planning
FineVision
[
56
]
3.5
​
M
3.5\text{M}
✓
✓
✓
✓
RefSpatial
[
68
]
2.5
​
M
2.5\text{M}
✓
✓
✓
✓
RoboInter-VQA
[
27
]
1.6
​
M
1.6\text{M}
✓
✓
✓
✓
✓
✓
✓
RoboPoint
[
61
]
1.3
​
M
1.3\text{M}
✓
✓
✓
✓
RoboAfford
[
50
]
765
​
K
765\text{K}
✓
✓
✓
✓
Robo2VLM
[
10
]
678
​
K
678\text{K}
✓
✓
ShareRobot
[
23
]
13
​
K
13\text{K}
✓
✓
✓
✓
✓
Table 3:
VLM datasets used for EgoSteer co-training
. This table details the sample sizes and the multi-modal and embodied knowledge domains covered by each dataset. This co-training mixture integrates general vision-language knowledge with interaction understanding and spatial-geometric priors, preserving the model’s inherent world knowledge while facilitating its ability to follow free-form instructions to generalize across diverse manipulation tasks.
Figure 11:
Statistical distribution of samples across VLM co-training datasets.
This section first introduces the VLM datasets utilized for co-training with VLA data,
i.e.
egocentric human videos and real-robot data, to preserve EgoSteer’s vision-language knowledge and ensure generalization (
Section
C.1
). We then present implementation details of EgoSteer (
Section
C.2
).
To preserve general vision-language reasoning capabilities while simultaneously cultivating robust robotic task comprehension, a
10.4
​
M
10.4\text{M}
-sample VLM co-training mixture is curated across seven datasets, ranging from open-world perception to embodied interaction grounding, to co-train EgoSteer. This mixture comprises four categories of data:
•
General VLM Pre-Training
: Incorporates FineVision
[
56
]
to prevent the catastrophic forgetting of open-world semantic concepts and preserve general visual-language reasoning.
•
Spatial Grounding
: Utilizes RefSpatial
[
68
]
and RoboPoint
[
61
]
for multi-step spatial referring, 2D visual grounding, and precise coordinate localization.
•
Embodied QA
: Integrates RoboInter-VQA
[
27
]
, Robo2VLM
[
10
]
, and ShareRobot
[
23
]
to support embodied question answering and temporal reasoning. This preserves the model’s capabilities in high-level task planning and causal scene understanding.
•
Affordance Perception
: Adopts RoboAfford
[
50
]
for fine-grained manipulation affordance prediction and spatial interaction grounding.
The sample distribution of these datasets is illustrated in
Figure
11
, with their covered multimodal, embodied knowledge domains detailed in
Table
3
. We standardize these datasets to comply with the conversational input format of Qwen3-VL. Specifically, we normalize 2D bounding box and point coordinates to Qwen3-VL’s native
[
0
,
1000
]
[0,1000]
scale and adopt its standard representation formats. To ensure training stability, we only utilize single-image samples and exclude those with context lengths exceeding our maximum budget. By co-training on this VLM mixture, EgoSteer preserves open-world vision-language understanding and reasoning while enhancing its comprehension of robotics-specific tasks, thereby assisting the policy in following open-ended instructions and generalizing to novel manipulation scenarios.
EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.
The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.
The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.
The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.
Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.
To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.
Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.
This section provides supplementary details for the experimental evaluations presented in
Section
6
, detailing the model training hyperparameters and per-task success rates. In particular,
Table
4
lists the hyperparameter configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
Table
5
presents the per-task success rates on the four evaluation tasks in
Section
6.2
to demonstrate the performance gains from DAgger refinement. For the scaling analysis in
Section
6.3
,
Table
6
lists the training hyperparameters of EgoSteer-3K/6K/9.6K pre-trained models, their respective post-training runs, and the baseline trained from scratch, with their corresponding task-specific success rates reported in
Table
7
. Additionally,
Table
8
compares the detailed success rates of our method against the Being-H0.5 and
π
0.5
\pi_{0.5}
baselines across ten tasks.
Table
10
specifies the training configurations of the ablation experiment from
Section
6.4
, including the EgoSteer-1K model and its three ablated variants,
No WM-objective
,
No training-RTC
, and
Noisy data
. Their respective performance comparison is provided in
Table
9
. Finally,
Table
11
specifies the hyperparameters for the few-shot fine-tuning of our pre-trained EgoSteer-9.6K on the two challenging, long-horizon tasks evaluated in
Section
6.5
, namely
Box-Folding
and
Cake-Unboxing
.
Hyperparameter
Pre-Training
Post-Training
Camera setup
Head
Head & Chest
Resolution
384
×
384
384\times 384
640
×
480
640\times 480
GPUs
128 A800
96 A800
Gradient accumulation
2
1
Global batch size
4608
384
Training steps
175K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Freeze-VLM steps
5000
0
Warmup steps
2000
2000
Training time
164 h
29 h
Table 4:
Training configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
During pre-training, the VLM backbone is frozen for the first 5,000 steps, during which AE and WM are warmed up for 2,000 steps; once it is unfrozen, VLM is warmed up for 2,000 steps.
Task
EgoSteer-DG
EgoSteer-FT
Stack tableware
80.0%
50.0%
Close laptop
70.0%
10.0%
Place phone on stand
50.0%
0.0%
Flip cup
50.0%
30.0%
Average
62.5%
22.5%
Table 5:
Success rates across four highly dexterous and failure-prone manipulation tasks, comparing EgoSteer-DG against EgoSteer-FT in
Section
6.2
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Camera setup
Head
Head
Head
Head & Chest
Head & Chest
Head & Chest
Head & Chest
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
GPUs
64 A800
64 A800
128 A800
64 A800
32 A800
64 A800
64 A800
Gradient accumulation
2
4
2
2
4
2
2
Global batch size
2304
4608
4608
512
512
512
512
Training steps
100K
100K
160K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 6:
Training configurations for the pre-training scaling study in
Section
6.3
.
Pre-Training Data (Hours)
Task
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Grasp object
80%
80%
70%
100%
Hand over object
70%
80%
80%
100%
Place items into container
40%
80%
90%
100%
Point at object
20%
40%
60%
70%
Place toy chick into slot
30%
0%
20%
40%
Pull out tissue
60%
20%
50%
80%
Push ball into box
0%
0%
10%
20%
Put garbage into trash bin
0%
60%
30%
30%
Stack paper cups
0%
20%
10%
40%
Stack tableware
0%
20%
10%
20%
Average
30%
40%
43%
60%
Table 7:
Per-task success rates for the pre-training scaling study in
Section
6.3
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Task
Ours
Being-H0.5
π
0.5
\pi_{0.5}
Grasp object
100%
80%
80%
Hand over object
100%
60%
20%
Place items into container
100%
50%
0%
Pour items out of box
50%
30%
0%
Place bread on tray
70%
80%
40%
Pull out tissue
80%
10%
30%
Place item at specific orientation
30%
30%
30%
Attach eraser to whiteboard
90%
50%
20%
Put garbage into trash bin
30%
0%
0%
Put tennis ball into bucket
90%
0%
0%
Average
74%
39%
22%
Table 8:
Per-task comparison between EgoSteer-9.6K and two VLA baselines, Being-H0.5 and
π
0.5
\pi_{0.5}
, in
Section
6.3
. All methods are post-trained on our real-robot dataset and evaluated on the same 10 tasks. Each task uses 10 randomized trials. Bold values highlight the best performance.
Task
Ours
No WM-objective
No training-RTC
Noisy data
Grasp object
60%
40%
30%
60%
Hand over object
40%
20%
30%
40%
Place items into container
50%
30%
50%
70%
Pour items out of box
60%
50%
10%
0%
Place bread on tray
70%
10%
60%
50%
Pull out tissue
20%
30%
60%
10%
Place item at specific orientation
20%
10%
10%
20%
Attach eraser to whiteboard
50%
80%
50%
40%
Put garbage into trash bin
30%
0%
0%
0%
Put tennis ball into bucket
40%
40%
90%
40%
Average
44%
31%
39%
33%
Table 9:
Per-task success rates for the ablation study in
Section
6.4
. EgoSteer-1K is compared with ablated variants. Each task uses 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
Ours
No WM-objective
No training-RTC
Noisy data
Ours
No WM-objective
No training-RTC
Noisy data
Camera setup
Head
Head
Head
Head
Head
Head
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
Global batch size
1152
1152
1152
1152
1152
1152
1152
1152
Training steps
30K
80K
30K
20K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 10:
Training hyperparameters for the ablation study in
Section
6.4
. All variants are pre-trained on 1K hours of egocentric data and post-trained on the same real-robot dataset, without DAgger refinement. Pre-training steps are selected at the lowest evaluation
L
1
L_{1}
loss for each pre-training run.
Hyperparameter
Box-Folding
Cake-Unboxing
Pre-training checkpoint
EgoSteer-9.6K at 155K steps
Camera setup
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
8 A800
8 A800
Gradient accumulation
1
1
Global batch size
144
144
Fine-tuning steps
44K
12K
Learning rate (VLM / AE / WM)
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Demonstrations
120
229
Table 11:
Few-shot fine-tuning hyperparameters of EgoSteer-9.6K for the two long-horizon dexterous tasks in
Section
6.5
.

## Appendix B
Details of the Robot Stack

This section details the implementation of the Robot Stack (
Section
B.1
), alongside the collection protocols and statistics of the
187
187
-hour real-robot dataset (
Section
B.2
).
The two physical embodiments utilized in our work are illustrated in
Figure
3
. The primary platform, referred to as the RealMan embodiment, consists of two 7-DoF RealMan RM75-6F robotic arms and two 6-DoF Ruiyan RY-H2 dexterous hands. It is equipped with one head-mounted and one chest-mounted Intel RealSense D455 camera, providing dual egocentric viewpoints to ensure a comprehensive visual field. The AgiBot G1 embodiment is adapted from an AgiBot G1 humanoid robot by replacing its default end-effectors with two 6-DoF Ruiyan RY-H2 hands. It features one head-mounted Intel RealSense D455 camera and two wrist cameras, the latter of which are unused. During experiments, the robot’s neck, waist, and mobile base are kept fixed. Both platforms conduct tabletop manipulation on an operational table. The RealMan platform serves as the primary embodiment, on which the entire
187
187
-hour dataset was collected. All experiments are conducted on this setup, with the sole exception of the few-shot adaptation experiment performed on the AgiBot G1.
For both platforms, the human wrist tracker, arm kinematic solvers, and joint control ROS 2 nodes operate at
100
100
Hz, while the glove, hand solvers, and control ROS 2 nodes run at
80
80
Hz. This high-frequency loop ensures near-zero latency, enabling intuitive bimanual teleoperation and the collection of fine-grained manipulation demonstrations. The cameras capture frames at
30
30
Hz. Data is recorded at these native frequencies and subsequently resampled to
30
30
Hz for training.
Robot data collection aims to ground the human manipulation priors learned from large-scale pre-training onto the target physical embodiment. To preserve and transfer this pre-trained knowledge, minimizing the domain gap between robot and human data is essential. Because the robotic palm is slightly longer than a human palm, we translate the robot’s wrist coordinate frame axially forward. This alignment ensures that the scale from the wrist to the fingertips matches human anatomical proportions. When converting raw robot data into training data, this coordinate transformation is applied alongside hand-eye calibration to map hand actions into the camera coordinate frame. Correspondingly, during real-time policy inference, the control stack executes the inverse transformation and hand-eye mapping to project predicted actions back to the robot’s physical coordinate space.
In practice, hand-eye calibration is prone to drift due to mechanical maintenance, wear, or collisions, with such discrepancies sometimes only identified
post
-acquisition. To ensure data quality, we introduce an offline verification and automated recalibration pipeline. RGB-D images from the camera are projected into a 3D point cloud, and the robot’s 3D mesh is rendered in the same space based on the hand-eye calibration, utilizing their spatial overlap for validation. Upon detecting misalignment, our pipeline runs FoundationPose
[
55
]
on the point cloud to estimate the robot hand’s 6D pose based on its URDF. Through FK, the calibration matrix is back-calculated for each frame. Averaging these matrices across frames and removing outliers robustly recovers the calibration parameters, preventing data degradation.
Below, we present the prompt template designed for Qwen3-VL-Flash
[
3
]
to generate multi-granularity language annotations for teleoperated demonstrations.
Robot Data Language Labeling Prompt
Although pre-training on egocentric human videos equips the model with rich dexterous manipulation priors, direct physical deployment is prevented by the embodiment gap across visual appearance, dynamics, and kinematics. Furthermore, because these pre-training labels are generated by automated depth and hand pose estimation models, their accuracy is constrained by current model capabilities, potentially introducing systematic biases into the pre-trained policy. Therefore, collecting real-robot teleoperation data aims to correct and ground these priors onto the target embodiment with high sample efficiency. Additionally, due to kinematic constraints on robotic degrees of freedom, the robot cannot perfectly replicate all fine-grained human hand movements. To achieve steerable manipulation, we must collect a highly diverse range of free-form tasks to maximize the coverage of basic manipulation primitives, guiding the model to efficiently transfer cross-domain knowledge and fostering robust compositional generalization and multi-task instruction following.
Based on these considerations, within the kinematic limits of RealMan’s degrees of freedom, we design
193
193
semantically distinct dexterous manipulation tasks. For each task, approximately
300
300
randomized, diverse demonstration trajectories, totaling around
1
1
hour, are collected under cluttered scenarios, constructing a high-quality real-robot dataset of
187
187
hours and
55
​
K
55\text{K}
trajectories.
These
193
193
tasks are classified into two major categories:
•
Common Tasks
(
56
56
tasks): Everyday manipulations that are readily achievable within the current robot hardware configuration and sensor limits, yielding high teleoperation success rates.
•
Long-Tail Tasks
(
137
137
tasks): Infrequent and physically challenging manipulations, such as contact-sensitive operations lacking tactile feedback, with lower collection success rates, primarily designed to ensure comprehensive semantic coverage of the dexterous manipulation space.
Furthermore, based on motion characteristics and physical interactions, all tasks are categorized into seven classes:
•
PnP-Easy
: Single-step tabletop pick-and-place where objects are easily graspable and the placement space is open.
•
PnP-Medium
: Non-planar or 3D spatial pick-and-place that requires precise operations related to containers, demanding higher control accuracy and spatial perception, such as “
put tennis ball into ball holder
”.
•
PnP-Hard
: Multi-step or high-precision pick-and-place sequences, such as “
stack paper cups
”.
•
Non-prehensile
: Actions not involving traditional finger grasping, including pushing, pulling, and pressing.
•
Reorient
: Operations involving rotation and reorientation, such as “
pour water
” or “
flip paper cups
”.
•
Bimanual
: Bimanual tasks requiring high synchronization and spatial coordination between arms and hands, such as “
plug cable into charger
”.
•
Contact-rich
: Operations involving frequent and complex physical contact interactions, requiring physical understanding, such as “
wipe whiteboard
”.
Throughout data collection, strict requirements are enforced on the randomness, diversity, and quality of each trajectory. Specifically, the tabletop features cluttered, unstructured scenes rather than pre-arranged, simplified environments, avoiding task identification from visual inputs alone. This design forces the model to deeply align language instructions with physical actions, encouraging it to learn general task semantics and execution goals rather than memorizing demonstration trajectories. Furthermore, given the highly randomized object configurations, operators are instructed to perform teleoperation in a natural, human-like manner. Consequently, different demonstrations of the same task exhibit substantial variations in execution trajectories, thereby covering a broader distribution of the mapping from human priors to the robot’s action space.
As illustrated in
Figure
8
, we analyze the
187
187
-hour real-robot dataset across several key dimensions, including the verb and noun word clouds with their top 30 frequency distributions, the task duration breakdown across seven manipulation categories, and the detailed duration statistics for the
56
56
common and
137
137
long-tail tasks. These statistics reveal a diverse vocabulary of actions and manipulated objects, alongside a balanced distribution across both manipulation categories and task durations. Encompassing rich semantic concepts and manipulation primitives, this dataset effectively grounds pre-trained human manipulation priors onto the RealMan embodiment. Additionally,
Figure
9
showcases the dual-view image sequences and language annotations of a teleoperation trajectory, while
Figure
10
visualizes sample tasks across each category.
(a)
Word cloud of nouns in annotation
(b)
Top 30 noun frequencies
(c)
Word cloud of verbs in annotation
(d)
Top 30 verb frequencies
(e)
Task duration by seven categories
(f)
Common task duration statistics
(g)
Long-tail task duration statistics
Figure 8:
Teleoperation dataset statistics. (1)
Figure
8
-
Figure
8
: Word clouds and top-30 frequency distributions of nouns and verbs in language annotations. (2)
Figure
8
: Task duration breakdown by seven manipulation categories (PnP-Easy/Medium/Hard, Non-prehensile, Reorient, Bimanual, Contact-rich). (3)
Figure
8
and
Figure
8
: Detailed duration statistics for common tasks (56 tasks) and long-tail tasks (137 tasks).
Figure 9:
Dual-view image sequence and three-level language annotations of a representative trajectory
. The frames illustrate a cluttered, randomized setup where the operator executes the task in a natural, human-like manner. The hierarchical language annotations describe the manipulation process from coarse to fine, assisting the model in aligning free-form instructions with physical actions.
Figure 10:
Representative task examples across seven manipulation categories
. These diverse tasks are executed in cluttered, randomized scenarios using natural, human-like manipulations, covering a wide range of action semantics and manipulation primitives to facilitate the efficient grounding of pre-trained egocentric human priors onto the physical robot.
Dataset
Samples
Captioning
Visual Question
Answering
Multiple-Choice
Question
Pointing
Bounding Box
Affordance
Trajectory
Spatial
Planning
FineVision
[
56
]
3.5
​
M
3.5\text{M}
✓
✓
✓
✓
RefSpatial
[
68
]
2.5
​
M
2.5\text{M}
✓
✓
✓
✓
RoboInter-VQA
[
27
]
1.6
​
M
1.6\text{M}
✓
✓
✓
✓
✓
✓
✓
RoboPoint
[
61
]
1.3
​
M
1.3\text{M}
✓
✓
✓
✓
RoboAfford
[
50
]
765
​
K
765\text{K}
✓
✓
✓
✓
Robo2VLM
[
10
]
678
​
K
678\text{K}
✓
✓
ShareRobot
[
23
]
13
​
K
13\text{K}
✓
✓
✓
✓
✓
Table 3:
VLM datasets used for EgoSteer co-training
. This table details the sample sizes and the multi-modal and embodied knowledge domains covered by each dataset. This co-training mixture integrates general vision-language knowledge with interaction understanding and spatial-geometric priors, preserving the model’s inherent world knowledge while facilitating its ability to follow free-form instructions to generalize across diverse manipulation tasks.
Figure 11:
Statistical distribution of samples across VLM co-training datasets.
This section first introduces the VLM datasets utilized for co-training with VLA data,
i.e.
egocentric human videos and real-robot data, to preserve EgoSteer’s vision-language knowledge and ensure generalization (
Section
C.1
). We then present implementation details of EgoSteer (
Section
C.2
).
To preserve general vision-language reasoning capabilities while simultaneously cultivating robust robotic task comprehension, a
10.4
​
M
10.4\text{M}
-sample VLM co-training mixture is curated across seven datasets, ranging from open-world perception to embodied interaction grounding, to co-train EgoSteer. This mixture comprises four categories of data:
•
General VLM Pre-Training
: Incorporates FineVision
[
56
]
to prevent the catastrophic forgetting of open-world semantic concepts and preserve general visual-language reasoning.
•
Spatial Grounding
: Utilizes RefSpatial
[
68
]
and RoboPoint
[
61
]
for multi-step spatial referring, 2D visual grounding, and precise coordinate localization.
•
Embodied QA
: Integrates RoboInter-VQA
[
27
]
, Robo2VLM
[
10
]
, and ShareRobot
[
23
]
to support embodied question answering and temporal reasoning. This preserves the model’s capabilities in high-level task planning and causal scene understanding.
•
Affordance Perception
: Adopts RoboAfford
[
50
]
for fine-grained manipulation affordance prediction and spatial interaction grounding.
The sample distribution of these datasets is illustrated in
Figure
11
, with their covered multimodal, embodied knowledge domains detailed in
Table
3
. We standardize these datasets to comply with the conversational input format of Qwen3-VL. Specifically, we normalize 2D bounding box and point coordinates to Qwen3-VL’s native
[
0
,
1000
]
[0,1000]
scale and adopt its standard representation formats. To ensure training stability, we only utilize single-image samples and exclude those with context lengths exceeding our maximum budget. By co-training on this VLM mixture, EgoSteer preserves open-world vision-language understanding and reasoning while enhancing its comprehension of robotics-specific tasks, thereby assisting the policy in following open-ended instructions and generalizing to novel manipulation scenarios.
EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.
The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.
The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.
The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.
Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.
To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.
Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.
This section provides supplementary details for the experimental evaluations presented in
Section
6
, detailing the model training hyperparameters and per-task success rates. In particular,
Table
4
lists the hyperparameter configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
Table
5
presents the per-task success rates on the four evaluation tasks in
Section
6.2
to demonstrate the performance gains from DAgger refinement. For the scaling analysis in
Section
6.3
,
Table
6
lists the training hyperparameters of EgoSteer-3K/6K/9.6K pre-trained models, their respective post-training runs, and the baseline trained from scratch, with their corresponding task-specific success rates reported in
Table
7
. Additionally,
Table
8
compares the detailed success rates of our method against the Being-H0.5 and
π
0.5
\pi_{0.5}
baselines across ten tasks.
Table
10
specifies the training configurations of the ablation experiment from
Section
6.4
, including the EgoSteer-1K model and its three ablated variants,
No WM-objective
,
No training-RTC
, and
Noisy data
. Their respective performance comparison is provided in
Table
9
. Finally,
Table
11
specifies the hyperparameters for the few-shot fine-tuning of our pre-trained EgoSteer-9.6K on the two challenging, long-horizon tasks evaluated in
Section
6.5
, namely
Box-Folding
and
Cake-Unboxing
.
Hyperparameter
Pre-Training
Post-Training
Camera setup
Head
Head & Chest
Resolution
384
×
384
384\times 384
640
×
480
640\times 480
GPUs
128 A800
96 A800
Gradient accumulation
2
1
Global batch size
4608
384
Training steps
175K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Freeze-VLM steps
5000
0
Warmup steps
2000
2000
Training time
164 h
29 h
Table 4:
Training configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
During pre-training, the VLM backbone is frozen for the first 5,000 steps, during which AE and WM are warmed up for 2,000 steps; once it is unfrozen, VLM is warmed up for 2,000 steps.
Task
EgoSteer-DG
EgoSteer-FT
Stack tableware
80.0%
50.0%
Close laptop
70.0%
10.0%
Place phone on stand
50.0%
0.0%
Flip cup
50.0%
30.0%
Average
62.5%
22.5%
Table 5:
Success rates across four highly dexterous and failure-prone manipulation tasks, comparing EgoSteer-DG against EgoSteer-FT in
Section
6.2
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Camera setup
Head
Head
Head
Head & Chest
Head & Chest
Head & Chest
Head & Chest
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
GPUs
64 A800
64 A800
128 A800
64 A800
32 A800
64 A800
64 A800
Gradient accumulation
2
4
2
2
4
2
2
Global batch size
2304
4608
4608
512
512
512
512
Training steps
100K
100K
160K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 6:
Training configurations for the pre-training scaling study in
Section
6.3
.
Pre-Training Data (Hours)
Task
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Grasp object
80%
80%
70%
100%
Hand over object
70%
80%
80%
100%
Place items into container
40%
80%
90%
100%
Point at object
20%
40%
60%
70%
Place toy chick into slot
30%
0%
20%
40%
Pull out tissue
60%
20%
50%
80%
Push ball into box
0%
0%
10%
20%
Put garbage into trash bin
0%
60%
30%
30%
Stack paper cups
0%
20%
10%
40%
Stack tableware
0%
20%
10%
20%
Average
30%
40%
43%
60%
Table 7:
Per-task success rates for the pre-training scaling study in
Section
6.3
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Task
Ours
Being-H0.5
π
0.5
\pi_{0.5}
Grasp object
100%
80%
80%
Hand over object
100%
60%
20%
Place items into container
100%
50%
0%
Pour items out of box
50%
30%
0%
Place bread on tray
70%
80%
40%
Pull out tissue
80%
10%
30%
Place item at specific orientation
30%
30%
30%
Attach eraser to whiteboard
90%
50%
20%
Put garbage into trash bin
30%
0%
0%
Put tennis ball into bucket
90%
0%
0%
Average
74%
39%
22%
Table 8:
Per-task comparison between EgoSteer-9.6K and two VLA baselines, Being-H0.5 and
π
0.5
\pi_{0.5}
, in
Section
6.3
. All methods are post-trained on our real-robot dataset and evaluated on the same 10 tasks. Each task uses 10 randomized trials. Bold values highlight the best performance.
Task
Ours
No WM-objective
No training-RTC
Noisy data
Grasp object
60%
40%
30%
60%
Hand over object
40%
20%
30%
40%
Place items into container
50%
30%
50%
70%
Pour items out of box
60%
50%
10%
0%
Place bread on tray
70%
10%
60%
50%
Pull out tissue
20%
30%
60%
10%
Place item at specific orientation
20%
10%
10%
20%
Attach eraser to whiteboard
50%
80%
50%
40%
Put garbage into trash bin
30%
0%
0%
0%
Put tennis ball into bucket
40%
40%
90%
40%
Average
44%
31%
39%
33%
Table 9:
Per-task success rates for the ablation study in
Section
6.4
. EgoSteer-1K is compared with ablated variants. Each task uses 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
Ours
No WM-objective
No training-RTC
Noisy data
Ours
No WM-objective
No training-RTC
Noisy data
Camera setup
Head
Head
Head
Head
Head
Head
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
Global batch size
1152
1152
1152
1152
1152
1152
1152
1152
Training steps
30K
80K
30K
20K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 10:
Training hyperparameters for the ablation study in
Section
6.4
. All variants are pre-trained on 1K hours of egocentric data and post-trained on the same real-robot dataset, without DAgger refinement. Pre-training steps are selected at the lowest evaluation
L
1
L_{1}
loss for each pre-training run.
Hyperparameter
Box-Folding
Cake-Unboxing
Pre-training checkpoint
EgoSteer-9.6K at 155K steps
Camera setup
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
8 A800
8 A800
Gradient accumulation
1
1
Global batch size
144
144
Fine-tuning steps
44K
12K
Learning rate (VLM / AE / WM)
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Demonstrations
120
229
Table 11:
Few-shot fine-tuning hyperparameters of EgoSteer-9.6K for the two long-horizon dexterous tasks in
Section
6.5
.

### B.1
Implementation Details

The two physical embodiments utilized in our work are illustrated in
Figure
3
. The primary platform, referred to as the RealMan embodiment, consists of two 7-DoF RealMan RM75-6F robotic arms and two 6-DoF Ruiyan RY-H2 dexterous hands. It is equipped with one head-mounted and one chest-mounted Intel RealSense D455 camera, providing dual egocentric viewpoints to ensure a comprehensive visual field. The AgiBot G1 embodiment is adapted from an AgiBot G1 humanoid robot by replacing its default end-effectors with two 6-DoF Ruiyan RY-H2 hands. It features one head-mounted Intel RealSense D455 camera and two wrist cameras, the latter of which are unused. During experiments, the robot’s neck, waist, and mobile base are kept fixed. Both platforms conduct tabletop manipulation on an operational table. The RealMan platform serves as the primary embodiment, on which the entire
187
187
-hour dataset was collected. All experiments are conducted on this setup, with the sole exception of the few-shot adaptation experiment performed on the AgiBot G1.
For both platforms, the human wrist tracker, arm kinematic solvers, and joint control ROS 2 nodes operate at
100
100
Hz, while the glove, hand solvers, and control ROS 2 nodes run at
80
80
Hz. This high-frequency loop ensures near-zero latency, enabling intuitive bimanual teleoperation and the collection of fine-grained manipulation demonstrations. The cameras capture frames at
30
30
Hz. Data is recorded at these native frequencies and subsequently resampled to
30
30
Hz for training.
Robot data collection aims to ground the human manipulation priors learned from large-scale pre-training onto the target physical embodiment. To preserve and transfer this pre-trained knowledge, minimizing the domain gap between robot and human data is essential. Because the robotic palm is slightly longer than a human palm, we translate the robot’s wrist coordinate frame axially forward. This alignment ensures that the scale from the wrist to the fingertips matches human anatomical proportions. When converting raw robot data into training data, this coordinate transformation is applied alongside hand-eye calibration to map hand actions into the camera coordinate frame. Correspondingly, during real-time policy inference, the control stack executes the inverse transformation and hand-eye mapping to project predicted actions back to the robot’s physical coordinate space.
In practice, hand-eye calibration is prone to drift due to mechanical maintenance, wear, or collisions, with such discrepancies sometimes only identified
post
-acquisition. To ensure data quality, we introduce an offline verification and automated recalibration pipeline. RGB-D images from the camera are projected into a 3D point cloud, and the robot’s 3D mesh is rendered in the same space based on the hand-eye calibration, utilizing their spatial overlap for validation. Upon detecting misalignment, our pipeline runs FoundationPose
[
55
]
on the point cloud to estimate the robot hand’s 6D pose based on its URDF. Through FK, the calibration matrix is back-calculated for each frame. Averaging these matrices across frames and removing outliers robustly recovers the calibration parameters, preventing data degradation.
Below, we present the prompt template designed for Qwen3-VL-Flash
[
3
]
to generate multi-granularity language annotations for teleoperated demonstrations.
Robot Data Language Labeling Prompt
Although pre-training on egocentric human videos equips the model with rich dexterous manipulation priors, direct physical deployment is prevented by the embodiment gap across visual appearance, dynamics, and kinematics. Furthermore, because these pre-training labels are generated by automated depth and hand pose estimation models, their accuracy is constrained by current model capabilities, potentially introducing systematic biases into the pre-trained policy. Therefore, collecting real-robot teleoperation data aims to correct and ground these priors onto the target embodiment with high sample efficiency. Additionally, due to kinematic constraints on robotic degrees of freedom, the robot cannot perfectly replicate all fine-grained human hand movements. To achieve steerable manipulation, we must collect a highly diverse range of free-form tasks to maximize the coverage of basic manipulation primitives, guiding the model to efficiently transfer cross-domain knowledge and fostering robust compositional generalization and multi-task instruction following.
Based on these considerations, within the kinematic limits of RealMan’s degrees of freedom, we design
193
193
semantically distinct dexterous manipulation tasks. For each task, approximately
300
300
randomized, diverse demonstration trajectories, totaling around
1
1
hour, are collected under cluttered scenarios, constructing a high-quality real-robot dataset of
187
187
hours and
55
​
K
55\text{K}
trajectories.
These
193
193
tasks are classified into two major categories:
•
Common Tasks
(
56
56
tasks): Everyday manipulations that are readily achievable within the current robot hardware configuration and sensor limits, yielding high teleoperation success rates.
•
Long-Tail Tasks
(
137
137
tasks): Infrequent and physically challenging manipulations, such as contact-sensitive operations lacking tactile feedback, with lower collection success rates, primarily designed to ensure comprehensive semantic coverage of the dexterous manipulation space.
Furthermore, based on motion characteristics and physical interactions, all tasks are categorized into seven classes:
•
PnP-Easy
: Single-step tabletop pick-and-place where objects are easily graspable and the placement space is open.
•
PnP-Medium
: Non-planar or 3D spatial pick-and-place that requires precise operations related to containers, demanding higher control accuracy and spatial perception, such as “
put tennis ball into ball holder
”.
•
PnP-Hard
: Multi-step or high-precision pick-and-place sequences, such as “
stack paper cups
”.
•
Non-prehensile
: Actions not involving traditional finger grasping, including pushing, pulling, and pressing.
•
Reorient
: Operations involving rotation and reorientation, such as “
pour water
” or “
flip paper cups
”.
•
Bimanual
: Bimanual tasks requiring high synchronization and spatial coordination between arms and hands, such as “
plug cable into charger
”.
•
Contact-rich
: Operations involving frequent and complex physical contact interactions, requiring physical understanding, such as “
wipe whiteboard
”.
Throughout data collection, strict requirements are enforced on the randomness, diversity, and quality of each trajectory. Specifically, the tabletop features cluttered, unstructured scenes rather than pre-arranged, simplified environments, avoiding task identification from visual inputs alone. This design forces the model to deeply align language instructions with physical actions, encouraging it to learn general task semantics and execution goals rather than memorizing demonstration trajectories. Furthermore, given the highly randomized object configurations, operators are instructed to perform teleoperation in a natural, human-like manner. Consequently, different demonstrations of the same task exhibit substantial variations in execution trajectories, thereby covering a broader distribution of the mapping from human priors to the robot’s action space.
As illustrated in
Figure
8
, we analyze the
187
187
-hour real-robot dataset across several key dimensions, including the verb and noun word clouds with their top 30 frequency distributions, the task duration breakdown across seven manipulation categories, and the detailed duration statistics for the
56
56
common and
137
137
long-tail tasks. These statistics reveal a diverse vocabulary of actions and manipulated objects, alongside a balanced distribution across both manipulation categories and task durations. Encompassing rich semantic concepts and manipulation primitives, this dataset effectively grounds pre-trained human manipulation priors onto the RealMan embodiment. Additionally,
Figure
9
showcases the dual-view image sequences and language annotations of a teleoperation trajectory, while
Figure
10
visualizes sample tasks across each category.
(a)
Word cloud of nouns in annotation
(b)
Top 30 noun frequencies
(c)
Word cloud of verbs in annotation
(d)
Top 30 verb frequencies
(e)
Task duration by seven categories
(f)
Common task duration statistics
(g)
Long-tail task duration statistics
Figure 8:
Teleoperation dataset statistics. (1)
Figure
8
-
Figure
8
: Word clouds and top-30 frequency distributions of nouns and verbs in language annotations. (2)
Figure
8
: Task duration breakdown by seven manipulation categories (PnP-Easy/Medium/Hard, Non-prehensile, Reorient, Bimanual, Contact-rich). (3)
Figure
8
and
Figure
8
: Detailed duration statistics for common tasks (56 tasks) and long-tail tasks (137 tasks).
Figure 9:
Dual-view image sequence and three-level language annotations of a representative trajectory
. The frames illustrate a cluttered, randomized setup where the operator executes the task in a natural, human-like manner. The hierarchical language annotations describe the manipulation process from coarse to fine, assisting the model in aligning free-form instructions with physical actions.
Figure 10:
Representative task examples across seven manipulation categories
. These diverse tasks are executed in cluttered, randomized scenarios using natural, human-like manipulations, covering a wide range of action semantics and manipulation primitives to facilitate the efficient grounding of pre-trained egocentric human priors onto the physical robot.
Dataset
Samples
Captioning
Visual Question
Answering
Multiple-Choice
Question
Pointing
Bounding Box
Affordance
Trajectory
Spatial
Planning
FineVision
[
56
]
3.5
​
M
3.5\text{M}
✓
✓
✓
✓
RefSpatial
[
68
]
2.5
​
M
2.5\text{M}
✓
✓
✓
✓
RoboInter-VQA
[
27
]
1.6
​
M
1.6\text{M}
✓
✓
✓
✓
✓
✓
✓
RoboPoint
[
61
]
1.3
​
M
1.3\text{M}
✓
✓
✓
✓
RoboAfford
[
50
]
765
​
K
765\text{K}
✓
✓
✓
✓
Robo2VLM
[
10
]
678
​
K
678\text{K}
✓
✓
ShareRobot
[
23
]
13
​
K
13\text{K}
✓
✓
✓
✓
✓
Table 3:
VLM datasets used for EgoSteer co-training
. This table details the sample sizes and the multi-modal and embodied knowledge domains covered by each dataset. This co-training mixture integrates general vision-language knowledge with interaction understanding and spatial-geometric priors, preserving the model’s inherent world knowledge while facilitating its ability to follow free-form instructions to generalize across diverse manipulation tasks.
Figure 11:
Statistical distribution of samples across VLM co-training datasets.
This section first introduces the VLM datasets utilized for co-training with VLA data,
i.e.
egocentric human videos and real-robot data, to preserve EgoSteer’s vision-language knowledge and ensure generalization (
Section
C.1
). We then present implementation details of EgoSteer (
Section
C.2
).
To preserve general vision-language reasoning capabilities while simultaneously cultivating robust robotic task comprehension, a
10.4
​
M
10.4\text{M}
-sample VLM co-training mixture is curated across seven datasets, ranging from open-world perception to embodied interaction grounding, to co-train EgoSteer. This mixture comprises four categories of data:
•
General VLM Pre-Training
: Incorporates FineVision
[
56
]
to prevent the catastrophic forgetting of open-world semantic concepts and preserve general visual-language reasoning.
•
Spatial Grounding
: Utilizes RefSpatial
[
68
]
and RoboPoint
[
61
]
for multi-step spatial referring, 2D visual grounding, and precise coordinate localization.
•
Embodied QA
: Integrates RoboInter-VQA
[
27
]
, Robo2VLM
[
10
]
, and ShareRobot
[
23
]
to support embodied question answering and temporal reasoning. This preserves the model’s capabilities in high-level task planning and causal scene understanding.
•
Affordance Perception
: Adopts RoboAfford
[
50
]
for fine-grained manipulation affordance prediction and spatial interaction grounding.
The sample distribution of these datasets is illustrated in
Figure
11
, with their covered multimodal, embodied knowledge domains detailed in
Table
3
. We standardize these datasets to comply with the conversational input format of Qwen3-VL. Specifically, we normalize 2D bounding box and point coordinates to Qwen3-VL’s native
[
0
,
1000
]
[0,1000]
scale and adopt its standard representation formats. To ensure training stability, we only utilize single-image samples and exclude those with context lengths exceeding our maximum budget. By co-training on this VLM mixture, EgoSteer preserves open-world vision-language understanding and reasoning while enhancing its comprehension of robotics-specific tasks, thereby assisting the policy in following open-ended instructions and generalizing to novel manipulation scenarios.
EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.
The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.
The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.
The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.
Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.
To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.
Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.
This section provides supplementary details for the experimental evaluations presented in
Section
6
, detailing the model training hyperparameters and per-task success rates. In particular,
Table
4
lists the hyperparameter configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
Table
5
presents the per-task success rates on the four evaluation tasks in
Section
6.2
to demonstrate the performance gains from DAgger refinement. For the scaling analysis in
Section
6.3
,
Table
6
lists the training hyperparameters of EgoSteer-3K/6K/9.6K pre-trained models, their respective post-training runs, and the baseline trained from scratch, with their corresponding task-specific success rates reported in
Table
7
. Additionally,
Table
8
compares the detailed success rates of our method against the Being-H0.5 and
π
0.5
\pi_{0.5}
baselines across ten tasks.
Table
10
specifies the training configurations of the ablation experiment from
Section
6.4
, including the EgoSteer-1K model and its three ablated variants,
No WM-objective
,
No training-RTC
, and
Noisy data
. Their respective performance comparison is provided in
Table
9
. Finally,
Table
11
specifies the hyperparameters for the few-shot fine-tuning of our pre-trained EgoSteer-9.6K on the two challenging, long-horizon tasks evaluated in
Section
6.5
, namely
Box-Folding
and
Cake-Unboxing
.
Hyperparameter
Pre-Training
Post-Training
Camera setup
Head
Head & Chest
Resolution
384
×
384
384\times 384
640
×
480
640\times 480
GPUs
128 A800
96 A800
Gradient accumulation
2
1
Global batch size
4608
384
Training steps
175K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Freeze-VLM steps
5000
0
Warmup steps
2000
2000
Training time
164 h
29 h
Table 4:
Training configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
During pre-training, the VLM backbone is frozen for the first 5,000 steps, during which AE and WM are warmed up for 2,000 steps; once it is unfrozen, VLM is warmed up for 2,000 steps.
Task
EgoSteer-DG
EgoSteer-FT
Stack tableware
80.0%
50.0%
Close laptop
70.0%
10.0%
Place phone on stand
50.0%
0.0%
Flip cup
50.0%
30.0%
Average
62.5%
22.5%
Table 5:
Success rates across four highly dexterous and failure-prone manipulation tasks, comparing EgoSteer-DG against EgoSteer-FT in
Section
6.2
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Camera setup
Head
Head
Head
Head & Chest
Head & Chest
Head & Chest
Head & Chest
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
GPUs
64 A800
64 A800
128 A800
64 A800
32 A800
64 A800
64 A800
Gradient accumulation
2
4
2
2
4
2
2
Global batch size
2304
4608
4608
512
512
512
512
Training steps
100K
100K
160K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 6:
Training configurations for the pre-training scaling study in
Section
6.3
.
Pre-Training Data (Hours)
Task
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Grasp object
80%
80%
70%
100%
Hand over object
70%
80%
80%
100%
Place items into container
40%
80%
90%
100%
Point at object
20%
40%
60%
70%
Place toy chick into slot
30%
0%
20%
40%
Pull out tissue
60%
20%
50%
80%
Push ball into box
0%
0%
10%
20%
Put garbage into trash bin
0%
60%
30%
30%
Stack paper cups
0%
20%
10%
40%
Stack tableware
0%
20%
10%
20%
Average
30%
40%
43%
60%
Table 7:
Per-task success rates for the pre-training scaling study in
Section
6.3
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Task
Ours
Being-H0.5
π
0.5
\pi_{0.5}
Grasp object
100%
80%
80%
Hand over object
100%
60%
20%
Place items into container
100%
50%
0%
Pour items out of box
50%
30%
0%
Place bread on tray
70%
80%
40%
Pull out tissue
80%
10%
30%
Place item at specific orientation
30%
30%
30%
Attach eraser to whiteboard
90%
50%
20%
Put garbage into trash bin
30%
0%
0%
Put tennis ball into bucket
90%
0%
0%
Average
74%
39%
22%
Table 8:
Per-task comparison between EgoSteer-9.6K and two VLA baselines, Being-H0.5 and
π
0.5
\pi_{0.5}
, in
Section
6.3
. All methods are post-trained on our real-robot dataset and evaluated on the same 10 tasks. Each task uses 10 randomized trials. Bold values highlight the best performance.
Task
Ours
No WM-objective
No training-RTC
Noisy data
Grasp object
60%
40%
30%
60%
Hand over object
40%
20%
30%
40%
Place items into container
50%
30%
50%
70%
Pour items out of box
60%
50%
10%
0%
Place bread on tray
70%
10%
60%
50%
Pull out tissue
20%
30%
60%
10%
Place item at specific orientation
20%
10%
10%
20%
Attach eraser to whiteboard
50%
80%
50%
40%
Put garbage into trash bin
30%
0%
0%
0%
Put tennis ball into bucket
40%
40%
90%
40%
Average
44%
31%
39%
33%
Table 9:
Per-task success rates for the ablation study in
Section
6.4
. EgoSteer-1K is compared with ablated variants. Each task uses 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
Ours
No WM-objective
No training-RTC
Noisy data
Ours
No WM-objective
No training-RTC
Noisy data
Camera setup
Head
Head
Head
Head
Head
Head
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
Global batch size
1152
1152
1152
1152
1152
1152
1152
1152
Training steps
30K
80K
30K
20K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 10:
Training hyperparameters for the ablation study in
Section
6.4
. All variants are pre-trained on 1K hours of egocentric data and post-trained on the same real-robot dataset, without DAgger refinement. Pre-training steps are selected at the lowest evaluation
L
1
L_{1}
loss for each pre-training run.
Hyperparameter
Box-Folding
Cake-Unboxing
Pre-training checkpoint
EgoSteer-9.6K at 155K steps
Camera setup
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
8 A800
8 A800
Gradient accumulation
1
1
Global batch size
144
144
Fine-tuning steps
44K
12K
Learning rate (VLM / AE / WM)
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Demonstrations
120
229
Table 11:
Few-shot fine-tuning hyperparameters of EgoSteer-9.6K for the two long-horizon dexterous tasks in
Section
6.5
.

#### B.1.1
Hardware Setup

The two physical embodiments utilized in our work are illustrated in
Figure
3
. The primary platform, referred to as the RealMan embodiment, consists of two 7-DoF RealMan RM75-6F robotic arms and two 6-DoF Ruiyan RY-H2 dexterous hands. It is equipped with one head-mounted and one chest-mounted Intel RealSense D455 camera, providing dual egocentric viewpoints to ensure a comprehensive visual field. The AgiBot G1 embodiment is adapted from an AgiBot G1 humanoid robot by replacing its default end-effectors with two 6-DoF Ruiyan RY-H2 hands. It features one head-mounted Intel RealSense D455 camera and two wrist cameras, the latter of which are unused. During experiments, the robot’s neck, waist, and mobile base are kept fixed. Both platforms conduct tabletop manipulation on an operational table. The RealMan platform serves as the primary embodiment, on which the entire
187
187
-hour dataset was collected. All experiments are conducted on this setup, with the sole exception of the few-shot adaptation experiment performed on the AgiBot G1.
For both platforms, the human wrist tracker, arm kinematic solvers, and joint control ROS 2 nodes operate at
100
100
Hz, while the glove, hand solvers, and control ROS 2 nodes run at
80
80
Hz. This high-frequency loop ensures near-zero latency, enabling intuitive bimanual teleoperation and the collection of fine-grained manipulation demonstrations. The cameras capture frames at
30
30
Hz. Data is recorded at these native frequencies and subsequently resampled to
30
30
Hz for training.

#### B.1.2
Aligning Robot Data with Egocentric Human Data

Robot data collection aims to ground the human manipulation priors learned from large-scale pre-training onto the target physical embodiment. To preserve and transfer this pre-trained knowledge, minimizing the domain gap between robot and human data is essential. Because the robotic palm is slightly longer than a human palm, we translate the robot’s wrist coordinate frame axially forward. This alignment ensures that the scale from the wrist to the fingertips matches human anatomical proportions. When converting raw robot data into training data, this coordinate transformation is applied alongside hand-eye calibration to map hand actions into the camera coordinate frame. Correspondingly, during real-time policy inference, the control stack executes the inverse transformation and hand-eye mapping to project predicted actions back to the robot’s physical coordinate space.

#### B.1.3
Hand-Eye Verification and Re-Calibration

In practice, hand-eye calibration is prone to drift due to mechanical maintenance, wear, or collisions, with such discrepancies sometimes only identified
post
-acquisition. To ensure data quality, we introduce an offline verification and automated recalibration pipeline. RGB-D images from the camera are projected into a 3D point cloud, and the robot’s 3D mesh is rendered in the same space based on the hand-eye calibration, utilizing their spatial overlap for validation. Upon detecting misalignment, our pipeline runs FoundationPose
[
55
]
on the point cloud to estimate the robot hand’s 6D pose based on its URDF. Through FK, the calibration matrix is back-calculated for each frame. Averaging these matrices across frames and removing outliers robustly recovers the calibration parameters, preventing data degradation.

#### B.1.4
Language Labeling Prompt

Below, we present the prompt template designed for Qwen3-VL-Flash
[
3
]
to generate multi-granularity language annotations for teleoperated demonstrations.
Robot Data Language Labeling Prompt
Although pre-training on egocentric human videos equips the model with rich dexterous manipulation priors, direct physical deployment is prevented by the embodiment gap across visual appearance, dynamics, and kinematics. Furthermore, because these pre-training labels are generated by automated depth and hand pose estimation models, their accuracy is constrained by current model capabilities, potentially introducing systematic biases into the pre-trained policy. Therefore, collecting real-robot teleoperation data aims to correct and ground these priors onto the target embodiment with high sample efficiency. Additionally, due to kinematic constraints on robotic degrees of freedom, the robot cannot perfectly replicate all fine-grained human hand movements. To achieve steerable manipulation, we must collect a highly diverse range of free-form tasks to maximize the coverage of basic manipulation primitives, guiding the model to efficiently transfer cross-domain knowledge and fostering robust compositional generalization and multi-task instruction following.
Based on these considerations, within the kinematic limits of RealMan’s degrees of freedom, we design
193
193
semantically distinct dexterous manipulation tasks. For each task, approximately
300
300
randomized, diverse demonstration trajectories, totaling around
1
1
hour, are collected under cluttered scenarios, constructing a high-quality real-robot dataset of
187
187
hours and
55
​
K
55\text{K}
trajectories.
These
193
193
tasks are classified into two major categories:
•
Common Tasks
(
56
56
tasks): Everyday manipulations that are readily achievable within the current robot hardware configuration and sensor limits, yielding high teleoperation success rates.
•
Long-Tail Tasks
(
137
137
tasks): Infrequent and physically challenging manipulations, such as contact-sensitive operations lacking tactile feedback, with lower collection success rates, primarily designed to ensure comprehensive semantic coverage of the dexterous manipulation space.
Furthermore, based on motion characteristics and physical interactions, all tasks are categorized into seven classes:
•
PnP-Easy
: Single-step tabletop pick-and-place where objects are easily graspable and the placement space is open.
•
PnP-Medium
: Non-planar or 3D spatial pick-and-place that requires precise operations related to containers, demanding higher control accuracy and spatial perception, such as “
put tennis ball into ball holder
”.
•
PnP-Hard
: Multi-step or high-precision pick-and-place sequences, such as “
stack paper cups
”.
•
Non-prehensile
: Actions not involving traditional finger grasping, including pushing, pulling, and pressing.
•
Reorient
: Operations involving rotation and reorientation, such as “
pour water
” or “
flip paper cups
”.
•
Bimanual
: Bimanual tasks requiring high synchronization and spatial coordination between arms and hands, such as “
plug cable into charger
”.
•
Contact-rich
: Operations involving frequent and complex physical contact interactions, requiring physical understanding, such as “
wipe whiteboard
”.
Throughout data collection, strict requirements are enforced on the randomness, diversity, and quality of each trajectory. Specifically, the tabletop features cluttered, unstructured scenes rather than pre-arranged, simplified environments, avoiding task identification from visual inputs alone. This design forces the model to deeply align language instructions with physical actions, encouraging it to learn general task semantics and execution goals rather than memorizing demonstration trajectories. Furthermore, given the highly randomized object configurations, operators are instructed to perform teleoperation in a natural, human-like manner. Consequently, different demonstrations of the same task exhibit substantial variations in execution trajectories, thereby covering a broader distribution of the mapping from human priors to the robot’s action space.
As illustrated in
Figure
8
, we analyze the
187
187
-hour real-robot dataset across several key dimensions, including the verb and noun word clouds with their top 30 frequency distributions, the task duration breakdown across seven manipulation categories, and the detailed duration statistics for the
56
56
common and
137
137
long-tail tasks. These statistics reveal a diverse vocabulary of actions and manipulated objects, alongside a balanced distribution across both manipulation categories and task durations. Encompassing rich semantic concepts and manipulation primitives, this dataset effectively grounds pre-trained human manipulation priors onto the RealMan embodiment. Additionally,
Figure
9
showcases the dual-view image sequences and language annotations of a teleoperation trajectory, while
Figure
10
visualizes sample tasks across each category.
(a)
Word cloud of nouns in annotation
(b)
Top 30 noun frequencies
(c)
Word cloud of verbs in annotation
(d)
Top 30 verb frequencies
(e)
Task duration by seven categories
(f)
Common task duration statistics
(g)
Long-tail task duration statistics
Figure 8:
Teleoperation dataset statistics. (1)
Figure
8
-
Figure
8
: Word clouds and top-30 frequency distributions of nouns and verbs in language annotations. (2)
Figure
8
: Task duration breakdown by seven manipulation categories (PnP-Easy/Medium/Hard, Non-prehensile, Reorient, Bimanual, Contact-rich). (3)
Figure
8
and
Figure
8
: Detailed duration statistics for common tasks (56 tasks) and long-tail tasks (137 tasks).
Figure 9:
Dual-view image sequence and three-level language annotations of a representative trajectory
. The frames illustrate a cluttered, randomized setup where the operator executes the task in a natural, human-like manner. The hierarchical language annotations describe the manipulation process from coarse to fine, assisting the model in aligning free-form instructions with physical actions.
Figure 10:
Representative task examples across seven manipulation categories
. These diverse tasks are executed in cluttered, randomized scenarios using natural, human-like manipulations, covering a wide range of action semantics and manipulation primitives to facilitate the efficient grounding of pre-trained egocentric human priors onto the physical robot.
Dataset
Samples
Captioning
Visual Question
Answering
Multiple-Choice
Question
Pointing
Bounding Box
Affordance
Trajectory
Spatial
Planning
FineVision
[
56
]
3.5
​
M
3.5\text{M}
✓
✓
✓
✓
RefSpatial
[
68
]
2.5
​
M
2.5\text{M}
✓
✓
✓
✓
RoboInter-VQA
[
27
]
1.6
​
M
1.6\text{M}
✓
✓
✓
✓
✓
✓
✓
RoboPoint
[
61
]
1.3
​
M
1.3\text{M}
✓
✓
✓
✓
RoboAfford
[
50
]
765
​
K
765\text{K}
✓
✓
✓
✓
Robo2VLM
[
10
]
678
​
K
678\text{K}
✓
✓
ShareRobot
[
23
]
13
​
K
13\text{K}
✓
✓
✓
✓
✓
Table 3:
VLM datasets used for EgoSteer co-training
. This table details the sample sizes and the multi-modal and embodied knowledge domains covered by each dataset. This co-training mixture integrates general vision-language knowledge with interaction understanding and spatial-geometric priors, preserving the model’s inherent world knowledge while facilitating its ability to follow free-form instructions to generalize across diverse manipulation tasks.
Figure 11:
Statistical distribution of samples across VLM co-training datasets.
This section first introduces the VLM datasets utilized for co-training with VLA data,
i.e.
egocentric human videos and real-robot data, to preserve EgoSteer’s vision-language knowledge and ensure generalization (
Section
C.1
). We then present implementation details of EgoSteer (
Section
C.2
).
To preserve general vision-language reasoning capabilities while simultaneously cultivating robust robotic task comprehension, a
10.4
​
M
10.4\text{M}
-sample VLM co-training mixture is curated across seven datasets, ranging from open-world perception to embodied interaction grounding, to co-train EgoSteer. This mixture comprises four categories of data:
•
General VLM Pre-Training
: Incorporates FineVision
[
56
]
to prevent the catastrophic forgetting of open-world semantic concepts and preserve general visual-language reasoning.
•
Spatial Grounding
: Utilizes RefSpatial
[
68
]
and RoboPoint
[
61
]
for multi-step spatial referring, 2D visual grounding, and precise coordinate localization.
•
Embodied QA
: Integrates RoboInter-VQA
[
27
]
, Robo2VLM
[
10
]
, and ShareRobot
[
23
]
to support embodied question answering and temporal reasoning. This preserves the model’s capabilities in high-level task planning and causal scene understanding.
•
Affordance Perception
: Adopts RoboAfford
[
50
]
for fine-grained manipulation affordance prediction and spatial interaction grounding.
The sample distribution of these datasets is illustrated in
Figure
11
, with their covered multimodal, embodied knowledge domains detailed in
Table
3
. We standardize these datasets to comply with the conversational input format of Qwen3-VL. Specifically, we normalize 2D bounding box and point coordinates to Qwen3-VL’s native
[
0
,
1000
]
[0,1000]
scale and adopt its standard representation formats. To ensure training stability, we only utilize single-image samples and exclude those with context lengths exceeding our maximum budget. By co-training on this VLM mixture, EgoSteer preserves open-world vision-language understanding and reasoning while enhancing its comprehension of robotics-specific tasks, thereby assisting the policy in following open-ended instructions and generalizing to novel manipulation scenarios.
EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.
The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.
The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.
The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.
Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.
To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.
Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.
This section provides supplementary details for the experimental evaluations presented in
Section
6
, detailing the model training hyperparameters and per-task success rates. In particular,
Table
4
lists the hyperparameter configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
Table
5
presents the per-task success rates on the four evaluation tasks in
Section
6.2
to demonstrate the performance gains from DAgger refinement. For the scaling analysis in
Section
6.3
,
Table
6
lists the training hyperparameters of EgoSteer-3K/6K/9.6K pre-trained models, their respective post-training runs, and the baseline trained from scratch, with their corresponding task-specific success rates reported in
Table
7
. Additionally,
Table
8
compares the detailed success rates of our method against the Being-H0.5 and
π
0.5
\pi_{0.5}
baselines across ten tasks.
Table
10
specifies the training configurations of the ablation experiment from
Section
6.4
, including the EgoSteer-1K model and its three ablated variants,
No WM-objective
,
No training-RTC
, and
Noisy data
. Their respective performance comparison is provided in
Table
9
. Finally,
Table
11
specifies the hyperparameters for the few-shot fine-tuning of our pre-trained EgoSteer-9.6K on the two challenging, long-horizon tasks evaluated in
Section
6.5
, namely
Box-Folding
and
Cake-Unboxing
.
Hyperparameter
Pre-Training
Post-Training
Camera setup
Head
Head & Chest
Resolution
384
×
384
384\times 384
640
×
480
640\times 480
GPUs
128 A800
96 A800
Gradient accumulation
2
1
Global batch size
4608
384
Training steps
175K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Freeze-VLM steps
5000
0
Warmup steps
2000
2000
Training time
164 h
29 h
Table 4:
Training configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
During pre-training, the VLM backbone is frozen for the first 5,000 steps, during which AE and WM are warmed up for 2,000 steps; once it is unfrozen, VLM is warmed up for 2,000 steps.
Task
EgoSteer-DG
EgoSteer-FT
Stack tableware
80.0%
50.0%
Close laptop
70.0%
10.0%
Place phone on stand
50.0%
0.0%
Flip cup
50.0%
30.0%
Average
62.5%
22.5%
Table 5:
Success rates across four highly dexterous and failure-prone manipulation tasks, comparing EgoSteer-DG against EgoSteer-FT in
Section
6.2
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Camera setup
Head
Head
Head
Head & Chest
Head & Chest
Head & Chest
Head & Chest
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
GPUs
64 A800
64 A800
128 A800
64 A800
32 A800
64 A800
64 A800
Gradient accumulation
2
4
2
2
4
2
2
Global batch size
2304
4608
4608
512
512
512
512
Training steps
100K
100K
160K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 6:
Training configurations for the pre-training scaling study in
Section
6.3
.
Pre-Training Data (Hours)
Task
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Grasp object
80%
80%
70%
100%
Hand over object
70%
80%
80%
100%
Place items into container
40%
80%
90%
100%
Point at object
20%
40%
60%
70%
Place toy chick into slot
30%
0%
20%
40%
Pull out tissue
60%
20%
50%
80%
Push ball into box
0%
0%
10%
20%
Put garbage into trash bin
0%
60%
30%
30%
Stack paper cups
0%
20%
10%
40%
Stack tableware
0%
20%
10%
20%
Average
30%
40%
43%
60%
Table 7:
Per-task success rates for the pre-training scaling study in
Section
6.3
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Task
Ours
Being-H0.5
π
0.5
\pi_{0.5}
Grasp object
100%
80%
80%
Hand over object
100%
60%
20%
Place items into container
100%
50%
0%
Pour items out of box
50%
30%
0%
Place bread on tray
70%
80%
40%
Pull out tissue
80%
10%
30%
Place item at specific orientation
30%
30%
30%
Attach eraser to whiteboard
90%
50%
20%
Put garbage into trash bin
30%
0%
0%
Put tennis ball into bucket
90%
0%
0%
Average
74%
39%
22%
Table 8:
Per-task comparison between EgoSteer-9.6K and two VLA baselines, Being-H0.5 and
π
0.5
\pi_{0.5}
, in
Section
6.3
. All methods are post-trained on our real-robot dataset and evaluated on the same 10 tasks. Each task uses 10 randomized trials. Bold values highlight the best performance.
Task
Ours
No WM-objective
No training-RTC
Noisy data
Grasp object
60%
40%
30%
60%
Hand over object
40%
20%
30%
40%
Place items into container
50%
30%
50%
70%
Pour items out of box
60%
50%
10%
0%
Place bread on tray
70%
10%
60%
50%
Pull out tissue
20%
30%
60%
10%
Place item at specific orientation
20%
10%
10%
20%
Attach eraser to whiteboard
50%
80%
50%
40%
Put garbage into trash bin
30%
0%
0%
0%
Put tennis ball into bucket
40%
40%
90%
40%
Average
44%
31%
39%
33%
Table 9:
Per-task success rates for the ablation study in
Section
6.4
. EgoSteer-1K is compared with ablated variants. Each task uses 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
Ours
No WM-objective
No training-RTC
Noisy data
Ours
No WM-objective
No training-RTC
Noisy data
Camera setup
Head
Head
Head
Head
Head
Head
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
Global batch size
1152
1152
1152
1152
1152
1152
1152
1152
Training steps
30K
80K
30K
20K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 10:
Training hyperparameters for the ablation study in
Section
6.4
. All variants are pre-trained on 1K hours of egocentric data and post-trained on the same real-robot dataset, without DAgger refinement. Pre-training steps are selected at the lowest evaluation
L
1
L_{1}
loss for each pre-training run.
Hyperparameter
Box-Folding
Cake-Unboxing
Pre-training checkpoint
EgoSteer-9.6K at 155K steps
Camera setup
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
8 A800
8 A800
Gradient accumulation
1
1
Global batch size
144
144
Fine-tuning steps
44K
12K
Learning rate (VLM / AE / WM)
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Demonstrations
120
229
Table 11:
Few-shot fine-tuning hyperparameters of EgoSteer-9.6K for the two long-horizon dexterous tasks in
Section
6.5
.

### B.2
Teleoperation Data Collection

Although pre-training on egocentric human videos equips the model with rich dexterous manipulation priors, direct physical deployment is prevented by the embodiment gap across visual appearance, dynamics, and kinematics. Furthermore, because these pre-training labels are generated by automated depth and hand pose estimation models, their accuracy is constrained by current model capabilities, potentially introducing systematic biases into the pre-trained policy. Therefore, collecting real-robot teleoperation data aims to correct and ground these priors onto the target embodiment with high sample efficiency. Additionally, due to kinematic constraints on robotic degrees of freedom, the robot cannot perfectly replicate all fine-grained human hand movements. To achieve steerable manipulation, we must collect a highly diverse range of free-form tasks to maximize the coverage of basic manipulation primitives, guiding the model to efficiently transfer cross-domain knowledge and fostering robust compositional generalization and multi-task instruction following.
Based on these considerations, within the kinematic limits of RealMan’s degrees of freedom, we design
193
193
semantically distinct dexterous manipulation tasks. For each task, approximately
300
300
randomized, diverse demonstration trajectories, totaling around
1
1
hour, are collected under cluttered scenarios, constructing a high-quality real-robot dataset of
187
187
hours and
55
​
K
55\text{K}
trajectories.
These
193
193
tasks are classified into two major categories:
•
Common Tasks
(
56
56
tasks): Everyday manipulations that are readily achievable within the current robot hardware configuration and sensor limits, yielding high teleoperation success rates.
•
Long-Tail Tasks
(
137
137
tasks): Infrequent and physically challenging manipulations, such as contact-sensitive operations lacking tactile feedback, with lower collection success rates, primarily designed to ensure comprehensive semantic coverage of the dexterous manipulation space.
Furthermore, based on motion characteristics and physical interactions, all tasks are categorized into seven classes:
•
PnP-Easy
: Single-step tabletop pick-and-place where objects are easily graspable and the placement space is open.
•
PnP-Medium
: Non-planar or 3D spatial pick-and-place that requires precise operations related to containers, demanding higher control accuracy and spatial perception, such as “
put tennis ball into ball holder
”.
•
PnP-Hard
: Multi-step or high-precision pick-and-place sequences, such as “
stack paper cups
”.
•
Non-prehensile
: Actions not involving traditional finger grasping, including pushing, pulling, and pressing.
•
Reorient
: Operations involving rotation and reorientation, such as “
pour water
” or “
flip paper cups
”.
•
Bimanual
: Bimanual tasks requiring high synchronization and spatial coordination between arms and hands, such as “
plug cable into charger
”.
•
Contact-rich
: Operations involving frequent and complex physical contact interactions, requiring physical understanding, such as “
wipe whiteboard
”.
Throughout data collection, strict requirements are enforced on the randomness, diversity, and quality of each trajectory. Specifically, the tabletop features cluttered, unstructured scenes rather than pre-arranged, simplified environments, avoiding task identification from visual inputs alone. This design forces the model to deeply align language instructions with physical actions, encouraging it to learn general task semantics and execution goals rather than memorizing demonstration trajectories. Furthermore, given the highly randomized object configurations, operators are instructed to perform teleoperation in a natural, human-like manner. Consequently, different demonstrations of the same task exhibit substantial variations in execution trajectories, thereby covering a broader distribution of the mapping from human priors to the robot’s action space.
As illustrated in
Figure
8
, we analyze the
187
187
-hour real-robot dataset across several key dimensions, including the verb and noun word clouds with their top 30 frequency distributions, the task duration breakdown across seven manipulation categories, and the detailed duration statistics for the
56
56
common and
137
137
long-tail tasks. These statistics reveal a diverse vocabulary of actions and manipulated objects, alongside a balanced distribution across both manipulation categories and task durations. Encompassing rich semantic concepts and manipulation primitives, this dataset effectively grounds pre-trained human manipulation priors onto the RealMan embodiment. Additionally,
Figure
9
showcases the dual-view image sequences and language annotations of a teleoperation trajectory, while
Figure
10
visualizes sample tasks across each category.
(a)
Word cloud of nouns in annotation
(b)
Top 30 noun frequencies
(c)
Word cloud of verbs in annotation
(d)
Top 30 verb frequencies
(e)
Task duration by seven categories
(f)
Common task duration statistics
(g)
Long-tail task duration statistics
Figure 8:
Teleoperation dataset statistics. (1)
Figure
8
-
Figure
8
: Word clouds and top-30 frequency distributions of nouns and verbs in language annotations. (2)
Figure
8
: Task duration breakdown by seven manipulation categories (PnP-Easy/Medium/Hard, Non-prehensile, Reorient, Bimanual, Contact-rich). (3)
Figure
8
and
Figure
8
: Detailed duration statistics for common tasks (56 tasks) and long-tail tasks (137 tasks).
Figure 9:
Dual-view image sequence and three-level language annotations of a representative trajectory
. The frames illustrate a cluttered, randomized setup where the operator executes the task in a natural, human-like manner. The hierarchical language annotations describe the manipulation process from coarse to fine, assisting the model in aligning free-form instructions with physical actions.
Figure 10:
Representative task examples across seven manipulation categories
. These diverse tasks are executed in cluttered, randomized scenarios using natural, human-like manipulations, covering a wide range of action semantics and manipulation primitives to facilitate the efficient grounding of pre-trained egocentric human priors onto the physical robot.
Dataset
Samples
Captioning
Visual Question
Answering
Multiple-Choice
Question
Pointing
Bounding Box
Affordance
Trajectory
Spatial
Planning
FineVision
[
56
]
3.5
​
M
3.5\text{M}
✓
✓
✓
✓
RefSpatial
[
68
]
2.5
​
M
2.5\text{M}
✓
✓
✓
✓
RoboInter-VQA
[
27
]
1.6
​
M
1.6\text{M}
✓
✓
✓
✓
✓
✓
✓
RoboPoint
[
61
]
1.3
​
M
1.3\text{M}
✓
✓
✓
✓
RoboAfford
[
50
]
765
​
K
765\text{K}
✓
✓
✓
✓
Robo2VLM
[
10
]
678
​
K
678\text{K}
✓
✓
ShareRobot
[
23
]
13
​
K
13\text{K}
✓
✓
✓
✓
✓
Table 3:
VLM datasets used for EgoSteer co-training
. This table details the sample sizes and the multi-modal and embodied knowledge domains covered by each dataset. This co-training mixture integrates general vision-language knowledge with interaction understanding and spatial-geometric priors, preserving the model’s inherent world knowledge while facilitating its ability to follow free-form instructions to generalize across diverse manipulation tasks.
Figure 11:
Statistical distribution of samples across VLM co-training datasets.
This section first introduces the VLM datasets utilized for co-training with VLA data,
i.e.
egocentric human videos and real-robot data, to preserve EgoSteer’s vision-language knowledge and ensure generalization (
Section
C.1
). We then present implementation details of EgoSteer (
Section
C.2
).
To preserve general vision-language reasoning capabilities while simultaneously cultivating robust robotic task comprehension, a
10.4
​
M
10.4\text{M}
-sample VLM co-training mixture is curated across seven datasets, ranging from open-world perception to embodied interaction grounding, to co-train EgoSteer. This mixture comprises four categories of data:
•
General VLM Pre-Training
: Incorporates FineVision
[
56
]
to prevent the catastrophic forgetting of open-world semantic concepts and preserve general visual-language reasoning.
•
Spatial Grounding
: Utilizes RefSpatial
[
68
]
and RoboPoint
[
61
]
for multi-step spatial referring, 2D visual grounding, and precise coordinate localization.
•
Embodied QA
: Integrates RoboInter-VQA
[
27
]
, Robo2VLM
[
10
]
, and ShareRobot
[
23
]
to support embodied question answering and temporal reasoning. This preserves the model’s capabilities in high-level task planning and causal scene understanding.
•
Affordance Perception
: Adopts RoboAfford
[
50
]
for fine-grained manipulation affordance prediction and spatial interaction grounding.
The sample distribution of these datasets is illustrated in
Figure
11
, with their covered multimodal, embodied knowledge domains detailed in
Table
3
. We standardize these datasets to comply with the conversational input format of Qwen3-VL. Specifically, we normalize 2D bounding box and point coordinates to Qwen3-VL’s native
[
0
,
1000
]
[0,1000]
scale and adopt its standard representation formats. To ensure training stability, we only utilize single-image samples and exclude those with context lengths exceeding our maximum budget. By co-training on this VLM mixture, EgoSteer preserves open-world vision-language understanding and reasoning while enhancing its comprehension of robotics-specific tasks, thereby assisting the policy in following open-ended instructions and generalizing to novel manipulation scenarios.
EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.
The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.
The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.
The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.
Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.
To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.
Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.
This section provides supplementary details for the experimental evaluations presented in
Section
6
, detailing the model training hyperparameters and per-task success rates. In particular,
Table
4
lists the hyperparameter configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
Table
5
presents the per-task success rates on the four evaluation tasks in
Section
6.2
to demonstrate the performance gains from DAgger refinement. For the scaling analysis in
Section
6.3
,
Table
6
lists the training hyperparameters of EgoSteer-3K/6K/9.6K pre-trained models, their respective post-training runs, and the baseline trained from scratch, with their corresponding task-specific success rates reported in
Table
7
. Additionally,
Table
8
compares the detailed success rates of our method against the Being-H0.5 and
π
0.5
\pi_{0.5}
baselines across ten tasks.
Table
10
specifies the training configurations of the ablation experiment from
Section
6.4
, including the EgoSteer-1K model and its three ablated variants,
No WM-objective
,
No training-RTC
, and
Noisy data
. Their respective performance comparison is provided in
Table
9
. Finally,
Table
11
specifies the hyperparameters for the few-shot fine-tuning of our pre-trained EgoSteer-9.6K on the two challenging, long-horizon tasks evaluated in
Section
6.5
, namely
Box-Folding
and
Cake-Unboxing
.
Hyperparameter
Pre-Training
Post-Training
Camera setup
Head
Head & Chest
Resolution
384
×
384
384\times 384
640
×
480
640\times 480
GPUs
128 A800
96 A800
Gradient accumulation
2
1
Global batch size
4608
384
Training steps
175K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Freeze-VLM steps
5000
0
Warmup steps
2000
2000
Training time
164 h
29 h
Table 4:
Training configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
During pre-training, the VLM backbone is frozen for the first 5,000 steps, during which AE and WM are warmed up for 2,000 steps; once it is unfrozen, VLM is warmed up for 2,000 steps.
Task
EgoSteer-DG
EgoSteer-FT
Stack tableware
80.0%
50.0%
Close laptop
70.0%
10.0%
Place phone on stand
50.0%
0.0%
Flip cup
50.0%
30.0%
Average
62.5%
22.5%
Table 5:
Success rates across four highly dexterous and failure-prone manipulation tasks, comparing EgoSteer-DG against EgoSteer-FT in
Section
6.2
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Camera setup
Head
Head
Head
Head & Chest
Head & Chest
Head & Chest
Head & Chest
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
GPUs
64 A800
64 A800
128 A800
64 A800
32 A800
64 A800
64 A800
Gradient accumulation
2
4
2
2
4
2
2
Global batch size
2304
4608
4608
512
512
512
512
Training steps
100K
100K
160K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 6:
Training configurations for the pre-training scaling study in
Section
6.3
.
Pre-Training Data (Hours)
Task
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Grasp object
80%
80%
70%
100%
Hand over object
70%
80%
80%
100%
Place items into container
40%
80%
90%
100%
Point at object
20%
40%
60%
70%
Place toy chick into slot
30%
0%
20%
40%
Pull out tissue
60%
20%
50%
80%
Push ball into box
0%
0%
10%
20%
Put garbage into trash bin
0%
60%
30%
30%
Stack paper cups
0%
20%
10%
40%
Stack tableware
0%
20%
10%
20%
Average
30%
40%
43%
60%
Table 7:
Per-task success rates for the pre-training scaling study in
Section
6.3
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Task
Ours
Being-H0.5
π
0.5
\pi_{0.5}
Grasp object
100%
80%
80%
Hand over object
100%
60%
20%
Place items into container
100%
50%
0%
Pour items out of box
50%
30%
0%
Place bread on tray
70%
80%
40%
Pull out tissue
80%
10%
30%
Place item at specific orientation
30%
30%
30%
Attach eraser to whiteboard
90%
50%
20%
Put garbage into trash bin
30%
0%
0%
Put tennis ball into bucket
90%
0%
0%
Average
74%
39%
22%
Table 8:
Per-task comparison between EgoSteer-9.6K and two VLA baselines, Being-H0.5 and
π
0.5
\pi_{0.5}
, in
Section
6.3
. All methods are post-trained on our real-robot dataset and evaluated on the same 10 tasks. Each task uses 10 randomized trials. Bold values highlight the best performance.
Task
Ours
No WM-objective
No training-RTC
Noisy data
Grasp object
60%
40%
30%
60%
Hand over object
40%
20%
30%
40%
Place items into container
50%
30%
50%
70%
Pour items out of box
60%
50%
10%
0%
Place bread on tray
70%
10%
60%
50%
Pull out tissue
20%
30%
60%
10%
Place item at specific orientation
20%
10%
10%
20%
Attach eraser to whiteboard
50%
80%
50%
40%
Put garbage into trash bin
30%
0%
0%
0%
Put tennis ball into bucket
40%
40%
90%
40%
Average
44%
31%
39%
33%
Table 9:
Per-task success rates for the ablation study in
Section
6.4
. EgoSteer-1K is compared with ablated variants. Each task uses 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
Ours
No WM-objective
No training-RTC
Noisy data
Ours
No WM-objective
No training-RTC
Noisy data
Camera setup
Head
Head
Head
Head
Head
Head
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
Global batch size
1152
1152
1152
1152
1152
1152
1152
1152
Training steps
30K
80K
30K
20K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 10:
Training hyperparameters for the ablation study in
Section
6.4
. All variants are pre-trained on 1K hours of egocentric data and post-trained on the same real-robot dataset, without DAgger refinement. Pre-training steps are selected at the lowest evaluation
L
1
L_{1}
loss for each pre-training run.
Hyperparameter
Box-Folding
Cake-Unboxing
Pre-training checkpoint
EgoSteer-9.6K at 155K steps
Camera setup
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
8 A800
8 A800
Gradient accumulation
1
1
Global batch size
144
144
Fine-tuning steps
44K
12K
Learning rate (VLM / AE / WM)
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Demonstrations
120
229
Table 11:
Few-shot fine-tuning hyperparameters of EgoSteer-9.6K for the two long-horizon dexterous tasks in
Section
6.5
.

## Appendix C
Details of EgoSteer

Figure 11:
Statistical distribution of samples across VLM co-training datasets.
This section first introduces the VLM datasets utilized for co-training with VLA data,
i.e.
egocentric human videos and real-robot data, to preserve EgoSteer’s vision-language knowledge and ensure generalization (
Section
C.1
). We then present implementation details of EgoSteer (
Section
C.2
).
To preserve general vision-language reasoning capabilities while simultaneously cultivating robust robotic task comprehension, a
10.4
​
M
10.4\text{M}
-sample VLM co-training mixture is curated across seven datasets, ranging from open-world perception to embodied interaction grounding, to co-train EgoSteer. This mixture comprises four categories of data:
•
General VLM Pre-Training
: Incorporates FineVision
[
56
]
to prevent the catastrophic forgetting of open-world semantic concepts and preserve general visual-language reasoning.
•
Spatial Grounding
: Utilizes RefSpatial
[
68
]
and RoboPoint
[
61
]
for multi-step spatial referring, 2D visual grounding, and precise coordinate localization.
•
Embodied QA
: Integrates RoboInter-VQA
[
27
]
, Robo2VLM
[
10
]
, and ShareRobot
[
23
]
to support embodied question answering and temporal reasoning. This preserves the model’s capabilities in high-level task planning and causal scene understanding.
•
Affordance Perception
: Adopts RoboAfford
[
50
]
for fine-grained manipulation affordance prediction and spatial interaction grounding.
The sample distribution of these datasets is illustrated in
Figure
11
, with their covered multimodal, embodied knowledge domains detailed in
Table
3
. We standardize these datasets to comply with the conversational input format of Qwen3-VL. Specifically, we normalize 2D bounding box and point coordinates to Qwen3-VL’s native
[
0
,
1000
]
[0,1000]
scale and adopt its standard representation formats. To ensure training stability, we only utilize single-image samples and exclude those with context lengths exceeding our maximum budget. By co-training on this VLM mixture, EgoSteer preserves open-world vision-language understanding and reasoning while enhancing its comprehension of robotics-specific tasks, thereby assisting the policy in following open-ended instructions and generalizing to novel manipulation scenarios.
EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.
The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.
The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.
The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.
Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.
To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.
Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.

### C.1
VLM Co-Training Data

To preserve general vision-language reasoning capabilities while simultaneously cultivating robust robotic task comprehension, a
10.4
​
M
10.4\text{M}
-sample VLM co-training mixture is curated across seven datasets, ranging from open-world perception to embodied interaction grounding, to co-train EgoSteer. This mixture comprises four categories of data:
•
General VLM Pre-Training
: Incorporates FineVision
[
56
]
to prevent the catastrophic forgetting of open-world semantic concepts and preserve general visual-language reasoning.
•
Spatial Grounding
: Utilizes RefSpatial
[
68
]
and RoboPoint
[
61
]
for multi-step spatial referring, 2D visual grounding, and precise coordinate localization.
•
Embodied QA
: Integrates RoboInter-VQA
[
27
]
, Robo2VLM
[
10
]
, and ShareRobot
[
23
]
to support embodied question answering and temporal reasoning. This preserves the model’s capabilities in high-level task planning and causal scene understanding.
•
Affordance Perception
: Adopts RoboAfford
[
50
]
for fine-grained manipulation affordance prediction and spatial interaction grounding.
The sample distribution of these datasets is illustrated in
Figure
11
, with their covered multimodal, embodied knowledge domains detailed in
Table
3
. We standardize these datasets to comply with the conversational input format of Qwen3-VL. Specifically, we normalize 2D bounding box and point coordinates to Qwen3-VL’s native
[
0
,
1000
]
[0,1000]
scale and adopt its standard representation formats. To ensure training stability, we only utilize single-image samples and exclude those with context lengths exceeding our maximum budget. By co-training on this VLM mixture, EgoSteer preserves open-world vision-language understanding and reasoning while enhancing its comprehension of robotics-specific tasks, thereby assisting the policy in following open-ended instructions and generalizing to novel manipulation scenarios.

### C.2
Implementation Details

EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.
The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.
The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.
The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.
Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.
To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.
Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.

##### Backbone Input Scheme.

EgoSteer treats the image observation history as a temporal video sequence to leverage the native video-processing capabilities of the Qwen3-VL-2B backbone. In practice, this history is downsampled to
6
6
frames at
1
1
FPS, covering a
5
5
s window, with proprioceptive states sampled at the corresponding timestamps. Language instructions and camera intrinsics are formatted as textual inputs, while the proprioceptive state history is encoded by a two-layer MLP and injected as continuous tokens. Because this proprioceptive history correlates strongly with target actions, the model is susceptible to shortcut learning, tending to ignore visual inputs and task instructions in favor of proprioception. To mitigate this, each frame of the proprioceptive history is replaced by a learnable mask token with a
75
%
75\%
probability during training, forcing the model to attend to the full multimodal context. Additionally, when utilizing dual-camera inputs during real-robot post-training, the chest-camera sequence is randomly dropped with a
50
%
50\%
probability to prevent over-reliance on chest-view observations.

##### Action Expert.

The action expert is built on a DiT architecture, comprising
14
14
layers with a hidden dimension of
1024
1024
and an intermediate size of
2816
2816
. It features
8
8
attention heads, each with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
300
300
M parameters.
The action expert operates on continuous action chunks of length
h
=
32
h=32
at a frequency of
30
30
Hz. Consistent with the VLM backbone, we apply Interleaved MRoPE for positional encoding. During pre-training, we set the delay
d
=
0
d=0
to disable training-time RTC, maximizing action supervision signals and learning rich human manipulation priors. During real-robot post-training, the simulated delay
d
d
is uniformly sampled as
d
∼
𝒰
​
(
[
0
,
5
]
)
d\sim\mathcal{U}([0,5])
to accommodate varying inference latencies during deployment. For the flow matching timestep
η
∈
[
0
,
1
]
\eta\in[0,1]
, the prefix action
𝐚
pre
\mathbf{a}_{\text{pre}}
is assigned
η
=
1
\eta=1
, representing no noise and excluding it from the loss computation. Conversely, the target suffix action
𝐚
suf
\mathbf{a}_{\text{suf}}
has its timestep sampled from the probability distribution
P
​
(
η
)
=
Beta
​
(
s
−
η
s
;
1.5
,
1
)
P(\eta)=\text{Beta}(\frac{s-\eta}{s};1.5,1)
with
s
=
0.999
s=0.999
[
5
]
. This timestep
η
\eta
is sinusoidally encoded, mapped through a two-layer MLP, and injected into the action expert via AdaLN-Zero.
Layer
ℓ
\ell
of the action expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
2
​
ℓ
f(\ell)=2\ell
after a linear projection. Specifically, let the query, key, and value of the
m
m
-th attention head in layer
ℓ
\ell
of the action expert be
𝐐
ℓ
,
m
AE
,
𝐊
ℓ
,
m
AE
,
𝐕
ℓ
,
m
AE
∈
ℝ
h
×
d
head
\mathbf{Q}_{\ell,m}^{\text{AE}},\mathbf{K}_{\ell,m}^{\text{AE}},\mathbf{V}_{\ell,m}^{\text{AE}}\in\mathbb{R}^{h\times d_{\text{head}}}
, and the corresponding key and value of the
m
m
-th head in layer
f
​
(
ℓ
)
f(\ell)
of the backbone be
𝐊
f
​
(
ℓ
)
,
m
B
,
𝐕
f
​
(
ℓ
)
,
m
B
∈
ℝ
N
B
×
d
head
\mathbf{K}_{f(\ell),m}^{\text{B}},\mathbf{V}_{f(\ell),m}^{\text{B}}\in\mathbb{R}^{N_{\text{B}}\times d_{\text{head}}}
, where
N
B
N_{\text{B}}
denotes the backbone’s input sequence length. The joint attention at layer
ℓ
\ell
for the
m
m
-th head is mathematically formulated as:
Softmax
​
(
1
d
head
​
𝐐
ℓ
,
m
AE
​
(
concat
​
[
𝐊
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
K
,
𝐊
ℓ
,
m
AE
]
)
T
)
​
concat
​
[
𝐕
f
​
(
ℓ
)
,
m
B
​
𝐖
ℓ
V
,
𝐕
ℓ
,
m
AE
]
,
\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{K}},\mathbf{K}_{\ell,m}^{\text{AE}}]\right)^{\text{T}}\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_{\ell}^{\text{V}},\mathbf{V}_{\ell,m}^{\text{AE}}],
(2)
where
𝐖
ℓ
K
,
𝐖
ℓ
V
∈
ℝ
d
head
×
d
head
\mathbf{W}_{\ell}^{\text{K}},\mathbf{W}_{\ell}^{\text{V}}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{head}}}
represent learnable projection matrices. This linear projection on the backbone’s key and value representations is designed to align the semantic spaces of the backbone and the action expert.
During real-world deployment, the simulated delay is set to
d
=
4
d=4
to cover physical inference latency. To achieve efficient closed-loop control, only the first
12
12
steps of the predicted
32
32
-step action chunk are retained. Subtracting the
4
4
prefix steps used for latency conditioning, the robot actually executes
8
8
new action steps per inference cycle. This high-frequency, asynchronous execution enables highly responsive control, rendering the system robust to dynamic manipulation tasks.

##### World Model Expert.

The world model expert is a lightweight Transformer comprising
4
4
layers, with its single-layer architecture identical to Qwen3’s text layer. It has a hidden dimension of
1024
1024
, an intermediate size of
4096
4096
, and
8
8
attention heads with a head dimension
d
head
=
128
d_{\text{head}}=128
, totaling approximately
70
70
M parameters. At layer
ℓ
\ell
, the world-model expert employs a joint attention mechanism, attending to both itself and the VLM backbone’s key-value cache from layer
f
​
(
ℓ
)
=
7
​
ℓ
f(\ell)=7\ell
after a linear projection, consistent with the formulation of the action expert.
For inputs, the relative camera motion
Δ
​
𝐓
∈
S
​
E
​
(
3
)
\Delta\mathbf{T}\in SE(3)
is flattened into a
16
16
-dimensional vector and encoded into a single continuous token via a two-layer MLP. Let the ground-truth DINOv3 features of the future frame
𝐈
t
+
h
−
1
\mathbf{I}_{t+h-1}
be
𝐙
∈
ℝ
H
v
×
W
v
×
C
DINO
\mathbf{Z}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. We utilize DINOv3 (ViT-L/16)
[
49
]
for feature extraction with an input resolution of
384
×
384
384\times 384
, yielding a spatial resolution of
H
v
=
W
v
=
24
H_{v}=W_{v}=24
and a feature dimension of
C
DINO
=
1024
C_{\text{DINO}}=1024
. To align with the spatial token-merge format of the backbone, the sequence length
L
𝐳
L_{\mathbf{z}}
of the input query vector
𝐳
\mathbf{z}
is configured to match the merged spatial resolution:
L
𝐳
=
H
v
′
×
W
v
′
=
H
v
2
×
W
v
2
=
144
L_{\mathbf{z}}=H^{\prime}_{v}\times W^{\prime}_{v}=\frac{H_{v}}{2}\times\frac{W_{v}}{2}=144
. The world model expert outputs
𝐘
^
∈
ℝ
H
v
′
×
W
v
′
×
d
WM
\hat{\mathbf{Y}}\in\mathbb{R}^{H^{\prime}_{v}\times W^{\prime}_{v}\times d_{\text{WM}}}
with a channel dimension
d
WM
=
1024
d_{\text{WM}}=1024
, which is subsequently mapped back to the original DINOv3 spatial resolution through a
2
×
2
2\times 2
linear upsampling projection layer, yielding the reconstructed feature map
𝐙
^
∈
ℝ
H
v
×
W
v
×
C
DINO
\hat{\mathbf{Z}}\in\mathbb{R}^{H_{v}\times W_{v}\times C_{\text{DINO}}}
. The world model objective is optimized via a mean squared error (MSE) loss:
ℒ
WM
=
1
H
v
⋅
W
v
​
∑
u
=
1
H
v
∑
v
=
1
W
v
‖
𝐙
u
,
v
−
𝐙
^
u
,
v
‖
2
2
.
\mathcal{L}_{\text{WM}}=\frac{1}{H_{v}\cdot W_{v}}\sum_{u=1}^{H_{v}}\sum_{v=1}^{W_{v}}\left\|\mathbf{Z}_{u,v}-\hat{\mathbf{Z}}_{u,v}\right\|_{2}^{2}.
(3)
Compared to direct image prediction in pixel space, regressing DINOv3 features preserves rich semantic information, naturally filtering out lighting variations and background noise to provide more stable gradient guidance for the VLM backbone. Furthermore, the world model expert adopts Interleaved MRoPE positional encoding, consistent with the VLM backbone, which enhances spatial-temporal awareness of multimodal sequences.

##### Attention Pattern.

The Qwen3-VL backbone employs causal attention. Both the action expert and the world-model expert jointly attend to their entire respective sequences and the entire backbone sequence. Crucially, the action expert and the world-model expert do not attend to each other.

##### Data Processing.

Due to significant variations in scale and quality across the 12 egocentric pre-training datasets, a heuristic sampling weight scheme is employed to balance different data sources. Specifically, each dataset
i
i
is assigned a subjective quality score
w
i
∈
[
1
,
10
]
w_{i}\in[1,10]
based on data quality. To mitigate scale discrepancies, this score is scaled by the square root of the total frame count
n
i
n_{i}
, yielding the final sampling weight
W
i
=
w
i
⋅
n
i
0.5
W_{i}=w_{i}\cdot n_{i}^{0.5}
. For data augmentation, ColorJitter is applied to the input images. Furthermore, all action dimensions, except for wrist rotations, are normalized to the range
[
−
1
,
1
]
[-1,1]
using their 1st and 99th percentiles estimated from a randomly sampled subset.

##### Joint Optimization Objective.

To balance the primary flow-matching task and the two auxiliary targets, the total training loss
ℒ
total
\mathcal{L}_{\text{total}}
is defined as:
ℒ
total
=
ℒ
CFM
+
ℒ
WM
+
0.05
​
ℒ
VLM
,
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CFM}}+\mathcal{L}_{\text{WM}}+0.05\mathcal{L}_{\text{VLM}},
(4)
where
ℒ
CFM
\mathcal{L}_{\text{CFM}}
is the action flow-matching loss,
ℒ
WM
\mathcal{L}_{\text{WM}}
is the world-model feature regression loss, and
ℒ
VLM
\mathcal{L}_{\text{VLM}}
is the autoregressive next-token prediction loss of the VLM. The loss weights are set to align the numerical scales of the three loss terms.

##### Training Infrastructure.

Both the pre-training corpus curated by EgoSmith and the real-robot dataset are stored sequentially as individual episodes in the WebDataset format. During data loading, training batches are randomly sampled from a large shuffle buffer of size
16
,
384
16,384
. This buffer is continuously filled by drawing samples from each dataset proportionally to its pre-defined weight. For each dataset, the data stream is maintained by randomly selecting and streaming WebDataset shards, caching samples via a sliding window, and applying a
20
%
20\%
random retention probability. By combining randomized shard reading, sample dropping, and large-buffer shuffling, this scheme ensures training randomness while leveraging the sequential streaming of WebDataset, thereby drastically reducing I/O pressure. Furthermore, in multi-node distributed training, we explicitly manage Python’s garbage collection to effectively mitigate training speed fluctuations and synchronization jitter across the cluster, ensuring high efficiency and stability.

## Appendix D
Experimental Details

This section provides supplementary details for the experimental evaluations presented in
Section
6
, detailing the model training hyperparameters and per-task success rates. In particular,
Table
4
lists the hyperparameter configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
Table
5
presents the per-task success rates on the four evaluation tasks in
Section
6.2
to demonstrate the performance gains from DAgger refinement. For the scaling analysis in
Section
6.3
,
Table
6
lists the training hyperparameters of EgoSteer-3K/6K/9.6K pre-trained models, their respective post-training runs, and the baseline trained from scratch, with their corresponding task-specific success rates reported in
Table
7
. Additionally,
Table
8
compares the detailed success rates of our method against the Being-H0.5 and
π
0.5
\pi_{0.5}
baselines across ten tasks.
Table
10
specifies the training configurations of the ablation experiment from
Section
6.4
, including the EgoSteer-1K model and its three ablated variants,
No WM-objective
,
No training-RTC
, and
Noisy data
. Their respective performance comparison is provided in
Table
9
. Finally,
Table
11
specifies the hyperparameters for the few-shot fine-tuning of our pre-trained EgoSteer-9.6K on the two challenging, long-horizon tasks evaluated in
Section
6.5
, namely
Box-Folding
and
Cake-Unboxing
.
Hyperparameter
Pre-Training
Post-Training
Camera setup
Head
Head & Chest
Resolution
384
×
384
384\times 384
640
×
480
640\times 480
GPUs
128 A800
96 A800
Gradient accumulation
2
1
Global batch size
4608
384
Training steps
175K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Freeze-VLM steps
5000
0
Warmup steps
2000
2000
Training time
164 h
29 h
Table 4:
Training configurations for both the pre-training and post-training phases in the main experiments of EgoSteer in
Section
6.1
.
During pre-training, the VLM backbone is frozen for the first 5,000 steps, during which AE and WM are warmed up for 2,000 steps; once it is unfrozen, VLM is warmed up for 2,000 steps.
Task
EgoSteer-DG
EgoSteer-FT
Stack tableware
80.0%
50.0%
Close laptop
70.0%
10.0%
Place phone on stand
50.0%
0.0%
Flip cup
50.0%
30.0%
Average
62.5%
22.5%
Table 5:
Success rates across four highly dexterous and failure-prone manipulation tasks, comparing EgoSteer-DG against EgoSteer-FT in
Section
6.2
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Camera setup
Head
Head
Head
Head & Chest
Head & Chest
Head & Chest
Head & Chest
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
640
×
480
640\times 480
GPUs
64 A800
64 A800
128 A800
64 A800
32 A800
64 A800
64 A800
Gradient accumulation
2
4
2
2
4
2
2
Global batch size
2304
4608
4608
512
512
512
512
Training steps
100K
100K
160K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 6:
Training configurations for the pre-training scaling study in
Section
6.3
.
Pre-Training Data (Hours)
Task
Scratch
EgoSteer-3K
EgoSteer-6K
EgoSteer-9.6K
Grasp object
80%
80%
70%
100%
Hand over object
70%
80%
80%
100%
Place items into container
40%
80%
90%
100%
Point at object
20%
40%
60%
70%
Place toy chick into slot
30%
0%
20%
40%
Pull out tissue
60%
20%
50%
80%
Push ball into box
0%
0%
10%
20%
Put garbage into trash bin
0%
60%
30%
30%
Stack paper cups
0%
20%
10%
40%
Stack tableware
0%
20%
10%
20%
Average
30%
40%
43%
60%
Table 7:
Per-task success rates for the pre-training scaling study in
Section
6.3
. Each task is evaluated over 10 randomized trials. Bold values highlight the best performance.
Task
Ours
Being-H0.5
π
0.5
\pi_{0.5}
Grasp object
100%
80%
80%
Hand over object
100%
60%
20%
Place items into container
100%
50%
0%
Pour items out of box
50%
30%
0%
Place bread on tray
70%
80%
40%
Pull out tissue
80%
10%
30%
Place item at specific orientation
30%
30%
30%
Attach eraser to whiteboard
90%
50%
20%
Put garbage into trash bin
30%
0%
0%
Put tennis ball into bucket
90%
0%
0%
Average
74%
39%
22%
Table 8:
Per-task comparison between EgoSteer-9.6K and two VLA baselines, Being-H0.5 and
π
0.5
\pi_{0.5}
, in
Section
6.3
. All methods are post-trained on our real-robot dataset and evaluated on the same 10 tasks. Each task uses 10 randomized trials. Bold values highlight the best performance.
Task
Ours
No WM-objective
No training-RTC
Noisy data
Grasp object
60%
40%
30%
60%
Hand over object
40%
20%
30%
40%
Place items into container
50%
30%
50%
70%
Pour items out of box
60%
50%
10%
0%
Place bread on tray
70%
10%
60%
50%
Pull out tissue
20%
30%
60%
10%
Place item at specific orientation
20%
10%
10%
20%
Attach eraser to whiteboard
50%
80%
50%
40%
Put garbage into trash bin
30%
0%
0%
0%
Put tennis ball into bucket
40%
40%
90%
40%
Average
44%
31%
39%
33%
Table 9:
Per-task success rates for the ablation study in
Section
6.4
. EgoSteer-1K is compared with ablated variants. Each task uses 10 randomized trials. Bold values highlight the best performance.
Pre-Training
Post-Training
Hyperparameter
Ours
No WM-objective
No training-RTC
Noisy data
Ours
No WM-objective
No training-RTC
Noisy data
Camera setup
Head
Head
Head
Head
Head
Head
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
64 A800
Global batch size
1152
1152
1152
1152
1152
1152
1152
1152
Training steps
30K
80K
30K
20K
60K
60K
60K
60K
Learning rate (VLM / AE / WM)
1
×
10
−
4
1{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
/
3
×
10
−
4
3{\times}10^{-4}
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Table 10:
Training hyperparameters for the ablation study in
Section
6.4
. All variants are pre-trained on 1K hours of egocentric data and post-trained on the same real-robot dataset, without DAgger refinement. Pre-training steps are selected at the lowest evaluation
L
1
L_{1}
loss for each pre-training run.
Hyperparameter
Box-Folding
Cake-Unboxing
Pre-training checkpoint
EgoSteer-9.6K at 155K steps
Camera setup
Head
Head
Resolution
384
×
384
384\times 384
384
×
384
384\times 384
GPUs
8 A800
8 A800
Gradient accumulation
1
1
Global batch size
144
144
Fine-tuning steps
44K
12K
Learning rate (VLM / AE / WM)
1
×
10
−
5
1{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
/
3
×
10
−
5
3{\times}10^{-5}
Demonstrations
120
229
Table 11:
Few-shot fine-tuning hyperparameters of EgoSteer-9.6K for the two long-horizon dexterous tasks in
Section
6.5
.