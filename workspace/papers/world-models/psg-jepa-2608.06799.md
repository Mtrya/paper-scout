# 2608.06799 (from arXiv HTML; MinerU fallback)



1]The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China
2]COCO Matrix
\contribution[∗]Equal contribution
\contribution[†]Project Leader
\contribution[‡]Corresponding author
\metadata[Project Page][https://haodong-yan.github.io/psg-jepa-project-page/](https://haodong-yan.github.io/psg-jepa-project-page/)

# Is Forward Prediction Enough? Physical State Grounding for JEPA World Models

Haodong Yan

  
Jiaguan Zhu

  
Mingyuan Jia

  
Ruiqing Yin

  
Junjie He

  
Zhide Zhong

  
Junfeng Li

  
Jinxuan Lu

  
Hengtao Li

  
Tianran Zhang

  
Jiayi Chen

  
Wenxuan Song

  
Wen Chen

  
Yuxiang Gao

  
Haoang Li

Affiliation: [

Affiliation: [

August 2026

###### Abstract

Learning structured and control-relevant latent representations remains a key challenge for world models. Recent JEPA-based world models learn action-conditioned predictive latent dynamics from observation sequences. However, their forward-prediction objectives do not explicitly enforce reliable identifiability of robot-centric physical state from individual latents or state changes from latent pairs, which can limit downstream planning and policy performance. We propose PSG-JEPA, a physically grounded JEPA world model that shapes its latent space with two complementary grounding objectives beyond forward prediction: grounding individual latents in robot proprioceptive state, and grounding latent pairs in multi-horizon joint-angle changes. Both objectives are applied only during training, leaving the inference architecture and computational cost unchanged. To comprehensively evaluate PSG-JEPA, we conduct experiments at three levels: (1) latent identifiability via probing, (2) goal-conditioned planning on frozen latents, and (3) policy learning in simulation and on a real robot. Experiments demonstrate that our PSG-JEPA consistently outperforms state-of-the-art latent world-model baselines at all three levels.

![Refer to caption](drafts/images/psg-jepa-2608.06799/teaser.png)

*Figure 1:
Overview of the proposed physical state grounding JEPA (PSG-JEPA).
(a) Forward-prediction JEPA world models (e.g., LeWM 21) learn a latent representation
space via action-conditioned forward prediction. However, this objective does not explicitly supervise the reliable decoding of robot-centric physical state or state change, leading to weak correlations between vision and action for downstream planners and policies.
(b) Our PSG-JEPA retains the action-conditioned forward pathway while
grounding individual latents in robot proprioceptive state and latent pairs in
multi-horizon joint-angle changes. These grounding objectives are implemented
through lightweight training-only state and transition heads.
(c) Compared to LeWM, our PSG-JEPA achieves stronger physical identifiability, higher
planning success under limited planner training, and higher policy success in simulation and on a real robot.*

## 1 Introduction

World models predict action-conditioned state transitions from observations 2; 4; 21; 22; 7; 8; 10.
Many recent world models 31; 18; 14; 7 rely on video generation, with VAE latents serving as their state representations.
However, the reconstruction objective of video VAEs prioritizes pixel-level accuracy and can preserve low-level appearance details that are often unnecessary for planning or policy learning 2; 25; 24; 32; 34.

In contrast, latent world models forgo pixel generation and learn action-conditioned dynamics in a compact feature space 9; 21; 2; 41; 20.
Early latent world models learn dynamics in a fixed pretrained representation space, as in DINO-WM 41, built on frozen DINOv2 features 26, and V-JEPA 2-AC, which uses a frozen video-pretrained JEPA encoder 2.
Such approaches benefit from strong visual priors, but physical information such as actions and robot states cannot shape the representation itself.
Recent end-to-end latent world models such as LeWM 21 jointly optimize the encoder and the action-conditioned predictor, allowing the forward-prediction objective to shape the latent space (see Fig. 1(a)).
However, this objective provides no direct supervision for recovering robot-centric physical state from individual latents or state changes from latent pairs.

To quantify this limitation, we investigate forward-trained representations through frozen probe evaluations, as shown in Fig. 1(c): single-latent probes do not reliably recover all robot-centric physical state variables, while pairwise probes recover physical state changes only partially.
These results characterize a robot-centric identifiability gap (probe-based recoverability of physical quantities from frozen latents) under forward-only training.
Control requires relating visual observations to the robot’s state and to how actions change it,
making vision–action alignment at the representation level itself a key property for
control 24.
When this information is not readily accessible from individual latents and latent pairs,
downstream planners and policies must learn vision–action alignment from their own task supervision.

Driven by this diagnosis, we introduce Physical State Grounding JEPA (PSG-JEPA), an end-to-end JEPA world model trained with complementary state-level and transition-level grounding objectives for individual latents and latent pairs, respectively (see Fig. 1(b)).
First, state grounding uses robot proprioceptive state $s_{t}$ to supervise each latent $z_{t}$.
Second, transition grounding utilizes the multi-horizon joint-angle changes
$\Delta q_{t,k}=q_{t+k}-q_{t}$, for all $k\in\{1,\dots,T{-}1\}$ in the training window, to supervise
the endpoint latent pairs $(z_{t},z_{t+k})$.
The two objectives constrain complementary aspects of the representation: state grounding links individual latents to robot state, while transition grounding links latent pairs to joint-angle changes across horizons.
The heads implementing these objectives are discarded after training, so PSG-JEPA introduces no inference-time overhead.
Together, the two objectives make state and state-change information readily accessible to downstream planners and policies.

In summary, our main contributions are as follows:

- •

We characterize a robot-centric identifiability gap in forward-trained LeWM representations: not all robot-centric physical state variables are reliably recoverable from individual latents, while physical state changes are only partially recoverable from latent pairs. This gap shifts the burden of vision–action alignment onto downstream planners and policies.
- •

We propose PSG-JEPA, which addresses the above gap by grounding the latent space in robot proprioception and multi-horizon joint-angle changes, without changing the backbone architecture or adding inference-time modules.
- •

Extensive experiments show that our PSG-JEPA outperforms state-of-the-art latent world-model baselines across latent identifiability, goal-conditioned planning, and policy learning.

## 2 Related Work

### 2.1 World Models

World models learn predictive dynamics that allow agents to reason about future outcomes under candidate actions 2; 4; 21; 22; 12.
One line of work couples world modeling with video generation, predicting future observations in RGB or in reconstruction-oriented latent spaces such as those of video VAEs 31; 30; 18; 14; 36; 13; 38; 29; 40; 39; 35; 34. These models generate visually rich future observations, but reconstruction objectives preserve low-level appearance detail that is often irrelevant for planning or policy learning 2; 25; 24; 32.
A second line learns compact action-conditioned latent dynamics for control, from Dreamer-style models 9; 10; 11; 33; 37 to world models over semantic or predictive features. DINO-WM 41 plans over frozen DINOv2 features 26, V-JEPA 2-AC adds an action-conditioned predictor on a frozen JEPA encoder 2, and LeWM trains the encoder and predictor end to end 21.
In the world-model baselines above, representations are optimized for reconstruction, reward prediction, or future-feature prediction. Their objectives do not explicitly enforce reliable identifiability of either robot-centric physical state or multi-horizon state changes. PSG-JEPA targets this limitation with state- and transition-level grounding objectives for end-to-end JEPA world models.

### 2.2 Joint Embedding Predictive Architecture

Joint Embedding Predictive Architectures (JEPAs) learn non-generative representations by predicting target embeddings rather than reconstructing observations.
I-JEPA 1 and V-JEPA 2; 22 instantiate this idea for image and video representation learning, while recent action-conditioned variants adapt JEPA objectives to latent world modeling and planning 2; 28; 21.

Because joint-embedding objectives admit collapsed solutions, JEPA training relies on explicit anti-collapse regularization, e.g., the principled isotropic-Gaussian regularizer (SIGReg) of LeJEPA 3.
Some JEPA-based world models 28 instead attach inverse-dynamics or action-decoder objectives to latent transitions, and these objectives serve a similar regularizing role.
They encourage action-discriminative representations by predicting actions from latent pairs, $(z_{t},z_{t+k})\mapsto a_{t:t+k-1}$.
However, inverse-dynamics supervision is ambiguous as a physical-transition signal: even from the same initial state, multiple action sequences can produce the same endpoint state change.
Our PSG-JEPA instead grounds latent pairs in the net joint-angle change $\Delta q_{t,k}=q_{t+k}-q_{t}$, which is uniquely determined by the endpoint states and fixed-dimensional at every horizon.
This gives each latent pair a single, well-defined physical transition target, making the correspondence between latent transitions and state change more readily learnable across horizons.

### 2.3 State-Aware Robot Representation Learning

A complementary line of work uses privileged robot state to shape visual representations for downstream
robot tasks. MCR 15 pre-trains a visual encoder on large robot datasets by aligning each
image feature with a temporal window of proprioceptive states and actions and by predicting actions
through a behavior-cloning head. RS-CL 16 regularizes a vision–language–action model at fine-tuning time so
that the latent similarity between two samples reflects their proprioceptive-state proximity.
RoboPEPP 6 pre-trains a single-image embedding-predictive encoder by masking
robot-joint regions, for robot pose and joint-angle estimation. We share their premise that the
proprioceptive state is freely available in robot data. However, none of these objectives regresses a physical transition over a temporally ordered,
same-trajectory latent pair, which is precisely what PSG-JEPA grounds by mapping each pair
$(z_{t},z_{t+k})$ to the net joint-angle change $\Delta q_{t,k}$ at multiple horizons.
PSG-JEPA also differs in scope and evidence: it characterizes and
addresses a robot-centric identifiability limitation in end-to-end JEPA world
models, and evaluates the remedy at three levels (identifiability, planning, and
policy learning) rather than through downstream performance alone.

## 3 Method

![Refer to caption](drafts/images/psg-jepa-2608.06799/method.png)

*Figure 2:
Illustration of PSG-JEPA grounding objectives.
For each image observation $o_{t+i}$, the shared encoder produces
$z_{t+i}=E_{\phi}(o_{t+i})$. $s_{t+i}$ is the aligned logged robot state, and
$q_{t+i}$ is its joint-angle component. The state head predicts
$\hat{s}_{t+i}=H_{s}(z_{t+i})$ from each latent, while the transition head predicts
$\Delta\hat{q}_{t+i,k}=H_{\Delta}(z_{t+i},z_{t+i+k})$ from every valid endpoint pair,
supervised by $\Delta q_{t+i,k}=q_{t+i+k}-q_{t+i}$. Here, $i$ indexes the
starting frame and $k$ the horizon. The figure shows a three-frame example. A
$T$-frame window uses all $k=1,\dots,T{-}1$.*

We introduce PSG-JEPA, a physical state grounding framework for end-to-end JEPA world models.
As illustrated in Fig. 1(b), PSG-JEPA preserves the forward JEPA prediction pathway while introducing two complementary physical grounding objectives: static state grounding for individual latents and dynamic transition grounding for latent pairs.
The grounding heads are used only during training and discarded during inference.
We first describe the retained forward JEPA world model (Sec. 3.1), then
introduce static state grounding (Sec. 3.2) and dynamic transition grounding
(Sec. 3.3), and finally present the overall training objective
(Sec. 3.4).

### 3.1 Preliminaries: Forward JEPA World Model

PSG-JEPA retains the JEPA encoder and causal action-conditioned predictor. For a training window with $T=C+1$
observations and $C$ actions, the encoder maps each observation to
$z_{t+i}=E_{\phi}(o_{t+i})$ for $i=0,\dots,T{-}1$, and the action-conditioned predictor $F_{\theta}$ uses the first $C$
latent–action pairs to produce teacher-forced one-step predictions:

|  | $$ \hat{z}_{t+1:t+C}=F_{\theta}\!\left(z_{t:t+C-1},a_{t:t+C-1}\right), $$ |  | (1) |
|---|---|---|---|

where each prediction $\hat{z}_{t+i+1}$ attends only to the prefix through time $t+i$. The forward objective is:

|  | $$ \mathcal{L}_{\mathrm{fwd}}=\frac{1}{C}\sum_{i=0}^{C-1}\operatorname{MSE}\!\left(\hat{z}_{t+i+1},z_{t+i+1}\right). $$ |  | (2) |
|---|---|---|---|

Following LeWM 21, we retain its SIGReg 3 anti-collapse regularizer without
modification. The resulting forward JEPA objective is:

|  | $$ \mathcal{L}_{\mathrm{JEPA}}=\mathcal{L}_{\mathrm{fwd}}+\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{SIGReg}}, $$ |  | (3) |
|---|---|---|---|

where $\lambda_{\mathrm{reg}}=0.09$ is the regularization weight.
This objective makes future features predictable conditioned on actions, but does not explicitly ground individual latents in robot state or latent pairs in physical state change.
PSG-JEPA adds the two grounding objectives to impose these complementary constraints.

### 3.2 Static State Grounding

As shown in Fig. 2, for a $T$-frame training window, a lightweight state head $H_{s}$ grounds every latent through:

|  | $$ \mathcal{L}_{\mathrm{static}}=\frac{1}{T}\sum_{i=0}^{T-1}\operatorname{MSE}\!\left(H_{s}(z_{t+i}),s_{t+i}\right), $$ |  | (4) |
|---|---|---|---|

where $s_{t}\in\mathbb{R}^{d_{s}}$ denotes the robot proprioceptive state, including joint angles, gripper state, and end-effector pose.
This supervision makes robot proprioceptive state more readily identifiable from individual latents. Because $H_{s}$ reads from the shared planning latent $z_{t+i}$ rather than a separate branch, minimizing $\mathcal{L}_{\mathrm{static}}$ encourages the encoder itself to expose physical state in the representation used downstream. Because physical state is a property of a single instant, static grounding operates on individual latents, complementing the pairwise transition grounding in Sec. 3.3.

### 3.3 Dynamic Transition Grounding

As illustrated in Fig. 2, dynamic grounding complements static grounding
with a shared transition head that directly predicts joint-angle changes across horizons.
Although the joint angles are part of $s_{t}$, static grounding supervises each endpoint independently and
does not directly optimize a pairwise readout of their change. A natural alternative is to supervise each pair with the intervening action sequence, as in inverse-dynamics objectives. However, this target is ill-posed as a physical-transition signal: its dimension grows with the horizon, and different action sequences can drive the robot between the same two configurations, so the same latent pair may map to many valid targets. In contrast, the overall joint-angle change between the two endpoints is uniquely determined by the endpoint states and keeps a fixed dimension across all horizons.

This motivates grounding latent pairs in the joint-angle change rather than the action sequence.
For each horizon $k\in\mathcal{K}=\{1,\dots,T{-}1\}$, let
$\mathcal{I}_{k}=\{0,\dots,T{-}k{-}1\}$ denote the valid starting offsets. For each
$i\in\mathcal{I}_{k}$, we define the $d_{q}$-dimensional joint-angle change:

|  | $$ \Delta q_{t+i,k}=q_{t+i+k}-q_{t+i}. $$ |  | (5) |
|---|---|---|---|

Here, $q_{t}\in\mathbb{R}^{d_{q}}$ is the vector of joint angles in $s_{t}$.
A transition head predicts
$\Delta\hat{q}_{t+i,k}=H_{\Delta}(z_{t+i},z_{t+i+k})$, with loss:

|  | $$ \mathcal{L}_{\mathrm{dynamic}}=\frac{1}{|\mathcal{K}|}\sum_{k\in\mathcal{K}}\frac{1}{|\mathcal{I}_{k}|}\sum_{i\in\mathcal{I}_{k}}\operatorname{MSE}\!\left(\Delta\hat{q}_{t+i,k},\,\Delta q_{t+i,k}\right). $$ |  | (6) |
|---|---|---|---|---|---|---|---|

All valid pairs contribute, and grounding every horizon rather than only adjacent pairs makes both short- and long-range physical change readable from latent endpoints. Because short horizons yield more pairs than long ones, we weight each horizon equally rather than each pair, so short-range transitions do not dominate the objective.

### 3.4 Overall Training Objective

The PSG-JEPA objective combines the retained forward JEPA objective with static and dynamic physical grounding:

|  | $$ \mathcal{L}_{\mathrm{PSG}}=\mathcal{L}_{\mathrm{JEPA}}+\lambda_{\mathrm{g}}\!\left(\mathcal{L}_{\mathrm{static}}+\mathcal{L}_{\mathrm{dynamic}}\right), $$ |  | (7) |
|---|---|---|---|

where $\lambda_{\mathrm{g}}=0.1$ is the grounding weight, shared by both grounding terms
and kept fixed across all benchmarks. After training, $H_{s}$ and $H_{\Delta}$ are discarded,
so downstream evaluations use the learned encoder without any test-time grounding heads.

## 4 Experiments

We evaluate PSG-JEPA on simulated and real-world robotic tasks, organizing the evaluation around four questions.
(Q1) Does physical grounding make forward-trained JEPA latents more physically identifiable?
(Q2) Do physically grounded representations improve goal-conditioned planning on frozen latents?
(Q3) Do physically grounded representations improve policy learning in simulation and on real robots?
(Q4) How does each grounding component contribute?

### 4.1 Experimental Setup

To ensure a controlled comparison, our PSG-JEPA retains exactly the JEPA encoder–predictor backbone of
LeWM 21. The model takes image observations and learns an encoder $E_{\phi}$
and an action-conditioned predictor $F_{\theta}$. We use $C=3$ context frames and
single-step prediction, giving training windows of $T=4$ frames and grounding horizons
$\mathcal{K}=\{1,2,3\}$.
PSG-JEPA and LeWM therefore differ only in the training-time grounding objectives applied to the latent,
and the proprioceptive quantities serve only as grounding targets or probe labels, never as inputs to
the encoder or predictor.

### 4.2 Methods for Comparison

We compare PSG-JEPA against three baselines. LeWM 21 is the
same-backbone forward-prediction baseline (SIGReg regularization, no grounding heads).
LeWMActionIDM is a variant of LeWM that we construct by adding a one-step
action-prediction (inverse-dynamics) objective on adjacent latent pairs,
$(z_{t},z_{t+1})\mapsto a_{t}$. It tests whether action-level supervision, the natural
alternative to our physical state grounding (Sec. 3.3), is
sufficient for the same benefits. DINOv2 26 is a pretrained
visual reference evaluated under the same downstream protocol: frozen features for
Q1–Q2, and a DINOv2-base encoder fine-tuned with the same policy head for Q3.

### 4.3 Physical Identifiability Probes (Q1)

We first assess whether physical grounding improves latent identifiability. Following
LeWM 21, we freeze the encoder and fit linear ridge and shallow
MLP probes using an episode-level train/test split. They measure linear and nonlinear recoverability,
respectively. Single-latent probes predict robot proprioception from $z_{t}$. Pairwise probes
predict joint velocity, gripper velocity, or action from $(z_{t},z_{t+1})$. We report Pearson $r$
on held-out test episodes.

#### Robot Proprioception Identifiability.

*Table 1:
Single-latent state probes on OGBench-Cube 27 proprioceptive quantities.
Cells report linear / MLP Pearson $r$ from frozen planning latents. Our PSG-JEPA shows stronger identifiability of robot proprioception.*

| Method | JointPos | EEPos | Gripper | EE-yaw |
|---|---|---|---|---|
| LeWM | 0.71 / 0.69 | 0.99 / 0.99 | 0.93 / 0.96 | 0.08 / 0.08 |
| DINOv2 | 0.73 / 0.72 | 0.99 / 0.94 | 0.84 / 0.84 | 0.51 / 0.50 |
| LeWMActionIDM | 0.75 / 0.72 | 1.00 / 0.99 | 0.96 / 0.98 | 0.11 / 0.10 |
| PSG-JEPA (ours) | 0.83 / 0.81 | 1.00 / 0.99 | 0.97 / 0.98 | 0.94 / 0.98 |

As shown in Table 1, forward-only training does not make robot
orientation readily identifiable from a single latent. LeWM and LeWMActionIDM leave EE-yaw
nearly unreadable ($r\leq 0.11$). Even with large-scale visual pretraining, DINOv2 recovers
EE-yaw only partially (MLP $r=0.50$), whereas our PSG-JEPA reaches $r=0.98$. This gap shows
that explicit state grounding makes orientation substantially more accessible in the latent. State
grounding also improves the linear recoverability of joint angles relative to LeWM
($r:0.71\to 0.83$).

#### Transition Identifiability.

*Table 2:
Transition probes on OGBench-Cube 27 from adjacent latent pairs
$(z_{t},z_{t+1})$. Cells report linear / MLP Pearson $r$. Our PSG-JEPA shows stronger transition identifiability.*

| Method | JointVel | GripVel | Action |
|---|---|---|---|
| LeWM | 0.68 / 0.66 | 0.44 / 0.47 | 0.74 / 0.76 |
| DINOv2 | 0.51 / 0.39 | 0.20 / 0.12 | 0.54 / 0.45 |
| LeWMActionIDM | 0.73 / 0.69 | 0.67 / 0.73 | 0.80 / 0.84 |
| PSG-JEPA (ours) | 0.75 / 0.75 | 0.69 / 0.76 | 0.80 / 0.86 |

We next use transition probes to assess whether physical state change can be recovered from latent
pairs, as shown in Table 2. Forward-only training leaves this information only
partially accessible. We evaluate this through velocities and actions, which describe
changes between frames. LeWM makes gripper velocity only weakly accessible (linear
$r=0.44$). Our PSG-JEPA matches or exceeds all baselines on each transition quantity.
In particular, our PSG-JEPA matches LeWMActionIDM in action recoverability despite not directly
using action prediction as an auxiliary objective. Even with large-scale visual
pretraining, DINOv2 recovers little gripper velocity
(0.20 / 0.12), whereas our PSG-JEPA reaches 0.69 / 0.76 without using gripper
velocity as an explicit auxiliary target. Together, these results support the
state/transition decomposition: physical state should be accessible from
individual latents, and physical change from latent pairs.

### 4.4 Goal-Conditioned Planning on Frozen Latents (Q2)

*Table 3:
Planning success rate (%) on OGBench-Cube and OGBench-Scene 27. Each method
uses the same GC-IDM 23 planning head trained on frozen latents with full or $25\%$
demonstration data (mean $\pm$ std over 3 planner seeds).*

OGBench-Cube, full data

| Epochs | 5 | 10 | 25 | 100 |
|---|---|---|---|---|
| LeWM | 80.7$\pm$1.9 | 83.3$\pm$0.8 | 84.2$\pm$2.1 | 89.7$\pm$0.2 |
| DINOv2 | 76.5$\pm$1.6 | 80.5$\pm$1.1 | 85.3$\pm$1.3 | 92.0$\pm$0.8 |
| LeWMActionIDM | 82.8$\pm$1.2 | 87.3$\pm$1.3 | 94.0$\pm$0.7 | 96.0$\pm$0.4 |
| PSG-JEPA | 95.0$\pm$0.7 | 92.7$\pm$1.9 | 94.5$\pm$2.1 | 98.7$\pm$1.2 |

OGBench-Cube, $25\%$ data

| Epochs | 5 | 10 | 25 | 100 |
|---|---|---|---|---|
| LeWM | 67.0$\pm$0.7 | 80.8$\pm$2.7 | 81.7$\pm$2.4 | 83.5$\pm$3.9 |
| DINOv2 | 60.8$\pm$5.1 | 70.0$\pm$0.8 | 79.8$\pm$1.0 | 85.8$\pm$1.7 |
| LeWMActionIDM | 63.2$\pm$1.8 | 77.5$\pm$1.5 | 82.5$\pm$1.9 | 91.5$\pm$1.1 |
| PSG-JEPA | 76.7$\pm$2.2 | 88.2$\pm$1.3 | 93.0$\pm$1.8 | 93.2$\pm$0.2 |

OGBench-Scene, full data

| Epochs | 5 | 10 | 25 | 100 |
|---|---|---|---|---|
| LeWM | 76.2$\pm$1.2 | 83.3$\pm$1.3 | 87.2$\pm$1.0 | 91.0$\pm$1.1 |
| DINOv2 | 72.2$\pm$1.9 | 73.2$\pm$1.0 | 83.2$\pm$1.2 | 90.2$\pm$1.4 |
| LeWMActionIDM | 80.3$\pm$0.6 | 85.8$\pm$0.6 | 90.5$\pm$0.4 | 94.7$\pm$0.6 |
| PSG-JEPA | 83.5$\pm$2.4 | 88.2$\pm$2.4 | 93.2$\pm$1.3 | 96.2$\pm$0.8 |

OGBench-Scene, $25\%$ data

| Epochs | 5 | 10 | 25 | 100 |
|---|---|---|---|---|
| LeWM | 62.2$\pm$1.2 | 70.0$\pm$1.8 | 76.7$\pm$0.9 | 83.0$\pm$1.1 |
| DINOv2 | 69.2$\pm$0.2 | 68.0$\pm$2.8 | 75.2$\pm$1.7 | 80.0$\pm$1.1 |
| LeWMActionIDM | 65.5$\pm$1.1 | 74.5$\pm$1.4 | 80.8$\pm$3.0 | 87.3$\pm$1.6 |
| PSG-JEPA | 69.5$\pm$1.1 | 76.2$\pm$1.9 | 83.3$\pm$1.9 | 89.7$\pm$0.9 |

We next evaluate whether physically grounded representations benefit goal-conditioned planning. For
each representation, we discard the training-time grounding heads, freeze the encoder, and train the
same GC-IDM planner 23, an amortized goal-conditioned inverse-dynamics model
implemented as a three-layer MLP with 512 hidden units and AdaLN-Zero conditioning. We evaluate
closed-loop goal reaching on OGBench 27. All methods are tested on the same 200 goal instances
using the OGBench goal-reaching criterion. To comprehensively characterize planning efficiency across optimization budgets
and data availability, we vary the GC-IDM training budget and the amount of
available demonstration data. Table 3 reports the
training-budget sweep at full and $25\%$ data, while
Fig. 3 further traces performance across data fractions.

##### Efficient planning under limited optimization and demonstration data.

Under limited optimization, our PSG-JEPA reaches $95.0\%$ success after only $5$
GC-IDM epochs (full data), compared with $80.7\%$ for LeWM. Even after $100$
epochs, LeWM reaches only $89.7\%$. Under limited demonstrations, with
the planner trained for $100$ epochs, PSG-JEPA reaches $84.5\%$ success from
only $5\%$ of the training data, a level LeWM requires roughly five times
more data to match (see Fig. 3(b)). At low GC-IDM
training budgets, both DINOv2 and LeWMActionIDM also trail PSG-JEPA
(see Table 3), showing that neither large-scale visual pretraining
nor action-level supervision alone yields equally efficient planning. Thus,
physical grounding makes the latent more readily usable under both limited
planner optimization and limited demonstrations.

##### Cross-environment validation on OGBench-Scene.

To test whether these gains are specific to OGBench-Cube, we retrain the three
learned world-model encoders (DINOv2 remains frozen) and repeat the identical
protocol on OGBench-Scene 27, a compound-goal manipulation task in which a cube, two
buttons, a drawer, and a window must all reach their goal configurations.
PSG-JEPA is best in every cell of the OGBench-Scene block of Table 3.
Similar to the results on OGBench-Cube, LeWMActionIDM is the closest baseline at
large planner budgets, while frozen DINOv2 is the hardest
representation to exploit at low planner budgets with full data. Along the data
axis, the planner trained on PSG-JEPA latents with half of the demonstrations
($94.7\%$) matches or exceeds every baseline planner trained on the full dataset.

##### Open-loop latent prediction.

Because GC-IDM 23 evaluates frozen representations without using the predictor $F_{\theta}$, we
separately examine the retained predictor on OGBench-Cube and
OGBench-Scene. Starting from three observed context frames, each model
recursively predicts future latents from logged actions, without teacher forcing,
on the same $512$ sampled trajectory segments per environment. At the shortest rollout
($5$ model steps, $25$ environment steps), PSG-JEPA already attains lower latent
MSE than the same-backbone LeWM baseline, at $0.0046$ vs. $0.0093$ on Cube and
$0.0208$ vs. $0.0269$ on Scene. The advantage grows at the longest rollout
($30$ model steps, $150$ environment steps), where PSG-JEPA lowers the MSE from
$0.1488$ to $0.0485$ on Cube (a $67\%$ reduction) and from $0.1608$ to
$0.0982$ on Scene (a $39\%$ reduction). PSG-JEPA has lower MSE at every reported horizon, so
grounding improves rather than degrades recursive predictive fidelity.

*(a) Success rate under different GC-IDM 23 training epochs, with full demonstration data.*

*(b) Success rate under different demonstration-data fractions, with the planner trained for $100$ epochs.*

*Figure 3: Efficient goal-conditioned planning under limited optimization and demonstration data. Markers indicate evaluated
points. Bands are $\pm 1$ std over $3$ planner seeds. Our PSG-JEPA leads along both axes, with the largest margins at small planner budgets and small
data.*

![Refer to caption](drafts/images/psg-jepa-2608.06799/real_robot.png)

*Figure 4:
Real-robot evaluation. (a) Dual-arm Cobot teleoperation
platform (Mobile ALOHA design). (b) Per-task success rate of our PSG-JEPA
versus the LeWM baseline on Place-to-Bread, Place-to-Plate, and Pour-Water.
Our PSG-JEPA improves over LeWM on all three tasks.*

### 4.5 Policy Learning (Q3)

Finally, we test whether the benefits of physical grounding extend beyond goal-conditioned planning to
policy learning, in simulation and on a real robot.

#### Simulated Benchmarks.

| Method | Success (%) |
|---|---|
| LeWM 21 | 77.7$\pm$0.5 |
| LeWMActionIDM | 82.6$\pm$2.2 |
| DINOv2 26 | 80.1$\pm$5.3 |
| PSG-JEPA (ours) | 85.3$\pm$3.9 |

*Table 4:
Policy learning on LIBERO-Goal (mean $\pm$ std over 3 seeds,
10 tasks $\times$ 50 rollouts per seed).
Our PSG-JEPA outperforms all compared baselines.*

We evaluate policy learning on LIBERO-Goal (10 goal-directed manipulation
tasks) 19. For each method, we train a task-conditioned OFT 17 action head while jointly
fine-tuning the encoder. The head has hidden width 1024 and four layers, takes two visual frames as
input, and predicts action chunks of eight steps. We train for 30 epochs and report mean $\pm$ std over
3 seeds, with 50 rollouts per task and seed (500 per seed, $1{,}500$ in total).

##### Policy performance on LIBERO-Goal.

As shown in Table 4, our PSG-JEPA reaches $85.3\%$ mean success
versus $77.7\%$ for LeWM ($+7.6$), and also exceeds LeWMActionIDM
($82.6\%$) and DINOv2 ($80.1\%$) under the same policy-learning protocol.
This result extends the planning results in Sec. 4.4: physically
grounded representations benefit not only goal-conditioned planners but also
task-conditioned policy learning across the ten LIBERO-Goal tasks. The
comparisons with LeWMActionIDM and DINOv2 further show that neither action-level
supervision nor large-scale visual pretraining alone provides the same benefit
for this policy-learning setting.

#### Real-World Robot.

We further evaluate PSG-JEPA on a physical dual-arm Cobot (AgileX
Robotics) that adopts the Mobile ALOHA system design 5. Each arm has six revolute joints
and a parallel gripper, and policies act from RGB observations captured
by three cameras: a front view and two wrist views (see Fig. 4(a)).
We study three tabletop manipulation tasks (Place-to-Bread,
Place-to-Plate, and Pour-Water) and collect $100$ teleoperated
demonstrations per task. Following the same policy-learning protocol as in
simulation, we adapt each representation with the shared action head and report
the average success rate over $50$ trials per task.

As shown in Fig. 4(b), our PSG-JEPA outperforms the LeWM baseline
on all three tasks: Place-to-Bread ($84\%$ vs. $62\%$), Place-to-Plate
($74\%$ vs. $58\%$), and Pour-Water ($80\%$ vs. $60\%$), i.e.,
$79.3\%$ versus $60.0\%$ on average. These results confirm that physically grounded
representations extend effectively to real-world manipulation.

### 4.6 Component Ablations (Q4)

To validate the effectiveness of the key components of our PSG-JEPA, we conduct an
ablation study on OGBench-Cube and LIBERO-Goal. We evaluate each variant across
latent identifiability, goal-conditioned planning, and policy
learning. Results are summarized in Table 5.
The w/o transition grounding variant retains only the single-latent state head, and
w/o state grounding removes the state head while retaining transition grounding.
The w/o multi-horizon (adjacent) and w/o multi-horizon (endpoint) variants replace the
multi-horizon joint-angle changes with only the adjacent ($k{=}1$) or the endpoint
(first$\to$last) horizon, respectively.

|  | Probe (mean lin. $r$) | Planning | Policy |
|---|---|---|---|
| Variant | state | transition | SR@5ep | LIBERO |
| LeWM | 0.68 | 0.62 | 80.7$\pm$1.9 | 77.7$\pm$0.5 |
| w/o transition | 0.93 | 0.72 | 81.3$\pm$2.4 | 80.3$\pm$3.9 |
| w/o state | 0.69 | 0.67 | 93.3$\pm$1.2 | 80.0$\pm$2.3 |
| w/o multi-horizon (adj.) | 0.86 | 0.74 | 93.5$\pm$1.9 | 81.5$\pm$3.1 |
| w/o multi-horizon (end.) | 0.86 | 0.75 | 93.7$\pm$2.0 | 81.2$\pm$2.7 |
| PSG-JEPA (full) | 0.94 | 0.75 | 95.0$\pm$0.7 | 85.3$\pm$3.9 |

*Table 5:
Component ablations across identifiability, planning, and policy learning.
State and transition report mean linear-probe $r$, i.e., robot proprioception
and transition identifiability. Planning reports success rate (%) on OGBench-Cube
(mean $\pm$ std over 3 planner seeds), and policy reports success rate (%) on
LIBERO-Goal (mean $\pm$ std over 3 seeds). Full PSG-JEPA
attains the best overall performance across the three evaluation levels.*

The ablations reveal distinct sensitivity patterns across the three evaluations.
Removing transition grounding produces the largest planning drop
($95.0\to 81.3$), whereas removing state grounding causes the largest drop in
mean linear state-probe performance ($0.94\to 0.69$). On LIBERO-Goal, every
ablated variant lowers policy success to roughly $80$–$81.5\%$, compared
with $85.3\%$ for the full model. The differences among the ablated
variants are within seed noise. On single-task Cube planning, full multi-horizon grounding gives a modest
improvement over the adjacent- and endpoint-only variants ($95.0\pm 0.7$ versus
$93.5\pm 1.9$ and $93.7\pm 2.0$), and a larger mean gap on policy
learning ($85.3$ versus $81.5$). Overall, every grounding component
contributes, and the full model is best across all three evaluation levels.

## 5 Conclusion

In this work, we introduced PSG-JEPA, an end-to-end JEPA world model that grounds individual
latents in robot proprioceptive state and latent pairs in multi-horizon joint-angle changes. The
grounding heads are used only during training, so deployment retains the original JEPA inference
cost, while the grounding objectives make state and state-change information more
readily identifiable from individual latents and latent pairs, respectively.
Across latent identifiability, goal-conditioned planning, and policy
learning, our PSG-JEPA achieves the best overall performance among all compared methods.
Ablations show distinct sensitivity patterns: removing state grounding most
strongly reduces state identifiability, removing transition grounding
produces the largest planning drop, and the full objective achieves the
highest multi-task policy success. Together, these results show that
physical grounding improves both the identifiability of latent representations
and their usefulness for embodied decision-making.

## References

- Assran et al. (2023)
M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas

Self-supervised learning from images with a joint-embedding predictive architecture.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 15619–15629.

Cited by: §2.2.
- Assran et al. (2025)
M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, S. Arnaud, A. Gejji, A. Martin, F. R. Hogan, D. Dugas, P. Bojanowski, V. Khalidov, P. Labatut, F. Massa, M. Szafraniec, K. Krishnakumar, Y. Li, X. Ma, S. Chandar, F. Meier, Y. LeCun, M. Rabbat, and N. Ballas

V-JEPA 2: self-supervised video models enable understanding, prediction and planning.

arXiv preprint arXiv:2506.09985.

External Links: 2506.09985,
[Link](https://arxiv.org/abs/2506.09985)

Cited by: §1,
§1,
§2.1,
§2.2.
- Balestriero and LeCun (2025)
R. Balestriero and Y. LeCun

LeJEPA: provable and scalable self-supervised learning without the heuristics.

arXiv preprint arXiv:2511.08544.

Cited by: §2.2,
§3.1.
- Dupoux et al. (2026)
E. Dupoux, Y. LeCun, and J. Malik

Why ai systems don’t learn and what to do about it: lessons on autonomous learning from cognitive science.

arXiv preprint arXiv:2603.15381.

Cited by: §1,
§2.1.
- Fu et al. (2024)
Z. Fu, T. Z. Zhao, and C. Finn

Mobile ALOHA: learning bimanual mobile manipulation with low-cost whole-body teleoperation.

arXiv preprint arXiv:2401.02117.

Cited by: §4.5.
- Goswami et al. (2025)
R. G. Goswami, P. Krishnamurthy, Y. LeCun, and F. Khorrami

Robopepp: vision-based robot pose and joint angle estimation through embedding predictive pre-training.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 6930–6939.

Cited by: §2.3.
- Guo et al. (2025)
Y. Guo, L. X. Shi, J. Chen, and C. Finn

Ctrl-World: a controllable generative world model for robot manipulation.

arXiv preprint arXiv:2510.10125.

Cited by: §1.
- Ha and Schmidhuber (2018)
D. Ha and J. Schmidhuber

World models.

arXiv preprint arXiv:1803.10122 2 (3), pp. 440.

Cited by: §1.
- Hafner et al. (2020)
D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi

Dream to control: learning behaviors by latent imagination.

In International Conference on Learning Representations,

External Links: [Link](https://openreview.net/forum?id=S1lOTC4tDS)

Cited by: §1,
§2.1.
- Hafner et al. (2019)
D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson

Learning latent dynamics for planning from pixels.

In International conference on machine learning,

pp. 2555–2565.

Cited by: §1,
§2.1.
- Hafner et al. (2023)
D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap

Mastering diverse domains through world models.

arXiv preprint arXiv:2301.04104.

Cited by: §2.1.
- Hou et al. (2026)
B. Hou, G. Li, J. Jia, T. An, X. Guo, S. Leng, H. Geng, Y. Ze, T. Harada, P. Torr, et al.

World model for robot learning: a comprehensive survey.

arXiv preprint arXiv:2605.00080.

Cited by: §2.1.
- Hu et al. (2023)
A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado

GAIA-1: a generative world model for autonomous driving.

arXiv preprint arXiv:2309.17080.

Cited by: §2.1.
- Huang et al. (2026)
Y. Huang, X. Lv, J. Xu, Z. Yu, J. Zhang, R. Hu, W. Feng, S. Zou, H. Xiao, Z. Zhou, et al.

PAIWorld: a 3D-consistent world foundation model for robotic manipulation.

arXiv preprint arXiv:2606.18375.

Cited by: §1,
§2.1.
- Jiang et al. (2025)
G. Jiang, Y. Sun, T. Huang, H. Li, Y. Liang, and H. Xu

Robots pre-train robots: manipulation-centric robotic representation from large-scale robot datasets.

In International Conference on Learning Representations,

Vol. 2025, pp. 81885–81905.

Cited by: §2.3.
- Kim et al. (2025)
T. Kim, J. Lee, M. Koo, D. Kim, K. Lee, C. Kim, Y. Seo, and J. Shin

Contrastive representation regularization for vision-language-action models.

arXiv preprint arXiv:2510.01711.

Cited by: §2.3.
- Li et al. (2026)
W. Li, G. Li, K. Maeda, T. Ogawa, and M. Haseyama

Predictive but not plannable: RC-aux for latent world models.

arXiv preprint arXiv:2605.07278.

Cited by: §4.5.
- Liao et al. (2025)
Y. Liao, P. Zhou, S. Huang, D. Yang, S. Chen, Y. Jiang, Y. Hu, J. Cai, S. Liu, J. Luo, et al.

Genie envisioner: a unified world foundation platform for robotic manipulation.

arXiv preprint arXiv:2508.05635.

Cited by: §1,
§2.1.
- Liu et al. (2023)
B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone

LIBERO: benchmarking knowledge transfer for lifelong robot learning.

In Advances in Neural Information Processing Systems,

Vol. 36, pp. 44776–44791.

Cited by: §4.5.
- Lyu et al. (2026)
J. Lyu, K. Liu, X. Zhang, H. Liao, Y. Feng, W. Zhu, T. Shen, J. Chen, J. Zhang, Y. Dong, et al.

Lda-1b: scaling latent dynamics action model via universal embodied data ingestion.

arXiv preprint arXiv:2602.12215.

Cited by: §1.
- Maes et al. (2026)
L. Maes, Q. L. Lidec, D. Scieur, Y. LeCun, and R. Balestriero

LeWorldModel: stable end-to-end joint-embedding predictive architecture from pixels.

arXiv preprint arXiv:2603.19312.

Cited by: Figure 1,
Figure 1,
§1,
§1,
§2.1,
§2.2,
§3.1,
§4.1,
§4.2,
§4.3,
Table 4.
- Mur-Labadia et al. (2026)
L. Mur-Labadia, M. Muckley, A. Bar, M. Assran, K. Sinha, M. Rabbat, Y. LeCun, N. Ballas, and A. Bardes

V-JEPA 2.1: unlocking dense features in video self-supervised learning.

arXiv preprint arXiv:2603.14482.

Cited by: §1,
§2.1,
§2.2.
- Nguyen et al. (2026)
H. Nguyen, X. Xu, and X. Huang

Latent geometry beyond search: amortizing planning in world models.

arXiv preprint arXiv:2605.08732.

Cited by: 3(a),
3(a),
§4.4,
§4.4,
Table 3,
Table 3.
- Nie et al. (2026)
D. Nie, F. Chen, Q. Lv, J. Kuang, X. Li, X. Cao, and X. Cai

LARY: a latent action representation yielding benchmark for generalizable vision-to-action alignment.

arXiv preprint arXiv:2604.11689.

Cited by: §1,
§1,
§2.1.
- Nilaksh et al. (2026)
Nilaksh, S. Jha, A. Zholus, and S. Chandar

Reconstruction or semantics? what makes a latent space useful for robotic world models.

External Links: 2605.06388,
[Link](https://arxiv.org/abs/2605.06388)

Cited by: §1,
§2.1.
- Oquab et al. (2023)
M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al.

DINOv2: learning robust visual features without supervision.

arXiv preprint arXiv:2304.07193.

Cited by: §1,
§2.1,
§4.2,
Table 4.
- Park et al. (2025)
S. Park, K. Frans, B. Eysenbach, and S. Levine

OGBench: benchmarking offline goal-conditioned RL.

In International Conference on Learning Representations,

Vol. 2025, pp. 94937–94982.

Cited by: §4.4,
§4.4,
Table 1,
Table 1,
Table 2,
Table 2,
Table 3,
Table 3.
- Sobal et al. (2025)
V. Sobal, W. Zhang, K. Cho, R. Balestriero, T. G. J. Rudner, and Y. LeCun

Stress-testing offline reward-free reinforcement learning: a case for planning with latent dynamics models.

In 7th Robot Learning Workshop: Towards Robots with Human-Level Abilities,

External Links: [Link](https://openreview.net/forum?id=jON7H6A9UU)

Cited by: §2.2,
§2.2.
- Song et al. (2026)
W. Song, Z. Zhou, H. Zhao, J. Chen, P. Ding, H. Yan, Y. Huang, F. Tang, D. Wang, and H. Li

ReconVLA: reconstructive vision-language-action model as effective robot perceiver.

In Proceedings of the AAAI Conference on Artificial Intelligence,

Vol. 40, pp. 18549–18557.

Cited by: §2.1.
- Team et al. (2026)
G. Team, A. Ma, B. Wang, B. Li, C. Ni, G. Li, G. Huang, G. Zhao, H. Li, H. Li, et al.

GigaWorld-1: a roadmap to build world models for robot policy evaluation.

arXiv preprint arXiv:2607.02642.

Cited by: §2.1.
- Team et al. (2025)
G. Team, A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, J. Zhu, K. Li, M. Xu, et al.

GigaWorld-0: world models as data engine to empower embodied AI.

arXiv preprint arXiv:2511.19861.

Cited by: §1,
§2.1.
- Tong et al. (2026)
S. Tong, D. Fan, J. Nguyen, E. Brown, G. Zhou, S. Qian, B. Zheng, T. Vallaeys, J. Han, R. Fergus, et al.

Beyond language modeling: an exploration of multimodal pretraining.

arXiv preprint arXiv:2603.03276.

Cited by: §1,
§2.1.
- Wu et al. (2023)
P. Wu, A. Escontrela, D. Hafner, P. Abbeel, and K. Goldberg

Daydreamer: world models for physical robot learning.

In Conference on robot learning,

pp. 2226–2240.

Cited by: §2.1.
- Yan et al. (2026a)
H. Yan, H. Yu, Z. Zhong, W. Yuan, X. Gong, Z. Luo, C. Heyu, J. Li, W. Song, S. Zhou, et al.

Open-world hand-object interaction video generation based on structure and contact-aware representation.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 16031–16041.

Cited by: §1,
§2.1.
- Yan et al. (2026b)
H. Yan, Z. Zhong, J. Zhu, J. He, W. Yuan, W. Song, X. Gong, Y. Cai, G. Zhao, X. Yan, et al.

S-VAM: shortcut video-action model by self-distilling geometric and semantic foresight.

arXiv preprint arXiv:2603.16195.

Cited by: §2.1.
- Yang et al. (2024)
S. Yang, Y. Du, S. K. S. Ghasemipour, J. Tompson, L. P. Kaelbling, D. Schuurmans, and P. Abbeel

Learning interactive real-world simulators.

In The Twelfth International Conference on Learning Representations,

External Links: [Link](https://openreview.net/forum?id=sFyTZEqmUY)

Cited by: §2.1.
- Zhao et al. (2025)
G. Zhao, X. Wang, Z. Zhu, X. Chen, G. Huang, X. Bao, and X. Wang

DriveDreamer-2: LLM-enhanced world models for diverse driving video generation.

In Proceedings of the AAAI Conference on Artificial Intelligence,

Vol. 39, pp. 10412–10420.

Cited by: §2.1.
- Zhao et al. (2026)
G. Zhao, Y. Wang, X. Wang, Z. Zhu, T. Yu, G. Huang, Y. Zai, J. Jiao, C. Xue, X. Wang, et al.

UniDriveDreamer: a single-stage multimodal world model for autonomous driving.

arXiv preprint arXiv:2602.02002.

Cited by: §2.1.
- Zhong et al. (2026)
Z. Zhong, J. Li, J. He, H. Yan, X. Gong, G. Zhao, Y. Cai, J. Gao, X. Yan, B. Liu, Y. Chen, L. Yang, and H. Li

DualCoT-VLA: visual-linguistic chain of thought via parallel reasoning for vision-language-action models.

arXiv preprint arXiv:2603.22280.

Cited by: §2.1.
- Zhong et al. (2025)
Z. Zhong, H. Yan, J. Li, X. Liu, X. Gong, T. Zhang, W. Song, J. Chen, X. Zheng, H. Wang, et al.

FlowVLA: visual chain of thought-based motion reasoning for vision-language-action models.

arXiv preprint arXiv:2508.18269.

Cited by: §2.1.
- Zhou et al. (2024)
G. Zhou, H. Pan, Y. LeCun, and L. Pinto

DINO-WM: world models on pre-trained visual features enable zero-shot planning.

arXiv preprint arXiv:2411.04983.

Cited by: §1,
§2.1.

