# 2608.14022 (from arXiv HTML; MinerU fallback)



1]CUHK
2]Tencent PCG
3]FDU
4]Shanghai AI Laboratory
5]HKUST
\contribution[*]Equal contribution
\contribution[†]Corresponding author
\contribution[]

 Contact: Xinye Li ()
\reportshorttitleForgeWM: Progressive Causal Training for Few-Step World Models
\metadata[Resources][Page](https://asdfo123.github.io/ForgeWM)   [Code](https://github.com/asdfo123/ForgeWM)   [Models](https://huggingface.co/ForgeWM)
\aftertitle

![[Uncaptioned image]](drafts/images/forgewm-2608.14022/forgewm_hero_compressed.png)

Figure 1: ForgeWM rollouts. Minecraft (top) under
keyboard–mouse control and an FPS domain (bottom) under gamepad control,
both produced by budget-specialized few-step students trained with the same
four-stage framework.

# ![[Uncaptioned image]](drafts/images/forgewm-2608.14022/forgewm-logo-clean.png)
ForgeWM: Progressive Causal Training for Few-Step Action-Conditioned
Video World Models

Xinye Li∗

  
Lingshuai Lin∗

  
Lei Wang

  
Liuzhou Zhang

  
Jialin Cui

  
Qingshan Li

  
Guanchu Wang

  
Qingbin Liu

  
Xi Chen

  
Jiang Bian

  
Wai Lam†

Affiliation: [

Affiliation: [

Affiliation: [

Affiliation: [

Affiliation: [

Email: xyli@se.cuhk.edu.hk

###### Abstract

Action-conditioned video world models require low-latency causal generation and reliable responses to game-native controls. Although causal distillation enables one- or few-step video synthesis, extending it to interactive world models remains challenging, as discrete keyboard states and continuous mouse motion must remain aligned with temporally compressed latent chunks during causal training and autoregressive rollout. We introduce ForgeWM, a progressive framework that transforms a bidirectional action-conditioned video generator into efficient few-step world models through domain adaptation, teacher-forced causal training, causal consistency distillation, and on-policy distribution matching with a bidirectional teacher. The resulting budget-specialized students operate at steady-state denoising budgets of 1, 2, and 4 steps. ForgeWM further supports a dual-path deployment protocol combining latency-critical interaction with optional replay-time refinement, where the one-step student re-noises and refines its saved draft. On paired Minecraft trajectories, ForgeWM leads the evaluated systems in Imaging Quality, reference-aligned motion-profile agreement, action-sign accuracy, and mouse-control accuracy, while achieving the lowest reference LPIPS; the same four-stage recipe transfers to gamepad-controlled FPS gameplay. Replay-time refinement matches four-step reference quality while remaining roughly three times closer to the experienced trajectory than regeneration from noise. These results demonstrate ForgeWM’s effectiveness for controllable few-step video generation.

## 1 Introduction

Generative video models can simulate future visual states from observations
and actions without an explicit graphics engine. Interactive deployment,
however, requires causal generation, persistent action responsiveness, and
few enough denoising steps to close the control loop. Recent systems
demonstrate autoregressive generation with keyboard, mouse, or camera-pose
control he2025matrixgame2; wang2026matrixgame3; gao2026lingbotworld2; dreamx2026world. Yet aggressive sampling compression feeds
visual, action, and cache errors into subsequent chunks, coupling fidelity,
controllability, and multi-chunk stability.
Figure 
![[Uncaptioned image]](drafts/images/forgewm-2608.14022/forgewm-logo-clean.png)
ForgeWM: Progressive Causal Training for Few-Step Action-Conditioned
Video World Models shows ForgeWM rollouts in two control domains.

The difficulty arises because causal generation changes both the input
distribution and model state. At inference, the denoiser conditions on
imperfect self-generated visual history rather than clean data, while action
histories and key–value caches must remain synchronized with the generated
latent chunks. Fewer denoising steps amplify errors in these states, which
then propagate autoregressively. This coupling is especially pronounced for
game-native controls: discrete keyboard states and continuous mouse motion
arrive at video-frame rate and enter the denoiser through dedicated pathways
rather than solely as a camera trajectory. A usable training framework must
therefore preserve the control interface, action-to-latent alignment, and
causal state-update protocol throughout adaptation and distillation.

Recent causal video methods huang2025selfforcing; zhao2026causalforcingpp; zheng2026causalrcm combine teacher-forced causalization, few-step
distillation, and on-policy self-rollout. ForgeWM studies how to preserve
frame-rate discrete and continuous game controls through this conversion as
clean context gives way to generated history.

To address this challenge, we introduce ForgeWM, a progressive
four-stage framework for few-step action-conditioned video world models. It
converts a bidirectional generator through domain adaptation, teacher-forced
causal training, causal consistency distillation, and on-policy distribution
matching. The bidirectional teacher and causal student share a base
initialization and meet during autoregressive self-rollout, where the former
supervises the consistency-initialized student. A modular action interface
preserves aligned keyboard-and-mouse conditioning throughout training. We
train budget-specialized 1-, 2-, and 4-step students, providing distinct
quality–latency operating points.

We evaluate ForgeWM against interactive world-model baselines at its native
1-, 2-, and 4-step operating points, and isolate test-time solver scaling
using a frozen one-step checkpoint. We further compare draft-preserving replay
with from-noise generation, separating online interaction from optional
replay-time refinement.

The contributions of this work are:

- •

We introduce a four-stage framework that converts a bidirectional
action-conditioned generator into budget-specialized few-step causal
world models.
- •

We preserve frame-aligned discrete and continuous game-native controls
across latent compression, causal training, and autoregressive rollout.
- •

We provide 1-, 2-, and 4-step operating points and a
same-checkpoint replay protocol that improves offline quality while
retaining the experienced trajectory.

## 2 Related Work

#### Interactive video world models.

Early diffusion-based simulators such as DIAMOND and GameNGen model game
observations conditioned on agent actions
alonso2024diamond; valevski2024gamengen, while Genie learns
action-controllable environments from unlabeled video bruce2024genie
and MineWorld targets real-time Minecraft interaction
guo2025mineworld. Recent systems extend this setting with frame-level
controls, long-horizon memory, and cross-game transfer
he2025matrixgame2; wang2026matrixgame3; tong2026scope.
LingBot-World 2.0 combines MoBA causal pretraining with consistency and
rollout-level distribution matching gao2026lingbotworld2, while
Cosmos 3 develops an omnimodal world-model backbone for Physical AI
nvidia2026cosmos3.

A complementary family parameterizes interaction through camera geometry.
PRoPE encodes relative projective transformations in attention
 li2025prope, minWM combines camera-controllable adaptation with causal
distillation zhao2026minwm, and WorldPlay uses dual action
representations with reconstituted context memory worldplay2025.
DreamX-World 1.0 extends this line with E-PRoPE, causal forcing, long-rollout
distribution matching, and geometry-retrieved scene memory
 dreamx2026world. Relative to these camera-, prompt-, or omnimodal
systems, ForgeWM centers frame-aligned keyboard-and-mouse controls across
budget-specialized causal students and draft-preserving replay.

#### Few-step autoregressive video distillation.

One- and few-step generation builds on consistency models, which learn direct
noise-to-data mappings song2023consistency, and distribution matching
distillation, which aligns a fast student’s distribution with that of a
diffusion teacher yin2024dmd. Recent video methods adapt these ideas to causal autoregressive generation. Diffusion Forcing assigns independent
noise levels to sequence tokens chen2024diffusionforcing, while
Self-Forcing reduces exposure bias through autoregressive self-rollout
huang2025selfforcing and AAPT provides an adversarial student-forcing
alternative for one-NFE generation lin2025aapt.

Causal Forcing initializes the student along a causal teacher trajectory
before asymmetric distribution matching zhu2026causalforcing.
Causal Forcing++ replaces offline ODE pairs with online causal consistency
distillation zhao2026causalforcingpp, and Causal-rCM unifies
teacher-forcing consistency learning with Self-Forcing distribution matching
zheng2026causalrcm. ForgeWM instantiates this progression in a
game-native training-to-deployment framework, producing separate 1-, 2-, and
4-step causal checkpoints with aligned discrete and continuous controls.

Most fixed-budget models can be evaluated with additional solver steps, but
quality need not improve outside their trained regime. AnyFlow instead learns
flow-map transitions for flexible test-time budgets gu2026anyflow;
ForgeWM trains budget-specialized students and separately evaluates a frozen
one-step checkpoint under off-budget schedules.

#### Replay-time and video refinement.

Adding noise to an existing sample and denoising it with a generative prior is
the basis of SDEdit meng2021sdedit. Recent video systems develop more
specialized correction mechanisms. AutoRefiner learns pathwise noise
refinement and a reflective cache for autoregressive video diffusion
 yu2025autorefiner; Pathwise Test-Time Correction uses the initial
frame to calibrate intermediate stochastic states without retraining
 xiang2026ttc; and SANA-WM employs a trained long-video refiner in a
two-stage minute-scale pipeline zhu2026sanawm. ForgeWM reconditions a saved rollout with the deployed low-step student at
a larger schedule, requiring no dedicated refiner or second checkpoint. We evaluate reference quality and draft retention separately.

![Refer to caption](drafts/images/forgewm-2608.14022/forgewm_overview_v7.png)

*Figure 2: Overview of ForgeWM. (A) Frame-aligned keyboard and
mouse controls condition latent chunks. (B) A shared base yields a
bidirectional teacher and budget-specialized causal students through
four-stage training. (C) Deployment separates low-latency interaction
from optional Replay-Time Refinement (Figure 6).*

## 3 Method

ForgeWM transforms a bidirectional action-conditioned video generator into
causal world models specialized for different sampling budgets.
Figure 2 summarizes its game-native action interface,
four-stage training graph, and dual-path deployment. Two branches start from
the same base generator: a bidirectional domain teacher and a causal student.
Their outputs meet during on-policy distribution matching over the
student’s autoregressive self-rollouts. We next describe
the shared generator and action interface, followed by the training and
deployment procedures.

### 3.1 Problem Setup and Base Generator

Let $x_{0:L}$ be a video, $x_{0}$ its initial observation, and
$a_{0:L}=(k_{0:L},m_{0:L})$ the corresponding control stream, where $k$ is a
discrete keyboard state and $m=(\Delta u,\Delta v)$ is continuous mouse
motion. A fixed VAE maps the video to a clean latent sequence $z$. Following
the Matrix-Game 2.0 image-to-video design he2025matrixgame2, the
generator receives the encoded first frame through both channel-wise latent
conditioning and a visual-context branch.

We adapt the base generator using the flow-matching objective
 lipman2022flow. Using the noise level $\sigma\in[0,1]$ as the path
coordinate and Gaussian noise $\epsilon$, we construct

|  | $$ z_{\sigma}=(1-\sigma)z+\sigma\epsilon,\qquad v^{*}(z_{\sigma},\sigma)=\epsilon-z, $$ |  | (1) |
|---|---|---|---|

and train the velocity predictor $v_{\theta}$ using

|  | $$ \mathcal{L}_{\mathrm{FM}}=\mathrm{E}_{z,\epsilon,\sigma}\!\left[w(\sigma)\left\|v_{\theta}(z_{\sigma},\sigma;c)-v^{*}(z_{\sigma},\sigma)\right\|_{2}^{2}\right], $$ |  | (2) |
|---|---|---|---|---|---|

where $c$ contains the initial-frame and action conditions. The induced clean
prediction is

|  | $$ \hat{z}_{\theta}\!\left(z_{\sigma},\sigma\right)=z_{\sigma}-\sigma\,v_{\theta}\!\left(z_{\sigma},\sigma\right), $$ |  | (3) |
|---|---|---|---|

which is the quantity the consistency and distribution-matching stages compare.
The first stage
uses bidirectional temporal attention over each training clip; later stages
change the temporal execution and training distribution while retaining this
conditioning interface.

### 3.2 Game-Native Action Conditioning

Following Matrix-Game 2.0 he2025matrixgame2, the
ActionModule injects frame-rate discrete actions and continuous
controls through separate pathways, without reducing them to camera poses or
pose-based encodings such as PRoPE li2025prope. Discrete inputs serve as
cross-attention keys and values, while windowed continuous inputs are fused
with visual features and processed by temporal attention.

The VAE compresses four video frames into one latent frame, and each
three-latent causal chunk therefore spans twelve video frames. We group actions
by latent interval and retain the preceding controls required by each token’s
temporal window. During rollout, the model maintains a visual key–value cache
together with separate keyboard and mouse caches at action-enabled blocks, so
each new chunk receives its complete aligned action window. The same alignment
and cache-update protocol is used throughout causal training, distillation,
and inference.

### 3.3 Four-Stage Causal Training

ForgeWM progressively changes the temporal execution pattern, sampling
objective, and conditioning-history distribution across four stages.

| Stage | Initialization | Training Context | Objective | Output |
|---|---|---|---|---|
| 0 | Base | Full-clip bidirectional | FM | Domain teacher |
| 1 | Base | Clean causal history | Causal FM | Causal teacher |
| 2 | Stage 1 | Clean causal history | Online CD | Few-step initializer |
| 3 | Stage 2 | Self-generated history | DMD | 1/2/4-step students |

*Table 1: Training stages.
Each stage changes either the temporal execution pattern,
sampling objective, or history distribution.*

Table 1 provides a roadmap of the four stages;
Figure 2(B) visualizes their computational dependencies,
and Appendix 6 reports the corresponding
optimization settings and stage-wise ablations.

#### Stage 0: bidirectional adaptation.

We first adapt the action-conditioned image-to-video generator to the target
game domain using $\mathcal{L}_{\mathrm{FM}}$. Full-clip bidirectional
attention learns the visual and control prior. We retain this
checkpoint as the frozen real denoiser used during Stage 3.

#### Stage 1: teacher-forced causal training.

In parallel to Stage 0, we initialize a second branch from the same base
generator and replace full temporal attention with block-wise causal
attention. Frames inside a latent chunk attend bidirectionally, while each
chunk can attend only to earlier chunks. Training concatenates the clean and
noisy token streams and applies a block mask under which noisy chunk $i$
attends to clean chunks $j<i$ and to itself, so the history is exact rather
than model-generated. A single noise level is drawn per chunk, giving the
teacher-forced objective

|  | $$ \mathcal{L}_{1}=\mathrm{E}\!\left[w(\sigma_{i})\left\|v_{\theta}\!\left(z^{(\sigma_{i})}_{i},\sigma_{i};c_{i},z_{<i}\right)-(\epsilon_{i}-z_{i})\right\|_{2}^{2}\right], $$ |  | (4) |
|---|---|---|---|---|---|

where $z_{<i}$ are the clean preceding chunks, $c_{i}$ is the initial-frame and
action condition for chunk $i$, and $w$ is the same flow-matching weighting as
Stage 0. This stage learns the causal execution pattern without yet exposing
the model to errors from its own rollout.

#### Stage 2: causal consistency initialization.

Starting from the Stage 1 causal checkpoint, we perform online causal
consistency distillation, building on consistency
distillation song2023consistency and its causal video adaptations
 zhao2026causalforcingpp. Generator, its
exponential-moving-average copy, and a frozen teacher are all initialized from
that checkpoint. We use a discrete grid of $N\!=\!48$ noise levels
$\sigma_{0}>\dots>\sigma_{N-1}$ and sample one adjacent pair $(\sigma_{i},\sigma_{i+1})$ per step. The frozen teacher takes one classifier-free-guided
Euler step from $\sigma_{i}$ towards $\sigma_{i+1}$,

|  | $$ \tilde{z}^{(\sigma_{i+1})}=z^{(\sigma_{i})}+\left(\sigma_{i+1}-\sigma_{i}\right)\left[v^{\emptyset}_{\mathrm{tch}}+\omega\left(v^{c}_{\mathrm{tch}}-v^{\emptyset}_{\mathrm{tch}}\right)\right], $$ |  | (5) |
|---|---|---|---|

with guidance weight $\omega$ and $v^{c}_{\mathrm{tch}}$, $v^{\emptyset}_{\mathrm{tch}}$
the conditional and unconditional teacher velocities. Because
$\sigma_{i+1}<\sigma_{i}$, the step moves towards the data end of the path. The
student’s original-level clean prediction is then matched to the EMA
stop-gradient at the teacher-advanced point:

|  | $$ \mathcal{L}_{2}=\mathrm{E}\!\left[\left\|\hat{z}_{\theta}\!\left(z^{(\sigma_{i})},\sigma_{i}\right)-\mathrm{sg}\!\left[\hat{z}_{\bar{\theta}}\!\left(\tilde{z}^{(\sigma_{i+1})},\sigma_{i+1}\right)\right]\right\|_{2}^{2}\right], $$ |  | (6) |
|---|---|---|---|---|---|

where $\bar{\theta}$ is the EMA parameter vector and $\mathrm{sg}[\cdot]$ denotes
stop-gradient. All three networks are conditioned on clean causal history at
this stage, which is what lets the teacher’s Euler step be evaluated in one
forward pass instead of an autoregressive unroll. This local consistency
objective initializes few-step sampling without an offline trajectory dataset.

#### Stage 3: on-policy distribution matching.

Finally, the student performs autoregressive self-rollout, so each new causal
chunk is conditioned on its own previously generated history. The student is
initialized from Stage 2, while the domain-adapted Stage 0 checkpoint provides
the real-distribution supervision. Let $\hat{z}$ be the clean latent
produced by a $K$-step self-rollout and let $\hat{z}^{(\sigma)}$ be a re-noised
copy of it. Following distribution matching distillation yin2024dmd and its
autoregressive video adaptations in Self-Forcing and
Causal Forcing huang2025selfforcing; zhu2026causalforcing, we form the
gradient direction from the disagreement between a frozen real denoiser
$\hat{z}_{\mathrm{real}}$ and a trainable fake denoiser
$\hat{z}_{\mathrm{fake}}$, which provide score-equivalent predictions under the
denoiser parameterization of Eq. (3),

|  | $$ g=\frac{\hat{z}_{\mathrm{fake}}\!\left(\hat{z}^{(\sigma)},\sigma\right)-\hat{z}_{\mathrm{real}}\!\left(\hat{z}^{(\sigma)},\sigma\right)}{\mathrm{mean}\!\left(\left|\hat{z}-\hat{z}_{\mathrm{real}}\!\left(\hat{z}^{(\sigma)},\sigma\right)\right|\right)}, $$ |  | (7) |
|---|---|---|---|---|---|

where the denominator averages the absolute deviation over all elements of each
sample and acts as an adaptive normalizer. We apply this direction through the
generator surrogate

|  | $$ \mathcal{L}_{3}=\mathrm{E}\!\left[\frac{1}{2}\left\|\hat{z}-\mathrm{sg}\!\left[\hat{z}-g\right]\right\|_{2}^{2}\right], $$ |  | (8) |
|---|---|---|---|---|---|

whose gradient with respect to $\hat{z}$ is exactly $g$; the fake denoiser is
updated concurrently with its own flow-matching loss on the student’s samples.
Training on the induced rollout
distribution reduces the clean-history mismatch left by teacher forcing. We
train separate budget-specialized students for
$K\in\{1,2,4\}$. Following the
first-frame enhancement strategy of yang2025asd, the 1- and 2-step
students use a fixed four-step schedule for the first generated latent chunk
and the matched $K$-step schedule thereafter; the 4-step student uses four
denoising evaluations throughout. Thus, “1-step”
and “2-step” denote the steady-state denoising budget rather than the total
number of generator calls for the complete rollout.

### 3.4 Budget-Specialized Interaction

For every generated chunk, the model follows its configured denoising schedule
and appends the clean prediction to the causal history. The primary deployment
uses the student matched to each target budget. We distinguish this native
evaluation from an off-budget analysis that freezes the one-step student and
changes only its test-time solver schedule. Because the low-step models retain
a four-evaluation first chunk, “one-step” and “two-step” refer to the
steady-state budget. The fixed-checkpoint step-scaling analysis reports both
generator evaluations and wall time under one consistent inference path.

#### Dual-path deployment.

ForgeWM uses the native low-step student for online interaction and optionally
applies a larger denoising budget to the saved rollout after interaction. The
replay path starts from the realized draft rather than from fresh noise and
reuses the recorded actions and refined causal prefix. The empirical motivation, formulation, and evaluation of this replay path are
presented together later under Inference Scaling and Replay-Time
Refinement (Section 4.3).

## 4 Experiments

### 4.1 Implementation Details

#### Training.

We initialize ForgeWM from the public Matrix-Game 2.0
lineage he2025matrixgame2 and train on 40,000 clips constructed from
GF-Minecraft yu2025gamefactory at $640\!\times\!352$. The final stage
produces students with steady-state budgets of one, two, and four denoising
updates. We also compare checkpoints from Stages 1–3 under a shared four-step
inference schedule.
Full optimization hyperparameters, stage-wise training details, and ablation
results are provided in Appendix 6.

#### Evaluation.

We compare ForgeWM with Matrix-Game 2.0 he2025matrixgame2 and the
HY-WorldPlay checkpoint of WorldPlay worldplay2025 using 77-frame
rollouts with shared initial frames and controls. Both baselines use four
denoising updates per chunk; ForgeWM uses four updates for the initial chunk
and its native one-, two-, or four-step budget thereafter. ForgeWM and
Matrix-Game 2.0 share a keyboard-and-mouse interface, while a deterministic
adapter maps the same controls to HY-WorldPlay’s parameterization.
Reference-aligned metrics use 1,000 paired trajectories; no-reference metrics
use 462 constant-action videos from 77 initial states. Chunk time is the mean
sampling time over non-initial chunks, excluding loading, VAE decoding, and
writing; FPS accounts for the 12-frame ForgeWM/Matrix-Game chunks and 16-frame
HY-WorldPlay chunks. Replay uses frozen ForgeWM-1 at $r=0.3$ with four updates; the direct
four-step baseline uses native ForgeWM-4 generation from noise. The user study includes 41
participants and 615 selections; CrossFPS uses 25 clips from each of seven
games. Full protocols are provided in Appendix 7.

#### Metrics.

We report VBench Imaging Quality (IQ), Aesthetic Quality (AQ), and Subject
Consistency huang2023vbench; LPIPS zhang2018lpips;
optical-flow profile similarity; and Mouse Accuracy based on GameWorld
Score zhang2025matrixgame. For keyboard control, we report KCtrl, a
camera-trajectory sign test on opposite-action pairs, in place of GameWorld
Score Keyboard Accuracy, whose real-frame-trained inverse-dynamics evaluator
can conflate control response with visual-domain similarity. For replay,
Replay LPIPS and $D_{\mathrm{draft}}$ use LPIPS with an AlexNet backbone on
the same 16 evaluated frames, excluding the initial frame. Replay LPIPS
compares the output with its paired reference, whereas
$D_{\mathrm{draft}}$ compares it with the saved draft. A blind three-way
study measures visual, action, and spatiotemporal preferences.

### 4.2 Interactive World Model Comparison

![Refer to caption](drafts/images/forgewm-2608.14022/qualitative_comparison_4frames.png)

*Figure 3: Qualitative comparison. Rollouts at frames 0, 25, 51,
and 76. Rows show the reference and three models. Controls are
annotated once on the reference: each overlay shows dominant WASD and
accumulated mouse-look to the next frame; the final frame has no
outgoing control overlay. Left/right: daytime forest stream/rainy
riverbank at night.*

| Model | Visual Quality | Temporal Quality | Action Controllability | Efficiency |
|---|---|---|---|---|
| IQ$\uparrow$ | LPIPS$\downarrow$ | AQ$\uparrow$ | Subj. Cons.$\uparrow$ | Flow Prof.$\uparrow$ | KCtrl$\uparrow$ | Mouse Acc.$\uparrow$ | Latency (ms)$\downarrow$ | FPS$\uparrow$ |
| Matrix-Game 2.0 | 0.6282 | 0.6443 | 0.4583 | 0.7349 | 0.9343 | 0.9156 | 0.7061 | 370.9 | 32.35 |
| HY-WorldPlay | 0.6133 | 0.6172 | 0.4855 | 0.9466 | 0.8288 | 0.9286 | 0.5818 | 2164.3 | 7.54 |
| ForgeWM-1 (1-step) | 0.6776 | 0.6529 | 0.4807 | 0.8279 | 0.9403 | 0.9545 | 0.7848 | 168.2 | 72.10 |
| ForgeWM-2 (2-step) | 0.6865 | 0.6171 | 0.4814 | 0.8349 | 0.9429 | 0.9740 | 0.8268 | 239.7 | 50.31 |
| ForgeWM-4 (4-step) | 0.6788 | 0.6168 | 0.4860 | 0.7613 | 0.9420 | 0.9740 | 0.8102 | 369.6 | 32.47 |

*Table 2: Comparison with interactive world models. Bold marks the
best quality/control value in each column; efficiency entries are not
bolded.*

Table 2 shows that ForgeWM variants attain
the best reported values in six of seven quality/control columns, while
ForgeWM-1 achieves the highest measured generation throughput. Performance
is not monotone in the denoising budget, motivating the fixed-checkpoint study
below. Subject Consistency should be interpreted cautiously because it can
favor conservative, low-motion videos; the reference-aligned Flow Profile
metric better reflects whether the requested motion pattern is reproduced.

ForgeWM-2 and ForgeWM-4 jointly rank first on KCtrl, indicating the highest action-sign accuracy under counterfactual
opposite-action pairs.
Figure 3 shows aligned rollouts from the
reference and the three models under a shared initial frame and control trace.

#### Human preference study.

Figure 4 reports a blind three-way study
comparing the four-step ForgeWM student with Matrix-Game 2.0
and HY-WorldPlay. In each comparison, the three clips shared
the same initial state and control trace; model identities were
hidden and their left-to-right order was randomized.

Each of the 41 participants evaluated five matched triplets per
criterion and selected one clip without a tie option, yielding
205 judgments per criterion and 615 in total. ForgeWM-4 receives
68.8% of visual-quality preferences, 57.6% of action-accuracy
preferences, and 55.6% of spatiotemporal-consistency preferences.
Its pooled preference share is 60.7%, with the largest margin in
visual quality. Appendix 7.3 provides recruitment,
interface, and instruction details.

*Figure 4: Human preferences.*

### 4.3 Inference Scaling and Replay-Time Refinement

#### Test-time step scaling.

To separate inference-time compute from checkpoint
specialization, we freeze ForgeWM-1 and vary only its
denoising schedule. Figure 5 shows that
performance remains stable beyond the native one-step budget.

Imaging Quality peaks at two steps, Subject Consistency at
four, and reference LPIPS improves through four to eight
steps before slightly regressing. Flow Profile and KCtrl
remain stable. More steps primarily increase motion
magnitude, latency, and computational cost rather than
directional control.

#### From scaling to refinement.

These trends reveal a deployment asymmetry: one step is
preferable for online interaction, whereas additional
denoising can improve reference-aligned quality without
sacrificing directional control. Rerunning from noise,
however, produces a new trajectory rather than improving
the rollout the user experienced. Replay-Time Refinement
instead applies the additional compute directly to the
saved draft.

*Figure 5: Test-time step scaling.
ForgeWM-1 under different denoising budgets; shading denotes
bootstrap 95% confidence intervals.*

#### Draft-preserving replay.

![Refer to caption](drafts/images/forgewm-2608.14022/01_replay_refine_reference.png)

*Figure 6: Replay-Time Refinement. The deployed student re-noises
and denoises its own saved chunks under aligned actions and the refined
causal prefix; each refined chunk is committed before the next is
processed.*

Given a saved rollout $\hat{z}_{1:B}$, we retain its initial observation,
actions, and latent chunks. Figure 6 summarizes
the resulting sequential procedure. Each draft chunk is re-noised at an
intermediate flow time $r_{i}$:

|  | $$ z_{i}^{(r_{i})}=(1-r_{i})\hat{z}_{i}+r_{i}\epsilon_{i},\qquad\epsilon_{i}\sim\mathcal{N}(0,I). $$ |  | (9) |
|---|---|---|---|

The deployed student then denoises it under the recorded action window and the
already-refined causal prefix:

|  | $$ z_{i}^{\mathrm{ref}}=\mathcal{R}_{\phi}^{\mathcal{S}(r_{i})}\left(z_{i}^{(r_{i})};x_{0},a_{\mathcal{W}_{i}},z_{<i}^{\mathrm{ref}}\right). $$ |  | (10) |
|---|---|---|---|

In the default setting, $\mathcal{R}_{\phi}$ is the frozen ForgeWM-1 draft
model, and $\mathcal{S}(r_{i})$ consists of four updates from $r_{i}=0.3$ to
zero. Replay therefore requires no parameter updates or separately trained
refiner.
Each refined chunk is committed before processing the next. The noise level
$r_{i}$ controls the trade-off between draft retention and visual revision.
Because replay occurs after interaction, it adds no online computation.

![Refer to caption](drafts/images/forgewm-2608.14022/replay_refinement_large_headers.png)

*Figure 7: Replay refinement versus direct generation.
Each row shares the same initial frame and controls.
(A) Saved ForgeWM-1 draft;
(B) replay-refined result;
(C) direct ForgeWM-4 generation from noise.*

| Method | Replay LPIPS$\downarrow$ | $D_{\mathrm{draft}}\downarrow$ |
|---|---|---|
| Draft-preserving methods |
| One-step draft (no refinement) | 0.6532 | — |
| Replay refinement (ours) | 0.6155 | 0.1970 |
| w/ four-step companion refiner | 0.6157 | 0.1960 |
| w/ four-step draft and refiner† | 0.6099 | 0.2361 |
| From-noise baselines |
| Direct ForgeWM-4 (from noise) | 0.6168 | 0.6187 |

*Table 3: Replay-Time Refinement.
Replay LPIPS is computed against the paired reference;
$D_{\mathrm{draft}}$ is LPIPS to the saved draft.
The † row is evaluated against its own four-step
draft and is therefore not directly comparable with the
remaining rows.*

#### Replay results.

Figure 7 illustrates the distinction between
refining a realized rollout and resampling a new one. Starting from the
one-step draft in (A), replay in (B) refines local visual
detail while preserving the draft’s viewpoint and scene layout. Direct
ForgeWM-4 generation in (C) also produces a high-quality rollout,
but its fresh-noise initialization can lead to a different realization,
visibly changing the viewpoint and object layout.
Table 7 quantifies this distinction: replay achieves
an LPIPS of 0.6155, comparable to 0.6168 for direct ForgeWM-4, while reducing
$D_{\mathrm{draft}}$ from 0.6187 to 0.1970. Using the four-step companion as
the refiner yields nearly the same trade-off, indicating that replay does not
require a separate checkpoint.

### 4.4 Recipe Transfer to FPS Gameplay

To test whether the training recipe extends to a different control space, we apply the same
four-stage progression to first-person-shooter (FPS) gameplay. This produces
ForgeWM-CrossFPS, a separately trained checkpoint. We adopt the cross-game FPS setting of
SCOPE tong2026scope, in
which controls are gamepad signals: two continuous analog sticks for movement
and look, together with discrete action buttons (fire, aim, jump, reload,
weapon switch, and melee). Using a similarly structured action interface and 4-stage
recipe without architectural changes yields a four-step causal student.

![Refer to caption](drafts/images/forgewm-2608.14022/crossfps_showcase_large_headers.png)

*Figure 8: CrossFPS rollouts. ForgeWM-CrossFPS on
(A) Halo Infinite, (B) Modern Warfare, and
(C) a science-fiction shooter. Colored overlays indicate active
gamepad controls.*

Figure 8 shows representative rollouts from three games. The macro-average
paired LPIPS is $0.656$, while the generated-to-reference motion ratio averages
$1.45$, suggesting a tendency toward stronger motion than the reference. These measurements use a
different domain and protocol from the Minecraft comparison;
Appendix 8 reports the per-game breakdown.

### 4.5 Extended Qualitative Rollouts

![Refer to caption](drafts/images/forgewm-2608.14022/appendix_long_strips.png)

*Figure 9: Extended qualitative rollouts from the four-step
Minecraft checkpoint. Each row shows one $22$ s causal rollout sampled at
evenly spaced timestamps, with elapsed time marked per frame. Rows span
biome and time of day and include both sustained forward motion and
sustained turning.*

Figure 9 shows five $22$ s rollouts, roughly
$3.5\times$ the horizon used for the quantitative protocol, selected to cover
different biomes, lighting conditions, and control patterns. Some sequences
exhibit a gradual loss of block structure or slowly spreading color artifacts
at later timestamps, illustrating the remaining long-horizon degradation modes
(Appendix 9).

### 4.6 Evaluation Scope

Our primary evaluation intentionally focuses on a controlled Minecraft
setting, where models can be assessed from matched initial states and mapped
control traces. Broader out-of-distribution generalization is therefore beyond
the scope of the current study. Because HY-WorldPlay uses a different control
parameterization, its effective motion scale depends on the deterministic
adapter. We consequently restrict direct quantitative comparisons to
Minecraft and present CrossFPS separately as a complementary evaluation of
recipe transfer to another gameplay domain.

## 5 Conclusion

We presented ForgeWM, a progressive causal training framework for
budget-specialized few-step world models that preserves frame-aligned
game-native controls. Across its
budget-specialized students, ForgeWM delivers the strongest overall
quality–control profile among the evaluated systems on paired Minecraft
trajectories.
ForgeWM further exposes
distinct 1-, 2-, and 4-step operating points and a replay-refinement path that
separates latency-critical interaction from optional quality-oriented
processing, where the deployed one-step student refines its own saved draft and
matches four-step reference quality without a second checkpoint. The same
training procedure also transfers to a gamepad-controlled FPS setting, while
our controlled quantitative comparison remains on Minecraft. Together, these
components provide an effective training and deployment framework for
controllable few-step video world models.

## References

\beginappendix

## 6 Training Details

### 6.1 Optimization and Stage-Wise Settings

#### Shared setup.

All stages train the Matrix-Game 2.0 image-to-video
lineage he2025matrixgame2 (a Wan2.1-T2V-1.3B backbone with
our action-conditioning module) on GF-Minecraft
clips yu2025gamefactory at $640\!\times\!352$ and $12$ frames per
second. A VAE with $4\times$ temporal compression maps each clip to a latent
sequence of $21$ latent frames ($16\!\times\!44\!\times\!80$), so a
three-latent causal chunk spans twelve video frames. We optimize with AdamW
($\beta=(0.0,0.999)$), mixed-precision bf16, gradient checkpointing,
and fully-sharded data parallelism across eight GPUs; the seed is fixed at $0$.
The flow-matching path uses $1000$ discretization steps with a timestep shift
of $5.0$; the consistency and distribution-matching stages apply
classifier-free guidance with weight $3.0$ to the frozen teacher. Learning
rates are held constant within each stage.

|  | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|
| Objective | $\mathcal{L}_{\mathrm{FM}}$ | $\mathcal{L}_{1}$ | $\mathcal{L}_{2}$ | $\mathcal{L}_{3}$ |
| Trainer | FM | causal FM | consist. distill. | DMD |
| Attention | bidir. | causal | causal | causal |
| Init | base | base | Stage 1 | Stage 2 |
| Blk. (latents) | 21 | 3 | 3 | 3 |
| Trainer iterations | 4k | 20k | 6k | 4k |
| Gen. lr | $2\mathrm{e}{-6}$ | $2\mathrm{e}{-5}$ | $2\mathrm{e}{-6}$ | $2\mathrm{e}{-6}$ |
| Critic lr | – | – | – | $4\mathrm{e}{-7}$ |
| Global batch | 8 | 8 | 8 | 8 |
| EMA | – | – | $0.99$/200 | $0.99$/200 |

*Table F1: Per-stage optimization for the reported Minecraft ForgeWM
lineage. “Blk.” is the number of latent frames per attention block
($21$ = full-clip bidirectional; $3$ = one causal chunk). “EMA” is the
generator exponential-moving-average decay and the step it starts (“–”
when disabled). Global batch is the effective batch after data-parallel
accumulation. All learning rates are constant.*

#### Stage-specific settings.

Table F1 lists the per-stage optimization.
Stage 0 adapts the base generator with full-clip bidirectional
attention over $21$-latent clips and no EMA; we freeze its $4{,}000$-update
checkpoint and reuse it as the real denoiser
$\hat{z}_{\mathrm{real}}$ in Stage 3. Stage 1 replaces full attention
with block-wise causal attention (three latent frames per block) and trains the
teacher-forced objective from the same base initialization; the reported
lineage uses the $20{,}000$-update checkpoint. Stage 2 initializes the
generator, its EMA copy, and the frozen teacher from that Stage 1 checkpoint and
runs online causal consistency distillation for $6{,}000$ updates over a
discrete grid of $N\!=\!48$ noise levels, sampling one adjacent pair per step.
Stage 3 initializes from the Stage 2 checkpoint and comprises
three separate $4{,}000$-trainer-iteration runs for
$K\!\in\!\{1,2,4\}$. The four-step student uses the denoising schedule
$[1000,750,500,250]$ for every generated chunk. Following
yang2025asd, the one- and two-step students use the same fixed
four-step schedule for the first generated chunk and their respective
budget-matched $K$-step schedules thereafter. Each run uses a local
attention window of six latent frames, equivalent to two three-latent
chunks. The critic is updated at every trainer iteration, whereas the
generator is updated once every five iterations. Figure F1
plots the per-stage training losses recorded during these runs. The
reported ForgeWM-1/2/4 models are the corresponding Stage 3
checkpoints.

The CrossFPS checkpoint reported in the main paper follows the same
sequence of four training objectives on gamepad-controlled data, using
the widened continuous-control interface described in
Appendix 8.2.

*Figure F1: Training losses across stages.
Raw values are shown as light traces, with exponential-moving-average
trends overlaid; the horizontal axis denotes trainer iterations. In
Stage 3, the critic is updated every iteration and the generator every
fifth iteration.
Top-left: Stage 0 bidirectional flow matching.
Top-right: Stage 1 teacher-forced causal flow matching.
Bottom-left: Stage 2 online causal consistency distillation.
Bottom-right: Stage 3 distribution matching, with the generator
(DMD) loss on the left axis and the critic (fake-denoiser) loss on the
right axis. Because the stages optimize different objectives, absolute
loss magnitudes are not comparable across panels; the curves document
optimization behavior within each stage only.*

### 6.2 Stage-Wise Inference Ablation

The body compares checkpoints from Stages 1–3 under a shared four-step
inference schedule; this section gives the protocol and results, and adds the
Stage 0 domain teacher as a non-causal reference point.

#### Protocol.

For Stages 1–3, we hold the inference path fixed and vary only the loaded
generator weights. All three checkpoints use the same chunked causal pipeline,
model configuration, four-step schedule $[1000,750,500,250]$, unrestricted
causal attention used for the main-table results, and identical
per-trajectory noise seeds. Stage 0 is bidirectional and has no chunked
few-step deployment; we therefore evaluate it in its native regime, using a
single full-clip block and a four-step UniPC schedule. Its row serves as a
non-causal domain-teacher reference and is not directly comparable to the
causal rows. All rows are scored on the same $1{,}000$ trajectories using the
paired LPIPS protocol of Appendix 7; intervals are
trajectory-level bootstrap 95% confidence intervals.

| Stage | Inference regime | LPIPS$\downarrow$ | IQ$\uparrow$ | AQ$\uparrow$ | SC$\uparrow$ |
|---|---|---|---|---|---|
| 0 | bidirectional teacher (ref.) | $0.814_{[.809,.819]}$ | $0.455$ | $0.463$ | $0.677$ |
| 1 | teacher-forced causal | $0.806_{[.799,.812]}$ | $0.508$ | $0.454$ | $0.700$ |
| 2 | causal consistency | $\mathbf{0.605}_{[.600,.610]}$ | $0.659$ | $0.483$ | $\mathbf{0.760}$ |
| 3 | distribution matching | $0.617_{[.613,.620]}$ | $\mathbf{0.716}$ | $\mathbf{0.489}$ | $\mathbf{0.760}$ |

*Table F2: Stage-wise inference ablation on $1{,}000$ paired
trajectories. LPIPS uses the AlexNet backbone on $16$ evaluated frames;
subscripts denote trajectory-level bootstrap 95% confidence intervals.
IQ, AQ, and SC are the corresponding VBench metrics. Stage 0 is a non-causal
domain-teacher reference evaluated in its native full-clip regime.*

#### Findings.

Table F2 isolates the contribution of each training
stage under a shared four-step evaluation protocol. Stages 0 and 1 are trained
as full-trajectory flow-matching models and are not intended for four-step
sampling; under this budget, they obtain LPIPS scores of $0.814$ and $0.806$,
respectively. Stage 2 produces the largest improvement, reducing LPIPS from
$0.806$ to $0.605$ with non-overlapping confidence intervals. This result
shows that causal consistency distillation, rather than teacher-forced
causalization alone, is the stage that enables effective few-step sampling.

Stage 3 introduces training on self-generated autoregressive histories. At
four steps, it obtains an LPIPS of $0.617$, with Stage 2 retaining a small
paired advantage of $-0.012$ ($[-0.015,-0.008]$). Its effect is more visible
in the no-reference metrics: Imaging Quality increases from $0.659$ to
$0.716$, while Subject Consistency remains unchanged at $0.760$ and Aesthetic
Quality changes only slightly. Thus, at the four-step budget, Stage 3 shifts
the trade-off toward sharper per-frame appearance rather than improving paired
reconstruction fidelity. Its incremental effect at one- and two-step budgets
is not isolated by this ablation.

## 7 Evaluation Protocol

### 7.1 Action Conversion Across Control Interfaces

The three compared systems do not expose the same control interface, so a single
recorded control trace is mapped onto each system’s own action parameterization
and every model is then driven from that mapped trace. The source trace is the
frame-rate stream used throughout this work: a discrete keyboard state and a
continuous two-dimensional mouse delta per video frame, recorded at
$640\!\times\!352$ and $12$ frames per second over the 77-frame comparison
window. ForgeWM and Matrix-Game 2.0 share this parameterization, so for those
two the mapping is the identity up to the sign convention of the vertical mouse
axis, which differs between the source recording and the model interface.
HY-WorldPlay additionally accepts a continuous camera parameterization, and the
conversion for that pathway integrates the per-frame mouse deltas into the
angular quantities its interface expects.

### 7.2 Metric Definitions

Table 2 reports nine columns. Let
$X_{n}=(x_{n,0},\ldots,x_{n,T-1})$ be generated clip $n$, let
$R_{n}=(r_{n,0},\ldots,r_{n,T-1})$ be its temporally aligned reference when a
reference is required, and let $N$ be the number of evaluated clips. All
reported dataset scores first average within a clip and then across clips. IQ
and AQ use the $462$ constant-action clips ($77$ scenes times six actions),
whereas SC, LPIPS, and Flow Profile use the $1{,}000$ shared-action rollouts.
Every metric reads exactly the first $77$ frames.

#### VBench metrics.

Imaging Quality (IQ), Aesthetic Quality (AQ), and Subject Consistency (SC) use
the standard VBench implementation huang2023vbench. Denote the MUSIQ
predictor by $q_{\rm MUSIQ}$, the LAION aesthetic linear predictor by
$q_{\rm aes}$, the normalized CLIP ViT-L/14 image feature by $c(\cdot)$, and the
normalized DINO ViT-B/16 feature by $d(\cdot)$. Their dataset-level scores are

|  | $$ {\rm IQ}=\frac{1}{NT}\sum_{n=1}^{N}\sum_{t=0}^{T-1}\frac{q_{\rm MUSIQ}(x_{n,t})}{100}, $$ |  | (11) |
|---|---|---|---|

|  | $$ {\rm AQ}=\frac{1}{NT}\sum_{n=1}^{N}\sum_{t=0}^{T-1}\frac{q_{\rm aes}(c(x_{n,t}))}{10}. $$ |  | (12) |
|---|---|---|---|

For Subject Consistency, let $[z]_{+}=\max(0,z)$ and define the local and
first-frame similarities

|  | $$ u_{n,t}=[d(x_{n,t-1})^{\top}d(x_{n,t})]_{+},\qquad v_{n,t}=[d(x_{n,0})^{\top}d(x_{n,t})]_{+}. $$ |  | (13) |
|---|---|---|---|

Then

|  | $$ {\rm SC}=\frac{1}{2N(T-1)}\sum_{n=1}^{N}\sum_{t=1}^{T-1}(u_{n,t}+v_{n,t}). $$ |  | (14) |
|---|---|---|---|

Thus, SC rewards both local frame-to-frame consistency and retention of the
first-frame subject appearance; it is not an identity-classification accuracy.

#### Reference-aligned visual and motion metrics.

Let

|  | $$ \mathcal{I}=\{1,6,11,\ldots,76\} $$ |  | (15) |
|---|---|---|---|

be the $16$ evaluated frame indices, excluding the given first frame, and let
$\ell_{\rm Alex}$ denote LPIPS with the AlexNet
backbone zhang2018lpips. We report

|  | $$ {\rm LPIPS}=\frac{1}{N|\mathcal{I}|}\sum_{n=1}^{N}\sum_{t\in\mathcal{I}}\ell_{\rm Alex}(x_{n,t},r_{n,t}), $$ |  | (16) |
|---|---|---|---|---|---|

where lower is better.

For Flow Profile similarity, let
$F(y_{t},y_{t+1})\in{\rm R}^{H\times W\times 2}$ denote the fixed dense
optical-flow estimator. For a video $Y$, define its temporal motion-magnitude
profile by

|  | $$ p_{k}(Y)=\frac{1}{|\Omega|}\sum_{(u,v)\in\Omega}\left\|F(\tilde{Y}_{4k},\tilde{Y}_{4(k+1)})_{u,v}\right\|_{2}, $$ |  | (17) |
|---|---|---|---|---|---|---|---|

for $k=0,\ldots,18$, and let
$\mathbf{p}(Y)=(p_{0}(Y),\ldots,p_{18}(Y))$.
Here $\tilde{Y}_{t}$ is frame $t$ converted to grayscale and resized to
$160\!\times\!88$ using area interpolation. The spatial evaluation region is
$\Omega=\Omega_{h}\times\Omega_{w}$, where
$\Omega_{h}=\{\lfloor 0.08H\rfloor,\ldots,\lfloor 0.82H\rfloor-1\}$
and
$\Omega_{w}=\{\lfloor 0.08W\rfloor,\ldots,\lfloor 0.92W\rfloor-1\}$.
This region suppresses the image borders and the HUD-heavy lower portion.
We compute $F$ using OpenCV Farneback flow with pyramid scale $0.5$, four
pyramid levels, window size $21$, four iterations, polynomial neighborhood
size $7$, polynomial standard deviation $1.5$, and flags set to zero.

The reported score is the mean paired cosine similarity

|  | $$ {\rm FlowProf}=\frac{1}{N}\sum_{n=1}^{N}\frac{\mathbf{p}(X_{n})^{\top}\mathbf{p}(R_{n})}{\|\mathbf{p}(X_{n})\|_{2}\|\mathbf{p}(R_{n})\|_{2}}. $$ |  | (18) |
|---|---|---|---|---|---|---|---|

For each summand, the cosine score is defined as zero when
$\|\mathbf{p}(X_{n})\|_{2}\|\mathbf{p}(R_{n})\|_{2}\leq 10^{-12}$.
Thus, Flow Profile compares the temporal pattern of motion magnitude rather
than optical-flow direction or absolute visual similarity.

#### Action controllability.

KCtrl uses constant-action counterfactual pairs from the same initial scene.
We sample frames $0,4,\ldots,76$, yielding $20$ sampled frames and $19$
consecutive relative poses. For action $a$, let
$v_{s,a,k}\in{\rm R}^{3}$ denote the translation component of relative pose
$k$ in scene $s$, where $k=0,\ldots,18$. Define the requested axis and sign as
$j(\mathrm{forward})=j(\mathrm{back})=z$,
$j(\mathrm{left})=j(\mathrm{right})=x$, and
$\eta_{\rm f}=\eta_{\rm r}=+1$,
$\eta_{\rm b}=\eta_{\rm l}=-1$. The net-direction indicator is

|  | $$ C_{s,a}={\rm I}\!\left[\eta_{a}\sum_{k=0}^{18}(v_{s,a,k})_{j(a)}>0\right], $$ |  | (19) |
|---|---|---|---|

where ${\rm I}[\cdot]$ denotes the indicator function.
With $S=77$ scenes, the main-table score averages the two opposite translation
pairs per scene:

|  | $$ {\rm KCtrl}=\frac{1}{2S}\sum_{s=1}^{S}\left(C_{s,{\rm f}}C_{s,{\rm b}}+C_{s,{\rm l}}C_{s,{\rm r}}\right). $$ |  | (20) |
|---|---|---|---|

Thus a pair receives credit only if both opposite commands yield the requested
net camera-motion sign; no test-time motion-magnitude threshold is used.

Mouse Accuracy follows the GameWorld/VPT inverse-dynamics evaluator
 zhang2025matrixgame. Let $g_{\rm IDM}(X)_{t}\in{\rm R}^{2}$ be its
predicted camera action at frame $t$, and let
$q$ be GameWorld’s nine-way camera-direction quantizer,
$q:{\rm R}^{2}\rightarrow\{0,\ldots,8\}$.
Write TL and TR for turn-left and turn-right,
respectively; their expected labels are $y_{\rm TL}=3$ and $y_{\rm TR}=4$. Over the two
turn actions, $S=77$ scenes, and $T_{\rm IDM}=76$ evaluated predictions per
clip, define the per-clip accuracy as
$\hat{y}_{s,a,t}=q(g_{\rm IDM}(X_{s,a})_{t})$ and

|  | $$ A_{s,a}=\frac{1}{T_{\rm IDM}}\sum_{t=0}^{T_{\rm IDM}-1}{\rm I}[\hat{y}_{s,a,t}=y_{a}]. $$ |  | (21) |
|---|---|---|---|

The reported score is

|  | $$ {\rm MouseAcc}=\frac{1}{2S}\sum_{s=1}^{S}\left(A_{s,{\rm TL}}+A_{s,{\rm TR}}\right). $$ |  | (22) |
|---|---|---|---|

This is a nine-way direction-classification accuracy under the commanded
constant mouse action, not a continuous regression error or a pixel-reference
metric.

#### Efficiency.

For measured chunk $j$, let $L_{j}$ be synchronized sampling time in
milliseconds and $C_{j}$ the number of newly generated frames (12 for ForgeWM
and Matrix-Game 2.0, and 16 for HY-WorldPlay). Over the $J$ timed chunks,

|  | $$ {\rm Latency}=\frac{1}{J}\sum_{j=1}^{J}L_{j},\qquad{\rm FPS}=\frac{1}{J}\sum_{j=1}^{J}\frac{1000C_{j}}{L_{j}}. $$ |  | (23) |
|---|---|---|---|

Loading, VAE decoding, and file writing are excluded. Efficiency columns are
reported for context and are not bolded in the table.

#### Profiling and resampling.

Efficiency is profiled on NVIDIA GPUs over 30 clips after three
warm-ups; the four-step ForgeWM/Matrix-Game comparison pools 90 measurements
from three GPUs, and efficiency reporting follows the same 30-clip protocol.
No-reference metrics use scene-level bootstrap and paired metrics use
trajectory-level bootstrap; plots report means with bootstrap 95% confidence
intervals. The fixed-checkpoint step-scaling study freezes ForgeWM-1 and
evaluates $1/2/4/8/16/32$ steps on identical scene–action–seed tuples.
Replay-refinement evaluation starts from a saved one-step draft and refines it
with the same frozen student under the recorded controls ($r=0.3$, four-step
schedule), comparing against the draft and against from-noise controls.

### 7.3 User Study Protocol

We recruited $41$ student volunteers to participate in the study. Each was shown a sequence of comparisons; in
every comparison three clips appeared side by side, one from the four-step
ForgeWM student, one from Matrix-Game 2.0, and one from HY-WorldPlay. All clips
in a comparison share the same source initial state and recorded control trace;
each model consumes that trace through its native or converted control interface
described above. Model identities were hidden and
the left-to-right placement was randomized independently for each comparison,
so position carried no information about which system produced a clip.

Each participant judged five comparisons per criterion under three criteria,
giving $41\times 5=205$ selections per criterion and $615$ in total. The
criteria were described to participants as follows. Visual Quality:
sharpness, naturalness, level of detail, and absence of artifacts.
Action Accuracy: whether the motion in the clip follows the direction and
the persistence of the control input. Spatiotemporal Consistency:
whether scene identity and geometry are preserved over time, and whether the
clip is free of flicker and warping. For each comparison and criterion the
participant selected exactly one of the three clips; there was no tie option,
which is why the three percentages for a criterion sum to $100\%$.

## 8 CrossFPS Evaluation and Adaptation

### 8.1 Cross-Domain FPS Evaluation

We sample 25 clips from each of the seven games in the official CrossFPS
evaluation split tong2026scope with a fixed seed, giving 175 clips. Equal
counts per game matter here: the released split is heavily unbalanced (its largest
title carries roughly $65\times$ the clips of its smallest), so an unweighted
mean over clips would report the largest title’s score under a cross-game label.
We therefore both sample equally and report a macro-average over games. Since
the released archive does not separately label its validation and test
partitions, the sample is drawn from the combined evaluation split and these
values are not directly comparable to the benchmark’s partition-specific
published results.

Each clip is replayed under its own recorded gamepad trace. Generation and
scoring follow the same protocol as the Minecraft experiments: 81 pixel frames
at $640\!\times\!352$ from the clip’s first frame, and paired LPIPS
(AlexNet backbone) on 16 frames sampled across the rollout, excluding the given
first frame. PSNR is computed and averaged over the same temporally aligned
frame pairs. Reference footage is temporally resampled to the same frame count.
Motion is measured as the mean dense optical-flow magnitude, the same estimator
as the main table’s flow column; flow ratio is the generated magnitude
divided by the reference magnitude, so $1.0$ means the rollout moves as much as
the reference.

| Game | LPIPS$\downarrow$ | PSNR$\uparrow$ | Flow | Ratio |
|---|---|---|---|---|
| Xonotic | 0.5828 | 11.21 | 13.88 | 1.66 |
| Modern Warfare III | 0.6352 | 11.61 | 8.15 | 1.35 |
| Modern Warfare | 0.6479 | 10.99 | 8.29 | 1.46 |
| Warzone | 0.6695 | 10.66 | 9.13 | 1.78 |
| Halo Infinite | 0.6730 | 9.75 | 11.23 | 1.46 |
| Halo | 0.6920 | 9.31 | 13.28 | 1.29 |
| Call of Duty | 0.6933 | 9.67 | 10.40 | 1.16 |
| Macro-average | 0.6562 | 10.46 | 10.62 | 1.45 |

*Table H1: Per-game CrossFPS results, 25 clips per game. Flow is
the mean dense optical-flow magnitude in pixels; ratio is generated over
reference, where $1.0$ matches the reference motion scale.*

Table H1 gives the breakdown. LPIPS spans
$0.583$–$0.693$ across titles. Flow ratios range from $1.16$ to $1.78$, with a
macro-average of $1.45$, indicating systematic over-response relative to the
reference motion scale.

### 8.2 CrossFPS Adaptation: Action Module and Training

The CrossFPS checkpoint reported in the main paper follows the same four-stage
training recipe and retains the same backbone and action-module topology.
Adapting to gamepad control requires only an interface-level change: the
continuous-control input width is increased from two to four channels, while
the data and control semantics differ from the Minecraft setting. This section
specifies that interface adaptation and the initialization of the added input
channels.

#### Action schema.

Minecraft control uses a two-dimensional mouse delta and a six-state keyboard
vector. CrossFPS is driven by a gamepad, so the continuous channel widens from
two to four dimensions – the $(x,y)$ deflection of the left analog stick
(movement) and the $(x,y)$ deflection of the right analog stick (camera look) –
while the discrete channel stays six-dimensional and is reinterpreted as the six
gamepad buttons (right trigger / fire, left trigger / aim-down-sights, and the
south / west / north / right-thumb buttons for jump, reload, weapon switch, and
melee). In the model configuration, this changes only
mouse_dim_in from $2$ to $4$. The backbone, action-module topology,
and all downstream dimensions remain unchanged; only the width of the first
continuous-control input projection is enlarged.

#### Where the change lands.

Continuous controls enter through the first layer of the action module’s mouse
branch, a linear layer over the windowed, temporally-grouped control vector
concatenated with the visual hidden state (the mouse_mlp input
projection). Widening the continuous input from two to
four channels enlarges only this input projection – from
$1536+2\!\times\!4\!\times\!3=1560$ to $1536+4\!\times\!4\!\times\!3=1584$ input units ($1536$ visual hidden units plus
mouse_dim_in $\times$ VAE temporal compression $\times$ window
control units) – leaving every downstream weight shape unchanged.

#### Initialization of the new channels.

A naive warm-start fails in two ways that we observed directly. First, the
shape filter used for checkpoint loading discards the entire input
projection when its input width changes, including the $1536$ columns that read
the visual hidden state and the columns for the two original control channels;
a model re-initialized this way collapses to a “use the hidden state, ignore
the control” solution and does not recover action controllability during
training. Second, keeping the visual columns but zero-initializing the
two new control channels leaves those channels without gradient signal at the
small Stage 0 learning rate: with no input variation reaching a zeroed column,
the symmetry is never broken and the two added channels stay effectively dead.

We therefore graft the input projection from the base checkpoint. The
$1536$ visual columns and the two original control channels are copied
verbatim, and each of the two new channels is initialized by copying one
of the pretrained control channels rather than from zero, so every channel
begins from a trained-scale, symmetry-broken state that gradient descent can
move. The zero-action-residual-at-initialization property is preserved not by
the input projection but by the action module’s output projection: the
base checkpoint ships the mouse and keyboard output projections zeroed, so at
the first update the widened action branch still contributes no residual and
the base video prior is left intact while the new interface is learned.

#### Training.

Stage 0 adapts the grafted generator on the merged CrossFPS corpus
(65,246 sharded clips across the seven titles) for $12{,}000$ updates.
Stage 1 initializes from this Stage 0 checkpoint, which already contains
the four-channel continuous-control projection. Stage 2 then initializes
from the resulting Stage 1 checkpoint, and Stage 3 initializes from
Stage 2; no further grafting is required. Stages 1–3 otherwise follow
the corresponding Minecraft objectives and optimization settings in
Table F1
($20{,}000$ teacher-forced updates, $6{,}000$ consistency-distillation
updates, and $4{,}000$ Stage 3 trainer iterations). The four-step
CrossFPS student uses the $[1000,750,500,250]$ schedule and is evaluated
in Table H1.

## 9 Limitations

We note three limitations that the results in this paper make visible.
Long-horizon drift. The quantitative protocol evaluates a
$77$-frame window; the extended rollouts of
Figure 9, roughly $3.5\times$ that horizon, show a
slow loss of block structure and spreading color artifacts at later timestamps.
Causalization and distillation reduce but do not eliminate the autoregressive
accumulation of error, and we do not claim indefinite rollout stability.
Motion over-response. On the cross-domain FPS split the generated flow
magnitude exceeds the reference by a macro-average ratio of $1.45$
(Table H1): the model tends to move more than the
recorded control implies, so magnitude fidelity lags directional and persistence
fidelity. Budget-dependent stage value.
At the four-step budget, Stage 3 increases Imaging Quality but does
not improve paired LPIPS relative to Stage 2. Its measured incremental
effect at this budget is therefore metric-dependent; the stage-wise
ablation does not isolate its incremental effect at one or two steps.

