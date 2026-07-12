# RynnWorld-4D: from 2D video prior to geometric, closed-loop bimanual manipulation

**Thread question:** What does it actually take to turn a large pretrained video
diffusion model into a physically grounded 4D world model that can drive a
real dexterous bimanual robot at a useful control frequency?  And does adding
explicit depth and optical flow inside a single denoising loop buy enough
consistency and action-relevance to beat RGB-only policies?

This thread reads the paper, clones the official code, and triangulates against
the closest neighboring work.

---

## 1. Core mechanism made concrete

RynnWorld-4D is built on **Wan2.2-TI2V-5B**, a 30-layer latent video diffusion
transformer.  The authors clone the RGB backbone into three modality branches:

* **RGB branch** – appearance.
* **Depth branch** – metric geometry, initialized from the RGB branch.
* **Flow branch** – pixel motion, initialized from the RGB branch.

All three branches share the same text cross-attention Key/Value projections
(because language is modality-agnostic) but keep **independent FFNs** and
**intra-modal self-attention**.

### 1.1 RGB-DF and 3D scene flow

The input to the model is a single RGB-D frame plus a text instruction.  The
output is three synchronized latent videos: RGB, depth, and optical flow.  With
camera intrinsics `K`, depth `D_t`, and optical flow `f_opt = (Δu, Δv)`, a pixel
`p_t = (u, v, 1)` lifts and tracks as:

```
P_t       = D_t(u, v) * K^{-1} p_t
P_{t+1}   = D_{t+1}(u+Δu, v+Δv) * K^{-1}(p_t + [Δu, Δv, 0])
f_3D      = P_{t+1} - P_t
```

This is the "projective 4D" bet: stay in 2D-aligned latents (so the model keeps
the scalability and generative prior of video diffusion) while still exposing
explicit geometry and per-point 3D motion.

### 1.2 Tri-branch Joint Cross-Modal Attention (JA)

Every third transformer layer (layers 0, 3, 6, …, 27 → 10 JA modules) does an
extra cross-modal attention step after the usual intra-modal self-attention.
For each branch `m`, the code computes one query and one shared key/value pair;
the query attends to the concatenated K/V of the other two branches:

```
Q_m = RMSNorm(QProj_m(LN^m(z_m + e_m)))
K/V_m = KVProj_m(LN^m(z_m + e_m))
A_m = Attn(RoPE(Q_m), RoPE(K_cross), V_cross)
```

where `K_cross = concat({K_j}_{j≠m})`.  Tokens are reshaped from
`(B, T·S, d)` to `(B·T, S, d)` so attention is **frame-wise**: RGB frame `i`
only attends to depth/flow frame `i`.  3D RoPE is applied to Q and K so
positions align across modalities.

A subtle but important detail: the output projection is **zero-initialized**, but
the scalar gate is **initialized to 1** and passed through `tanh`.  The code
comments explain why: if both the output projection and the gate start at zero,
their gradients mutually depend on each other being non-zero and the joint path
locks at a saddle point.  With `OutProj=0` and `gate=1`, the initial residual is
still zero (so the Stage-1 checkpoint is preserved), but gradients immediately
flow into the output projection and can then tune the gate.

### 1.3 Phased training and shared noise

The training recipe has three stages:

| Stage | Fusion mode | Trainable parts | Notes |
|-------|-------------|-----------------|-------|
| 1 | `none` (independent branches) | All branches | `loss_weight_flow=0.5` because the first flow frame is just a zero-flow latent. |
| 2 | `joint` | Only the JA modules, RMSNorms, per-modality LayerNorms, gates, and modality embeddings; backbone **frozen**. | JA every 3 layers, frame-wise, 3D RoPE, branch dropout `p=0.2`. |
| 3 | `joint` | Entire model | Full fine-tuning, branch dropout `p=0.05`, cosine learning-rate schedule. |

The trainer samples **one shared Gaussian noise** for all three branches, so
their denoising trajectories stay aligned.  The flow-matching objective is:

```
L_total = E[ ||v_video - (ε - z0_video)||²
           + ||v_depth - (ε - z0_depth)||²
           + λ_flow ||v_flow - (ε - z0_flow)||² ]
```

with supervision applied only to frames `[1:]` (the first frame is the clean
conditioning latent).  Branch dropout randomly replaces the noisy depth or flow
latent (frames `>0`) with fresh Gaussian noise, forcing the JA modules to
reconstruct the missing modality from the visible ones.  The RGB branch is never
dropped; it is the appearance anchor.

### 1.4 RynnWorld-4D-Policy: inverse dynamics from internal 4D latents

At inference the world model is **frozen**.  Given the current RGB-D frame and
text instruction, it is run for a **single forward pass at diffusion timestep
`t=500`** and the hidden state after block 15 is extracted from all three
branches.  Those features are concatenated along the channel dimension
(condition_dim = 3 × 3072 = 9216), reshaped, and fed into a **Video_Former**
(Perceiver resampler: 384-dim latent space, 336 learnable latents, 21 frames,
6 layers).  The compressed representation conditions a small **flow-matching
policy head** that outputs a chunk of 10 actions (54-dimensional Tianji dual-arm
+ Wuji hand) via 4-step Euler ODE integration.

Because the heavy world model runs only once per replan cycle, the policy head
adds ~1% of the latency.  The whole cycle is ~1.1 s on an RTX 5090, giving an
effective control frequency of ~9 Hz by executing the 10-action chunk while the
next plan is computed in parallel.

---

## 2. External signals

### 2.1 Official code exists and is usable

The prompt said a clone attempt failed, but the repository
`alibaba-damo-academy/RynnWorld-4D` is publicly available and cloned successfully
into `code/rynnworld-4d/`.  The code is Apache-2.0 licensed and includes:

* the tri-branch transformer (`core/finetune/models/wan_i2v/module.py` and
  `module_joint.py`),
* the phased trainer (`core/finetune/models/wan_i2v/rynnworld4d_trainer.py`),
* the inference pipeline (`core/inference/rynnworld4d.py`),
* the policy head (`rynnworld4d_policy/policy_models/vpp_policy.py`,
  `Video_Former.py`, `flow_matching.py`),
* launch scripts for all three training stages and the policy.

This lets us verify details the paper only sketches:

* The released Stage-2 and Stage-3 scripts set `--joint_unidirectional True`,
  i.e. depth and flow attend **only to video**, while the paper’s equation (5)
  is fully bidirectional.  The code also includes an optional
  `joint_gate_video_decay` cosine schedule that fades the video-side injection
  to zero during Stage 3 to protect RGB quality.  These are sensible
  engineering choices, but they are not the same as the paper’s main narrative.
* Branch dropout probabilities in the scripts are `0.2` (Stage 2) and `0.05`
  (Stage 3), while the paper reports `0.2` and `0.1`.
* The policy config extracts features at block 15 and uses early-exit at layer
  20 to skip the trailing transformer blocks, saving compute.

### 2.2 Neighbor comparison

See `code/compare_4d_world_models.md` for a full table.  The short version:

* **TesserAct** also adds geometry to video diffusion (RGB-D-N) and learns
  inverse dynamics, but it reconstructs a 4D scene via optimization *after*
  generation, and it does not expose optical flow as an explicit kinetic cue.
* **4DNeX** generates RGB+XYZ pointmaps from a single image, but it is a
  single-image-to-4D method with no action head and no closed-loop robot
  evaluation.
* **Cosmos 3** is the broadest comparison: an omnimodal MoT that natively
  reasons over language, image, video, audio, and action tokens.  RynnWorld-4D
  is narrower and more robot-centric: it bets everything on RGB-DF co-generation
  and a lightweight inverse-dynamics head on internal latents.
* **LingBot-VA / LingBot-World** are causal video-action world models on the
  Wan family.  They are action-conditioned *generators*; RynnWorld-4D is a
  *predictive encoder* whose latents are read by a separate policy head.

### 2.3 Minimal architectural probe

`code/rynnworld4d_minimal_probe.py` isolates the joint-attention and
shared-noise flow-matching steps in a tiny, runnable module (PyTorch required).
It is not a reproduction, but it makes the training-step mechanics inspectable.

---

## 3. Findings

### 3.1 What the paper shows

* **4D generation quality:** on a 50-sequence held-out test set, RynnWorld-4D
  achieves `δ_1 = 0.610` for depth (vs. 0.327 for 4DNeX and 0.279 for
  TesserAct) and `AEPE = 0.170` for optical flow.  Independent branches drop to
  `AbsRel = 0.737` and `AEPE = 0.247`, confirming that joint attention matters.
* **Large-scale pre-training matters:** training only on task-specific robot
  data sends AEPE from 0.170 to 0.729.
* **Policy benefits from 4D latents:** replacing the predictive encoder with a
  ResNet-18 drops Dual Picking success from 94.29% to 71.43%.  Adding depth
  helps spatial-precision tasks; adding flow helps temporal-coordination tasks.
* **Real-world bimanual dexterous tasks:** on six tasks (Dual Picking, Block
  Pushing, Hand-over, Bimanual Lifting, Lid Placement, Bowl Stacking),
  RynnWorld-4D-Policy outperforms Diffusion Policy, π₀, and π₀.₅.  Standout
  numbers: Hand-over 88.57%, Dual Picking 94.29%, Lid Placement / Bowl Stacking
  65.71%.

### 3.2 What the code adds to the story

* The released recipe is more conservative than the paper suggests: it uses
  **unidirectional** joint attention and a cosine decay on RGB-side injection.
  This is a plausible reason the RGB quality stays high while depth/flow still
  improve, but it means the "mutual cross-modal interaction" claim should be
  read as "depth/flow are strongly regularized by RGB" rather than a fully
  symmetric three-way fusion.
* The policy does not wait for a decoded video.  It reads **mid-diffusion
  internal features** at `t=500`, which is the real speed trick: the world model
  is used as a frozen 4D vision encoder, not as a full future simulator that
  must be unrolled at every control step.

---

## 4. Uncertainties and limitations

* **Pseudo-label quality.**  Depth and flow are generated by Depth-Anything 3
  and DPFlow, not ground truth.  The geometric metrics therefore measure
  consistency with other estimators, not with the real world.
* **Small test sets and trial counts.**  World-model metrics are on 50 held-out
  sequences; real-world policy evaluation is 35 trials per task on a single
  hardware platform.  Cross-lab replication is needed.
* **AEPE measured on color-coded flow videos.**  The paper computes AEPE in the
  normalized RGB color space of the Middlebury-encoded flow videos, which is
  unusual and may not reflect true pixel displacement error.
* **Modest control frequency.**  9 Hz effective is far below a 500 Hz low-level
  controller.  The paper argues that spatial-volume features and 10-action
  chunking compensate, but fast contact transients and deformable objects are
  likely still hard.
* **Replanning is coarse-grained.**  The robot executes 10 actions open-loop
  before the world model re-plans.  Small bumps during that 1.1 s window can be
  corrected only at the next cycle.
* **Egocentric-only.**  The model is trained and evaluated from a single
  first-person camera; multi-view consistency and multi-robot collaboration are
  explicitly left open.
* **Paper/code mismatch on unidirectional attention and dropout probability**
  should make us cautious about which exact configuration produced the reported
  numbers.  The code is the more reliable artifact.

---

## 5. Suggested report claims

* **Main claim:** RynnWorld-4D demonstrates a concrete, buildable path to turn a
  2D video diffusion prior into a geometrically grounded 4D world model for
  real robot control, by adding depth/flow branches, sparse position-aware joint
  attention, and an inverse-dynamics head that reads internal latents in one
  forward pass.
* **Strongest evidence:** internal ablations showing that shared denoising +
  joint attention + modality adaptation each materially improve depth/flow
  accuracy, and that the predictive 4D latent space beats a static ResNet-18
  encoder for downstream manipulation.
* **Weakest evidence:** the absolute real-world success-rate comparisons, because
  of small trial counts, a single hardware platform, task-specific training
  data, and the lack of a standardized public benchmark.  The policy numbers are
  promising but should be treated as a proof-of-concept, not a settled ranking.
* **Open question:** whether the RGB-DF representation is the right trade-off
  long-term, or whether a denser 3D representation (point clouds, Gaussians,
  meshes) will eventually win for contact-rich manipulation.
