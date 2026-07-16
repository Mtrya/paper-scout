# FlowWAM (arXiv 2607.13017) — Deep-dive findings

**Thread:** `runs/2026-07-16-densereward-terrazero-flowwam/flowwam-2607.13017/`  
**Paper:** `papers/world-models/flowwam-2607.13017.md`  
**Repos cloned:**
- `code/FlowWAM` — policy + RoboTwin training/inference
- `code/FlowWAM_WorldArena` — WorldArena world-model training/inference

---

## 1. Core idea

FlowWAM treats dense optical flow as the **action representation** for World Action Models.
Instead of numerical action tokens or learned latent actions, it encodes per-pixel
motion into RGB-format images via an HSV color wheel, then processes the flow video
with the same VAE + DiT as the RGB video.

**HSV encoding (Eq. 1 in paper, implemented in `reversible_flow_codec.py`):**
- Hue → `atan2(v,u)` mapped to `[0,1]` then `[0,179]` (8-bit) or `[0,65535]` (16-bit)
- Saturation → `||flow|| / max_magnitude`
- Value → constant 1.0 (white in the 8-bit OpenCV path)
- The paper/appendix use a **25 px magnitude cap** and **0.5 px noise threshold**

The codec is reversible: the magnitude ceiling is stored in a sidecar `.meta` file
(and redundantly in the bottom-right 8 pixels of the PNG). We ran a pure-numpy
probe of the 16-bit path in `code/flow_codec_probe.py`; for a constant 10 px
rightward flow the round-trip error is ~0.001 px. At very small magnitudes the
angle is undefined, which is why the pipeline zeros out sub-0.5 px vectors.

**Two operating modes:**
- **Policy mode:** both RGB and flow latents are noised and jointly denoised;
  the predicted flow is decoded into actions by the action expert.
- **World-model mode:** flow latents are set to the clean VAE encoding of a
target motion trajectory and held fixed; only RGB is denoised.

---

## 2. Dual-stream architecture

**Implementation files:**
- `code/wan_video_dit_dual_stream.py` — `FlowStreamModule`
- `code/wan_video_dual_stream.py` — `model_fn_wan_video_dual_stream` and `_dual_stream_block_fn`

What the code actually does:
- RGB latents go through the original `dit.patch_embedding` and `dit.head`.
- Flow latents go through a separate `flow_patch_embedding` and `flow_head`,
  both initialized as **deep copies** of the RGB patch embedding and head.
- A **learnable stream embedding** `stream_embed` of shape `(1,1,dit.dim)` is
  added to flow tokens after patchification so the shared blocks can tell the
  two streams apart.
- Inside each DiT block the RGB and flow tokens are **concatenated for joint
  self-attention**, split back, then each stream runs the same cross-attention
  and FFN.
- RoPE is computed independently for each stream's spatial/temporal grid.

**Static parameter estimate:** the FlowStream adds only ~400 k parameters to the
5B Wan2.2 DiT (see `code/param_count_note.md`).

**Important discrepancy:** the policy repo (`FlowWAM`) contains the
`stream_embed`, but the WorldArena repo (`FlowWAM_WorldArena`) **does not**
include it (see `patches/flow_stream_stream_embed.diff`). The paper's Appendix
A.1 explicitly says the learnable flow-token embedding is used; the released
WorldArena code therefore deviates from the paper description.

---

## 3. Action expert

**Implementation file:** `code/action_dit.py` (`ActionExpertIDM`)

The action expert is a separate 780 M-parameter transformer that:
- Takes a noisy action chunk `(B, N, action_dim=14)` as input.
- Cross-attends at every layer to the **per-layer hidden states** of the video
  DiT (RGB + flow concatenated). The mapping is 1:1 with the 30 Wan layers by
  default (`_layer_map`).
- Conditions on the current proprioceptive state (`qpos`, 14D). The default
  `proprio_mode="text"` projects qpos to the T5 text dimension and appends it to
  the instruction context.
- Predicts a flow-matching **velocity** (`pred_target="velocity"`) by default;
  `x0` is also supported.
- Uses 1D RoPE over the action sequence (`action_pos_mode="rope"`).

**Train/test alignment via noise mixing (Eq. 2):**
During training, the latents fed to the video DiT for the action-expert
conditioning are noised with probability `cond_noise_prob=0.5` and the noise
level is passed to the action expert as an auxiliary `cond_timestep` embedding.
This matches the paper's description.

At inference (`flow_action_server.py`):
1. Denoise RGB + flow latents with the dual-stream DiT (25 steps by default).
2. Capture per-layer features from the generated latents at timestep 0.
3. Denoise the action chunk with the action expert (10 steps by default) using
the captured features.

The actions are z-score normalized over the whole training set
(`action_norm_stats.npz`); the server denormalizes before returning absolute
qpos.

---

## 4. Training recipe

### Policy training (`code/train.sh`, `code/flow_action_train.py`)

- Base model: **Wan2.2-TI2V-5B** (VAE and T5 frozen, DiT fine-tuned).
- Data: RoboTwin 2.0 `aloha-agilex_clean_50` + `aloha-agilex_randomized_500`,
  50 tasks, 3 cameras (head, left wrist, right wrist) tiled into a 320×384
  T-shape.
- Temporal: `num_frames=33` action steps, `num_video_frames=9`, `visual_stride=4`
  (1 video frame per 4 action steps; latent length = 3).
- Flow: RAFT, robot-only mode by default, `flow_max_magnitude=25`,
  `flow_motion_boost=2.0`.
- Losses: video flow-matching loss with
  `flow_loss_weight=0.1`, `action_loss_weight=1.0`.
- Motion-aware reweighting: implemented in `flow_action_train.py` lines 501–506
  and 729–735, using Eq. 4 from the paper.
- Reference augmentation: first latent frame is perturbed with Gaussian noise
  (`ref_aug_strength=0.1`) to mimic autoregressive rollouts.
- IDM conditioning: `cond_noise_prob=0.5`, `cond_detach=false` (action loss also
  flows into the video backbone), `cond_layer_stride=1`.
- Optimizer: AdamW, lr=1e-4, weight decay=1e-2, 32 H100 GPUs.

### World-model training (`code/world_model_train.sh`, `code/world_model_module.py`)

- Same Wan2.2-TI2V-5B backbone + FlowStream.
- Data: RoboTwin 640×480 head-camera frames + robot-only flow, 121 output frames.
- The flow latents are **kept clean** (`flow_noise = zeros_like(flow_z)`) and
  fed to the model as conditioning. The shell script sets
  `FLOW_LOSS_WEIGHT=0.0`, so only the RGB stream is supervised. This is
  consistent with the paper's world-model mode, where flow is observed rather
  than generated.
- Autoregressive training: each episode is split into overlapping chunks; the
  last decoded frame of chunk `i` becomes the reference for chunk `i+1`.

### Two-stage / action-free pretraining

The paper claims a **video-only pretraining stage on EgoDex** before joint
robot training. **We could not find any EgoDex pretraining code or data loader
in either repository.** The only "stage 1" artifact released is the WorldArena
`flowwam_worldarena_stage1.safetensors`, which is trained on RoboTwin, not
EgoDex. The policy `train.sh` can warm-start from a checkpoint via
`RESUME_CHECKPOINT`, but no stage-1 script is provided. This is a significant
reproducibility gap for the action-unlabeled pretraining claim.

---

## 5. Experiments — claims and evidence

### RoboTwin 2.0 policy

Paper reports **92.94% Clean / 92.14% Random** average success over 50 tasks,
outpacing π0.5, X-VLA, Motus, GigaWorld-Policy, X-WAM and Fast-WAM.

Code evidence:
- `code/train.sh` trains on both clean and randomized variants.
- `code/FlowWAM/inference/robotwin_policy/eval_all.sh` lists all 50 tasks and
  dispatches them to the inference server.
- The eval writes results to `${ROBOTWIN_ROOT}/eval_result/<task>/flowwam/<task_config>/`.

We did not run the full eval (no RoboTwin env / data in this harness).

### WorldArena world modeling

Paper reports **EWMScore 63.71**, with the largest gain on **Trajectory
Accuracy** (64.26) vs. text/numerical/image-action baselines.

Code evidence:
- `code/world_model_inference.py` implements flow-conditioned autoregressive
  generation over 121 frames.
- It also loads and applies a **SeedVR stage-2 refiner** by default
  (`refiner/runtime.py`, `refiner/temporal_blend.py`), blending the dual-stream
  output with a refined video. The paper's main WorldArena number likely
  includes this refiner; this should be noted when comparing to baselines that
  do not use an external refiner.

### Real-world manipulation

Paper reports 75.7% average success on 7 Franka / ARX tasks. The real-world
setup is described in Appendix E; we found no real-world code in the repos.

---

## 6. Code-vs-paper discrepancies and blockers

| Item | Paper claim | Code observation |
|------|-------------|------------------|
| **Flow encoding** | HSV color wheel, invertible | Confirmed in `reversible_flow_codec.py`; magnitude cap 25 px, threshold 0.5 px. |
| **Dual-stream joint attention** | RGB + flow share DiT blocks, joint self-attention, stream-specific adapters | Confirmed in policy code; **WorldArena repo omits the `stream_embed` learnable token**, contradicting Appendix A.1. |
| **Action expert size** | ~780 M params | Confirmed by static count of `ActionExpertIDM`. |
| **Noise mixing for train/test alignment** | p=0.5 noise on cond latents | Confirmed in `flow_action_train.py` lines 782–799. |
| **Motion-aware reweighting** | Eq. 4 with α=2.0 | Confirmed in `flow_action_train.py`. |
| **Two-stage training / EgoDex pretraining** | Stage 1 on EgoDex without action labels | **No EgoDex code or loader in either repo.** Released "stage1" checkpoint is RoboTwin-only. |
| **WorldArena result** | EWMScore 63.71 | WorldArena inference uses an external **SeedVR refiner** by default; the number is not purely from the dual-stream model. |
| **Flow for world model** | Flow latents held clean as conditioning | Confirmed; training loss on flow is zeroed in `world_model_train.sh`. |

**Blockers encountered:**
- No PyTorch / cv2 in the harness, so we could not instantiate the model or run
  RAFT. Diagnostics were limited to static reading and a pure-numpy flow-codec
  probe.
- No RoboTwin simulator or dataset available, so policy success rates could not
  be reproduced.
- No EgoDex data or stage-1 script, so the action-unlabeled pretraining claim
  cannot be independently verified from the released artifacts.

---

## 7. Related-work comparison

FlowWAM sits in the third line of WAM action representations from the paper:
image-space conditioning signals. What is genuinely new vs. the baselines it
cites:

- **Motus** — learns *latent* actions from frame transitions; flow is only an
  auxiliary cue. FlowWAM instead exposes an explicit dense flow video as both
  prediction target and conditioning signal.
- **GigaWorld-Policy / Fast-WAM** — use numerical action tokens or direct action
  decoding without a generated intermediate motion plan. FlowWAM decodes actions
  from a generated dense flow plan.
- **X-WAM** — couples multi-view RGB-D generation with numerical state/action
  latents. FlowWAM replaces the depth/action stream with a single flow video
  that shares the RGB latent space.
- **Cosmos Policy** — fine-tunes a video model with action tokens appended to
  visual latents. FlowWAM's flow is a *visual* stream, not a token stream, so it
  can be pretrained on action-unlabeled video.
- **UWM / CoVAR** — co-generate video and action via shared latents or joint
  diffusion heads. FlowWAM casts action as a video prediction problem inside the
  same DiT.
- **LangToMo / DAWN / FLIP / EC-Flow** — use flow or pixel motion as an
  intermediate or auxiliary variable, but keep it outside the video generator.
  FlowWAM's claim is that flow is the *primary* action modality *inside* the
  video generator.

The genuine novelty is therefore architectural: optical flow is not a
side-output or a planning intermediate, but a full video stream that is
**generated** in policy mode and **conditioned on** in world-model mode, using
the same pretrained VAE and DiT blocks.

---

## 8. What the parent report should say

**Key findings to carry forward:**
1. FlowWAM's central design is implemented as described in the policy repo:
   HSV flow encoding, dual-stream Wan2.2 DiT with joint attention, a 780 M
   action expert conditioned on per-layer video features, and noise mixing for
   train/test alignment.
2. The numbers are impressive (92.94% / 92.14% on RoboTwin, 63.71 EWMScore on
   WorldArena) and the codebase is unusually complete for a video-generator-based
   robotics paper.
3. **Caveats:**
   - The WorldArena codebase lacks the `stream_embed` that the paper describes,
     suggesting either an implementation fork or an undocumented ablation.
   - The reported WorldArena score relies on an external SeedVR refiner, so the
     dual-stream model alone is not solely responsible for the number.
   - The EgoDex action-unlabeled pretraining stage is described but not released;
     only the RoboTwin joint-training code is present. The pretraining scalability
     claim therefore rests on the paper's private pipeline, not on reproducible
     public code.
4. The idea is well motivated and the code matches the main policy-side claims.
   The biggest open question is how much of the gain comes from the flow
   representation itself versus the large Wan2.2 video prior and the
   robot-only flow preprocessing; the ablations in the paper point to the flow
   representation, but independent reproduction would be valuable.

---

## 9. Preserved evidence

All durable files are in this thread directory:

- `code/wan_video_dit_dual_stream.py` — FlowStreamModule (policy repo)
- `code/wan_video_dual_stream.py` — dual-stream forward with joint attention
- `code/action_dit.py` — action expert implementation
- `code/flow_action_train.py` — joint training loop and losses
- `code/train.sh` — policy training configuration
- `code/reversible_flow_codec.py` — HSV flow codec
- `code/flow_action_server.py` — inference server
- `code/world_model_module.py` — WorldArena world-model training
- `code/world_model_train.sh` — WorldArena training configuration
- `code/world_model_inference.py` — WorldArena inference (includes refiner)
- `code/flow_codec_probe.py` — pure-numpy round-trip diagnostic
- `code/param_count_note.md` — static parameter-count analysis
- `patches/flow_stream_stream_embed.diff` — shows WorldArena repo is missing the
  learnable stream embedding
