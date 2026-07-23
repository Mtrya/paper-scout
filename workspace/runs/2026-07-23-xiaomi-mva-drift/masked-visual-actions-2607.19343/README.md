# Deep thread: Masked Visual Actions for Unified World Modeling (arXiv 2607.19343)

Date: 2026-07-23. Paper text: `papers/world-models/masked-visual-actions-2607.19343.md`.
Code probe: `code/mva-probe/` (clone of github.com/HadiZayer/masked-visual-actions @ main,
plus `code/mva-probe/diffsynth-src/` partial fetch of DiffSynth-Studio @ pinned commit 3743b130 — git fetch failed twice on TLS, so the single needed file `diffsynth/pipelines/wan_video.py` was pulled via raw.githubusercontent and archived here as `code/wan_video_pinned_3743b130.py`).

## Actions taken

1. Cloned the released repo and read every file (`README.md`, `inference/infer.py`,
   `inference/download_weights.py`, `training/train_control.sh`). Earlier "empty body" fetches
   were spurious — the repo is real and complete (4 files + LICENSE).
2. Downloaded both released LoRA weights from `HadiZayer/masked-visual-actions` via hf-mirror
   (2,453,763,192 bytes each, size-verified against Content-Length; SHA256 in `code/weights_sha256.txt`).
   Stored at `code/mva-probe/checkpoints/`.
3. Ran a weight-level probe (`code/inspect_lora.py`, results `code/lora_probe_results.json`,
   plot `../assets/mva_lora_perblock_norms.png`).
4. Verified the conditioning path at source level in the pinned DiffSynth pipeline.
5. Read the full paper incl. appendix (VLM judge protocol, data mixture, rubrics).

## 1. How the interface actually works (from code, not prose)

- The "masked visual action" is an ordinary RGB video passed as `control_video` to the stock
  Wan2.2-Fun-A14B-Control pathway. In `WanVideoUnit_FunControl.process`
  (`code/wan_video_pinned_3743b130.py:514-528`): the control video is VAE-encoded and
  **channel-concatenated** with the noisy latents at the DiT input (`x = cat([x, y], dim=1)`,
  line 1252) before patchify. The paper's "concatenation" claim is exactly the base model's
  native control channel — no architectural change at all.
- **There is no explicit binary-mask channel.** The extra `y` channels are zero-filled at
  inference (line 524). The mask exists only implicitly in pixel space: unrevealed regions are
  uniform gray; the revealed entity (robot render or segmentation cut-out) keeps its pixels.
- The `reference_image` (first real frame) is VAE-encoded and prepended as an **extra token
  frame** via `dit.ref_conv` (lines 1269-1274), not channel-concatenated.
- Mask construction (paper §4.1 only — the renderer is **not released**, README says
  "Tools for rendering the URDF robot control videos from DROID episodes are coming soon"):
  (a) segmentation-based: SAM ("A robotic arm" prompt) cut-out of the robot from DROID videos;
  (b) rendering-based: URDF mesh rendered from recorded joint states with refined camera
  extrinsics (PointWorld protocol), robot rendered **translucent with bright-red gripper
  fingers** to beat self-occlusion. Trained only on active-entity (robot) masks; inverse
  modeling (reveal object motion) is a **zero-shot** generalization of the same checkpoint.
- Temporal structure: dense, not sparse — the control video covers the full trajectory
  (81 frames, uniform resample in `infer.py`), 480x832, 15 fps.
- Released `infer.py` only exercises the **forward** direction (robot render in → scene out).
  Inverse-mode control videos (object masks) would have to be built by the user.

## 2. Weight-level probe findings (`code/lora_probe_results.json`)

Both files: 800 tensors = 40 blocks x {self_attn, cross_attn} x {q,k,v,o} + {ffn.0, ffn.2},
A/B pairs, **all exactly rank 256**, BF16, 1,226,833,920 params per expert (~8.8% of the 14B
expert — a very fat LoRA). Structure matches `train_control.sh` targets exactly.

- **The interface lives in the early blocks.** Per-block ||BA||_F peaks at blocks 2-3
  (~165 high expert) and decays monotonically to ~12 at block 39. Top-5: blocks {2,3,0,4,6}
  (high), {1,4,2,0,3} (low). The conditioning wiring is front-loaded; late blocks are barely
  touched.
- **Per-entry update RMS** (norm normalized by matrix size): in the **high-noise expert**,
  `cross_attn.k` (0.01236) and `cross_attn.q` (0.01060) are the largest — cross-attention
  (text-conditioning pathway) is rewired most, consistent with learning to rely on the visual
  control channel over the (deliberately generic) text prompt. In the **low-noise expert** the
  distribution is flatter, led by `self_attn.q` (0.00997).
- **High vs low experts:** per-block norm profiles correlate at r = 0.912; the high-noise
  expert carries ~14% more total update mass (mean 25.36 vs 22.18). Both experts learned the
  same interface layout; the high-noise (global structure) expert does more of the work.

## 3. Paper-vs-code consistency

| Claim (paper) | Released artifact | Verdict |
|---|---|---|
| rank-256 LoRA, Wan-Fun-Control 2.2 14B | weights all rank 256, both experts | match |
| concat conditioning, gray fill for masked regions | channel-concat in pinned pipeline; gray fill only in paper prose (renderer unreleased) | mechanism match; fill not code-verifiable |
| 15h data, ~10k steps, batch 4, 8xH200, 4 days | train_control.sh: 5 epochs, lr 1e-4, 480x832, targets q,k,v,o,ffn.0,ffn.2 | consistent (steps/epochs not cross-checkable without the CSV) |
| "release code, data, weights" | HF repo has weights only; no data, no renderer, no eval code | partial |
| Ablations | Table 2 ablates conditioning *signal* (masked vs skeleton vs EEF). **No mask-density ablation, no temporal-sparsity ablation, no entity-masking-fraction sweep.** Entity generalization is shown (BEHAVIOR R1-Pro bimanual, custom gripper) but not ablated quantitatively by mask type. | gap |

## 4. Policy-evaluation evidence quality

- **Simulation (Fig 9):** r = 0.982 — but this is **7 task-level points** (RoboCasa atomic
  tasks: close microwave/fridge/toaster/dishwasher, open drawer/dishwasher, coffee setup mug),
  one open-loop Diffusion Policy, 10 rollouts/scene, manual rubric scoring. It is a correlation
  across a handful of per-task success rates, **not** rank correlation across checkpoints or
  policies. Consistent positive bias (imagination overestimates success).
- **Real world (Fig 10):** 4 tasks x 20 demos, per-trial rubric progress (appendix H rubrics),
  paired demo-vs-imagination distributions "closely match" with the same positive bias. No
  correlation coefficient reported for the real-world case.
- **Planning (Fig 8):** Best-of-N (N=10) with a Gemini 3.1 Pro judge (strict anti-hallucination
  rubric, appendix E: ghost-contact / coasting / teleport penalties, lexicographic ranking) gives
  +7% to +26% success over base policy on 6 RoboCasa tasks; monotone-ish in N (1→10).
- **Action extraction (Fig 11):** zero-shot inverse modeling + learned IDM hits 90% on
  CoffeeServeMug vs DP 50% / ACT 80% / SmolVLA 85%, 20 trials each; baselines see only the
  first generated frame — a favorable comparison setup for the video model.
- Verdict: directionally convincing demos, thin statistics. The r=0.982 headline rests on 7
  points from one policy class; usable as evidence that imagined rollouts *track* execution,
  not that they *rank* policies.

## 5. Local reproduction feasibility (RTX 4060 Ti 16GB / 30GB RAM)

Not attempted; documented blocker instead.

- Weight footprint: 2 x 14B DiT experts (bf16 ~28GB each) + T5-XXL (~11GB) + VAE → ~67GB;
  single expert alone exceeds 16GB VRAM.
- `infer.py --low-vram` uses DiffSynth disk offload (`offload_device="disk"`): each of 50
  denoising steps x 2 experts streams ~28GB from disk ≈ **2.8TB minimum disk traffic per
  81-frame clip**; 30GB RAM cannot cache even one expert. Est. hours per clip on NVMe, plus a
  ~70GB base-model download (ModelScope). The pinned commit's inference path is bf16-only;
  no supported fp8/int8 variant of Wan2.2-Fun-A14B-Control.
- Conclusion: code+weights-level probe (done here) is the right depth for this hardware.

## Evidence files

- `code/inspect_lora.py`, `code/lora_probe_results.json` — probe script + full JSON
- `code/weights_sha256.txt` — integrity record for both LoRAs
- `code/wan_video_pinned_3743b130.py` — pinned DiffSynth pipeline (conditioning path)
- `../assets/mva_lora_perblock_norms.png` — per-block LoRA update norms, both experts
- `../assets/mva_interface_fig3_4.png` — paper Fig 3+4 (applications + mask construction)
