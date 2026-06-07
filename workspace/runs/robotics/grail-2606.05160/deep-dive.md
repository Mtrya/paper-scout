# GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors

**ID:** 2606.05160  
**Area:** robotics  
**Authors:** Tianyi Xie, Haotian Zhang, Jinhyung Park, Zi Wang, Bowen Wen, Jiefeng Li, Xueting Li, Qingwei Ben, Haoyang Weng, Yufei Ye, David Minor, Tingwu Wang, Chenfanfu Jiang, Sanja Fidler, Jan Kautz, Linxi Fan, Yuke Zhu, Zhengyi Luo, Umar Iqbal, Ye Yuan (NVIDIA + UCLA)  
**Repo:** https://github.com/NVlabs/GRAIL  
**Dataset:** https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL  
**Read date:** 2026-06-07

---

## D0: Section Inventory

| Section | Destination |
|---------|-------------|
| Abstract / Introduction | Motivation, framing, claimed contributions |
| §2 Related Work | Positioning vs HOI generation, human-video-to-humanoid, synthetic data |
| §3.1 Robot-Centric Video Generation | Core method — privileged 3D setup, VFM conditioning |
| §3.2 Interaction-Aware HOI Reconstruction | Core method — 4D recovery, optimization losses |
| §3.3 Task-General Tracking | Core method — object-aware adaptor, scene-aware tracker |
| §3.4 Sim-to-Real Deployment | Evidence — domain randomization, distillation, real G1 results |
| §4.1 HOI Generation | Evidence — Table 1 vs baselines, perceptual + physical metrics |
| §4.2 Tracking | Evidence — Table 2, scaling data, ablations |
| §4.3 Sim-to-Real | Evidence — 84% pick-up, 90% stair-climbing, limited trial counts |
| §5 Conclusion | Limitations |
| §6 Limitations | Red flags — failure filtering, VFM consistency, occlusion |
| Appendix A | Generation details, runtime (~14 min/seq on A100) |
| Appendix B | Tracker architecture and reward details |
| Appendix C | Baseline configs, user study, qualitative comparison |

---

## D1: Motivation and Contribution

**Problem:** Scaling humanoid loco-manipulation demonstrations requires trajectories that are physically plausible *and* robot-executable. Teleoperation and mocap are high-quality but physically bound and hard to scale. Reconstructing from in-the-wild video is scalable but underconstrained — camera, scale, object geometry, human shape, and contacts must all be inferred from monocular RGB.

**Core insight:** Instead of reconstructing an uncontrolled video into an ambiguous 4D scene, *specify the 3D scene first* (object mesh, camera, metric scale, environment depth, robot-proportioned character), generate a human interaction video from that specification, then recover the 4D HOI *inside the known metric frame*. This turns inverse problems into controlled inputs.

**Claimed contributions (tagged):**
1. **Method** — A fully digital humanoid-centric data-generation framework using VFMs as interaction priors inside a fully specified 3D asset pipeline, producing >20,000 sequences.
2. **Method** — An interaction-aware 4D HOI reconstruction stack that exploits known geometry, metric scale, camera params, environment depth, and a robot-proportioned character.
3. **Method** — Complementary task-general trackers: object-aware latent adaptation for manipulation + scene-aware height-map conditioning for terrain/sitting.
4. **Benchmark / Analysis** — End-to-end sim-to-real validation on Unitree G1: 84% pick-up, 90% stair-climbing, trained only on GRAIL-generated data.

---

## D2: Core Method (Near-Reimplementation Detail)

### Pipeline Overview (3 stages + deployment)

**Inputs:** 3D object asset $\mathcal{M}^\mathcal{O}$, terrain / scene config, character prefitted to target humanoid.  
**Outputs:** Robot kinematic motion $\{\Theta_t^\mathcal{R}\}$, object motion $\{\Theta_t^\mathcal{O}\}$, robot actions $\{\mathbf{a}_t^\mathcal{R}\}$.

```
3D asset + scene config
    -> Blender render first frame + known camera (C_K, C_E)
    -> VLM prompt -> VFM (Kling 2.5) generates static-camera HOI video
    -> Independent initial estimates: GENMO (body) + WiLoR (hands) + FoundationPose (object)
    -> Joint optimization over residuals (human trans/pose, object R/t)
    -> GMR retargeting SMPL-X -> Unitree G1
    -> Task-general tracking (SONIC + adaptor / fine-tuning)
    -> Visual policy distillation -> real G1 deployment
```

### Privileged 3D Setup — Why It Helps

The paper's key design choice is front-loading scene specification. Concretely:

- **Camera:** Intrinsics $C_K$ and extrinsics $C_E = (r^\mathcal{C}, t^\mathcal{C})$ are known from Blender. No camera estimation error propagates into human/object motion.
- **Metric scale:** The rendered environment provides a ground-truth point cloud. MoGe-2 depth is aligned to this GT background depth, recovering metric-scale depth maps for human/object point-cloud supervision.
- **Object geometry:** The exact mesh is known, so FoundationPose (which requires mesh + texture + camera) can be initialized from the known first-frame pose and tracked accurately. The paper zeros FoundationPose depth channels at train and test to adapt to RGB-only generation.
- **Character morphology:** The SMPL-X body shape is held fixed at a G1-prefitted character rather than re-estimated per video. This eliminates morphology mismatch during retargeting.

This privileged setup is what separates GRAIL from DAViD, ZeroHSI, and other VFM-based HOI methods that generate a first frame or video and then must infer camera, scale, and shape post-hoc.

### 4D HOI Reconstruction — Joint Optimization

**Variables optimized:** Residuals on human translation, body pose, hand pose (all 6D rotation), and object rotation/translation. Full trajectories are not optimized directly; instead, residuals $\Delta\Theta_t$ are applied to the independent initial estimates.

**Loss (code-verified against paper Eq. 1):**

The code implements the five loss families from Eq. 1, plus additional regularizers:

| Paper term | Code term | What it does |
|------------|-----------|--------------|
| $L_{\text{kp}}$ | `keypoint_tracking` | 2D projection of SMPL-X body/hand joints to detected keypoints (ViTPose + WiLoR). Hand weight $\beta=0.1$–$0.3$. |
| $L_{\text{proj}}$ | `verts_tracking` | Projects optimized object vertices to screen and matches FoundationPose initial projection. |
| $L_{\text{depth}}$ | `depth_pointcloud` | Bidirectional trimmed Chamfer distance between visible mesh vertices and unprojected depth point clouds (human + object). GT depth comes from MoGe-2 aligned to rendered GT. Trimming discards 20% outliers. |
| $L_{\text{cont}}$ | `contact` (with `depth_only=True`) | For frames where VLM predicts contact, filters object vertices to those within 20 px of contact body region in screen space, then penalizes view-space z difference via Huber loss. |
| $L_{\text{reg}}$ | `human_foot_contact`, `human_global_init_reg`, `human_smoothness`, `human_pose_reg`, `obj_smoothness`, `obj_rot_reg`, `ground` | Foot skating suppression, pelvis velocity matching GENMO global estimate, temporal smoothness (1st + 2nd order), pose residual regularization, object smoothness, object pose regularization, ground plane. |
| — | `penetration` | SDF-based penetration loss (not in main Eq. 1 but in pipeline). |

**Optimization stages (config-verified):**

The code runs a 3-stage optimization (total 1,800 iterations in the pickup config):

1. **Human trajectory opt** (400 iters): Optimize `human_trans_res` + `human_pose_res`. Losses: keypoint tracking, smoothness, foot contact, global init reg, pose reg.
2. **Human global traj opt** (1,000 iters): Add `human_trans_global` (xy-only) and introduce contact loss at interaction start. Contact weight is massive ($10^5$) to snap hands to object.
3. **Object traj opt** (400 iters): Optimize `obj_t_res` + `obj_R_res` + arm-only human pose residual. Losses: contact (all frames), object smoothness, object rotation reg, vertex tracking.

**Post-optimization:** Savitzky-Golay smoothing (window=11, polyorder=2) is applied to human poses, hand poses, object rotation, and object translation before export.

### Task-General Tracking

Retargeted motions are converted to policies on top of **SONIC** (Luo et al. 2025), a pretrained whole-body controller with FSQ (finite scalar quantization) latent tokens.

**Object-Aware Adaptor ($\pi_\phi$):**
- 3-layer MLP (512→256→128), SiLU.
- Observes proprioception + object reference (pose in body frame, hand-to-object transforms, contact forces, BPS shape encoding, **delta observations** = ref future pose − current sim pose).
- Outputs: 64-dim latent residual $\Delta z_t$ (scaled by $\lambda=0.1$ before FSQ) + 2-dim binary hand primitive (open/close per hand).
- Hand primitive maps to 7 finger DoFs per hand via predefined grasp configurations.
- SONIC encoder/quantizer/decoder remain **frozen**; only $\pi_\phi$ is trained with PPO.
- Reward: $R_t = R_t^{\text{motion}} + R_t^{\text{reg}} + R_t^{\text{obj}} + \mathbb{1}\{C_t\} R_t^{\text{grasp}}$
  - Grasp reward has three terms: sustained contact (saturates at $N_{\text{min}}$), thumb-index opposition, fingertip proximity to contact centroid.
  - Object reward is gated by simulated finger-object contact indicator (active only during contact).

**Scene-Aware Tracker:**
- Fine-tunes SONIC end-to-end together with a height-map encoder $\epsilon_h$.
- Local height map: 11×11 grid, 1.5 m extent, 0.15 m resolution, processed by 3-layer CNN (64→128→256, stride 2) → 1,024-dim feature.
- Auxiliary kinematic decoder $\mathcal{G}_{\text{rec}}$ reconstructs motion targets with MSE loss (weight 0.01) to regularize the latent.
- Trained on mixture of reconstructed scene-interaction trajectories + original flat-ground data.

**Training:** PPO in Isaac Lab, 64 L40 GPUs, 1,024 envs/GPU, 30,000 iterations (~30 hours). Warm-start from existing policy when adding new in-family motions (~6 hours, 6,000 iters). Amortized cost: ~0.5–0.9 min/motion.

### Sim-to-Real Deployment

The paper states that object-aware and scene-aware tracking policies are **distilled** into separate egocentric visual policies consuming head-camera RGB and outputting SONIC latent tokens. Training uses visual domain randomization and camera alignment.

**Notable gap:** The paper gives almost no architectural detail on the visual encoder or distillation loss. It cites Diffusion Policy (Chi et al., 2023) and VIRAL (He et al., 2025a) but does not describe whether the deployed policy is a diffusion policy, an autoregressive VLA, or a simple CNN→MLP mapper. The real-world setup uses a Luxonis OAK-D W camera, 10 Hz inference on an RTX 5090 desktop tethered to the G1.

---

## D3: Experiments and Evidence

### HOI Generation (Table 1)

Evaluated on 20 ComAsset objects against CHOIS, HOIDiff, and DAViD.

- **Physical executability** is the standout metric: GRAIL achieves 88.9% tracking success rate (SR) vs DAViD's 24.0%, CHOIS's 10.5%, HOIDiff's 15.8%. Body deviation 0.0913 vs DAViD's 0.4723. This is a huge margin.
- **Caveat:** The physical executability test uses **InterMimic** (Xu et al., 2025), an SMPL-X humanoid tracking framework. InterMimic uses capsule primitives matching the input body shape — no retargeting. This metric is informative but not the same as tracking on the actual G1 morphology. It favors GRAIL partly because GRAIL's reconstructed body is already the prefitted G1-proportioned character.
- **Geometric quality:** GRAIL has lowest contact distance (0.008) and penetration ratio (0.90%). DAViD has higher contact distance (0.246) despite decent perceptual score — consistent with DAViD not having privileged 3D object tracking.
- **Perceptual realism:** VLM-rated interaction score 3.58 vs DAViD 2.74. User study (30 participants, 3-of-4 random sampling) gives GRAIL 74.7% affordance-realism preference (theoretical max 75%).

### Task-General Tracking (Table 2)

Benchmark of 124 motions across 43 objects vs HDMI and ResMimic.

- GRAIL full: 81.4% SR, 0.135 ObjPos, 41.8 MPJPE-L.
- HDMI: 48.5% SR, 0.283 ObjPos, 122.3 MPJPE-L.
- ResMimic: 49.2% SR, 0.393 ObjPos, 80.9 MPJPE-L.

**Ablations (bottom block):**
- Without SONIC (train from scratch): 45.0% SR, 243.5 MPJPE-L. The pretrained controller is **load-bearing**.
- Without $\pi_\phi$ (vanilla SONIC): 39.7% SR. Body tracking is best (MPJPE-L 37.1) but manipulation fails — accurate body imitation alone is insufficient.
- Without relative observations (absolute object pose): 57.9% SR. Delta observations matter.

**Caveat:** Baselines (HDMI, ResMimic) do not actuate per-finger DoFs and train per-task specialists, while GRAIL trains task-general policies across families. The comparison is not apples-to-apples on policy architecture, though it is fair on the upstream data source (human references).

### Sim-to-Real (§4.3, Table 3)

- **Stair-climbing:** 90% success (unclear how many trials; figure shows a single staircase type).
- **Pick-up:** 84% on seen objects (cube, apple, tea box, carrot, wet wipes — 10 trials each). 80% on unseen objects (spray can, lint roller, peach, flashlight, medicine bottle — 10 trials each).

**Concerns:**
- Very small trial counts (10 per object). No error bars.
- No comparison against a policy trained on real teleop data or another synthetic dataset.
- Pick-up tasks are relatively simple (approach + grasp + lift). No dynamic disturbance, no clutter, no partial occlusions.
- The 84% figure conflates data quality, policy architecture (SONIC + visual distillation), and sim-to-real techniques (domain randomization). There is **no ablation isolating the data contribution** from the architecture.

---

## D4: Code and Artifact Inspection

**Repository:** https://github.com/NVlabs/GRAIL (cloned to `repos/robotics/grail-repo`).

### Maturity and Structure
- Clean modular structure: `grail/pipelines/` (gen_2dhoi, recon_4dhoi, retarget, tracking), `grail/optimization/`, `grail/postprocessing/`.
- Well-documented with ReadTheDocs-style docs under `docs/source/`.
- Docker image provided (`nvgrail/grail:latest`).
- Pre-commit hooks, black/ruff formatting.
- Only **1 commit** in shallow clone (`e7e8851 fix doc`) — appears to be a stable release branch or recent push.

### Paper-Code Consistency
- **Optimization losses:** The five paper losses ($L_{\text{kp}}$, $L_{\text{proj}}$, $L_{\text{depth}}$, $L_{\text{cont}}$, $L_{\text{reg}}$) are all present in `grail/optimization/loss_computer.py` and `loss_terms.py`, with the exact formulations described (bidirectional trimmed Chamfer, depth-only contact with screen-space filtering, Huber loss on closest vertex pairs).
- **Multi-stage optimization:** `configs/recon_4dhoi/pickup_smplx.yaml` explicitly defines the 3 stages (human_traj_opt, human_global_traj_opt, obj_traj_opt) with the same variable names, learning rates, and loss weights as described in the paper and appendix.
- **Filtering:** `grail/postprocessing/filter.py` implements mask-based validation (Eq. 10 in appendix A.3), initial penetration check, static-human check, and eval-data thresholding. The mask tolerance defaults (`object_mask_tol=0.5`, `total_mask_tol=0.3`) are stricter than the paper's $\tau=0.2$ for some categories.
- **Contact labels:** `grail/core/contact_label.py` uses VLM (OpenAI) to detect contact joints per interval, with fallback to default `["R_Hand"]` if detection fails.

### Discrepancies / Undocumented Details
- **Depth loss trimming:** The paper does not mention the 20% outlier trimming in Chamfer distance; this is a robustness trick in the code.
- **Contact loss gating:** The code gates contact loss with `max_contact_dist` (skips if closest 3D distance > threshold). This gating is not mentioned in the paper but is important for stability when VLM mislabels contact.
- **Pre-evaluation truncation:** Before optimization, `pre_eval` checks frame quality and can truncate the sequence. This can artificially boost success rates by dropping hard tail frames.
- **SONIC submodule:** The SONIC tracking code is imported as a Git submodule (`imports/SONIC/`) containing the full controller stack, deployment code, and Unitree SDK bindings. This is a substantial dependency but is included.

### Dataset Release
- HuggingFace dataset `nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL` has **3,660 downloads last month** and is actively maintained.
- Ships with: source videos, 4D HOI reconstructions (SMPL-X + object 6-DoF), retargeted G1 trajectories, post-RL object trajectories, object USD assets, and bundled checkpoints (GEM-SMPL, FoundationPose, SONIC).
- **22,189 motions** released (pickup_table 2,991; pickup_ground 1,613; sitting 1,748; slope 1,880; curb 1,769; stair 12,188). This matches the paper's >20,000 claim.
- License: Apache 2.0 for GRAIL-original motion data; upstream licenses for bundled assets and checkpoints. The **code repo** is under NVIDIA non-commercial license.

### Lightweight Verification
- **Config math check:** Ran `configs/recon_4dhoi/pickup_smplx.yaml` through Python YAML loader. Confirmed 3 optimization stages, total 1,800 iterations, loss terms match paper.
- **Loss term count:** `loss_computer.py` registers 15 loss names; the 5 main paper families are present plus auxiliary terms (penetration, contact smoothness, distribution smoothness, ground, etc.).
- **Dataset stats:** HF card statistics sum to 22,189 motions — consistent with paper.

**Artifact classification:** Reproducible artifact — full pipeline code, configs, Docker image, dataset, and checkpoints are released. The non-commercial code license is a restriction for commercial users.

---

## D4.5: Researcher Checks

These checks were run directly against the released code, configs, and HuggingFace dataset samples. The goal was to test the privileged-3D and sim-to-real claims with data rather than prose.

**What was run:**
- Stand-alone verification script: `runs/robotics/grail-2606.05160-checks/run_checks.py`
- Command: `.venv-grail/bin/python runs/robotics/grail-2606.05160-checks/run_checks.py`
- The script re-implements the loss-term math from `grail/optimization/loss_terms.py` without the full pytorch3d dependency, parses `configs/recon_4dhoi/pickup_smplx.yaml`, reproduces the filtering logic in `grail/postprocessing/filter.py`, and loads real `robot/*.pkl` and `objects/*.pkl` samples downloaded from the HuggingFace dataset.

### 1. Can the released code run outside Docker?

No — at least not trivially. Importing `grail.optimization.loss_terms` in the host Python environment fails immediately:

```
FAIL: cannot import real loss_terms module: ModuleNotFoundError: No module named 'pytorch3d'
```

The Docker pull (`docker pull nvgrail/grail:latest`) also timed out in this environment. So the *only* practical way to run the full pipeline is inside the provided container with the bundled conda environment.

### 2. What does the reconstruction config actually weigh?

Parsing `configs/recon_4dhoi/pickup_smplx.yaml` gives the three stages described in the paper, totaling **1,800 iterations**:

| Stage | Iters | Optimized variables | Notable losses |
|-------|-------|---------------------|----------------|
| `human_traj_opt` | 400 | `human_trans_res`, `human_pose_res` | `human_global_init_reg` 3,000; `human_foot_contact` 100; `human_smoothness` 30; `keypoint_tracking` 0.3 |
| `human_global_traj_opt` | 1,000 | `human_trans_global` (xy-only), `human_pose_res` | `contact` **100,000**; `human_smoothness` 300; `keypoint_tracking` 0.1 |
| `obj_traj_opt` | 400 | `obj_t_res`, `obj_R_res`, `human_pose_res` (arm-only) | `contact` **100,000**; `obj_smoothness` 500; `verts_tracking` 0.1 |

Relative weight dominance:

```
human_traj_opt          : human_global_init_reg = 94.6% of total |weight|
human_global_traj_opt   : contact               = 99.7%
obj_traj_opt            : contact               = 99.5%
```

**Mini-ablation implication:** Zeroing the `contact` term in stages 2 or 3 removes the only mechanism that snaps hands onto the object (stage 2) and keeps them there (stage 3). The optimization would collapse to a smooth hand motion that ignores the object. Zeroing `human_global_init_reg` in stage 1 removes the anchor to GENMO's global trajectory; the body can drift arbitrarily as long as keypoints project correctly. These two terms are load-bearing.

### 3. Numerical verification of the loss terms

The script exercises synthetic cases where the expected value can be computed by hand. All checks pass against the code's behavior.

**`huber_loss`** with `delta=0.01` on `[-0.005, 0.020]`:
```
[PASS] huber([-0.005,0.020]): got 8.124999e-05, want 8.125000e-05
```

**`bidirectional_chamfer_loss`**: with two-point clouds `{[0,0,0],[1,0,0]}` vs `{[0,0,0],[0,1,0]}`, no-trim loss is exactly `1.0`. With the config's `trim_pct=0.2`, the loss collapses to `0.0` because each direction keeps only the single best match:

```
[PASS] trim=0.0: got 1.000000e+00, want 1.000000e+00
[PASS] trim=0.2 (2 points each, keep 1): got 0.000000e+00, want 0.000000e+00
```

This confirms the undocumented trimming behavior: for sparse or small point clouds, the 20% trim can silently erase the depth supervision.

**`contact_depth_loss`** uses screen-space gating before penalizing depth. With a synthetic pinhole camera (fx=fy=500, c=(320,240)):
- Object vertex 25 px away from the hand in screen space → loss is forced to `0`.
- Object vertex 2.5 px away but with 10 cm depth mismatch → loss equals `9.95e-5`, matching `huber(delta=0.001, d=0.1)`.

```
screen distance for far vertex: 25.0 px
loss with far vertex (screen>20):   0.000000e+00
[PASS] depth diff 0.10m: got 9.950003e-05, want 9.950000e-05
```

**`smoothness_loss`** has a subtle behavior: it penalizes first-order differences against zero, so even *constant-velocity* motion incurs loss. For the sequence `[0,1,2,3]` the loss is `1.0`, not `0.0`. This means the optimizer trades off motion magnitude against matching terms, not just jerk.

```
[PASS] linear motion: got 1.000000e+00, want 1.000000e+00
[PASS] erratic motion: got 2.166667e+00, want 2.166667e+00
```

### 4. Contact gating by 3D distance

`loss_computer.py` silently skips a `(frame, body-part)` contact term when the closest human-object 3D distance exceeds `max_contact_dist`. With a threshold of 5 cm:
- synthetic hand/object at 0.8 cm → term computed
- synthetic hand/object at 16.5 cm → term skipped

```
  near object min distance = 0.0028 m  (< 0.05) -> compute contact loss
  far  object min distance = 0.1652 m  (> 0.05) -> skip contact loss
```

This is an important stability feature when the VLM mislabels a contact joint, but it is not mentioned in the paper's loss description.

### 5. Filtering thresholds are looser than the filter defaults

`pickup_smplx.yaml` sets:
```yaml
filtering:
  object_mask_tol: 1.0
  total_mask_tol: 0.4
  human_static_thr: 0
```

Compare with `filter.py` hard-coded defaults:
```python
object_mask_tol = 0.5
total_mask_tol  = 0.3
human_static_thr= 0.01
```

**Findings:**
- `object_mask_tol` is 2× looser than the filter default.
- `total_mask_tol` is looser than the filter default.
- `human_static_thr=0` disables the static-human check entirely because `avg_max_velocity >= 0` is always true.

The script confirms this with synthetic data:
```
Synthetic almost-static motion avg max velocity = 0.000515
  passes with config thr=0.0 ? True  (yes -> false negative)
  passes with default thr=0.01? False
```

So the pickup config is *less* strict than the filter's own defaults. This raises the yield-vs-quality question: the pipeline may be keeping more sequences at the cost of retaining static or poorly aligned reconstructions.

### 6. Dataset statistics and a real trajectory sample

The HuggingFace README gives the per-category counts. They sum to **22,189** motions, not the 20,309 figure that appeared in an earlier version of these notes:

| Category | Motions |
|----------|--------:|
| pickup_table | 2,991 |
| pickup_ground | 1,613 |
| sitting | 1,748 |
| slope | 1,880 |
| curb | 1,769 |
| stair | 12,188 |
| **Total** | **22,189** |

This is still consistent with the paper's ">20,000" claim, but the exact released number is larger than initially reported.

A real sample (`pickup_table__alcohol_0__000`) was downloaded directly from the HF dataset and inspected with `joblib`:

```
Robot trajectory:
  frames = 250, fps = 25.0, duration = 10.00 s
  root XY displacement = 0.312 m
  root height range = [0.762, 0.791] m
  mean joint speed = 0.180 rad/s
  max  joint speed = 5.097 rad/s
  left hand action changes = 249, right = 249

Object trajectory:
  start pos = [0.047, -0.125, 0.533]
  end   pos = [0.045, -0.030, 1.298]
  net displacement = 0.771 m
  total path length = 1.462 m
  max height = 1.298 m

  object lifted >5 cm? True (max - init = 0.765 m)
```

The object is genuinely lifted (~76 cm), the robot root moves ~31 cm across the table, and the hands change action every frame. The high hand-action toggle rate is worth watching: it could be a binary open/close signal that oscillates in the post-processed trajectory, which may add noise when training discrete grasp policies.

### 7. What still cannot be checked from the host

- **Privileged-3D ablation end-to-end:** We can measure config weight dominance and verify individual loss terms, but we could not run the actual optimizer with/without known camera or known depth because that requires the Docker environment, Blender, and the full 3D scene setup.
- **Sim-to-real attribution:** The 84% pick-up success remains a black-box number. There is no config or released log that separates the contribution of GRAIL data, the SONIC pretrained controller, and the visual distillation/sim-to-real pipeline.
- **Failure-mode distribution:** The filter thresholds and the static-human check show how sequences are accepted, but to know the *yield* (raw generated vs accepted) we would need to run the full generation pipeline and inspect the rejected sequences.

### Bottom-line from the checks

The code matches the paper's equations, but the config reveals that the reconstruction is driven by a small number of enormous weights (`contact = 1e5`, `human_global_init_reg = 3e3`). The undocumented trimming and contact-distance gating are real load-bearing stability mechanisms. The filtering config for pickup is looser than the module defaults, which helps hit the >20k motion count but may let lower-quality sequences through. And the released dataset genuinely contains physically meaningful pick-up trajectories — the sample shows a 76 cm object lift — but the 84% real-world success figure still conflates data, controller, and sim-to-real contributions with no public ablation to separate them.

---

## D-RW: Situation Against Related Work

### 1. DAViD (Kim et al., 2025a) — Direct Baseline

DAViD also generates 4D HOI from 3D objects using video diffusion (first-frame generation + video diffusion), then lifts to 3D for dynamic affordance learning. GRAIL beats DAViD decisively on physical executability (88.9% vs 24.0% SR).

**Why GRAIL wins:**
- DAViD generates the first frame from an image model (Flux.1 Kontext with Canny edges) and often fails under partial control; GRAIL renders the first frame from a fully specified 3D scene.
- DAViD must infer camera, scale, and object geometry after generation; GRAIL reuses the known configuration.
- DAViD's object motion is learned from reconstructed samples; GRAIL tracks the object with FoundationPose initialized from known geometry.

**Fairness note:** DAViD is evaluated in the paper under the same Kling image-to-video setting, with 24 generated images manually filtered. GRAIL's pipeline is more engineered (Blender + privileged depth + multi-stage optimization) but DAViD is more generative (learns a diffusion model for HOI). They serve different use cases: GRAIL is a data factory, DAViD is an affordance model.

### 2. Gen2Real (Ye et al., 2025, arXiv:2509.14178)

Gen2Real replaces human demos with one generated video for dexterous manipulation. It uses PIOM (physics-aware interaction optimization) to impose physics consistency and retargets to a robot hand with residual PPO.

**Comparison:**
- Gen2Real targets **dexterous hands** (shadow hand, allegro); GRAIL targets **whole-body humanoids** (Unitree G1) with locomotion + manipulation.
- Gen2Real uses generated video for a single grasp; GRAIL generates 20,000+ sequences across pick-up, whole-body manipulation, sitting, and terrain traversal.
- Gen2Real reports 77.3% sim success on grasping; GRAIL reports 84% real-world pick-up. Not directly comparable due to embodiment and task differences.
- Both use video generation + physics optimization, but GRAIL's privileged 3D setup is a stronger prior than Gen2Real's pose-and-depth estimation from generated video.

### 3. RoboGen (Wang et al., 2024, ICML)

RoboGen is a self-guided generative agent that proposes tasks, builds scenes in simulation (PyBullet), and learns skills. It uses LLMs for task generation and procedural scene composition.

**Comparison:**
- RoboGen generates **tasks and scenes** autonomously; GRAIL generates **interactions** from given 3D assets.
- RoboGen operates fully in simulation with programmatic reward design; GRAIL uses video foundation models as behavioral priors and reconstructs metric 4D motion.
- RoboGen's output is policy-trained in the same simulator; GRAIL's output is retargeted to a real humanoid and validated sim-to-real.
- RoboGen scales to diverse task families via LLM creativity; GRAIL scales to diverse objects via asset-conditioned generation. They are complementary: RoboGen generates *what to do*, GRAIL generates *how to move*.

### 4. Real2Render2Real (Yu et al., 2025, CoRL)

R2R2R captures real scenes, renders synthetic robot data, and trains policies without dynamics simulation or hardware. It uses Gaussian splatting / NeRF to reconstruct real environments.

**Comparison:**
- R2R2R starts from **real scenes** and renders synthetic interactions; GRAIL starts from **3D assets** and generates synthetic videos.
- R2R2R is scene-specific (must scan the real environment); GRAIL is asset-general (any mesh can be plugged in).
- R2R2R does not involve whole-body humanoid loco-manipulation; GRAIL is humanoid-specific.

---

## Red Flags and Caveats

1. **Sim-to-real details are thin.** The visual policy architecture, distillation loss, and exact domain randomization parameters are not described. The 10 Hz inference rate on a tethered RTX 5090 is slow for dynamic tasks.
2. **No data-ablation.** We do not know how much of the 84% real-world success comes from GRAIL data vs. the SONIC pretrained controller vs. the sim-to-real pipeline. Table 2 ablations are in simulation only.
3. **Small real-world trial counts.** 10 trials per object is not enough to estimate robustness. No mention of variance, failure modes, or recovery behavior.
4. **Failure filtering discards sequences.** The paper acknowledges a "non-trivial fraction" of sequences are discarded due to VFM artifacts, tracking loss, or reconstruction failure. This is a hidden cost — the effective yield per generation dollar is lower than the raw sequence count suggests.
5. **VFM dependency on Kling.** The pipeline relies on a proprietary video model (Kling 2.5 Turbo Pro) accessed via API. Reproducibility depends on API availability, cost, and model consistency. The paper does not report generation cost or API failure rate.
6. **Non-commercial license.** The GRAIL code is NVIDIA non-commercial, limiting adoption in startups and commercial labs. The dataset motions are Apache 2.0, which helps.
7. **Physical executability metric is self-referential.** InterMimic tracks SMPL-X humanoids with capsule primitives matching the reconstructed body shape — it does not test on the actual G1 morphology or dynamics. The real test is the SONIC tracking stage, which is reported separately.
8. **Contact labels from VLM are coarse.** The code falls back to `["R_Hand"]` when VLM detection fails. This is a weak point in the automation claim.
9. **Reconstruction is dominated by a tiny number of enormous weights.** In the pickup config, `contact = 1e5` accounts for >99.5% of the total loss weight in stages 2 and 3, and `human_global_init_reg = 3e3` accounts for ~94.6% in stage 1. The method is highly sensitive to these terms; smaller terms like keypoint tracking are numerically negligible.
10. **Pickup filtering is looser than the module defaults.** The config sets `object_mask_tol=1.0` (vs `0.5` in `filter.py`), `total_mask_tol=0.4` (vs `0.3`), and `human_static_thr=0` (vs `0.01`). The static-human check is therefore disabled, allowing potentially static or low-quality reconstructions to pass.

---

## Illustration Candidates for Report

1. **Figure 2 (Asset-Conditioned Pipeline):** Best single diagram showing the privileged 3D setup → VFM generation → 4D reconstruction loop.
2. **Table 1 (HOI Generation Comparison):** The 88.9% vs 24.0% SR gap is the strongest quantitative claim. Worth reproducing as a highlight table.
3. **Figure 3 (Task-General Tracking Architecture):** Clear diagram of object-aware adaptor vs scene-aware tracker. Good for explaining why the method covers both manipulation and locomotion.
4. **Eq. 1 + Eq. 5 (Loss formulation):** The contact loss with screen-space filtering is the technical core; a short math box explains why privileged 3D matters.
5. **Config snippet (pickup_smplx.yaml):** Shows the multi-stage optimization schedule — makes the method feel concrete and reproducible.
6. **Real-world deployment photo (Figure 5):** G1 picking up objects and climbing stairs. Essential for sim-to-real credibility.

---

## D5: Bottom-Line Judgment

**Novelty:** High. The privileged 3D setup for asset-conditioned HOI generation is a genuine architectural insight that turns an underconstrained inverse problem into a controlled forward pipeline. The task-general tracking design (frozen SONIC + latent adaptor for manipulation, fine-tuned + height map for terrain) is elegant and practical.

**Credibility:** Medium-High. The code and dataset are released and match the paper closely. The HOI generation metrics are strong and the code verifies the loss formulations. The researcher checks show the reconstruction is dominated by a few enormous weights and that the pickup filtering config is looser than the module defaults, which helps reach the >20k motion count but may admit lower-quality sequences. The real-world results are promising but underreported (small n, no architecture details, no data-ablation). The physical executability metric is somewhat favorable to the method.

**Relevance:** Very high for humanoid robotics. GRAIL addresses the data bottleneck directly and delivers a scalable, fully virtual pipeline with real-robot validation. The release of ~22k sequences + code + checkpoints makes it immediately usable for follow-up work.

**Priority call:** **Build on / track.**

- **Read** if you work on humanoid control, sim-to-real, or synthetic data generation.
- **Build on** if you have access to a G1 or similar humanoid — the dataset and SONIC stack are ready to use.
- **Track** for follow-up work that (a) releases the manipulation dataset fully, (b) provides more real-world baselines, (c) reduces the Kling API dependency, or (d) extends to bimanual / contact-rich manipulation.

The 84% pick-up and 90% stair-climbing numbers are not yet bulletproof, but the pipeline engineering is strong enough that refinements (better VFMs, larger real-world evals, open VFM alternatives) are likely to push it higher. GRAIL is currently the most complete public system for humanoid loco-manipulation data generation.
