# Embodied-R1.5 Deep-Dive: From Pointing Specialist to Embodied Foundation Model

**Run:** 2026-06-11-alebench-worldpilot-embodiedr1-nextforcing  
**Paper:** Embodied-R1.5: Evolving Physical Intelligence via Embodied Foundation Models (arXiv:2606.11324)  
**Authors:** Yifu Yuan et al. (Tianjin University, Tencent Hunyuan)  
**Code:** `code/embodiedr1/` (cloned from https://github.com/pickxiguapi/Embodied-R1.5)  
**Investigator:** Paper Scout subagent

---

## Top-line thesis

Embodied-R1.5 is the most systematic open-source attempt yet to unify embodied reasoning in a single 8B VLM. It does not invent a new architecture; instead, it executes an extremely deliberate **data-system + training-recipe + evaluation-infrastructure** play. The core bet is that a VLM backbone with sufficiently internalized spatial cognition, planning, and pointing can substitute for large-scale action pretraining when adapted downstream into a VLA. The empirical results on SimplerEnv (92.4% vs π₀.₅'s 72.7%) and LIBERO (97.3% without action pretraining) strongly support this hypothesis in the evaluated settings. What is less certain is whether the released artifacts fully match the paper's claims—particularly around the closed-loop PGC autonomy framework and the completeness of the training data release.

---

## 1. What is an Embodied Foundation Model (EFM), and how it differs from VLA

The paper draws a sharp architectural distinction:

| | **EFM (Embodied-R1.5)** | **VLA (e.g., π₀, OpenVLA)** |
|---|---|---|
| **Output modality** | Text tokens: reasoning, plans, coordinates, trajectories | Continuous actions (delta joint positions, end-effector poses) |
| **Pretraining data** | Diverse embodied VQA, pointing, planning, spatial reasoning | Large-scale robot demonstration datasets (hundreds of thousands of action trajectories) |
| **Action generation** | Indirect: points/trajectories → low-level skill executor | Direct: model outputs actions end-to-end |
| **Generalization mechanism** | Reasoning-driven: understands *why* an action makes sense | Imitation-driven: reproduces statistical action patterns |
| **Parameter count** | 8B (single unified model) | Often 10B+ with separate action heads |

The EFM→VLA extension in Embodied-R1.5 is minimal: attach a lightweight DiT-B flow-matching action expert to the frozen (or lightly tuned) VLM backbone. The hypothesis, validated in the experiments, is that because the VLM has already internalized *what* to do and *where* to interact, the action head only needs to learn a simple mapping from understood intent to continuous motor commands. This is conceptually close to the System-2-reasons/System-1-acts dual-system architecture discussed in the paper.

**Key insight:** The EFM is not a replacement for VLAs; it is an *upstream substrate* that makes VLA training cheaper and more sample-efficient. The LIBERO ablation (Figure 9 in the paper) shows Embodied-R1.5 backbones converging 10pp faster than Qwen3-VL-8B at 10K steps, confirming genuine capability transfer rather than warm-start.

---

## 2. The three capability dimensions

Embodied-R1.5 organizes embodied intelligence into a progressive chain: **Cognition → Planning → Pointing**. This is not an arbitrary task collection; the paper frames it as a reasoning pipeline where each dimension feeds the next.

### 2.1 Cognition & Spatial Reasoning
This is the *perceptual foundation*. Four sub-capabilities:
- **Spatial relation understanding:** above/inside/between, occlusion, visibility
- **Metric spatial reasoning:** distances, sizes, free-space feasibility
- **3D scene perception:** monocular depth, cross-view consistency
- **Object & scene cognition:** robot-centric viewpoint understanding

**Benchmark coverage:** ERQA, OpenEQA, CV-Bench, EmbSpatial, SAT, RoboSpatial, BLINK, VSIBench. Embodied-R1.5 achieves 70.2% average, with particularly strong RoboSpatial (69.7%)—a robot-view spatial dataset that general-purpose VLMs struggle with.

### 2.2 Planning & Correction
This is the *executive control* layer. Again four sub-capabilities:
- **Long-horizon task decomposition:** break "make milk tea" into 10 sub-steps
- **Next-step planning:** given progress, what's the immediate next action instruction
- **Process detection:** is the current sub-task complete?
- **Error localization & correction:** diagnose failure, generate recovery plan

The correction capability is the most novel relative to prior work. The paper constructs ~800K correction samples along a 2×3 taxonomy: planning failures vs execution failures × detection vs localization vs correction. This structured failure awareness is what enables the closed-loop PGC framework.

**Benchmark coverage:** RoboVQA (61.0%), EgoPlan-2 (53.8%), Cosmos-Reason (69.3%), RoboFAC (77.2%). The +24.0pp margin over Gemini-Robotics-ER-1.5 on planning average is the paper's most decisive win.

### 2.3 Pointing & Location
This is the *action interface*. Four pointing types:
- **REG (Referring Expression Grounding):** "the red mug" → point
- **RRG (Region Referring Grounding):** "empty region that can hold a plate" → point
- **OFG (Object Functional Grounding):** "grasp the mug by its handle" → point on functional part
- **VTG (Visual Trace Generation):** ordered point sequences for 2D/3D trajectories (both end-effector flow and object flow)

Pointing is where Embodied-R1.5 dominates most severely: 72.8% average across 9 benchmarks, with Part-Afford at 82.9% and RoboAfford at 80.0%. The OFG capability is the critical enabler for zero-shot real-robot manipulation—without understanding functional parts, grasping tools by the wrong end fails immediately.

---

## 3. The three automated data construction pipelines

The paper's data system totals >15B tokens across 34 datasets. Three proprietary automated pipelines expand coverage in critical gaps:

### Pipeline 1: 3D Scene Annotation for Spatial Reasoning (ER1.5-Spatial, ~20K)
**Goal:** Generate tabletop-level spatial reasoning QA from single RGB images, because existing 3D datasets (ScanNet, HM3D) operate at room-level navigation granularity.

**Actual implementation** (`Embodied-Label/src/`):
1. **Resize** to ViT-multiple (step1)
2. **Semantic tagging** with dual backend: Qwen3-VL (open-domain, long-tail objects) or RAM++ (fast, broad coverage), with optional VLM-based synonym merging and cleaning (step2)
3. **Geometry estimation** via MoGE-2 (metric-scale depth, normals, intrinsics) (step3)
4. **Instance segmentation** via Grounded-SAM-2: GroundingDINO boxes conditioned on per-image labels → SAM2 mask refinement (step4)
5. **3D lifting** with edge-aware filtering (depth-gradient outlier removal) and voxel downsampling (step5)
6. **Dominant normal estimation** via RANSAC on normal map, with upward-direction bias (step6)
7. **World-frame alignment** via histogram peak detection for table height, Gram-Schmidt orthogonalization for camera-to-world transform (step7)
8. **QA generation** from structured scene graphs (step8)

**Quality controls embedded at each stage:**
- Semantic: filter images with <2 valid objects; reject if robot arm misidentified as manipulable object
- Segmentation: confidence ≥0.3, mask area between 0.5%–50% of image
- 3D lifting: edge-filter thickness=1, discontinuity tolerance=0.04; SOR outlier removal
- World frame: inlier ratio ≥30% for dominant plane; verify object bottoms within 7cm of z=0

**Code verdict:** This is a **genuinely well-engineered pipeline**. The 8-step design is modular, each step is independently runnable, and quality controls are explicit. The use of MoGE-2 for metric-scale depth (no post-hoc scale recovery) is the critical enabler for distance-estimation QA. The bash runner (`scripts/run_pipeline.bash`) handles conda environment switching because Grounded-SAM-2 and Qwen3-VL have incompatible transformers versions—a pragmatic acknowledgment of real dependency hell.

### Pipeline 2: Failure-Aware Data Construction for Correction (ER1.5-Correction, ~800K)
**Goal:** Structured failure annotations for closed-loop autonomy, because existing datasets are almost exclusively success demonstrations.

**Taxonomy:** 2 stages × 3 cognitive levels = 6 QA types:
- Planning failures: step omission, redundancy, swap, object error, action/location replacement
- Execution failures: interruption, wrong object, wrong action, operation failure
- Cognitive levels: detection (binary) → localization (which step/type) → correction (recovery plan)

**Construction strategies:**
- **Planning:** Apply 5 structured perturbation operators to correct sub-task plans. Each perturbation simultaneously generates all 3 cognitive-level QA pairs. Correct plans are retained as positive samples.
- **Execution:** Three complementary strategies: (1) truncate successful videos at sub-task boundaries to simulate interruption; (2) replace object/action descriptions to create mismatches; (3) inject physics perturbations in ManiSkill simulation.

**Code verdict:** The pipeline logic is described in detail in Appendix B.2, but **the actual perturbation code is not in the released repository**. The `EasyR1/ER1.5_scripts/rft_train.sh` references datasets like `ER1.5_RoboVQA_image.json` and `ER1.5_spatialssrl_image_qa.json`, but the data generation scripts themselves are absent. The README notes: "Some of the training dataset mappings are not yet complete and will be updated soon." This is a material gap for reproducibility.

### Pipeline 3: Functional Affordance & Trajectory Data (ER1.5-Pointing)
**Goal:** Scale OFG and VTG data where part-level annotation is prohibitively expensive.

**OFG strategies:**
- Simulation-based: render PartNet-Mobility / PRISM objects from multiple views; part labels are free in simulation
- Human interaction data: extract contact regions from hand-object interaction datasets
- Existing dataset restructuring: HandAL, PACO-LVIS, InstructPart unified into OFG format

**VTG strategies:**
- Robot-centric: project 3D end-effector poses (DROID) onto 2D image plane; for datasets without poses, fine-tune a Detectron2 end-effector detector
- Object-centric: Co-Tracker3 tracking for real videos; physics-engine pose sequences for simulation

**Code verdict:** The trajectory extraction code is not present in the repo. The `embodied_reward.py` implements the VTG evaluation metric (RMSE-based piecewise-linear decay), confirming the authors understand the measurement, but the data generation pipeline itself is unreleased.

---

## 4. The multi-task balanced RL recipe

### 4.1 Two-stage training
- **Stage 1 (SFT):** Full-parameter fine-tune Qwen3-VL-8B-Instruct on the full 15B-token corpus for 1 epoch. LR 2×10⁻⁶, cosine decay, 10% warmup, global batch 512, context length 8192. Vision encoder is **jointly trained** (not frozen)—the paper argues embodied visual distributions differ enough from general pretraining to require deep adaptation.
- **Stage 2 (RFT):** Initialize from SFT checkpoint, run RL for 2 epochs. LR 3×10⁻⁶, 8 parallel rollouts per prompt, clip ratios [0.2, 0.28], KL-in-reward with β=0.01.

### 4.2 The RL algorithm: MBPO, not GRPO
Here is the most important finding from code inspection. The paper discusses GRPO extensively (§4.2.1), and the config file (`Embodied-R1.5_config.yaml`) contains GRPO-like hyperparameters. However, the actual `adv_estimator` is set to **`mbpo`**:

```yaml
algorithm:
  adv_estimator: mbpo
  online_filtering: true
  filter_key: overall
  filter_low: 0.01
  filter_high: 0.99
```

Looking at `verl/trainer/core_algos.py`, the MBPO implementation (lines 220–262) is:

```python
@register_adv_estimator(AdvantageEstimator.MBPO)
def compute_mbpo_outcome_advantage(token_level_rewards, response_mask, index, eps=1e-6):
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
    batch_std = scores.std()          # ← entire mixed batch
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (batch_std + eps)
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns
```

Compare with vanilla GRPO (same file, lines 176–217):

```python
for idx in id2score:
    id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
    id2std[idx] = torch.std(torch.tensor(id2score[idx]))   # ← per-group
for i in range(bsz):
    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
```

**What the paper calls "global batch reward normalization" is literally `scores.std()` over the entire batch.** This is elegant: it preserves intra-group relative ordering (which rollout is better for a given prompt) while unifying gradient magnitudes across heterogeneous tasks. It requires no task labels, no EMA history, and no per-task statistics that become unstable for low-resource tasks.

Our probe (`code/probe_mbpo_vs_grpo.py`) demonstrates the effect on a toy mixed batch:

| Task | Raw scores | GRPO adv | MBPO adv |
|---|---|---|---|
| Pointing (A) | 0.80, 0.90, 0.85, 0.95 | −1.34, +0.45, −0.45, +1.34 | −0.19, +0.06, −0.06, +0.19 |
| Planning (B) | 0.10, 0.00, 0.20, 0.00 | +0.30, −0.91, +1.51, −0.91 | +0.06, −0.19, +0.31, −0.19 |

GRPO blows up Task B's tiny variance into ±1.0 advantages, while compressing Task A's signal. MBPO uses batch_std≈0.41, giving both tasks stable, proportionate updates. The paper's claim is correct and the implementation is clean.

### 4.3 Difficulty-aware filtering and dynamic filtering
The config also enables:
- **`online_filtering: true` with `filter_low: 0.01` and `filter_high: 0.99`**: Automatically masks degenerate groups where all rollouts receive identical rewards. This prevents zero-gradient updates.
- **Difficulty-aware data filtering** (described in paper, not visible in config): Retain ~200K medium-difficulty samples from the SFT corpus based on rollout pass rates. The idea is that trivially easy samples waste gradient budget, and impossibly hard ones provide no useful signal.

### 4.4 Reward design
The reward function (`EasyR1/ER1.5_scripts/reward_function/embodied_reward.py`) implements five families with a unified piecewise-linear decay framework:

```python
def phi(d, tau_p, tau_z):
    if d < tau_p: return 1.0
    elif d >= tau_z: return 0.0
    else: return (tau_z - d) / (tau_z - tau_p)
```

| Family | Metric | Thresholds (τp, τz) | Notes |
|---|---|---|---|
| Exact match | Binary | — | Multiple choice, numerical (1 decimal), math (symbolic verify via `math_verify`) |
| IoU | Box overlap | — | Natural continuous reward in [0,1] |
| Point distance | Nearest-neighbor avg dist | 40, 150 px | Count penalty δc=0.3 if predicted point count ≠ ground truth |
| Trajectory RMSE | Per-point RMSE after linear interpolation | 50, 120 px (2D); depth MAE 0.1, 0.4m | Length-mismatch penalty δl=0.35; single-point outputs get zero reward (anti-hacking) |
| Semantic similarity | Skywork-Reward-V2-Qwen3-8B via SGLang | Sigmoid temp=1.0 | Fallback to BLEU-1/2/3 average if RM server unavailable |
| Format | `<answer>...</answer>` tag + JSON structure | — | Weight λ=0.1 combined with accuracy reward |

**Code verdict:** The reward implementation is **thorough and battle-tested**. It handles markdown code blocks, escaped quotes, flexible multiple-choice matching (accepts "A", "A.dog", "A) dog"), polygon containment for segmentation-grounded points, and anti-hacking guards (single-point trajectories get zero reward). The use of `math_verify` for symbolic equivalence is a nice touch that avoids brittle string matching on mathematical answers.

One discrepancy: the paper says point-distance thresholds are (40, 150), but `embodied_reward.py` uses `perfect_threshold=40.0, zero_threshold=150.0` for point tasks and `perfect_threshold=50.0, zero_threshold=120.0` for trace tasks. The paper's Table 4 caption also mentions different thresholds for VABench-V vs ShareRobot-V. The reward code uses fixed global thresholds, so per-benchmark tuning may happen elsewhere or may be a minor inconsistency.

---

## 5. The PGC closed-loop framework

### 5.1 What the paper claims
The Planner-Grounder-Corrector (PGC) framework is described as a **minimalist stateless harness** where a single Embodied-R1.5 instance asynchronously serves three roles:
- **Planner:** Decomposes long-horizon tasks, performs next-step planning
- **Grounder:** Adaptively orchestrates pointing capabilities (REG, RRG, OFG, VTG) to produce spatial commands for a low-level skill executor
- **Corrector:** Asynchronously monitors execution state (SUCCESS / PROCESS / FAIL), triggers replanning on failure

A FIFO Memory buffer records fixed-rate image samples and sub-task status annotations. The framework is validated on real-robot tasks: making milk tea (10 steps), stacking cups (6 steps), sweeping garbage (cyclic), and goods picking from shelves (open-vocabulary).

### 5.2 What the code actually contains
The released repository contains **only a basic reflection demo** in `inference/reflection.py`:

```python
def plan_with_reflection(client, case, max_rounds=3):
    for round_idx in range(1, max_rounds + 1):
        plan_text = inference(client, planning_case)
        reflection_text = inference(client, reflection_case)
        if "correct" in reflection_text.lower() and "incorrect" not in reflection_text.lower():
            return plan_text
        feedback_text = reflection_text
    return plan_text
```

This is a **simple planning→reflection loop** over at most 3 rounds. It lacks:
- No asynchronous corrector process
- No FIFO memory buffer with fixed-rate image sampling
- No low-level skill executor interface
- No grounder capability orchestration (the demo only does planning)
- No state machine for SUCCESS/PROCESS/FAIL
- No sub-task completion detection

The `inference/` directory contains useful utilities (`point_utils.py` for decoding heterogeneous point formats, `vllm_offline_example.py` for batch inference, `hf_example.py` for HF Transformers), but **no PGC runtime**.

**Honest assessment:** The PGC framework is described compellingly in the paper and the qualitative demos (Figures 7–8) are impressive. However, the released code does not include the autonomy harness. This is a significant gap: researchers wanting to build on the closed-loop results cannot reproduce the milk-tea or cup-stacking experiments from the repository alone. The README mentions a project page with videos, but the system code is missing.

---

## 6. Code investigation: what's actually released vs what's claimed

| Claim | Status | Notes |
|---|---|---|
| Model weights | ✅ Released | `IffYuan/Embodied-R1.5` on HuggingFace |
| SFT training code | ✅ Released | LLaMA-Factory scripts (`scripts/train/`) |
| RFT training code | ✅ Released | EasyR1 scripts (`EasyR1/ER1.5_scripts/`) |
| Multi-task reward functions | ✅ Released | `embodied_reward.py` + `embodied_reward_relax.py` (identical copies) |
| Embodied-Label pipeline | ✅ Released | Full 8-step pipeline with bash runner |
| Evaluation data (EmbodiedEvalKit) | ⚠️ Separate repo | https://github.com/pickxiguapi/EmbodiedEvalKit (not in this repo) |
| VLA training code | ❌ Not released | Uses external starVLA framework; only checkpoints released |
| PGC closed-loop framework | ❌ Not released | Only basic `reflection.py` demo |
| Complete training datasets | ⚠️ Partial | RFT script lists 26 datasets, but README says "some dataset mappings are not yet complete" |
| Failure-aware data generation | ❌ Not released | Pipeline 2 logic described in paper but code absent |
| Trajectory extraction pipeline | ❌ Not released | Pipeline 3 logic described but code absent |

**Net assessment:** This is a **strong open-source release by current standards** in embodied AI, but it is not a complete reproduction package. The training infrastructure, reward functions, and one data pipeline (Embodied-Label) are present and usable. The autonomy framework and two data pipelines are described but not shipped. The 15B-token corpus is referenced but not fully downloadable in preprocessed form.

---

## 7. What this run learned that wasn't obvious from the abstract

### 7.1 The MBPO advantage estimator is the real innovation, not GRPO
The abstract and introduction emphasize "multi-task balanced RL recipe" without naming the specific algorithm. A reader might assume GRPO with tweaks. The actual implementation uses **MBPO** (`adv_estimator: mbpo`), which replaces per-group std with batch std. This is a simple, stateless, and effective solution to heterogeneous multi-task RL that does not require task labels or EMA state. It deserves more prominence in the paper.

### 7.2 The vision encoder is jointly trained in SFT but the SFT config freezes it
Here is a subtle inconsistency. The paper states (§4.1): "The vision encoder is jointly trained with the LLM backbone without freezing." But the SFT config (`scripts/train/sft_config.yaml`) says:

```yaml
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
```

The RFT config (`Embodied-R1.5_config.yaml`) correctly sets `freeze_vision_tower: false`. This suggests either: (a) the SFT config in the repo is not the exact one used for the paper's SFT stage, or (b) the paper's statement applies only to RFT. Given that the visual distribution shift argument is strongest at SFT time (first exposure to embodied data), this is worth clarifying.

### 7.3 "Adaptive thinking" is emergent, not engineered
The paper notes that the model learns to allocate reasoning tokens on demand: near-zero reasoning for pointing tasks, thorough chains for planning. This is a genuine emergent property of RL with outcome rewards that only evaluate final answers. The format requirement (`<answer>...</answer>`) is the only structural constraint; no explicit CoT prompting or reasoning-stage curriculum is used. This is elegant, but it also means the model's reasoning depth is stochastic and task-dependent—fine for benchmarks, potentially risky for safety-critical deployment.

### 7.4 The pointing dominance is real, but spatial reasoning margins are thin
Embodied-R1.5's pointing numbers are genuinely dominant (+17–40pp over Qwen3-VL on some benchmarks). But on spatial reasoning, the margins over Gemini-2.5-Pro and GPT-5.4 are small or nonexistent. The paper acknowledges this candidly: "strong general-purpose VLMs already exhibit excellent spatial reasoning." The value add is not raw spatial IQ; it is **unification**—having pointing, planning, and correction in one model without catastrophic forgetting.

### 7.5 The VLA results rely on a very specific action-head design
The flow-matching DiT-B action expert with 32 learnable future query tokens is not a generic adapter. It requires the starVLA framework, and the paper's strong results may not transfer to simpler action-head architectures (e.g., MLP projection, diffusion policy). The claim that "internalized embodied reasoning can effectively substitute for action data scaling" is validated only for this specific DiT-B + flow-matching setup.

---

## 8. Limitations and honest assessment

### Strengths
1. **Systematic unification.** The three-dimensional capability taxonomy is well-motivated and the paper delivers SOTA on 16/24 benchmarks within it.
2. **Data engineering depth.** Embodied-Label is a genuinely useful standalone tool for 3D scene annotation from RGB. The quality controls are thoughtful.
3. **Training recipe transparency.** The MBPO normalization, difficulty filtering, and five-family reward design are clearly described and verifiable in code.
4. **Strong empirical validation.** The VLA ablations (Figure 9) are particularly convincing because they isolate backbone contribution from action-head confounds.
5. **Open weights and partial code.** Even incomplete, this is among the most open releases in embodied VLM research.

### Weaknesses and open questions
1. **PGC framework unreleased.** The closed-loop autonomy claims cannot be independently verified or built upon without the harness code.
2. **2D-only perception.** The model operates on RGB images; native 3D inputs (point clouds, depth maps) are not integrated despite the 3D scene annotation pipeline generating them. This limits performance in heavy occlusion.
3. **Real-robot sample size.** The zero-shot real-robot experiments report n=6 trials per task. This is suggestive but not statistically rigorous. The impressive qualitative videos (milk tea, cup stacking) are not accompanied by quantitative success-rate distributions.
4. **Action-pretraining substitution claim is bounded.** It holds for LIBERO and SimplerEnv with DiT-B flow matching, but may not generalize to: (a) contact-rich tasks without force-aware policies, (b) out-of-distribution robot morphologies, (c) tasks requiring fine force control. The ForceFlow citation (Zhang et al., 2026b) acknowledges this by adding a force-aware policy handover.
5. **Reward model dependency.** Open-ended planning/correction rewards rely on Skywork-Reward-V2-Qwen3-8B served via SGLang. The released `embodied_reward.py` has `USE_MODEL_FOR_OPEN_ENDED = False` by default, falling back to BLEU. This means the out-of-the-box RFT recipe may not replicate the exact planning/correction RL gains without setting up the RM server.
6. **Data completeness.** The note that "some dataset mappings are not yet complete" means the exact 15B-token mixture cannot be reproduced today.

### Verdict
Embodied-R1.5 is a **high-quality, ambitious, and mostly honest piece of work**. The core scientific claim—that unified embodied reasoning internalized in a VLM can reduce downstream VLA data requirements—is well-supported by the evidence presented. The gaps between paper and code (PGC framework, two data pipelines) are significant but do not undermine the benchmark results, which are evaluated through the separate EmbodiedEvalKit. For the community, the immediate value is: (a) a strong 8B embodied VLM checkpoint, (b) a reusable 3D annotation pipeline, (c) a verified multi-task RL recipe. The long-horizon autonomy story is inspiring but remains a demo until the PGC harness is released.

---

## 9. Probe artifacts

Two small probes are preserved under `code/`:

- **`probe_mbpo_vs_grpo.py`** — Reconstructs the MBPO advantage computation from `core_algos.py` and contrasts it with vanilla GRPO on a toy mixed batch, showing how batch-level std normalization stabilizes gradient magnitudes across heterogeneous tasks.
- **`probe_reward_functions.py`** — Plots the piecewise-linear decay curves for point-distance and trajectory-RMSE rewards, making the paper's threshold choices直观. (Requires matplotlib; run in your own environment.)

---

## 10. Addendum: What Does "Reasoning" Actually Mean in Embodied-R1.5?

See `ADDENDUM_reasoning.md` for a second-pass investigation. Core finding: the RL training objective is structurally incapable of teaching reasoning quality—the reward discards all reasoning text and scores only final answers. The "adaptive thinking" claim is contradicted by explicit prompt injection in the training code (`EasyR1/verl/utils/dataset.py`), and the System 2/System 1 framing is a metaphor, not an architecture.

---

*End of deep-dive. Last updated: 2026-06-11.*
