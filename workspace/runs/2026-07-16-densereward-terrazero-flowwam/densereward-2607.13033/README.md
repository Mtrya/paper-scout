# DenseReward: Dense Reward Learning via Failure Synthesis for Robotic Manipulation

**arXiv:** 2607.13033  
**Thread:** `runs/2026-07-16-densereward-terrazero-flowwam/densereward-2607.13033/`  
**Investigator:** research subagent

## Research questions

1. How does DenseReward synthesize failure trajectories? What are the five manipulation phases and six failure modes, and how are perturbations implemented?
2. How are per-timestep dense rewards computed? What happens after failure events?
3. What is the model architecture, inputs/outputs, and use of historical frames?
4. What do the experiments actually show, and are the comparisons fair?
5. Is there public code, data, or models?
6. How does it compare with closest related work?

---

## 1. Failure-synthesis pipeline

The paper decomposes manipulation into five canonical phases (§3.1):

| Phase | Trigger / boundary |
|-------|-------------------|
| **Reach** | Robot moves end-effector toward the target object. |
| **Grasp** | Begins when gripper contacts the object. |
| **Lift** | Begins when the object is off the table. |
| **Move** | Transport object toward target location. |
| **Place** | Begins when end-effector enters proximity radius `d_place` around the target. |

The automated pipeline (§3.1):
1. Randomly initialize object + container on a table.
2. Use **GraspNet** to predict up to `N=50` grasp pose candidates from multi-view RGB-D.
3. Feed candidates to **CuRobo** for collision-aware motion planning and select a feasible one.
4. Execute six motion segments matching the five phases; phase boundaries are detected automatically from simulator state.

Six failure modes are induced by targeted perturbations (§3.2):

| Mode | Perturbation | Reward shape described in paper |
|------|--------------|--------------------------------|
| **Success** | None | Monotonic rise to 1.0. |
| **Collision** | Motion planned **with collision avoidance disabled** → robot hits object/table. | Mountain-shaped: rises to collision, then decays. |
| **Miss** | **Offset grasp target pose** → gripper closes in air, no stable grasp. | Mountain-shaped: rises until failed grasp, then decays. |
| **Fall** | **Random rotation perturbations** applied during Move phase → gripped object loses stability and falls. | Mountain-shaped: rises to drop event, then decays. |
| **Smooth** | **Small Gaussian joint noise** injected at every timestep → jittery but completing trajectory. | Scaled/penalised reward (lower than clean success). |
| **Recover** | Collision occurs, then motion planner **replans a clear path** to complete the task. | Dip during collision, then resumes climbing. |

Validity checks (Appendix B) filter physically invalid episodes:
- planning feasibility,
- grasp/lift height thresholds for success/recovery,
- stricter holding height for fall trajectories,
- collision must displace object without lifting,
- miss must leave object pose nearly unchanged,
- final placement distance threshold,
- recovery must show a failed attempt followed by successful replanning.

The dataset contains **26,579 episodes / 7,560,942 frame-level samples** (Appendix A):
- DROID: 2,986 episodes (1,500 success, 1,486 failure)
- Isaac Sim: 12,481 episodes (success + collision/miss/fall/smooth/recover)
- RoboSuite: 9,287 episodes (3,366 success, 5,921 failure)
- LIBERO: 1,825 episodes across four splits

---

## 2. Dense reward computation

The paper does **not** give an explicit closed-form reward function. It describes the labels as:

> "per-timestep reward scores that reflect task progress throughout execution, capturing successful progress, partial completion, degradation after failure events, and recovery from temporary failures."

Key properties:
- Rewards are scalar values `r_t ∈ [0, 1]`.
- Values are stored/rounded to **three decimal places**.
- They are **frame-level** (not trajectory-level) and **phase-aware**.
- After a failure event the reward is **not simply zero**; it reflects partial progress (mountain-shaped curves) or a temporary dip (recover).

Because the exact formula is omitted, I reconstructed schematic reward curves from the qualitative rules in §3.2. The probe is in `code/reward_curves.py` and produces `code/dense_reward_curves.png` (also copied to `runs/2026-07-16-densereward-terrazero-flowwam/assets/dense_reward_curves.png`).

The curves implement:
- **Success**: linear progress `t/T`.
- **Collision / Miss / Fall**: rise proportional to phase progress up to the failure event, then exponential decay.
- **Smooth**: success trajectory scaled by a penalty with small noise.
- **Recover**: normal rise, low flat plateau during the collision window, then linear recovery to 1.0.

These are illustrative; the actual dataset labels could use a different interpolation or decay rule.

---

## 3. Model architecture

DenseReward is built on **Qwen3-VL-4B-Instruct** and fine-tuned with **LoRA rank 16** for 10 epochs on 8× H100, batch size 32, using the ms-swift framework (Appendix D).

**Inputs:**
- Task language instruction `l`.
- Current visual observation (RGB image).
- **Two historical frames** as temporal context (ablation in §4.5).

**Output:**
- A single scalar floating-point reward for the current frame, formatted to three decimal places via a system prompt.

The model is trained with dense frame-level reward supervision, not trajectory-level binary labels. The ablation on historical frames shows MAE improves from 0.096 (0 frames) → 0.088 (1 frame) → **0.081** (2 frames), then degrades slightly to 0.086 with 3 frames.

---

## 4. Experiments

### 4.1 Dense reward prediction (Table 1)
Metric: **mean absolute error (MAE)**.

| Model | Overall | DROID | Isaac Sim | RoboSuite | LIBERO |
|-------|---------|-------|-----------|-----------|--------|
| Qwen3-VL-4B-Instruct | 0.289 | 0.532 | 0.285 | 0.195 | 0.478 |
| Qwen3-VL-8B-Instruct | 0.293 | 0.538 | 0.305 | 0.180 | 0.502 |
| Molmo2-4B | 0.282 | 0.506 | 0.282 | 0.187 | 0.478 |
| Molmo2-8B | 0.335 | 0.480 | 0.307 | 0.303 | 0.455 |
| RoboReward-4B | 0.275 | 0.534 | 0.269 | 0.179 | 0.470 |
| RoboReward-8B | 0.230 | 0.484 | 0.185 | 0.172 | 0.431 |
| Robometer | 0.366 | 0.521 | 0.328 | 0.345 | 0.468 |
| **DenseReward (ours)** | **0.081** | **0.259** | **0.081** | **0.051** | **0.044** |

**Supported claims:** DenseReward reduces MAE dramatically on its own benchmark, especially on simulated sources. The gap is largest on Isaac Sim, RoboSuite, and LIBERO.

**Potential issues / unsupported details:**
- The benchmark is internally constructed by the authors; no independent test set is described.
- The exact train/test split is not reported.
- General-purpose VLMs are used zero-shot without task-specific fine-tuning, so the comparison measures in-domain fine-tuning vs. out-of-the-box VLM performance, which is not a fair like-for-like baseline in the usual sense. RoboReward-8B is the most comparable prior method and still trails by a large margin.
- The paper reports only MAE; no calibration, correlation, or per-mode breakdown is shown.

### 4.2 Model predictive control (Table 2)
Setup: sampling-based MPC in Isaac Lab, 28 candidate actions per step (27 Cartesian directions + gripper open/close). Metric: minimum end-effector-to-object distance (lower is better).

| Model | Can | Cup | Lemon | Avg. |
|-------|-----|-----|-------|------|
| RoboReward-4B | 0.199 | 0.307 | 0.295 | 0.267 |
| RoboReward-8B | 0.314 | 0.270 | 0.317 | 0.300 |
| VLAC-2B | 0.316 | 0.346 | 0.380 | 0.347 |
| VLAC-8B | 0.351 | 0.360 | 0.363 | 0.358 |
| **DenseReward** | **0.219** | **0.181** | **0.288** | **0.229** |

DenseReward wins on average and on cup/lemon. The MPC task is essentially a local reaching test; it does not evaluate grasping, lifting, or placing. Fairness note: all models receive the same action samples and simulator setup.

### 4.3 RL fine-tuning on LIBERO (Figure 5)
- Actor: π₀ supervised-fine-tuned on LIBERO.
- Algorithm: PPO via RLinf.
- Reward: `r_t = α·r_t^sim + β·r_t^model`, with `α=1.0`, `β=C/T_max`, `C=5` action-chunk size.
- Claim: DenseReward improves final success on LIBERO-Spatial and LIBERO-10, matches on LIBERO-Object, and provides competitive learning curves.
- No numerical table is provided; only learning curves are shown. I could not verify the exact success rates.

### 4.4 Real-world RL with DSRL (Figure 6)
- Platform: DROID / Franka Research 3 + Robotiq 2F-85 + ZED cameras.
- Tasks: "stack the cups" and "put ball in basket".
- Algorithm: DSRL (latent-noise steering of frozen π₀).
- Reward: `r_t = -1 + r_t^model`; final transition uses `r_T = r_T^model` if successful.
- Training budget: 20k steps (~20 rollouts) for cups; 10k steps (~10 rollouts) for basket.
- Reported success rates:
  - stack cups: 40% → 80%
  - put ball in basket: 30% → 70%

**Supported:** Large relative improvements with tiny rollout budgets.
**Weaknesses:** Only 10 evaluation trials per policy, only two tasks, no error bars, no ablation on reward source vs. just the DSRL step penalty.

### 4.5 Ablations
- Removing all failure trajectories raises MAE from 0.0809 → 0.1312. This supports the claim that synthesized failures are critical for fine-grained reward learning.
- Historical-frame ablation: 0 frames MAE 0.096; 1 frame 0.088; 2 frames 0.081; 3 frames 0.086. Two frames chosen as default.

---

## 5. Public code, data, models

**Current status: NOT YET RELEASED.**

Evidence gathered:
- Project page `https://dense-reward.github.io/` contains explicit HTML comments: `<!-- TODO: fill in arXiv / code / dataset URLs when available -->`.
- The page buttons for "Code" and "Dataset" link to `#` (placeholders).
- GitHub org `dense-reward` exists but contains only the website repository `dense-reward.github.io` (created 2026-07-09). No code repository, no dataset repository.
- GitHub-wide search for `DenseReward` returns unrelated repositories (`yinyueqin/DenseRewardRLHF-PPO`, `SSKKai/RLBench-SAC-with-DenseReward`).
- Hugging Face search (models API `search=DenseReward`) returns unrelated `cj453/dense_reward_trainer_*` models; no DenseReward model or dataset.
- The paper abstract and website state an intent to release, but no artifacts are public as of 2026-07-16.

**Blocker:** I could not trace official code, inspect configs, run the training pipeline, or load trained weights. All quantitative results are taken from the paper and project page.

---

## 6. Comparison with related work

| Work | What it does | Key difference from DenseReward |
|------|--------------|--------------------------------|
| **RoboReward** (Lee et al., arXiv 2601.00675) | General-purpose VL reward models for robotics trained on OXE + RoboArena. Generates negatives via **counterfactual relabeling** and **temporal clipping** of successful demos. Trajectory-level / sparse progress signal. | DenseReward synthesises failures **physically in simulation** (collisions, drops, etc.) and provides **per-timestep dense** rewards, not relabeled demo snippets. |
| **SARM** (Chen et al., arXiv 2509.25358) | Stage-aware reward model: jointly predicts task stage + subtask progress from language-annotated human demos. Used for RA-BC and DiffQL. | SARM relies on **human language subtask annotations** and focuses on long-horizon deformable-object tasks. DenseReward is **fully automated** (no human labels) and targets pick-place with explicit failure synthesis. |
| **AHA** (Duan et al., arXiv 2410.00371) | VLM for failure detection and reasoning; procedurally perturbs successful sim demos to build a failure dataset. | AHA is primarily a **failure detector/explainer**; DenseReward is a **dense progress reward model**. Both use simulated perturbations, but DenseReward assigns continuous phase-aware rewards rather than categorical failure labels. |
| **Robo-Dopamine** (Tan et al., arXiv 2512.23703) | Step-aware process reward model from multi-view inputs; policy-invariant reward shaping; trained on 3,400+ hours. | Robo-Dopamine is a much larger industrial-scale system with multi-view fusion and theoretical reward-shaping analysis. DenseReward is simpler (single-view + history, Qwen3-VL-4B, simulation-generated labels) and explicitly models recovery/suboptimal motion. |
| **RoboMeter** (Liang et al., arXiv 2603.02115) | Combines frame-level progress loss with trajectory-comparison preference loss; trained on RBM-1M (1M+ trajectories). | RoboMeter scales via **preference comparisons across trajectories**. DenseReward scales via **automatic physics-based failure synthesis** and dense per-frame labels without pairwise annotations. |

**What is genuinely new in DenseReward:**
1. A **fully automated, physics-based failure-synthesis pipeline** that produces diverse, physically realistic failure modes (collision, miss, fall, smooth, recover) without human teleoperation or relabeling.
2. **Dense, phase-aware reward labels** that capture partial progress, degradation after failures, and recovery — rather than binary or trajectory-level labels.
3. Integration of the resulting data into a compact Qwen3-VL-4B model, showing strong empirical gains on both reward prediction and downstream RL/MPC.

---

## Evidence preserved

- `code/reward_curves.py` — schematic reconstruction of the six reward-curve shapes from the paper's qualitative rules.
- `code/dense_reward_curves.png` — generated figure.
- `runs/2026-07-16-densereward-terrazero-flowwam/assets/dense_reward_curves.png` — report-facing copy.
- `papers/robotics/densereward-2607.13033.md` — full paper markdown.
- Screenshots/inspection notes in this README (no separate screenshots needed; placeholders and API responses are described above).

## How to rerun

```bash
cd runs/2026-07-16-densereward-terrazero-flowwam/densereward-2607.13033/code
python3 reward_curves.py
```

Requirements: `numpy`, `matplotlib`.

---

## Claims supported vs. unsupported

| Claim | Status | Notes |
|-------|--------|-------|
| Five-phase decomposition + six failure modes | **Supported** | Explicit in §3.1-3.2 and Appendix B. |
| Perturbation implementations (collision disabled, grasp offset, rotation noise, joint noise, replan) | **Supported** | Listed in §3.2. |
| Dense reward captures progress / degradation / recovery | **Supported qualitatively** | No explicit formula given; reconstructed schematically. |
| MAE 0.081 vs. baselines on internal benchmark | **Reported, not independently verified** | No public model/dataset to test. |
| MPC minimum-distance improvement | **Reported** | Fair shared setup, but task is limited to reaching. |
| LIBERO RL improvement | **Reported** | Only curves shown, no final success-rate table. |
| Real-world DSRL 40%→80% / 30%→70% | **Reported, small sample** | 10 trials, 2 tasks, no public logs. |
| Failure data improves MAE | **Supported by ablation** | 0.0809 → 0.1312 when removed. |
| Code / dataset / model release | **Unsupported as of now** | Website has placeholders only. |

## Takeaway for the parent report

DenseReward is a clean, well-motivated contribution: it replaces hand-labeled or relabeled failure data with **physics-based failure synthesis** and replaces sparse/trajectory-level rewards with **dense, phase-aware per-timestep rewards**. The empirical gains on reward prediction are large, and the downstream RL/MPC results point in the right direction. The biggest caveats are (1) **no public artifacts yet**, so the numbers cannot be independently checked, and (2) the reward-prediction benchmark is internally constructed, and several baselines are zero-shot VLMs rather than fine-tuned competitors. Compared to RoboReward and SARM, the genuine novelty is the **automatic simulation pipeline for diverse failure modes + dense progress labels**; compared to Robo-Dopamine/RoboMeter, the scope is narrower and the scale smaller, but the mechanism is more transparent and reproducible once code is released.
