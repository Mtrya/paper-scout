# ACE-Ego-0 Deep-Dive Evidence

**Paper:** ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining (arXiv 2606.17200)  
**Run:** 2026-06-17-aceego-actworld-motionvla  
**Subagent scope:** investigate the core mechanism, gather external signals, build a concrete probe, and preserve report-ready material.

---

## What question does ACE-Ego-0 raise?

Can large-scale egocentric human video be used as *action-level* supervision for a VLA policy, rather than just visual pretraining? The paper argues yes, but only if two mismatches are resolved simultaneously:

1. **Representation mismatch** — human hands and robots live in different coordinate frames, kinematic structures, and control frequencies.
2. **Supervision-quality mismatch** — pseudo-actions reconstructed from video are noisy, while robot sensor logs are clean.

ACE-Ego-0's distinctive answer is to unify the representation (camera-space actions + morphology tokens + time-aligned chunks) and to treat human pseudo-actions through a *reliability-aware auxiliary loss* rather than the primary robot action loss.

---

## What I learned

- **Core idea:** The paper converts 1.48K hours of egocentric video (Ego4D, EgoExo4D, EPIC-KITCHENS-100, HOI4D, EgoDex, Xperience-10M) into robot-format pseudo-action trajectories via a five-stage pipeline (clip curation → ego-interaction filtering → 3D hand reconstruction → action parameterization → quality control). Combined with 4.53K hours of robot/sim data, this yields a 6.0K+ hour pretraining corpus.
- **Unified action representation:** All end-effector/hand trajectories are expressed in the head-camera frame (Eq. 1); human hands are parameterized by wrist origin, palm-plane 6D orientation, and thumb-to-palm gripper proxy. Cross-embodiment morphology tokens (URDF graph embeddings for robots, learned surrogate embeddings for humans) condition only the action expert, leaving the VLM backbone embodiment-agnostic. Time-aligned action chunking uses a fixed physical duration `T*` with `H_d = round(f_d T*)` rather than a fixed frame count.
- **Reliability-aware training:** Human pseudo-actions are routed through an auxiliary Huber loss weighted by `W_{t,j} = rho_j * w_data(d,h) * w_step(t,h)` (Eq. 6/20). Position channels get `rho=1.0`; rotation/gripper get `rho=0.001`; step-level weights attenuate jumps/jerks exceeding dataset-hand 95th-percentile thresholds.
- **Key numbers:**
  - RoboCasa GR1 TableTop: **72.8%** avg success (next best: DIAL 70.2%, JoyAI-RA 63.2%).
  - RoboTwin 2.0: **91.12%** Easy / **90.62%** Hard (next best: JoyAI-RA 90.48% / 89.28%).
  - Real ARX bimanual: **78.3%** avg over 6 tasks vs π0.5 71.7%, GR00T-N1.7 35.6%.
  - Ablations on RoboCasa: removing reliability-aware loss drops −3.6% (largest), removing morphology tokens −1.9%, removing time-aligned chunking −1.1%.
  - Human-augmented fine-tuning on Sweep Cubes with 34 robot demos: 10% → 40% success.

---

## External signals

- **Project page:** https://acerobotics-vla.github.io/ACE-Ego/ exists and mirrors the paper results.
- **Code / data release:** The paper lists `https://github.com/ACERobotics-VLA/ACE-Ego` but the repository returned **HTTP 404** at investigation time (2026-06-17). No code, weights, or processed pseudo-action dataset are publicly available yet.
- **Hugging Face / other mirrors:** No public Hugging Face model/dataset page found for ACE-Ego-0.
- **Neighbor comparison:**
  - **ActiveMimic (arXiv 2606.06194)** — also uses Ego4D, but its key claim is that *active perception* (camera/viewpoint motion) is the missing signal in egocentric pretraining. It jointly models 27D camera+wrist actions and shows real-robot transfer on AGIBOT G1. ACE-Ego-0 instead fixes the camera frame and treats viewpoint change as extrinsic, focusing on cross-embodiment action alignment and label reliability.
  - **EgoMimic (ICRA 2025)** — reconstructs wrist trajectories from egocentric video and retargets to a bimanual robot. It is closer to a data-collection/imitation pipeline than a large-scale mixed-source VLA pretraining framework.
  - **MotoVLA (arXiv 2509.19958)** — pretrains on human RH20T video plus robot data through a 3D dynamic point-cloud prediction stage, then aligns to robot actions. It avoids direct pseudo-action regression by using point clouds as an intermediate representation, whereas ACE-Ego-0 directly regresses camera-space pseudo-actions with explicit reliability weighting.
  - **GR00T-N1 / π0.5** — strong closed/generalist VLA baselines; ACE-Ego-0 beats both on RoboCasa and real ARX tasks.

---

## Probe

`code/reliability_weighting_probe.py` is a self-contained Python script that concretizes the reliability-aware human auxiliary loss (Sec. 3.2 / Appendix A.5). It runs without any data download.

### What it does

1. Generates a synthetic clean robot bimanual action chunk in the 22-D unified layout (position + 6D rotation + gripper + activity flag, per hand).
2. Corrupts it into a pseudo-action-labeled "human" chunk with realistic estimation noise: position jitter, sparse rotation spikes, and gripper bias/noise.
3. Computes the hierarchical reliability weight `W_{t,j} = rho_j * w_data(d,h) * w_step(t,h)` exactly as described in the paper.
4. Compares the auxiliary Huber loss under (a) naive uniform weighting and (b) reliability-aware weighting.
5. Saves a four-panel figure to `../assets/ace-ego-0-reliability-probe.png`.

### How to rerun

```bash
cd runs/2026-06-17-aceego-actworld-motionvla/ace-ego-0-2606.17200/code
python3 -m venv venv
source venv/bin/activate
pip install matplotlib numpy
python reliability_weighting_probe.py
```

The existing `venv/` in this directory already contains the required packages.

### Representative output

```text
Synthetic episode length     : 64 steps
Position channel prior rho   : 1.0
Rotation/gripper prior rho   : 0.001
Dataset prior left/right     : 0.85/0.7
Naive auxiliary loss         : 0.0353
Reliability-aware loss       : 0.0002
Effective supervision mass   : pos=288.8 (99.8%), rot=0.6 (0.2%), grip=0.1 (0.0%)
```

The probe demonstrates that the reliability weighting concentrates >99% of the effective human supervision mass on the position channels, so the auxiliary loss is ~176× smaller than naive uniform weighting when the model has large errors on the noisy rotation/gripper channels.

---

## Preserved artifacts

| Path | Description |
|------|-------------|
| `code/reliability_weighting_probe.py` | Self-contained probe implementing the reliability-aware auxiliary loss. |
| `code/venv/` | Local Python environment with numpy+matplotlib for rerunning the probe. |
| `../../assets/ace-ego-0-reliability-probe.png` | Four-panel figure: trajectory reliability, per-dimension weights, loss decomposition, and naive vs reliability-aware loss. |

---

## Limitations / blockers

- **No public code or checkpoints:** The listed GitHub repository is not live, so the exact data pipeline, model weights, and training code cannot be inspected or reproduced independently.
- **Heavy compute:** Pretraining uses 128×A800 GPUs for 200K steps; fine-tuning uses 16×A800 GPUs. Full reproduction is out of reach without similar infrastructure.
- **Pseudo-action fidelity:** The paper itself notes that rotation and fine-grained finger motion remain noisy; the reliability-aware objective currently supervises these channels only weakly (`rho=0.001`).
- **Evaluation scope:** Results are reported on tabletop/bimanual tasks; whole-body humanoid control, mobile manipulation, and deformable-object tasks are not tested.
