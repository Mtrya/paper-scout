# PhysisForcing deep-dive thread

**Paper:** PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation (arXiv 2606.28128)  
**Thread id:** `physisforcing-2606.28128`  
**Run id:** `2026-07-11-gigaworld-vla-corrector-physis`

## Core idea

PhysisForcing is a fine-tuning recipe for DiT-based video world models. It argues that the dominant failures in robot manipulation rollouts are *local* (discontinuous gripper/object trajectories, deformation) and *relational* (broken contact, a pushed object staying still, a grasped object drifting). Instead of supervising all pixels uniformly with the standard flow-matching loss, it:

1. Extracts **physics-informative regions** from the ground-truth clip with an off-the-shelf point tracker (CoTracker3) and a depth map (Depth-Anything-V2).
2. Adds two auxiliary losses on a **single middle DiT block** hidden feature:
   - **Pixel-level trajectory alignment** \( \mathcal{L}^{\mathrm{phy}}_{\mathrm{pix}} \): the DiT feature is interpreted as per-point trajectories via soft-argmax from a first-frame query, and the predicted trajectories are supervised against CoTracker3 reference trajectories inside the physics mask.
   - **Semantic-level relational alignment** \( \mathcal{L}^{\mathrm{phy}}_{\mathrm{sem}} \): the pairwise cosine-similarity matrix of DiT tokens inside the physics mask is matched to the relation matrix of a frozen self-supervised video encoder (V-JEPA 2).
3. All auxiliary models are discarded at inference, so there is **zero extra inference cost**.

The full training objective is:

\[
\mathcal{L} = \mathcal{L}_{\mathrm{FM}} + \lambda_{\mathrm{pix}} \mathcal{L}^{\mathrm{phy}}_{\mathrm{pix}} + \lambda_{\mathrm{sem}} \mathcal{L}^{\mathrm{phy}}_{\mathrm{sem}}
\]

where \(\mathcal{L}_{\mathrm{FM}}\) is the standard flow-matching loss. The authors apply this to Wan2.2-I2V-A14B, Wan2.2-TI2V-5B, and Cosmos3-Nano.

## Experimental design

The reported evidence spans three video-generation benchmarks and two policy-learning settings:

- **R-Bench** (650 image-text pairs): task-oriented and embodiment-specific dimensions; evaluated with an MLLM-as-judge pipeline plus pixel/tracking metrics.
- **PAI-Bench robot domain** (174 real-world image-prompt pairs): domain score (physical-semantic plausibility) and quality score.
- **EZS-Bench** (196 zero-shot OOD robot-task-scene combinations): quality and domain scores.
- **Policy learning:** PhysisForcing-trained Wan2.2-TI2V-5B is plugged into Fast-WAM for RoboTwin 2.0 tasks and into the WorldArena action-planner protocol, where closed-loop success rises from 16.0% to 24.0%.

The headline R-Bench gains are +22.3% for PF-Wan over the Wan2.2-I2V-A14B base and +9.2% for PF-Cosmos over Cosmos3-Nano; most of the lift over vanilla fine-tuning is on task-oriented dimensions (Manipulation, Spatial, Long-horizon, Reasoning). Ablations show the two losses are complementary and that applying them only inside the physics mask beats uniform supervision.

## Code inspection

The paper points to `https://github.com/DAGroup-PKU/PhysisForcing.github.io`, which is only a project-page stub. A separate repository `https://github.com/DAGroup-PKU/PhysisForcing` exists but, at the time of this run, lists inference and training code as **"Coming soon"**.

What is actually in the repository:

- `pf_wan/` — a self-contained **inference bundle** for the PF-Wan (Wan2.2-A14B) model. It contains the generation engine, sampler/schedule code, FSDP + Ulysses sequence-parallel helpers, and a config (`pf_wan_i2v.jsonc`). The README explicitly says the physics gains are "baked in" to the checkpoint and there is nothing extra to do at inference.
- `pf_cosmos/` — a full Cosmos3-Nano code tree, but no PhysisForcing-specific files (no loss modules, no training scripts named after the method).

We preserved a snapshot of the actually usable `pf_wan` inference code in `code/pf_wan_snapshot/`. The takeaway from the code inspection is consistent with the paper claim: PhysisForcing is a training-time only modification; the released artifact is an inference bundle with a fine-tuned checkpoint.

## Concrete probe: NumPy implementation of the forcing mechanism

Because the training code is not yet released, we built a compact pure-NumPy diagnostic that reproduces the two auxiliary losses exactly as described in the paper. The probe is in `code/physisforcing_probe.py` and its outputs are in `code/probe_outputs/`.

What the probe does:

1. Generates a synthetic 8-frame, 16×16 video: a foreground square translates right while a static sinusoidal background stays fixed.
2. Computes the **physics mask** from dense query trajectories and a synthetic depth map, using the same depth-weighted motion threshold described in the paper.
3. Builds a **teacher feature map** that encodes object identity and absolute coordinates.
4. Corrupts the teacher with noise to act as an imperfect reconstruction target.
5. Trains a tiny linear student feature map under four loss combinations:
   - MSE only (to the noisy target)
   - MSE + pixel-level trajectory loss
   - MSE + semantic-level relation loss
   - MSE + both physics losses

### Probe results (final values)

| Variant | MSE to clean teacher | Trajectory error \(L^{\mathrm{phy}}_{\mathrm{pix}}\) | Relation error \(L^{\mathrm{phy}}_{\mathrm{sem}}\) |
|---|---:|---:|---:|
| MSE only | 0.024 | 69.79 | 0.014 |
| MSE + pixel | 3.896 | **21.97** | 0.134 |
| MSE + semantic | 0.025 | 77.42 | **0.008** |
| MSE + pixel + semantic | 1.902 | **19.45** | 0.072 |

The pattern mirrors the paper's ablation: the pixel loss strongly reduces local trajectory error, the semantic loss strongly reduces relational error, and the combined loss finds a balance that improves both physics-aware metrics relative to reconstruction alone. The MSE-only variant overfits the noisy target and therefore achieves low clean MSE but poor physical consistency.

The probe does **not** validate the full 14B/16B models; it makes the forcing mechanism concrete and shows that the two auxiliary objectives actually pull features in the claimed directions.

Figures:

- `assets/mask_and_training.png` — the synthetic video with the physics mask overlay, the selected reference trajectories, and the optimization objectives.
- `assets/metric_curves.png` — MSE-to-teacher, trajectory error, and relation error curves for the four variants.

## Neighbor comparison

We triangulated against two nearby world-model papers in the same run pool:

- **GigaWorld-1** (2607.02642) reframes the world-model problem as *policy evaluation*. Its central finding is that a good evaluator is not the model with the prettiest frames but the one that stays action-faithful over long horizons. PhysisForcing and GigaWorld-1 share the goal of long-horizon physical consistency, but PhysisForcing pursues it through a training-time auxiliary loss, whereas GigaWorld-1 pursues it through data curation, action encoding, memory design, and evaluator-focused post-training on a dedicated benchmark (WMBench).
- **RynnWorld-4D** (2607.06559) makes geometry and motion explicit by co-generating RGB, depth, and optical flow in a tri-branch diffusion model. It changes the *representation* of the world model itself. PhysisForcing leaves the backbone architecture and inference pipeline unchanged and instead regularizes an intermediate DiT feature; it is closer in spirit to a regularizer than to a new modality architecture.
- **ABot-PhysWorld** (cited in the paper) applies DPO with a physics-aware discriminator. That is a *post-hoc preference-alignment* approach; PhysisForcing is positioned as a *training-time prevention* mechanism with dense, localized gradients.

## What the evidence supports

- The auxiliary losses are well-defined, localized, and model-agnostic; they can in principle be dropped into any DiT-based video generator.
- The ablation pattern (pixel improves trajectories, semantic improves relations, region focus beats uniform supervision) is mechanistically plausible and reproduced in the toy probe.
- The released `pf_wan` inference bundle confirms the "no inference overhead" claim: it is a standard Wan2.2 generation pipeline with a PhysisForcing-tuned checkpoint.
- The benchmark tables show consistent gains across backbones and benchmarks, with the largest gains on the physical-semantic domain scores (PAI-Bench domain, EZS-Bench domain) rather than on pure visual-quality scores.

## Blockers and uncertainties

- **Training code is not released.** We could not verify the exact loss implementation, the per-backbone layer choice, the feature projection MLPs, or the hyperparameters \(\lambda_{\mathrm{pix}}\), \(\lambda_{\mathrm{sem}}\), and the adaptive threshold.
- **No public checkpoints or eval harness in the repo.** The inference bundle requires downloading `backbone.pth` from Hugging Face, which we did not attempt.
- **The probe is a toy.** It demonstrates the loss mechanism but does not reproduce the reported +22.3% R-Bench lift, nor does it validate the auxiliary models' behavior on real robot videos.
- **Comparison with baselines is limited to the paper's tables.** We could not independently re-run R-Bench or PAI-Bench.

## How to rerun

```bash
cd /home/alpheratz/Projects/paper-scout/workspace
python3 code/physisforcing_probe.py
```

The script writes `metrics.json`, `mask_and_training.png`, and `metric_curves.png` to `runs/2026-07-11-gigaworld-vla-corrector-physis/physisforcing-2606.28128/code/probe_outputs/`.

## Preserved artifacts

- `code/physisforcing_probe.py` — NumPy diagnostic implementing the physics mask, pixel-level trajectory loss, and semantic-level relation loss.
- `code/metrics.json` — per-step metrics for the four training variants.
- `code/pf_wan_snapshot/` — curated files from the official inference bundle (`README.md`, `requirements.txt`, `pf_wan_i2v.jsonc`, `generate_i2v.py`, `dit.py`).
- `assets/mask_and_training.png` and `assets/metric_curves.png` — report-facing figures.
