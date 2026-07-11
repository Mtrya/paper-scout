# VLA-Corrector (arXiv 2607.01804) — Deep-dive thread

**Focus question:** Does the proposed detect-and-correct mechanism meaningfully reduce open-loop errors in action-chunked VLA policies, and how lightweight is it in practice?

**Answer in one sentence:** Truncation is the workhorse; the external latent-dynamics corrector is genuinely small, but the public implementation adds several thresholds/heuristics not described in the paper, and full evaluation is blocked by missing checkpoints and the inability to run the LeRobot fork end-to-end without its dependencies.

---

## 1. What the paper claims

VLA-Corrector targets the **fixed action-horizon blind spot** of action-chunked VLAs. Its four pieces are:

1. **External latent-dynamics corrector** `M_φ`: a ~40 MLP trained on frozen VLA visual features to predict the short-horizon latent residual `ΔZ = Z_{t+k} − Z_t` induced by the executed action.
2. **Latent-space Vision Monitor (LVM)**: compares the predicted residual with the observed residual using `E_t = 1 − CosSim(ΔZ^pred, ΔZ^real)`.
3. **Event-triggered truncation**: when `E_t` exceeds a robust median+MAD threshold for `p` consecutive steps, discard the remaining queued actions and replan.
4. **Online Gradient Guidance (OGG)**: on the single recovery replan, backprop through the corrector to nudge the first action so its predicted latent effect aligns with `ΔZ_corr = ΔZ_exp − ΔZ_dev`.

Reported MetaWorld results (π₀.₅, horizon 50): baseline 48.70 % success, +Truncation 60.35 %, +Truncation+OGG 64.35 %. The paper emphasizes that the VLA backbone stays frozen and the corrector is lightweight.

---

## 2. Code inspection

**How the repo was obtained.** Direct `git clone` from `https://github.com/ZJU-OmniAI/vla-corrector` timed out in this environment, so the relevant files were pulled through the GitHub REST API (`gh api`) and placed under `code/vla-corrector/`. Curated snippets are preserved in this thread under `code/official_snippets/`.

**Key implementation files examined:**

| File | What it contains |
|------|------------------|
| `src/siglip_dynamics/MLP.py` | `SiglipResidualMLP`: action embedding + residual MLP blocks, outputs `ΔZ`. |
| `src/siglip_dynamics/config.py` | `SiglipMLPConfig` with scales `4m / 20m / 100m` and loss types `mse / cosine / both`. |
| `src/siglip_dynamics/inference/guidance_injector.py` | `GuidanceInjector`: stores `z_baseline` and `target_delta_z`, computes cosine guidance loss. |
| `src/siglip_dynamics/inference/circuit_breaker.py` | A simple `CircuitBreaker` with median+MAD threshold. |
| `src/siglip_dynamics/inference/safety_module.py` | `SafetyModuleLoader`: lazy-loads the trained corrector and predicts `ΔZ`. |
| `src/lerobot/safety/siglip_dynamics_mlp.py` | `SiglipDynamicsPredictor`: higher-level wrapper that picks checkpoints and instantiates MLP/Transformer/DiT variants. |
| `src/lerobot/scripts/lerobot_eval_modified_detection.py` | The actual evaluation entry point; contains a much richer `RobustThresholdState` and `robust_threshold_step`. |
| `src/lerobot/policies/pi05_modified/modeling_pi05_modified.py` | Modified π₀.₅ policy that injects the OGG gradient into the flow-matching velocity field. |

**Code-paper discrepancies found:**

- **The monitor is more complex than the paper description.** The evaluation script (`lerobot_eval_modified_detection.py`) uses an EWMA-smoothed inconsistency score, a bootstrap phase, a base noise floor, a jump trigger, a hard-retrigger margin, and separate on/off thresholds. The simple `CircuitBreaker` in `siglip_dynamics/inference/circuit_breaker.py` is **not** the code path used at evaluation time. The paper only describes median+MAD + persistence.
- **Corrector architectures include Transformer and DiT variants**, not just the MLP described in the paper. The default config scale is `M20`, but the corresponding custom widths `[2048, 2048, 2048, 2048]` actually yield ~38–40 M parameters (the README still says 38–42 M), so the “M20” label is a misnomer.
- **OGG has extra knobs.** The code exposes `guidance_apply_every`, `guidance_loss_objective` (e.g. `attract_delta_z_correction`), `guidance_eta`, and `guidance_compare_baseline`. The paper describes a single default `η = 1` and the attract-to-corrective-direction loss.
- **The repo does not ship any checkpoints, datasets, or fine-tuned VLA weights.** The README explicitly states that users must supply `<POLICY_CHECKPOINT>` and `<CORRECTOR_CHECKPOINT>`. This means the public code cannot reproduce the reported numbers without additional artifacts.

---

## 3. Synthetic probe

Because the real checkpoints and LeRobot environment are unavailable, I built a self-contained toy probe that implements the same mechanism in pure NumPy. It is **not** a reproduction of the paper; it is a sanity check that the detect-and-correct loop behaves as advertised.

**What it does:**

- A 2-D point-mass must track a target.
- A fixed visual encoder maps state → 32-D latent token.
- A small residual MLP corrector is trained on *normal* trajectories to predict `ΔZ` from `(Z_t, a_t)`.
- During rollout, the monitor queues predicted residuals `k=5` steps ahead and compares them with observed residuals.
- On trigger, the remaining action queue is cleared and a recovery replan is generated. With OGG enabled, the first action of the replan is refined by a few gradient steps on the corrector’s cosine loss.

**How to rerun:**

```bash
cd runs/2026-07-11-gigaworld-vla-corrector-physis/vla-corrector-2607.01804/code
python3 probe_vla_corrector.py
```

**Results (200 episodes, success threshold = 0.8):**

| Method | Mean final distance | Success rate | Avg. policy calls | Success / call | Avg. truncations |
|---|---:|---:|---:|---:|---:|
| Baseline H=5 | 0.964 | 39.5 % | 16.00 | 0.025 | 0.00 |
| Baseline H=10 | 1.375 | 18.0 % | 8.00 | 0.022 | 0.00 |
| Baseline H=20 | 1.743 | 25.5 % | 4.00 | 0.064 | 0.00 |
| Baseline H=50 | 6.773 | 0.0 % | 2.00 | 0.000 | 0.00 |
| + Truncation (H=20) | 1.423 | 35.5 % | 4.98 | 0.071 | 1.18 |
| + Truncation + OGG (H=20) | 1.362 | 33.5 % | 5.03 | 0.067 | 1.24 |

**Interpretation:**

- Long horizons amplify open-loop error after the perturbation (H=50 is essentially lost).
- Truncation recovers most of the closed-loop benefit of a short horizon while keeping policy-call count close to the long-horizon baseline.
- OGG’s additional gain is small in this toy setting; this matches the paper’s ablation where truncation alone closes most of the gap and OGG adds a few points.
- The toy corrector has **12,640 parameters** and **≈25 K FLOPs** per forward pass, versus **2,208 parameters / 4 K FLOPs** for the toy visual encoder. The probe therefore demonstrates the *shape* of a lightweight add-on, even though it is not scaled to real VLA sizes.

Figures are saved in the run-level `assets/` directory:

- `probe_trajectories.png`: one representative episode for baseline H=20, +Truncation, and +Truncation+OGG.
- `probe_metrics.png`: success rate, policy calls, and success-per-call across the horizon sweep.

---

## 4. Neighbor comparison

Two closely related recent papers are already in the pool:

1. **Look Before You Leap (arXiv 2607.03751)** also keeps the VLA backbone frozen, but its lever is **action evaluation**: it distills MCTS search into a lightweight Q-value model and reranks multiple candidate actions at test time. VLA-Corrector instead monitors *execution drift* and intervenes during the open-loop window. The two approaches are complementary: one improves the action selected at each query, the other decides *when* the current query has gone stale.

2. **LaMem-VLA / Dual Latent Memory (arXiv 2607.07608)** addresses temporal context by weaving short-term and long-term latent memory tokens directly into the VLA reasoning sequence. It requires end-to-end training of the memory modules. VLA-Corrector is cheaper: it adds an external monitor and only modifies inference, leaving the backbone weights untouched. The trade-off is that VLA-Corrector cannot create recovery behaviors the backbone cannot represent, whereas LaMem-VLA can, in principle, learn them.

3. **Plain action-chunking baseline.** The paper’s own horizon sweep is the most important baseline. It shows that simply increasing `H` lowers policy calls but reliably hurts success; VLA-Corrector’s value is shifting the Pareto curve rather than picking a better fixed point.

---

## 5. Blockers and uncertainties

- **No runnable official checkpoint.** The repo is a LeRobot fork without bundled model weights or trained corrector checkpoints, so I could not run the actual evaluation or trace a real action-chunk generation end-to-end.
- **Git-over-HTTPS was blocked** in this environment; code was recovered via the GitHub API, but large binary assets (videos, presentation PDFs) were skipped.
- **Dependency wall.** The full environment requires the LeRobot ecosystem, MuJoCo/MetaWorld/LIBERO, and the base VLA checkpoints. Installing that stack was out of scope for a single deep-dive turn.
- **Implementation complexity gap.** The paper’s concise median+MAD description is wrapped in a much heavier thresholding state machine in the released evaluation script. It is unclear how much of the reported gain comes from the core idea versus the extra heuristics (EWMA, jump trigger, hard retrigger, base noise floor).

---

## 6. Durable artifacts

- **Paper cache:** `papers/vla/vla-corrector-2607.01804.md`
- **Thread README:** this file
- **Probe:** `runs/2026-07-11-gigaworld-vla-corrector-physis/vla-corrector-2607.01804/code/probe_vla_corrector.py`
- **Probe outputs:** `code/summary.json`, plus `probe_trajectories.png` and `probe_metrics.png` copied to `runs/2026-07-11-gigaworld-vla-corrector-physis/assets/`
- **Official code snippets:** `code/official_snippets/` (MLP, config, guidance injector, circuit breaker, safety module, SiglipDynamicsPredictor, evaluation entry point, modified π₀.₅ policy)
