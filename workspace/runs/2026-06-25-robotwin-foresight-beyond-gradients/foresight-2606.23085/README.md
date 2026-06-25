# Foresight Deep-Dive Report

**Paper:** Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents  
**arXiv:** 2606.23085  
**Run ID:** 2026-06-25-robotwin-foresight-beyond-gradients  
**Author team:** Haoran Zhang*, Yifu Lu*, Boyang Wang, Xuhui Kang, Yen-Ling Kuo, Zezhou Cheng, Mengdi Wang, Odest Chadwicke Jenkins

---

## Research question

Can action-conditioned world-model latents reliably detect failures in long-horizon manipulation, and at what practical cost?

**Short answer:** Yes — but reliability is strongly horizon- and architecture-dependent, and the practical cost is dominated by a 1.3B-parameter world-model backbone that adds ~183 ms of latency per policy replan. The causal Transformer detector is the strongest variant; FCP calibration gives statistical control over false positives on successful rollouts, but its guarantees require the calibration distribution to match deployment.

---

## 1. The world-model backbone: V-JEPA 2-AC

### What "action-conditioned" means here

Foresight builds on **V-JEPA 2-AC** (Assran et al., 2025). The model is a *latent* action-conditioned world model: it does not generate pixels; it predicts future **representations** in the latent space of a frozen visual encoder, conditioned on the robot's intended action chunk.

At each replan step `t`:

- `c_t` = observation context (sliding window of 8 frames, 256×256).
- `A_t = (a_t, a_{t+1}, ..., a_{t+H-1})` = policy-predicted action chunk.
- `z_t^h = Pool(f_phi(c_t))` — hidden latent summarizing what is currently observed.
- `z_t^p = Pool(g_psi(z_t^h, A_t))` — predicted latent summarizing what the world model expects to happen under the proposed actions.

Foresight uses `z_t^p` as the failure signal, not `z_t^h`. An ablation (Table 13) shows predicted latents outperform hidden latents on LIBERO-Long across all detector architectures, confirming that the action-conditioning is doing real work: failures are often not visually anomalous in isolation, but are mismatches between intended action and observed state transition.

### Training / fine-tuning

The visual encoder is **frozen**. Only the action-conditioned predictor `g_psi` is trained on robot rollouts from each benchmark. Training uses a combined objective (Appendix 7):

```
L = L_teacher_forcing + L_autoregressive_rollout
```

Code evidence from `facebookresearch/vjepa2/app/vjepa_droid/train.py` (preserved in `code/vjepa2_ac_training_loop.py`):

```python
# Teacher forcing: context = all but last frame
_z, _a, _s, _e = z[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1]
z_tf = _step_predictor(_z, _a, _s, _e)

# Autoregressive rollout: n=2 future steps
_z = torch.cat([z[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
for n in range(1, auto_steps):
    ...
loss = L1(z_tf, h) + L1(z_ar, h)
```

The target representations `h` are LayerNorm-normalized outputs of a momentum target encoder. The predictor itself has 24 transformer layers, 1024-d embeddings, 16 attention heads, and ~305M trainable parameters. The full backbone (frozen ViT-Giant encoder + trained predictor) is ~1.3B parameters.

Per-benchmark training settings from the paper:

| Setting | GPUs | Effective batch size | Peak LR | Epochs |
|---|---|---|---|---|
| LIBERO-Long | 1× H200 | 256 | 2×10⁻⁴ | 200 |
| ManiSkill-Long / BEHAVIOR-1K | 2× H200 | 512 | 2×10⁻⁴ | 200 |
| Real-world | 2× H200 | 32 | 5×10⁻⁵ | 200 |

So the action-conditioned predictor is not a small head; it is a large transformer trained from scratch per benchmark/policy family. The encoder is shared and frozen.

---

## 2. From per-timestep latents to a failure score

### Feature extraction

At every policy replan step:

1. Encode the current observation context `c_t` with the frozen ViT-Giant encoder.
2. Average-pool the 256 spatial patch tokens per frame to get `z_t^h` (1408-d).
3. Feed `z_t^h` and the policy's action chunk `A_t` to the trained predictor.
4. Average-pool the predictor's output patch tokens to get `z_t^p` (1408-d).
5. Build the timestep token `u_t = W z_t^p + p_t`, where `p_t` is a fixed sinusoidal positional encoding and `W` is a learned linear projection.

### Causal sequence model

The detector `D_theta` is a **causal Transformer**. It processes the sequence `U_{≤t} = {u_1, ..., u_t}` and outputs a per-timestep failure score `s_t ∈ [0, 1]`. The causal mask ensures `s_t` uses only information available up to `t`.

Paper architecture (Appendix 7):

- Input dim: 1408
- 2 layers, hidden dim 256, 4 attention heads
- Feedforward dim: 1024 (= 4 × 256)
- Dropout: 0.1
- L2 regularization λ = 10⁻²
- Adam optimizer, LR 10⁻⁴, 300 epochs

Three detector variants are compared: MLP, LSTM, and causal Transformer. The Transformer is consistently best on long-horizon benchmarks; the MLP is near chance on real-world data, showing that temporal aggregation is essential.

### Training signal

The detector is trained with **trajectory-level binary labels only**. Because failure timestamps are unknown, every timestep in a rollout inherits the rollout's final success/failure label `y ∈ {0, 1}`. Early-detection weighting is applied to encourage high scores before or during failure events.

Reconstructed training objective (from Section 4.3):

```
L = Σ_t w_t · BCE(s_t, y)
```

where `w_t` is larger for earlier timesteps.

---

## 3. Functional conformal prediction (FCP) for adaptive thresholding

FCP converts the continuous score `s_t` into a binary alarm with a **time-varying threshold** `δ_t` calibrated on held-out successful rollouts.

### Construction (Appendix 9)

Given `n` successful calibration trajectories `{s_t^(i)}`:

1. Compute the mean score trajectory:
   ```
   μ_t = (1/n) Σ_i s_t^(i)
   ```
2. Compute a time-varying modulation `σ_t` (standard deviation of scores at `t`).
3. For each calibration rollout, compute the nonconformity score:
   ```
   R_i = sup_t (s_t^(i) - μ_t) / σ_t
   ```
4. Let `q̂` be the `(1 - α)`-quantile of `{R_i}`.
5. The threshold is:
   ```
   δ_t = μ_t + q̂ · σ_t
   ```

A failure alarm fires at the first timestep where `s_t ≥ δ_t`.

### Distribution assumptions

Under **exchangeability** of successful calibration rollouts, FCP guarantees that the probability of falsely flagging a truly successful rollout is controlled at level `α`. This is a marginal, finite-sample guarantee; it does not require parametric assumptions on the score distribution.

**Practical caveat:** The guarantee holds only if the calibration distribution matches deployment. If the robot, camera, lighting, or task distribution shifts, the threshold may become conservative or anti-conservative. The paper explicitly notes this limitation.

### Operating point selection

The authors sweep:

```
α ∈ {0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90}
```

and select the `α` that maximizes cross-validation balanced accuracy. Selected values vary by benchmark: e.g., 0.02 for LIBERO-Long and ManiSkill-Long, 0.20 for BEHAVIOR-1K, and 0.10 for real-world experiments.

---

## 4. Simulation evaluation

### Benchmarks and policies

| Benchmark | #Tasks | Robot | Avg. steps | Policies |
|---|---|---|---|---|
| LIBERO-Long | 10 | Franka | 253 | OpenVLA, π₀-FAST |
| ManiSkill-Long | 4 | Franka | 1,484 | π₀-FAST |
| BEHAVIOR-1K | 4 | R1Pro (mobile) | 8,557 | revised π₀.₅ |

### Metrics

- **ROC-AUC:** computed from the rollout-level score `s̄ = max_t s_t`. Threshold-independent.
- **Balanced accuracy:** `BalAcc = ½(TPR + TNR)`, computed after calibrating `δ_t` on the calibration split and selecting the best `α`.

Both metrics use 3-fold cross-validation.

### Main results (Table 2)

| Method | LIBERO ROC-AUC | LIBERO BalAcc | ManiSkill ROC-AUC | ManiSkill BalAcc | BEHAVIOR ROC-AUC | BEHAVIOR BalAcc |
|---|---|---|---|---|---|---|
| FAIL-Detect | 0.90±0.02 | 0.82±0.06 | 0.71±0.02 | 0.50±0.01 | 0.54±0.06 | 0.52±0.01 |
| SAFE-LSTM | 0.91±0.02 | 0.88±0.02 | 0.82±0.01 | 0.74±0.01 | 0.72±0.02 | 0.64±0.05 |
| RND | 0.90±0.02 | 0.83±0.04 | 0.83±0.02 | 0.68±0.18 | 0.65±0.01 | 0.54±0.04 |
| Gauge | 0.88±0.01 | 0.81±0.06 | 0.80±0.02 | 0.77±0.03 | 0.61±0.03 | 0.60±0.03 |
| **Foresight-Transformer** | **0.89±0.02** | **0.94±0.06** | **0.84±0.03** | **0.80±0.10** | **0.76±0.02** | **0.78±0.02** |

The gains are largest on BEHAVIOR-1K, the longest benchmark (avg. 8,557 steps). Foresight-Transformer improves balanced accuracy by **+0.14** and ROC-AUC by **+0.04** over the best baseline. This is the core empirical claim: action-conditioned world-model features are especially useful over long trajectories where failures develop gradually.

### What failure modes are detected in simulation?

Qualitative examples (Section 13):

- **LIBERO-Long Task 5:** robot drops the book mid-execution; alarm fires before termination.
- **ManiSkill-Long Task 3 (Stack 3 Cubes):** robot fails to stack the red cube on the blue cube.
- **BEHAVIOR-1K Task 47 (Cook Hot Dogs):** robot fails to grasp the first hot dog; alarm raised early.

These are execution-level deviations — the policy intended an action, but the world state did not evolve as predicted.

---

## 5. Real-robot validation

### Setup

| Robot | Tasks | Policy | #Episodes | Avg. steps |
|---|---|---|---|---|
| ReactorX-200 | pick banana + lion toy; arrange objects; pick up lego | ACT | 3 × 40 | ~1,150 |
| ReactorX-200 | same tasks | π₀.₅ | 3 × 40 | ~1,190 |
| ReactorX-200 | same tasks | SmolVLA | 3 × 40 | ~1,192 |
| Franka Emika Panda | pick 3 toys | GR00T N1.5 | 44 | ~1,727 |

Camera: RealSense D435. Rollouts were teleoperated / policy-collected and labeled with final success/failure.

### Results (Table 3, ROC-AUC)

| Method | ReactorX/ACT | ReactorX/π₀.₅ | ReactorX/SmolVLA | Franka/GR00T N1.5 |
|---|---|---|---|---|
| FAIL-Detect | 0.85±0.07 | 0.64±0.06 | 0.71±0.05 | 0.88±0.05 |
| SAFE-LSTM | 0.70±0.07 | 0.75±0.14 | 0.43±0.10 | 0.79±0.10 |
| RND | 0.86±0.04 | 0.78±0.06 | 0.82±0.03 | 0.64±0.15 |
| **Foresight-Transformer** | **0.93±0.01** | **0.87±0.03** | 0.79±0.09 | **0.89±0.10** |

Foresight-Transformer wins in 3 of 4 settings and is the only method that transfers across robot embodiments (ReactorX → Franka). The MLP detector again performs near chance (0.50–0.59), reinforcing that frame-independent classification is insufficient for real-world long-horizon monitoring.

### Cross-policy generalization (Table 4)

Training on one policy and testing on another:

| Train | Test | ROC-AUC | BalAcc |
|---|---|---|---|
| π₀.₅ → ACT | ACT | **0.94±0.02** | 0.82±0.08 |
| SmolVLA → ACT | ACT | 0.92±0.04 | 0.73±0.07 |
| π₀.₅ → SmolVLA | SmolVLA | 0.67±0.02 | 0.62±0.01 |
| ACT → π₀.₅ | π₀.₅ | 0.56±0.07 | 0.52±0.03 |
| π₀-FAST → OpenVLA | OpenVLA | 0.64±0.02 | 0.90±0.01 |

Transfer is **feasible but asymmetric**. Training on π₀.₅ transfers well to ACT because π₀.₅ rollouts include broader behaviors and recovery trajectories. Training on ACT transfers poorly to π₀.₅ because the ACT detector has never seen recovery behavior. This is important: the detector is not purely policy-agnostic; its transfer strength depends on whether the training policy covers the target policy's failure modes.

---

## 6. Cost analysis: latency, compute, and deployability

All measurements on a single NVIDIA H200 GPU (Table 14).

| Component | Parameters | Latency (ms) |
|---|---|---|
| World-model encoder (ViT-G/16, 8 frames) | 1,012 M | 122.54±0.08 |
| Action-conditioned predictor | 305 M | 60.19±0.07 |
| **Feature extraction subtotal** | **1,317 M** | **182.73±0.11** |
| Failure detector (MLP) | 0.4 M | 0.08±0.00 |
| Failure detector (LSTM) | 3.4 M | 0.16±0.00 |
| Failure detector (Transformer) | 2.0 M | 0.91±0.02 |
| **Total Foresight-Transformer** | **1,317 M** | **183.64±0.02** |

### Key cost facts

- The world-model backbone accounts for **over 99%** of inference time.
- The detector head is negligible regardless of architecture.
- For π₀-FAST, Foresight runs once per action chunk (every 16 control steps), so the relevant budget is the replan interval, not the control step.

### Is it deployable for closed-loop fast control?

**Qualified yes for chunk-based policies, no for high-frequency reactive control.**

- ~183 ms per replan is acceptable when the policy already replans every 100–1000 ms.
- It is too slow for closed-loop servoing at hundreds of Hz.
- On-device / edge deployment is challenging: a 1.3B-parameter model needs substantial GPU memory and power.
- The paper lists this as a key limitation: "computational cost and latency of pretrained world models ... may limit applicability to highly reactive or agile tasks requiring fast closed-loop control."

A practical deployment would likely run the detector asynchronously or only at replan boundaries, not synchronously with every control tick.

---

## 7. Code evidence and what could not be verified

### What was found

The official **Foresight** code repository is **not publicly available**. See `BLOCKER.md` for the full search log.

The official **V-JEPA 2-AC** repository is public at `https://github.com/facebookresearch/vjepa2` and was cloned and inspected. It is the actual code that produces the `z_t^p` representation Foresight consumes. Key preserved snippets are in `code/`:

- `code/vjepa2_ac_predictor_arch.py` — predictor architecture (24-layer causal transformer, action/state tokens, RoPE, projection back to 1408-d).
- `code/vjepa2_ac_causal_mask.py` — block-causal attention mask; shows why "action-conditioned" is more than concatenation.
- `code/vjepa2_ac_training_loop.py` — teacher-forcing + autoregressive rollout objective.
- `code/vjepa2_inference_wrapper.py` — how the encoder/predictor are called at inference.
- `code/foresight_detector_pseudocode.py` — faithful reconstruction of the causal Transformer detector and FCP thresholding from the paper.
- `code/diagnostic_shape_trace.py` / `diagnostic_shape_trace_pure.py` — shape checks for the full pipeline; the pure-Python version runs without PyTorch.

### What the paper says but code cannot confirm

Without the official Foresight repo, the following integration details remain taken on author authority:

- Exact early-detection weighting schedule.
- How variable-length rollouts are padded / packed for the causal Transformer.
- How action chunks are aligned across policies with different action dimensions and replan intervals.
- How the FCP threshold is extrapolated when test rollouts are longer than calibration rollouts.
- Exact data splits and random seeds for the 3-fold cross-validation.

These are not minor details; they affect reproducibility and fair comparison.

---

## 8. Limitations and critical assessment

### Strengths

1. **Unified interface.** Foresight needs only observations and action chunks; no policy logits, hidden states, or uncertainty heads. This makes it genuinely policy-interface-agnostic.
2. **Long-horizon scaling.** The BEHAVIOR-1K result is the most impressive: a 0.14 BalAcc improvement over baselines on 8,500-step rollouts.
3. **Statistical calibration.** FCP provides a principled way to set time-varying thresholds and control false positives on successful rollouts.
4. **Real-world transfer.** Works across ReactorX and Franka embodiments, suggesting the learned representations capture execution-level rather than purely visual cues.

### Weaknesses and caveats

1. **No public code.** As of this run, the implementation cannot be inspected or reproduced from author-provided artifacts.
2. **Heavy backbone.** 1.3B parameters and ~183 ms latency make edge deployment hard.
3. **Calibration fragility.** FCP guarantees assume exchangeability and distribution match. Real-world domain shift (lighting, wear, new objects) is not studied.
4. **Failure labels only at rollout level.** The method cannot localize *which* subgoal failed without further annotation or analysis.
5. **Asymmetric cross-policy transfer.** The method is not automatically policy-agnostic; training policy coverage matters.
6. **Limited failure-mode taxonomy.** The paper reports aggregated success/failure; it does not systematically characterize which failure categories (grasp misses, collisions, navigation errors, semantic mistakes) are easiest or hardest to detect.

### What remains uncertain

- How sensitive is the method to the quality of the action-conditioned predictor? If V-JEPA 2-AC is poorly fine-tuned on a new robot/domain, does failure detection degrade gracefully or catastrophically?
- Can the detector be trained with far fewer rollouts? The real-world experiments use 40–44 episodes per task; is that the minimum or just what was collected?
- Could a smaller world model (e.g., V-JEPA 2 ViT-L) achieve similar detection performance with much lower latency? The paper uses only ViT-Giant.

---

## 9. Bottom line

Foresight demonstrates that **action-conditioned predicted latents from a large pretrained world model are a strong signal for long-horizon manipulation failure detection**. The causal Transformer detector and FCP thresholding are the right architectural choices; the gains are largest on the longest, most compositional tasks. However, the practical cost is real — a 1.3B-parameter backbone running on an H200-class GPU — and the absence of public source code means several integration details remain unverified. For closed-loop monitoring of chunk-based VLA policies, Foresight is a promising but computationally expensive solution; for high-rate reactive control, it is not yet practical.

---

## Evidence index

| Claim | Source |
|---|---|
| V-JEPA 2-AC architecture and training objective | `facebookresearch/vjepa2` (cloned), preserved in `code/` |
| Foresight pipeline and detector specs | Paper Sections 4.1–4.4, Appendix 7 |
| FCP threshold construction | Paper Appendix 9 |
| Simulation metrics and results | Paper Table 2, Appendix 8 |
| Real-robot results | Paper Table 3, Table 4, Section 5.4 |
| Latency / cost numbers | Paper Section 14, Table 14 |
| No official Foresight code | Search results documented in `BLOCKER.md` |
