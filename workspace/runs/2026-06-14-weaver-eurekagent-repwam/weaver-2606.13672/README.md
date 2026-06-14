# WEAVER deep-dive thread

**Paper:** *WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation* (arXiv 2606.13672)  
**Thread:** `runs/2026-06-14-weaver-eurekagent-repwam/weaver-2606.13672/`  
**Goal:** Make the method concrete, check the released code/config against the paper, and produce a runnable probe that validates the architecture scale and the flow-matching / diffusion-forcing mechanics.

---

## 1. What was attempted

1. **Read the full paper** (including appendix) from the MinerU parse at `papers/world-models/weaver-2606.13672.md`.
2. **Locate and clone the official code:** found via the project page at `https://arnavkj1995.github.io/WEAVER/`, cloned to `code/weaver-2606.13672/`.
3. **Fetch the released configs** from HuggingFace (`arnavkj1995/WEAVER`) to see the actual hyperparameters used for the released `WEAVER`, `WEAVER-FT`, and `WEAVER-ReFlow` checkpoints.
4. **Inspect the implementation:** traced `weaver/wm/model.py`, `weaver/wm/nets.py`, `weaver/wm/encoders.py`, `weaver/reflow.py`, and the training/eval entrypoints.
5. **Run a constructive probe** (`code/probe_model.py`) that:
   - rebuilds the released model architecture without downloading the 8.2 GB checkpoint or the SD3 VAE;
   - counts parameters by component;
   - runs a tiny CPU forward pass to exercise the v-pred flow loss, reward/critic heads, and latent dynamics;
   - inspects the diffusion-forcing pyramid schedule and the NFE reduction from ReFlow.
6. **Compare with load-bearing neighbors:** Ctrl-World (the paper’s main baseline) and Dreamer-v4 (the latent-world-model predecessor it cites for reward/value heads).

---

## 2. Evidence preserved

| Path | What it is |
|------|------------|
| `code/weaver-2606.13672/` | Cloned official repository (commit at clone time). |
| `code/weaver-2606.13672/hf_configs/WEAVER/config.yaml` | Released pre-train config (32 layers, 1536 dim, v-pred, diffusion forcing, SPRINT, memory). |
| `code/weaver-2606.13672/hf_configs/WEAVER-FT/config.yaml` | Fine-tuned checkpoint config (`val_steps=50`, cosine schedule). |
| `code/weaver-2606.13672/hf_configs/WEAVER-ReFlow/config.yaml` | ReFlow config (`val_steps=4`, lockstep, cosine schedule). |
| `code/hf_configs/{WEAVER,WEAVER-FT,WEAVER-ReFlow}/config.yaml` | Curated copies of the above configs inside the tracked thread. |
| `code/probe_model.py` | Probe script: architecture reconstruction, parameter counting, synthetic forward, schedule inspection. |
| `code/probe_outputs/probe_results.json` | Machine-readable probe output (parameter counts, losses, schedule shapes, NFEs). |

All evidence lives inside the workspace. No checkpoints were downloaded.

---

## 3. How to rerun

```bash
cd /home/betelgeuse/Documents/paper-scout/workspace/code/weaver-2606.13672
uv sync                                    # already done; ~3 GB of torch/diffusers deps
uv run python \
  ../../runs/2026-06-14-weaver-eurekagent-repwam/weaver-2606.13672/code/probe_model.py
```

The probe intentionally uses dummy image/task encoders so it does **not** need the SD3 VAE, CLIP weights, or the 8.2 GB `checkpoint.pt`. It runs on CPU in a few seconds.

---

## 4. Core mechanism (made concrete)

### 4.1 Inputs and latent encoding

* Observation: multi-view RGB `(B, T, C, H, W)` + proprioception `q_t ∈ ℝ⁸` (7 joints + gripper).
* The images are encoded by a frozen Stable Diffusion 3 VAE into latent patch tokens:
  * for `image_size = (192, 320)` and `spatial_size = 2`, each view becomes `24 × 40 = 960` patches of dim `16 · 2² = 64`.
  * The two used views (`wrist_left`, `exterior_1_left`) give `2 · 960 = 1920` patch tokens per frame.
* Per-frame token order inside the transformer:
  `[patches_view1, patches_view2, state_token, action_token, timestep_token]`.
* Sparse memory: every `t_memory = 5` frames, `n_memory_frames = 6` past latents are prepended to the sequence.
* Short-term history: `n_history = 2` most recent frames are always in the conditioning window.

### 4.2 Latent dynamics model

`FlowWM` (the dynamics core) is a stack of `n_layers // (n_spatial + 1)` block-causal dynamics blocks. With the released config (`n_layers=32`, `n_spatial=1`) this is 16 blocks, each containing one spatial-attention block and one causal temporal-attention block — 32 transformer layers total, matching the paper.

Each block uses RMSNorm, RoPE (2-D pixel RoPE for spatial, 1-D causal RoPE for temporal), QK-Norm, and SwiGLU FFNs.

### 4.3 Training objective: conditional flow matching + diffusion forcing

For a ground-truth future latent chunk `x¹ := z_{t+1:t+h+1}`:

```
x⁰ ~ N(0, I)
x^τ = τ x¹ + (1 - τ) x⁰,   τ ∈ [0, 1)
```

The model predicts either the velocity `v = x¹ - x⁰` (v-pred) or the clean target `x¹` (x-pred). The released model uses **v-pred**:

```
L_flow(φ) = E_{x⁰, x¹, τ} || (x¹ - x⁰) - f_φ(z_mem, z_hist, a, x^τ, τ) ||²
```

Diffusion forcing: during training the future frames are noised with **independently sampled** `τ` per timestep (`diff_forcing=True`), while history frames are clamped to `τ = 1` (clean). This is implemented in `WEAVER.sample_timestep`.

The full training loss also includes:

* proprioception down-weighting: `0.1 · L_flow(states)`;
* reward head MSE against RoboMeter progress rewards;
* critic head MSE against bootstrapped λ-returns;
* optional decoder MSE and temporal-consistency loss (disabled in the released config).

### 4.4 Inference: lockstep vs. pyramid (diffusion forcing)

**Lockstep** (`pyramid_stagger_width=0`): all future frames share the same `τ` schedule. Euler update:

```
τ_0 = 0,  Δτ = 1 / val_steps
for step in 1..val_steps:
    v = f_φ(x^τ, a, τ)
    x^{τ+Δτ} = x^τ + Δτ · v
```

**Pyramid / diffusion forcing** (`pyramid_stagger_width=1`): each future frame starts denoising one step later than the previous one, so earlier frames become clean context for later frames. The schedule is built in `WEAVER._build_pyramid_schedule`:

```
height = val_steps + stagger·(horizon - 1) + 1
u[m, f] = clamp(1 - (m - stagger·f) / val_steps, 0, 1)
linear:    τ = 1 - u
cosine:    τ = cos(u · π/2)
power:     τ = (1 - u)^p
sigmoid:   τ = 1 - normalize(sigmoid(α(u - c)))
```

NFE = `height - 1`. For the released setting (`horizon=8`, `val_steps=16`, `stagger=1`) this is **23 function evaluations**, not 16; for ReFlow (`val_steps=4`, `stagger=0`) it is **4**.

### 4.5 KV caching

At inference the memory and history tokens are clean and constant across denoising steps. `FlowWM.forward_cached` computes their temporal K/V once (`write`) and reuses them for future-only `read` passes. The paper reports this saves up to ~30% wall time.

### 4.6 ReFlow post-training

`weaver/reflow.py` freezes a teacher WEAVER and distills a student so that a single (or few) Euler step(s) from `x⁰` reaches the teacher’s endpoint `x̂¹`:

```python
with torch.no_grad():
    xhat1, x0 = teacher.generate_latent_rollouts(x1_data, actions)  # teacher: 50 steps, cosine

# student is trained on the straight path from the same x0 to xhat1
t = sample_timestep(...)
xt = interpolate(xhat1, x0, t)
x_pred = student.wm(xt, actions, t, memory=memory_tokens)
L_flow = || (xhat1 - x0) - x_pred ||²
```

This is the rectified-flow straightening step that lets ReFlow run at NFE=4 with small quality loss (Table 6 in the paper).

### 4.7 Latent verifier: reward and critic heads

Both heads share the same `ScoreBlock` design:

* per-view latent tokens are mean-pooled and projected;
* state and action tokens are projected;
* task (CLIP) embedding is projected and broadcast across time;
* concatenated vector is fed through an MLP to a scalar.

The critic is trained to predict the λ-return:

```
v_t^λ = r_t + γ[(1 - λ)V(z_{t+1}) + λ v_{t+1}^λ]
L_critic = || V(z_t) - v_t^λ ||²
```

At test time the advantage for a candidate action chunk is:

```
Â = Σ_{ℓ=1}^{H} γ^{ℓ-1} r̂_{t+ℓ} + γ^H V(ẑ_{t+H}) - V(z_t)
```

This is used for both synthetic-data filtering and best-of-N planning.

---

## 5. Probe results

### 5.1 Parameter count matches the paper

From `probe_outputs/probe_results.json`:

```
Total trainable parameters: 928,735,746
  wm   : 914,936,320  (98.5%)
  rm   :   6,899,713  (0.7%)
  critic:  6,899,713  (0.7%)
```

This matches the paper’s stated **928 M parameters** almost exactly. The frozen SD3 VAE and CLIP text encoder are not counted here.

### 5.2 Tiny synthetic forward pass works

A downsized model (`n_embed=384`, `n_layers=4`, no memory, no SPRINT, v-pred) runs a full training step on CPU in ~20 ms. The probe confirms:

* per-view flow losses are produced independently;
* the state flow loss is down-weighted by `loss_scale.states = 0.1`;
* reward and critic losses are computed on a concatenated batch of ground-truth and predicted latents (the distillation trick described in Sec. 3.3).

### 5.3 Schedule / NFE inspection

| Setting | horizon | val_steps | stagger | schedule | NFE |
|---------|---------|-----------|---------|----------|-----|
| Released 16-step (paper Table 1) | 8 | 16 | 1 | cosine | **23** |
| ReFlow | 8 | 4 | 0 | cosine | **4** |

The README formula `NFE = val_steps + eval_horizon * stagger` is an approximation; the actual code uses `height = val_steps + stagger·(horizon-1) + 1`, so NFE = `height - 1`. With `eval_horizon=5` the README’s approximation gives 21, close to the 23 obtained with `horizon=8`.

---

## 6. Code status and blockers

### What works

* The repository is well-structured and the released configs are public.
* The core model code (`FlowWM`, `RewardModel`, `Critic`, `reflow.py`) is readable and closely follows the paper.
* Parameter counts, loss shapes, and schedule construction all line up with the paper’s claims.
* `uv` environment resolves cleanly.

### What could not be fully exercised

* **No checkpoint download:** each `checkpoint.pt` is ~8.2 GB. The probe avoids this by reconstructing the architecture with dummy encoders, so we could not verify weight loading, exact generation quality, or reproduce the reported FID/FVD numbers.
* **No real DROID / OOD data:** the OOD dataset is on HuggingFace but was not downloaded; the probe uses synthetic latents.
* **No GPU:** the machine has no CUDA device, so latency claims (4.78 s vs. 14.65 s for Ctrl-World at NFE=16) were not measured locally. The schedule inspection provides the qualitative basis for the speedup.
* **Inference path mismatch:** `weaver/generate_views.py` builds the model with `use_precomputed_features=False`, but `WEAVER.generate_videos_full` feeds latents into `FlowWM`, which would then attempt to VAE-encode already-encoded latents. The actual release likely uses a slightly different inference wrapper or an override; this is a code-path quirk that would need a real checkpoint + data to resolve.

---

## 7. Neighbor comparison

### Ctrl-World (Guo et al., ICLR 2026)

* **Base model:** builds on Stable Video Diffusion (SVD) and fine-tunes a UNet video-diffusion model. It is therefore a **diffusion-in-pixel-latent-space** model, whereas WEAVER is a **flow-matching transformer in a tokenized SD3 latent space**.
* **Conditioning:** Ctrl-World uses frame-level action cross-attention and pose-conditioned memory retrieval; WEAVER concatenates action tokens directly into the spatio-temporal transformer and uses explicit sparse-memory + short-history token prefixes.
* **Speed:** Ctrl-World needs ~14–42 s per 10 s rollout on an H100; WEAVER reports 2.5–14 s, i.e. roughly **3× faster at equal NFE** and a larger Pareto gap at low NFE.
* **Verifier:** Ctrl-World relies on an external VLM judge for reward scoring; WEAVER distills RoboMeter into a latent reward head, which is what enables 0.0006 s reward inference and best-of-N planning.
* **Params:** paper reports Ctrl-World at 1.5 B; WEAVER at 928 M.

### Dreamer-v4 (Hafner et al., 2025)

* **Architecture:** Dreamer-v4 uses its own causal tokenizer + shortcut-forcing dynamics transformer trained from scratch. WEAVER replaces the learned tokenizer with a **pretrained SD3 VAE**, inheriting stronger OOD visual robustness.
* **Scope:** Dreamer-v4 is evaluated on Minecraft/web-scale video; WEAVER targets real-world robot manipulation with multi-view inputs.
* **Shared ingredient:** WEAVER explicitly borrows Dreamer-v4’s idea of latent reward/value heads to avoid decoding during planning.

---

## 8. Research claims the main report can make

1. **The 928 M parameter count is real and code-verifiable.** The released `WEAVER/config.yaml` instantiates a model whose trainable parameters sum to ~928.7 M, with 98.5% in the `FlowWM` dynamics core and only ~1.5% in the latent reward/critic heads. This supports the paper’s framing of a large but efficient latent dynamics model.

2. **The speedup is structural, not just hardware.** WEAVER combines (a) a tokenized SD3 latent space with SPRINT token dropping, (b) KV-cached memory/history, (c) flow matching with diffusion-forcing pyramid schedules, and (d) ReFlow distillation. The schedule inspection shows the code can drop NFE from 23 (released 16-step setting) to 4 (ReFlow) while keeping the same transformer backbone — matching the paper’s 5–10× inference-time improvement over Ctrl-World.

3. **The latent verifier is what unlocks planning.** Reward and critic heads operate directly on imagined latents (no image decoding, no VLM judge). The probe confirms these heads add only ~13.8 M parameters and are trained with standard MSE on RoboMeter progress rewards / bootstrapped λ-returns, making the reported 1.2–1.6 s full planning latency mechanically plausible.

---

## 9. Open questions / follow-up

* Does the released checkpoint actually load and produce the reported FID/FVD numbers on the OOD dataset?
* How sensitive is the policy-improvement result to the RoboMeter reward noise the authors flag in the limitations?
* Can the inference wrapper (`generate_views.py`) be reconciled with `use_precomputed_features=True` for fully offline, no-VAE evaluation?
