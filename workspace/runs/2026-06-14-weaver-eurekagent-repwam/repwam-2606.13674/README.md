# RepWAM deep-dive thread

**Paper:** RepWAM: World Action Modeling with Representation Visual-Action Tokenizers (arXiv 2606.13674)  
**Thread directory:** `runs/2026-06-14-weaver-eurekagent-repwam/repwam-2606.13674/`  
**Investigator:** Paper Scout reading agent  
**Date:** 2026-06-14

---

## What was attempted

1. **Read the full MinerU-parsed paper** at `papers/world-models/repwam-2606.13674.md`.
2. **Clone and inspect the official code** from `https://github.com/wdrink/RepWAM`.
3. **Make the method concrete** by writing down the key equations and implementing a minimal runnable probe of the core components.
4. **Run a constructive probe** on synthetic data to exercise the visual tokenizer losses, the Latent Action Tokenizer (IDM+FDM with transport operator), and the causal WAM flow-matching objective.
5. **Compare with load-bearing neighbors:** LAPA (discrete latent actions, VQ-VAE) and Lingbot-VA (WAN2.2 VAE + causal diffusion transformer).

---

## Key findings

### Official code status — BLOCKER

The GitHub repository `wdrink/RepWAM` is currently a **project-page placeholder**. It contains only:

- `README.md` with the open-source plan
- `assets/` with paper figures and demo GIFs

The README explicitly states:

> - [2026/06/12] ~~Paper release.~~
> - [2026/06] Inference code release.
> - [2026/06] Code and model weights release.

**No implementation, checkpoints, or training configs are available yet.** Therefore no end-to-end trace of the official model, no weight loading, and no reproduction of the reported RoboTwin / real-robot numbers was possible.

### What the method actually is (made concrete)

RepWAM is a **representation-centric world action model** built in three stages:

**Stage 1 — Representation Visual-Action Tokenizer (RepViTok)**

A visual tokenizer is a ViT autoencoder trained with

```
L_vis = L_rec + lambda_align * L_align
L_rec  = lambda_1 * ||o - o_hat||_1 + lambda_perc * L_perc(o, o_hat) + lambda_gan * L_gan(o_hat)
L_align = || avg(W_align * z) - avg(G(o)) ||_2^2
```

where `G(o)` is a frozen visual foundation model (paper cites Perception Encoder) and `z` are the video latents.

On top of the frozen visual latents, a **Latent Action Tokenizer (LAT)** couples an inverse dynamics model `q_phi` with a forward dynamics model `f_psi`:

```
l_t            = q_phi(z_t, z_{t+1})                       # IDM,  l_t in R^{d_l}, d_l << d_v
K_t, delta_t   = f_psi(z_t, l_t)                           # FDM
z_hat_{t+1}    = K_t @ z_t + delta_t                       # transport + residual
```

`K_t in R^{L x L}` is a soft transport matrix (left-multiplied over spatial tokens), inspired by optical flow in token space; `delta_t in R^{L x d_v}` captures non-transportable changes.

The LAT is trained with forward next-latent prediction and backward consistency:

```
L_fwd  = sum_t || z_hat_{t+1} - z_{t+1} ||_2^2
L_cons = sum_t || z_hat_t       - z_t       ||_2^2     # by running LAT on reversed pair
```

**Stage 2 — Causal WAM pretraining**

The model is a causal diffusion transformer over visual-action chunks:

```
u_{t:t+k} = [ z_{t:t+k}, l_{t:t+k-1} ]
s         = [ c, z_1, u_{t1:t1+k}, ..., u_{tN:tN+k} ]
```

A block-causal mask lets each chunk attend only to the language prefix `c`, the initial visual context `z_1`, and previous chunks. Visual and action tokens share attention weights but use modality-specific feed-forward networks.

Training uses conditional flow matching:

```
x_alpha = (1 - alpha) * epsilon + alpha * u
dot_x   = u - epsilon
L_FM    = E[ || F_theta^v(x_alpha, alpha, s_<t) - dot_x^v ||_2^2
          + lambda_a * || F_theta^a(x_alpha, alpha, s_<t) - dot_x^a ||_2^2 ]
```

**Stage 3 — Adaptation to robot actions**

The WAM is first pretrained on video + latent-action tokens from unlabeled / weakly-labeled data (AgiBot, ~100 G tokens), then fine-tuned on robot-trajectory data (~300 G tokens from AgiBot, RoboMIND, RoboCOIN, InternA1) to map the latent actions to embodiment-specific motor commands.

Key hyperparameters from the paper:

| Component | Setting |
|---|---|
| Visual encoder/decoder | 12-layer Transformer, hidden dim 768 |
| Visual latent dim `d_v` | 96 |
| Latent action dim `d_l` | 4 |
| LAT IDM / FDM | 4-layer MLP, hidden size 256 |
| WAM world expert | 30 layers, 1536 (1.3B) or 3072 (5B) hidden dim |
| WAM action expert | 30 layers, 768 hidden dim (~350M extra params) |
| Reconstruction weights | λ₁=1, λ_perc=1, λ_gan=0.1 |
| Alignment weight | λ_align=1 |
| Action loss weight | λ_a=1 |
| Optimizer | Muon, peak lr 1e-2, bfloat16, grad clip 2.0 |

### Constructive probe — what we built and what it shows

Because the official code is unreleased, we implemented a **minimal, CPU-runnable probe** of the core mechanisms on synthetic moving-blob videos.

**Probe file:** `code/repwam_probe.py`

It exercises:

1. **Toy ViT autoencoder** with reconstruction + semantic-alignment-style loss.
2. **LAT with full L×L transport matrix** and residual, trained with forward + backward consistency.
3. **Toy causal WAM** with block-causal attention and conditional flow matching over visual-action chunks.

**Synthetic data:** 128 videos of a Gaussian blob translating horizontally, vertically, or diagonally on an 8×8 grid, with `d_v=8` and temporal downsampling by 4 (so `T'=5` latent frames per video).

**How to rerun:**

```bash
cd /home/betelgeuse/Documents/paper-scout/workspace
python3 -m venv code/.repwam_probe_venv
./code/.repwam_probe_venv/bin/pip install torch torchvision matplotlib
REPWAM_PROBE_OUT=runs/2026-06-14-weaver-eurekagent-repwam/assets \
  ./code/.repwam_probe_venv/bin/python3 \
  runs/2026-06-14-weaver-eurekagent-repwam/repwam-2606.13674/code/repwam_probe.py
```

(The virtual environment is intentionally placed under the ignored `code/` directory; only the probe script and its outputs are preserved in the run packet.)

Runtime is ~2 minutes on a CPU.

**Probe outputs (preserved in the run-level `assets/` directory):**

| File | Description |
|---|---|
| `runs/2026-06-14-weaver-eurekagent-repwam/assets/repwam_probe_losses.png` | Tokenizer and WAM training loss curves |
| `runs/2026-06-14-weaver-eurekagent-repwam/assets/repwam_probe_transport.png` | Learned transport matrix `K` and residual norm for 4 test samples |
| `runs/2026-06-14-weaver-eurekagent-repwam/assets/repwam_probe_summary.txt` | Parameter counts and final LAT forward MSE |

**Probe results:**

- Visual tokenizer + LAT train stably; LAT forward MSE on held-out samples is `< 1e-4`.
- The toy WAM flow-matching loss decreases from ~2.3 to ~0.15 over 15 epochs.
- The transport matrices are diffuse on this toy task (the model can satisfy the objective with a near-uniform blur plus a localized residual), which is expected for unconstrained synthetic motion but still validates the `K @ z + delta` math.

The probe does **not** validate the paper's real-robot or RoboTwin claims — it only confirms that the described equations form a coherent, trainable objective.

### Comparison with neighbors

We inspected the open-source codebases of the two closest baselines the paper engages with.

**LAPA (Latent Action Pretraining from Videos, Ye et al. 2025)**

- Repository: `code/lapa/`
- Latent actions are **discrete** codes from a VQ-VAE/NSVQ model (`laq/laq_model/latent_action_quantization.py`).
- The encoder maps two frames to first/last tokens; a neural-scale VQ layer produces a small discrete codebook index.
- The decoder reconstructs the future frame with cross-attention conditioned on the quantized action.
- LAPA uses these codes as **pretraining targets** for a VLA; at deployment the model outputs real robot actions directly.
- **Contrast with RepWAM:** RepWAM's latent actions are continuous (`d_l=4`), live in the same semantic space as the visual tokens, and are modeled jointly with future visuals by the WAM rather than fed to a separate VLA head. RepWAM also uses an explicit transport+residual FDM rather than a VQ reconstruction decoder.

**Lingbot-VA (Causal World Modeling for Robot Control, Robbyant/Ant Group, 2026)**

- Repository: `code/lingbot-va/`
- Uses the **WAN2.2 VAE** (`diffusers.AutoencoderKLWan`) as the visual tokenizer — a reconstruction-oriented video generation VAE, not a semantic-alignment tokenizer.
- The transformer is a causal diffusion model with block-causal masking implemented via `flex_attention` (`wan_va/modules/model.py`).
- Training jointly denoises **latent video patches** and **action tokens** with flow matching (`wan_va/train.py`, `compute_loss`).
- The action tokens are real robot actions, not separately learned latent actions.
- **Contrast with RepWAM:** Lingbot-VA inherits pretrained WAN2.2 weights and a reconstruction-oriented latent space; RepWAM trains from scratch with a semantic visual-action tokenizer and an intermediate latent-action pretraining stage. The paper's ablation (Table 3) shows that swapping WAN2.2 VAE for RepViTok improves RoboTwin success by ~8 points Easy / ~7 points Hard.

---

## Preserved evidence

| Path | Content |
|---|---|
| `code/repwam_probe.py` | Runnable toy implementation of RepWAM's tokenizer, LAT, and causal WAM flow matching |
| `runs/2026-06-14-weaver-eurekagent-repwam/assets/repwam_probe_losses.png` | Training curves |
| `runs/2026-06-14-weaver-eurekagent-repwam/assets/repwam_probe_transport.png` | Learned transport matrices and residuals |
| `runs/2026-06-14-weaver-eurekagent-repwam/assets/repwam_probe_summary.txt` | Parameter counts and final MSE |
| `code/repwam-2606.13674/` | Cloned placeholder repository (README + assets only) |
| `code/lapa/` | Cloned LAPA codebase for neighbor comparison |
| `code/lingbot-va/` | Cloned Lingbot-VA codebase for neighbor comparison |

---

## What this means for the main report

### Claims the report can make responsibly

1. **RepWAM's core design is coherent and distinctive.** The paper proposes a visual tokenizer supervised by both reconstruction and semantic alignment, plus a continuous latent-action tokenizer that decomposes state transitions into a transport matrix and residual. A minimal reimplementation of these equations trains stably on synthetic data.

2. **The method is positioned against two clearly different baselines.** LAPA learns discrete VQ latent actions as pretraining targets for a VLA; Lingbot-VA uses a pretrained WAN2.2 reconstruction VAE and models real actions directly. RepWAM instead learns continuous semantic latent actions jointly with future visuals in a from-scratch causal WAM.

3. **Empirical claims are currently unverifiable from external signals.** The official code and weights are scheduled for release in June 2026 but were not available at the time of this investigation. The reported RoboTwin 2.0 and real-robot success rates, as well as the WAN2.2 VAE ablation, rest on the paper's tables and the (forthcoming) release.

### Limitations and blockers to flag

- **No official code or checkpoints:** the GitHub repo is a placeholder. The ablation numbers in Tables 2–4 and the real-robot results in Figure 2 could not be independently reproduced or inspected.
- **Toy probe only:** the synthetic moving-blob experiment validates the math but says nothing about manipulation semantics, instruction following, or sim-to-real transfer.
- **No pretrained visual foundation model:** the probe uses a dummy alignment target instead of a real Perception Encoder-style teacher.
- **Related-code inspection is architectural, not experimental:** we did not train or evaluate LAPA/Lingbot-VA ourselves; the comparison is based on reading their source.

---

## Notes for future re-investigation

- Re-run this thread once `wdrink/RepWAM` releases code + weights.
- Priority checks when code is available:
  1. Verify the exact LAT architecture (IDM/FDM layer counts, how `K_t` is parameterized, whether there are sparsity/entropy constraints).
  2. Inspect the visual tokenizer alignment loss implementation and the frozen teacher model.
  3. Reproduce the WAN2.2 VAE vs. RepViTok ablation on a small held-out subset.
  4. Check the causal chunking and flow-matching code for the world/action experts.
  5. Load pretrained weights and run the tokenizer on sample frames to inspect latent-action visualizations like Figure 4.
