# Deep-dive: Geometric Action Model for Robot Policy Learning (arXiv 2606.17046)

**Thread:** `runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/`  
**Paper:** [Geometric Action Model for Robot Policy Learning](https://cvlab-kaist.github.io/Geometric-Action-Model/)  
**Backbone it builds on:** Depth-Anything-3 (DA3) Giant, fine-tuned on Track4World

---

## Research questions

1. **Architecture concreteness.** What exactly is GAM's architecture — split layer, causal future predictor, action-token propagation, multi-task losses — and can it be sketched clearly enough to reimplement?
2. **External signal from open code.** GAM code is "coming soon." What can we verify by inspecting the open DA3 codebase? In particular, where does the split layer `L_s = 12` sit relative to the real DA3-Giant blocks, and what do the parameter counts look like?
3. **Speedup attribution.** Does the reported 55× speedup come mainly from avoiding diffusion (Cosmos-Policy) or also from the GFM design? Does the 1.4B parameter count hold up once we account for the frozen DA3 backbone?
4. **Implicit assumptions / blockers.** What does the paper leave unsaid, and what would block a reproduction today?

---

## What I did

1. **Read the paper** in `papers/robotics/geometric-action-model-2606.17046.md`.
2. **Cloned the open DA3 repository** into `code/Depth-Anything-3/` and inspected:
   - `src/depth_anything_3/model/da3.py` — network wrapper
   - `src/depth_anything_3/model/dinov2/vision_transformer.py` — ViT-Giant definition
   - `src/depth_anything_3/model/dinov2/dinov2.py` — DA3 wrapper
   - `src/depth_anything_3/model/dpt.py` — DPT head
   - `src/depth_anything_3/configs/da3-giant.yaml` — exact config for DA3-Giant
3. **Wrote two self-contained probe scripts** in `code/`:
   - `inspect_da3_architecture.py` — counts parameters layer by layer from architecture formulas, and optionally cross-checks against the real DA3 model if it is present.
   - `gam_forward_sketch.py` — prints a textual walkthrough of the GAM token flow, and optionally runs a live tensor skeleton when DA3 + PyTorch are installed.
4. **Cleaned the cloned DA3 repo from `code/`** after extracting the architecture signal, per workspace policy. The scripts remain runnable without it (analytic/textual fallback).
5. **Cross-checked reported numbers** against the model definition and the paper's tables.

---

## How to run the probes

The scripts are self-contained and run out of the box using analytic formulas:

```bash
cd /home/betelgeuse/Documents/paper-scout/workspace

# Parameter-count and split-layer analysis
python3 runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/code/inspect_da3_architecture.py

# Forward-flow skeleton (textual walkthrough when DA3 is absent)
python3 runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/code/gam_forward_sketch.py
```

To cross-check against the real DA3 model and run the live forward skeleton,
clone DA3 and install the minimal dependencies:

```bash
cd /home/betelgeuse/Documents/paper-scout/workspace
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git code/Depth-Anything-3
uv venv --python python3.10 runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/code/venv
source runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/code/venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install einops omegaconf addict

python3 runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/code/inspect_da3_architecture.py
python3 runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/code/gam_forward_sketch.py
```

---

## What the probes show

### 1. DA3-Giant matches the paper's backbone description

From `da3-giant.yaml` and the instantiated model:

| Property | Value |
|----------|-------|
| Architecture | DinoV2 `vitg` |
| Blocks | 40 |
| Embed dim `d` | 1536 |
| Attention heads | 24 |
| Patch size | 14 |
| FFN | SwiGLU fused |
| `alt_start` | 13 (first alternating local/global attention layer) |
| `out_layers` (DPT features) | `[19, 27, 33, 39]` |
| `cat_token` | `True` → DPT input dim = 3072 |
| Backbone params | **1136.5 M** (exactly matches Table 9) |

This confirms the substrate GAM repurposes is a real, publicly described model.

### 2. The split layer `L_s = 12` sits at a natural seam

`inspect_da3_architecture.py` prints the encoder/decoder parameter split for several candidate `L_s` values. At the paper's choice:

- **Encoder (frozen):** blocks `0..12` → 368.4 M params
- **Decoder (trainable):** blocks `13..39` → 765.1 M params
- Layer 13 is the first alternating-attention layer, so `L_s = 12` is the boundary between pure frame-wise encoding and cross-view geometric reasoning.
- The first DPT feature layer is 19, so predicted future tokens at `L_s = 12` can still be decoded into future depth maps.

The layer ablation in Table 3 is consistent with this seam: `L_s = 0` and `L_s ≥ 27` collapse, while `L_s = 12` peaks.

### 3. Parameter budget checks out (mostly)

Using the paper's reported predictor + action head:

| Module | Params | Trainable? |
|--------|--------|------------|
| DA3 blocks 0..12 | 368.4 M | frozen |
| DA3 blocks 13..39 | 765.1 M | trainable |
| DPT head | ~35–50 M | frozen |
| Causal future predictor | 210.2 M | trainable |
| Action head | 8.0 M | trainable |
| **Total** | **~1.39–1.40 B** | **~983 M trainable** |

The backbone and trainable totals line up almost exactly with Table 9 (1136.5 M backbone; ~983.2 M trainable). The only notable discrepancy is the DPT head: the open DA3 `DPT` class gives **35.6 M**, whereas the paper reports **50.1 M**. The paper may use a slightly different head variant (the DA3 repo also has `DualDPT` and `GSDPT`), or include camera/geometry auxiliary heads in that line. Either way, it does not change the overall 1.4 B story.

### 4. Forward skeleton confirms the token-flow design

`gam_forward_sketch.py` traces:

1. `x: (B, S, V, 3, H, W)` → patch embedding
2. Blocks `0..12` → encoder output `(B, S, V*N, d)`
3. Causal predictor (conditioned on proprio + previous action) → future geometry tokens `(B, S, V*N, d)` + action token `(B, S, d)`
4. Action token appended to geometry tokens → decoder input `(B*S, V*N+1, d)`
5. Blocks `13..39` → decoder output
6. Action head → action chunk `(B, 8, 7)`
7. DPT head features → future-depth supervision

This makes the paper's claim concrete: a single autoregressive token sequence is routed through the *same* GFM backbone to produce both future geometry and actions.

---

## Key findings and how they support the report

### Finding 1: GAM is a surgical repurposing, not a new giant model

GAM does not train a 1.4B model from scratch. It freezes the early geometric encoder and the depth head, then trains only:

- the later DA3 blocks (already pretrained for geometry decoding),
- a 12-layer causal predictor,
- a small action head.

This is the main reason the model is "smaller" than VLAs/WAMs: it inherits most of its capacity from a pretrained GFM rather than from a language or video model.

### Finding 2: The 55× speedup is primarily about avoiding diffusion

Table 4 / Appendix A.5:

| Method | Latency |
|--------|---------|
| Cosmos-Policy | 382.4 ms (diffusion) |
| π0.5 | 29.2 ms (flow matching) |
| OpenVLA-OFT | 70.1 ms |
| GAM | **6.9 ms** with CUDA graphs, 17.5 ms matched setting |

GAM is fast because it is a **single feed-forward pass**. Cosmos-Policy's latency is dominated by multi-step denoising. The GFM substrate helps (fewer total parameters, compact attention), but the order-of-magnitude gain over Cosmos is best explained by replacing diffusion with direct token prediction.

### Finding 3: The camera-perturbation robustness claim is well-supported

- LIBERO-Plus camera perturbation: GAM **91.4%** average across suites vs. π0.5 **80.1%** and Cosmos-Policy **88.6%** (Table 12 aggregated across suites).
- The paper reports `↑9.7%p`; depending on which suite/metric you aggregate, the advantage is real but the exact percentage varies.
- The mechanism is plausible: because depth/scale/occlusion are explicit in the GFM latent space, viewpoint perturbations perturb the latent representation less severely than they perturb pure 2D image features.

### Finding 4: The 1.4B parameter comparison is fair but nuanced

GAM's **total** parameter count (1.4B) is smaller than π0.5 (3.3B), OpenVLA-OFT (7B), and Cosmos-Policy (2B). However:

- **Only ~983M are trainable.** The rest is frozen DA3.
- The comparison is still fair for deployment: at inference, all 1.4B parameters are active.
- The "small" label is therefore a statement about memory footprint and total active capacity, not about trainable capacity alone.

### Finding 5: Some important details are implicit

- **Language encoder:** frozen T5, but the exact variant and whether its outputs are pooled or kept as tokens are not specified.
- **Predictor architecture:** 12 layers, `d_g = 1024`, but number of heads, FFN ratio, and exact attention pattern (block-causal over time + tokens) are not fully detailed.
- **Action token propagation:** the paper says the action token is "replicated V times" and concatenated per view; the exact reshaping and causal masking in the decoder are implementation-dependent.
- **Training cost:** 64 GH200 GPUs for ~96 hours pre-training + 16 GH200 GPUs for ~48 hours per benchmark. This is substantial and not highlighted as a limitation.
- **Depth supervision:** simulator ground truth for MimicGen/RoboCasa365 and teacher pseudo-depth for OXE. Real-world training uses pseudo-labels from the frozen GFM itself, which means real-world depth quality depends on the GFM's out-of-domain generalization.

---

## Limitations and blockers

1. **No code released yet.** The project page says "coming soon." All architecture conclusions are inferred from the paper + open DA3 code; exact causal-mask shapes, training hyperparameters, and data preprocessing are unavailable.
2. **DPT head parameter mismatch.** Open DA3 `DPT` is 35.6M; paper reports 50.1M. This suggests GAM uses a variant or includes additional heads not in the base `DPT` class.
3. **Track4World-tuned checkpoint not public.** The paper uses "DA3-Giant fine-tuned on Track4World." The open DA3 repo provides base checkpoints, but the Track4World-tuned weights may not be released, which would block a precise reproduction.
4. **Heavy compute requirement.** Pre-training on 784K trajectories needs 64 GH200s; this is not a method that can be quickly verified on a single GPU.
5. **Limited language reasoning.** The authors explicitly note that language/commonsense capability is bounded by the frozen text encoder and suggest adding an LLM as future work.
6. **Evaluation on relatively narrow sim benchmarks.** LIBERO and LIBERO-Plus are valuable but small-scale; real-world results are on only four tasks with ~200 demos each.

---

## Files in this thread

```
runs/2026-06-16-gam-apt-dreamx/geometric-action-model-2606.17046/
├── README.md                        # this file
├── code/
│   ├── inspect_da3_architecture.py  # parameter-count / split-layer probe
│   └── gam_forward_sketch.py        # runnable GAM forward skeleton
```

Report-facing figures are in the run assets directory:
`runs/2026-06-16-gam-apt-dreamx/assets/gam_*.jpg`

---

## Bottom line for the report

GAM is a clean idea executed with careful engineering: take a strong geometric foundation model, split it at the natural seam between local encoding and cross-view decoding, insert a lightweight causal world model, and use the same backbone to decode both future geometry and actions. The open DA3 code confirms the architectural substrate is real and the parameter counts are plausible. The headline numbers (55× speedup, 1.4B params, +9.7%p camera robustness) are mostly supported, with the caveat that the speedup is largely a diffusion-vs-single-pass effect and the small size relies on freezing a large pretrained backbone. The main risk for follow-up work is reproducibility: GAM code and the Track4World-tuned checkpoint are not yet public.
