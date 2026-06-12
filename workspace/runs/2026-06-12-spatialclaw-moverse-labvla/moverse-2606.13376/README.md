# MoVerse (arXiv 2606.13376) — Deep Thread

**Paper:** *MoVerse: Real-Time Video World Modeling with Panoramic Gaussian Scaffold*  
**Authors:** Yang Zhou, Ziheng Wang, Yuqin Lu, Haofeng Liu, Jun Liang, Shengfeng He, Jing Li  
**Thread path:** `runs/2026-06-12-spatialclaw-moverse-labvla/moverse-2606.13376/`

## What I attempted

1. Read the full paper (`papers/world-models/moverse-2606.13376.md`) and traced the three-stage pipeline:
   - **Stage I:** NFOV image → gravity-aligned 360° ERP panorama with topology-aware diffusion.
   - **Stage II:** panorama → persistent 3D Gaussian scaffold via ERP-aware initialization + residual prediction.
   - **Stage III:** scaffold renderings → photorealistic roaming video via a bidirectional teacher distilled to a causal autoregressive student.
2. Checked for released code, models and project page.
3. Compared MoVerse with the closest neighbors: single-image world models, panoramic generation, and feed-forward Gaussian novel-view synthesis.
4. Built two small probes that make the paper’s key geometry choices concrete:
   - Stage I circular/shift-equivariant latent encoding.
   - Stage II spherical back-projection, latitude-aware scale, and angular–inverse-depth residual composition.

## What I found

### Official artifacts

- **GitHub repo:** `https://github.com/Orange-3DV-Team/MoVerse` (cloned to `code/MoVerse/`). At the time of investigation it contains only a `README.md`, an icon, and a pipeline figure. The authors state: *“The release of this code and pretrained model are subject to a corporate compliance and security review.”* So there is **no runnable implementation or checkpoint yet**.
- **Project page:** `https://orange-3dv-team.github.io/MoVerse/` — has demo videos, a 360° panorama viewer, and a 3D scaffold viewer. It confirms the 8 FPS RTX 4090 deployment claim and the three-stage interface.
- **Hugging Face papers entry** (`hf papers info 2606.13376`) lists the same project page and GitHub repo, with 3 stars and **“Model: Coming Soon.”**

Because the official code is a placeholder, this thread is built on the paper text plus a humble reconstruction of the Stage II geometry engine.

### Neighbors / external signals

| Method | Relation to MoVerse | What the signal shows |
|--------|--------------------|-----------------------|
| **WonderWorld** (CVPR 2025, arXiv 2406.09394) | Single-image interactive 3D scene generation. Uses Fast Gaussian Surfels (FLAGS) and guided depth diffusion, generating connected scenes in <10 s on an A6000. Code is public (`KovenYu/WonderWorld`). | Confirms the trend of “single image → explorable 3D scene,” but WonderWorld is iterative/scene-by-scene, not a panoramic-then-video streaming design. |
| **CAT4D** (CVPR 2025, arXiv 2411.18613) | Single-video → 4D scene via multi-view video diffusion + deformable 3DGS optimization. | Shows that a strong video diffusion model can be a teacher for a 3D representation, but CAT4D optimizes a per-scene 4DGS rather than distilling a real-time causal renderer. |
| **SHARP** (arXiv 2512.10685, Apple) | Feed-forward monocular 3DGS predictor with depth-guided residual Gaussian prediction. Code and weights public (`apple/ml-sharp`, `huggingface.co/apple/Sharp`). | MoVerse explicitly follows SHARP’s residual prediction principle; the novelty is the ERP/spherical adaptation (latitude-aware scale, angular–inverse-depth residuals, horizontal closure). |
| **Wan2.1-T2V-1.3B** | Open text-to-video diffusion transformer. Public (`Wan-Video/Wan2.1`, `Wan-AI/Wan2.1-T2V-1.3B`). | MoVerse’s Stage III teacher is initialized from this checkpoint and distilled into a causal student with shared RoPE and a MemRoPE-style KV cache. |
| **MVSplat / pixelSplat / PanSplat / PanoSplatt3R** | Feed-forward Gaussian splatting from sparse or panoramic views. | The Stage II literature is active; MoVerse differs by targeting **single NFOV → 360° panorama → 3DGS** as an offline scaffold for a streaming video renderer. |

### Reconstruction / probes

All preserved code is in `code/` and produced artifacts are in `assets/`.

#### Probe 1 — Stage I topology (`code/probe_stagei_shift_equivariance.py`)

Demonstrates that a convolution with **circular horizontal padding** satisfies the shift-equivariance loss in the paper (Eq. 11), while zero padding introduces a boundary seam. Outputs:

- `assets/stagei_probe/shift_equivariance_probe.png`
- `assets/stagei_probe/erp_periodicity.png`

Numerical result on the toy latent: circular padding error ≈ `0.0`, zero padding error ≈ `0.035`.

#### Probe 2 — Stage II ERP-to-Gaussian scaffold (`code/probe_stageii_erp_to_gaussians.py`)

Implements the spherical back-projection, latitude-aware scale initialization, and angular–inverse-depth residual composition described in Sec. 2.3 of the paper:

- Builds a synthetic ERP panorama + depth map of a box room.
- Samples an ERP grid and computes `μ_k = D_k · d(θ_k, φ_k)` (Eq. 17).
- Initializes scale as `s_k ∝ D_k · cos φ_k` with a near-pole lower bound (Eq. 18).
- Applies small residuals in `(Δθ, Δφ, Δz)` and recomposes the center via inverse-depth softplus (Eq. 21–22).
- Renders a novel pinhole view with a tiny front-to-back point splatter.

Outputs:

- `assets/stageii_probe/01_input_erp_depth.png`
- `assets/stageii_probe/02_gaussian_centers_3d.png`
- `assets/stageii_probe/03_novel_view_residual.png`

Observed quantities from the probe:

- Mean Gaussian scale at the equator (`|φ|<0.2`): **0.196**
- Mean Gaussian scale near the poles (`|φ|>1.2`): **0.018**
- Mean center displacement after residuals: **1.27** units

The latitude scaling is non-trivial: without `cos φ`, Gaussians at the ERP poles would be far too large for their actual spherical footprint.

## What it means for the report

MoVerse’s central design move is **factorization**: put durable, panoramic 3D geometry in an explicit scaffold (offline), and use a lightweight causal video model only for local observation refinement (online). This is why the system can claim real-time roaming: the heavy lifting (panorama generation, 3D lifting) happens once, and the online renderer streams blocks of four frames with a bounded KV cache.

The strongest research question the paper raises is whether this factorization is actually robust: Stage I hallucinates the whole 360° world from a single NFOV image, and Stage II/III do not have a feedback path to correct a bad panorama. My investigation learned that:

1. The technical glue is real: ERP spherical back-projection + latitude-aware scale + angular–inverse-depth residuals is a coherent way to adapt feed-forward Gaussian prediction to panoramas.
2. The real-time claim is plausible on paper because the scaffold carries long-range memory, but it cannot be verified independently until the optimized causal student and the TAEHV decoder are released.
3. The closest open system, **WonderWorld**, solves a similar user-facing problem with a very different representation (layered surfels, iterative generation), while **SHARP** provides the strongest open precedent for the Stage II residual 3DGS mechanism.

## Limitations / blockers

- **No runnable code or weights.** The GitHub repo is a placeholder; the actual model is undergoing compliance review. This blocks end-to-end verification, profiling, and ablation.
- **No quantitative numbers in the paper** (PSNR/LPIPS tables, user-study metrics, dataset sizes) for direct comparison with neighbors.
- **Probe is a toy.** The reconstruction does not use the actual diffusion model, depth estimator, or differentiable Gaussian rasterizer; it only validates the geometric update rules.
- **Stage III is the hardest to probe without the Wan2.1-T2V checkpoint and the distilled student.** The MemRoPE cache and TAEHV decoder details are described but not testable here.

## Assets/figures that best illustrate the thread in the report

- `assets/moverse_repo_pipeline_overview.png` — the three-stage pipeline from the official repo.
- `assets/stagei_probe/shift_equivariance_probe.png` — why circular padding matters for ERP topology.
- `assets/stageii_probe/01_input_erp_depth.png` — the Stage I→II interface (ERP + depth).
- `assets/stageii_probe/02_gaussian_centers_3d.png` — the lifted 3D Gaussian scaffold.
- `assets/stageii_probe/03_novel_view_residual.png` — effect of angular–inverse-depth residuals on novel-view rendering.
- Paper figures in `runs/2026-06-12-spatialclaw-moverse-labvla/assets/moverse-2606.13376/`, especially:
  - `5e0c10fcc30b50b66690c90d811c6eb68488cb7b9336ebc8c1b111c015f2bf94.jpg` (Fig. 1 pipeline)
  - `a7b185a95251764f7e4ed4cf9371c200b5b260b06fb9b5f53b5f74887f7d8310.jpg` (Fig. 3 canonicalization)
  - `08a1fb49b216464b563bd3b7ac0015947086cf0f15679f03847e70c780bf1d85.jpg` (Fig. 4 Stage II)
  - `097250f9e3fea757f1d6c4cad7f4d91d8a768c69b9e1248070d76e14158e6996.jpg` (Fig. 5 Stage III)
  - `c00e32ec6799c0d51b6e4c90ef8627435bbad0b9fc622e384b1608433fd001bf.jpg` (Fig. 6 full-pipeline results)

## How to rerun the probes

```bash
cd runs/2026-06-12-spatialclaw-moverse-labvla/moverse-2606.13376/code
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib numpy
python probe_stagei_shift_equivariance.py
python probe_stageii_erp_to_gaussians.py
```

Both scripts write their outputs to `../assets/stagei_probe/` and `../assets/stageii_probe/`.
