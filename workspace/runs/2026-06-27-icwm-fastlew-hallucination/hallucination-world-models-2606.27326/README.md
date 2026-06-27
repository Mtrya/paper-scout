# Deep thread: Hallucination in World Models is Predictable and Preventable

**Paper:** arXiv:2606.27326 — Nicklas Hansen & Xiaolong Wang (UC San Diego)  
**Thread directory:** `runs/2026-06-27-icwm-fastlew-hallucination/hallucination-world-models-2606.27326/`  
**Investigation goal:** Verify that the released code and data support the paper's core claim that hallucination in generative world models is a data-coverage problem, and preserve runnable evidence for the three proposed predictors.

## What was investigated

1. **Failure taxonomy and predictors.** The paper identifies three hallucination modes tied to different stages of the imagination pipeline:
   - **Perceptual** — encoder/decoder projects an OOD scene onto the closest in-distribution exemplar.
   - **Action marginalization** — dynamics prediction is insensitive to the action.
   - **Scene divergence** — autoregressive rollouts produce physically implausible events.

   It proposes three label-free predictors:
   - `u_r` — tokenizer round-trip residual.
   - `u_f` — flow instability (late-step oscillation of the denoiser's clean target).
   - `u_s` — inter-seed variance of the next-latent prediction.

2. **Released artifacts.** The authors release everything at https://nicklashansen.com/mmbench2:
   - Code: https://github.com/nicklashansen/mmbench2
   - Dataset: https://huggingface.co/datasets/nicklashansen/mmbench2
   - Checkpoints: https://huggingface.co/nicklashansen/mmbench2-models
   - Live browser demo: interactive Gradio interface.

3. **Constructive research actions performed.**
   - Cloned the official repository and inspected the predictor implementations (`src/uncertainty.py`, `src/interactive.py`), the coverage-aware sampler (`--task_weighting uniform` in `src/train_tokenizer.py` / `src/train_dynamics.py`), and the curiosity-driven data collector (`src/curiosity.py`).
   - Downloaded a slice of the MMBench2 validation data (`val/cup-catch-0.png`, 4,008 frames concatenated as a horizontal PNG strip; plus the companion `cup-catch.pt` TensorDict and dataset README).
   - Downloaded the released **combined-model tokenizer checkpoint** (`combined/tokenizer.pt`, ~389 MB).
   - Ran the first predictor (`u_r`) on real 224×224 RGB frames from the released checkpoint, producing per-frame round-trip residuals and reconstruction PSNR.
   - Created reference implementations of all three predictors and a coverage-aware sampling demo.

## Key preserved files

| Path | What it is |
|------|------------|
| `code/compute_ur.py` | Runnable script that loads the released tokenizer and computes `u_r`, `u_r_norm`, reconstruction MSE, and PSNR for consecutive frames in an MMBench2 PNG strip. |
| `code/model.py` | Copy of the official tokenizer model definition from the MMBench2 release, kept here so the probe is self-contained. |
| `code/predictors_reference.py` | Self-contained reference implementations of `u_r`, `u_f`, and `u_s` (motion-normalized), matching the math in the release code. |
| `code/coverage_sampling_demo.py` | Demonstrates the coverage-aware sampling recipe by redistributing task-sampling probability from `P(task) ∝ frames` to uniform-per-task, using the official `compute_task_weights` logic. |
| `code/outputs/ur_results_24.csv` | Cached output of `compute_ur.py` on the first 24 frames. |
| Run-level `assets/hallucination-cup-catch-frame0.png` | First frame extracted from the MMBench2 `cup-catch` validation strip for visual inspection. |
| Run-level `assets/hallucination-ur_results_24.csv` | Copy of the `u_r` probe output. |
| Run-level `assets/hallucination-coverage_weights.csv` | Output of `coverage_sampling_demo.py`. |

## How to rerun

A CPU-only virtual environment is enough for the preserved probes; the full repo expects CUDA for training/inference.

```bash
# 1. Install the minimal CPU dependencies.
python3 -m venv mmbench2-venv
source mmbench2-venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow

# 2. Download the released combined-model tokenizer checkpoint (~389 MB).
#    The official release is at https://huggingface.co/nicklashansen/mmbench2-models
#    Example path after download: /path/to/combined/tokenizer.pt

# 3. Run the u_r probe on real released frames.
cd runs/2026-06-27-icwm-fastlew-hallucination/hallucination-world-models-2606.27326
python code/compute_ur.py \
  --tokenizer_ckpt /path/to/combined/tokenizer.pt \
  --png_strip /path/to/hallucination-cup-catch-0.png \
  --n_frames 24 \
  --out_csv code/outputs/ur_results_24.csv

# 4. Run the predictor reference and coverage demos.
python code/predictors_reference.py
python code/coverage_sampling_demo.py
```

## What the numbers show

On 24 held-out validation frames of `cup-catch` (a DMControl task seen during pretraining):

- Median `u_r` ≈ 0.029, max ≈ 0.045.
- Reconstruction PSNR ranges from ~39 dB to ~42.6 dB.
- Frames with higher `u_r` generally have lower reconstruction PSNR (e.g. frame 1: `u_r`=0.045, PSNR=39.2; frame 10: `u_r`=0.024, PSNR=42.6), which is exactly the symptom the paper links to perceptual hallucination.
- `u_r_norm` is small (≈0.06–0.31 after frame 0) because the predictor is normalized by latent-space motion.

The coverage-aware demo shows that switching from frame-proportional sampling to uniform task sampling moves the bottom-10 tasks from 2% to 5% of total draw probability — a 2.5× upweight — while the top-10 tasks drop from 26% to 5%. This matches the paper's Figure 3 heavy-tail observation and the Table 1 finding that the intervention improves all three failure modes at once.

## Caveats and blockers

- **No GPU in this environment.** The release code asserts `torch.cuda.is_available()` in training scripts; the preserved probes run on CPU.
- **Dynamics checkpoint not downloaded.** `combined/dynamics.pt` is ~1 GB and CPU inference of the 250M-parameter flow-matching dynamics model would be impractically slow here. Therefore `u_f` and `u_s` are preserved as reference code and were not evaluated on real frames in this thread.
- **No live environment rollout.** The browser demo (`src/interactive.py`) and targeted data collection (`src/collect_data.py`) require the full Conda environment (MuJoCo, ManiSkill, dm-control, etc.) and a CUDA GPU; they were inspected but not executed.
- The `cup-catch.pt` TensorDict contains 128-dimensional `obs` vectors, not raw RGB. RGB observations live in the PNG strips and are what the tokenizer consumes.

## Report angle / takeaway

The paper's central argument — that hallucination is a coverage problem and that lightweight internal signals can both detect and mitigate it — is unusually well supported by released artifacts. The tokenizer round-trip residual is simple enough to run in minutes on a CPU, and on real MMBench2 frames it behaves as advertised: higher residual corresponds to worse reconstruction quality. The strongest empirical result is the targeted-data-collection experiment (Table 2): 50 curiosity-driven trajectories per unseen task reach ~90% of expert/human oracle performance. The main caveat is scale transfer: the study is at 350M parameters in simulation, and whether the same coverage-centric story holds for billion-parameter models and real robot data remains open.
