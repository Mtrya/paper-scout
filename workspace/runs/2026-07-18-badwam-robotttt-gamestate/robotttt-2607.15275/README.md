# RoboTTT (arXiv 2607.15275) — Deep-Dive Thread

**Paper**: *RoboTTT: Context Scaling for Robot Policies* — Jiang, Chebotar, Zheng, Hu, Ge, Wu, Dai, Reed, Fei-Fei, Zhu, Fan (NVIDIA / Stanford / UT Austin). Project page with videos: https://research.nvidia.com/labs/gear/robottt/ . **No official code release** as of 2026-07-18 (checked project page, arXiv, GitHub search; page links paper + videos only). The backbone, GR00T N1.7, exists as open source but the RoboTTT modifications (TTT layers in the DiT action head, sequence training recipe) are not public.

## 1. The mechanism, precisely

RoboTTT inserts one **TTT layer after the self-/cross-attention of each of the 16 DiT layers** of GR00T N1.7's action head (DiT 538M params; +~10M per TTT layer → 690M total). Attention operates **within** a timestep (register tokens `R_t` (N=16 learned), proprio token `q_t`, noised action tokens `Ã_t`, cross-attending to that step's VL tokens `Φ_t`); TTT layers operate **across** timesteps on the concatenated per-step outputs. VL tokens are *not* passed through TTT (compute); the 16 register tokens carry VL information across time.

**Inner loop** (paper Eq. 1–2): fast weights `W` parameterize a small *fast model* `f_W` (2-layer MLP, GeLU). Per timestep:

```
W_t ← W_{t-1} − η ∇_W || f_{W_{t-1}}(K_t) − V_t ||²        (update: 1 SGD step on self-supervised reconstruction)
O_t = f_{W_t}(Q_t)                                          (apply: retrieve)
```

- `K, V, Q` come from learned projections `θ_K, θ_V, θ_Q` of the token stream; `θ`s and the **learned fast-weight init `W_0`** are ordinary parameters (W₀ meta-learned through gradients-of-gradients).
- Inner optimizer: plain SGD; **learnable per-channel η on top of base LR 0.1** (App. A.1). Muon etc. left to future work.
- **tanh gating** preserves pretrained capability: `O = tanh(α) ⊙ O_TTT + O_attn`, `α ∈ R^d` init 0.001.
- Inference: rollout starts from W₀, fast weights update on each observation and propagate; per-step cost is **constant in context length** (fixed-size weight-space state), unlike KV-cached attention whose per-step cost grows linearly.

**Training recipe**: sequence flow matching (Eq. 5) with (a) **sequence action forcing** — noise level τ_t sampled *independently per action chunk*, τ = s(1−u), u~Beta(1.5,1), s=0.999; without it training is unstable; (b) **TBPTT** — gradients flow within segments, fast weights carried across but detached at boundaries, so memory is set by segment length, not sequence length; W₀ still gets gradients through the first segment. Pretrain 30K steps on 16× GB200 (only the new TTT/GDN layers tuned, backbone frozen), context gradually increased to 8K; post-train per task at 1K context, 20K steps, all params.

**Context-exploiting capabilities** via loss masking (masked steps = pure context, update fast weights, no action loss):
- **One-shot imitation from human video**: prepend a human demo video (loss masked) to a robot trajectory of the same configuration; at test time one in-context video of an *unseen* configuration steers the policy. Result: 6/10 successes; GDN baseline 0/10.
- **DAgger Distillation**: robot failures = context, human corrections = targets. Distills the failure→correction mapping into fast weights → on-the-fly recovery. +36% (RoboTTT) vs +9% (standard DAgger avg) from the same 100-trajectory pool.

## 2. Paper's key quantitative results

| Result | Numbers |
|---|---|
| Main eval (rubric task-completion, avg of 3 bimanual assembly tasks) | RoboTTT **79%** vs GR00T N1.7 42% (+87%), N1.7 Hist. 49%, GDN 56% |
| Full successes (Table 1) | Pup Go Car 9/20, Circuit 13/20, Gear Bot 2/10; best baseline ≤8/20; **only RoboTTT ever completes Gear Bot** (5-min, 10-stage) |
| Context scaling (Fig. 8) | pretrain ctx 128→8K: RoboTTT ~0.31→**0.715** (+63% vs 1K's 43.9%), GDN flat ~0.42–0.45; no saturation at 8K |
| One-shot human-video imitation (Circuit, Table 2) | **6/10** RoboTTT vs 0/10 GDN |
| Perturbation robustness (Table 3) | roof 15/20 (GDN 13/20, short-ctx ≤10/20); tire 18/20 (tie with GDN) |
| Ablations (Fig. 12) | no seq-action-forcing: collapse; TTT-linear: −27% vs MLP fast model; +action tokens +23%, +register tokens +18% (registers alone don't help GR00T) |
| Deployment | YAM bimanual, 4× RealSense D405, RTX 5090 workstation, **30 Hz control** |

## 3. Constructive probe (code/)

No official code exists, so we re-implemented the *mechanism* faithfully at toy scale and stress-tested the paper's central mechanism-level claims:

> (i) fast-weight recurrent state supports steady gains from longer context; (ii) per-step inference cost stays constant as context grows; (iii) nonlinear fast model (TTT-MLP) > linear recurrent state (delta rule / GDN-lite).

**Toy task** (`task.py`): 2D point-mass with latent per-episode task parameters — target `g~U(±0.8)²`, gain `k~logU(0.5,2.5)` — driven by a *nonlinear* expert `a_t = clip(k(g−x_t)(1+0.5‖g−x_t‖), ±1)`, dynamics `x += 0.15·a`, observations `o = x + N(0, 0.15²)` (heavy). Episodes are multi-phase ("repeated reach trials", phase=48 steps, teleport between phases, same latent `(g,k)`) — the toy analogue of multi-stage assembly with state aliasing. Policy sees `[o_t, a_{t−1}]`, predicts `a_t` at every step. Context helps two ways: identify `(g,k)` from earlier trials, and temporally denoise `x_t` using the dynamics. `dt=0.15` was chosen *deliberately*: with small dt the task collapses into trivial local extrapolation (`a_t ≈ 0.925·a_{t−1}` regardless of task, no-context R²≈0.95) — a shortcut we caught and removed during probe design (see §5).

**Models** (`models.py`, all d=64, 2 layers, ~50K params each):
- **TTT-MLP** — exact Eq. 1–2: per-step inner gradient on reconstruction MSE, 2-layer-GeLU fast model, learned W₀, learnable η (base 0.1), update-then-apply, TBPTT training (seg 64) exactly as the paper.
- **Gated delta rule** — linear-attention/GDN-lite analogue: matrix state, gated delta update. Note the delta rule *is* one online gradient step on a *linear* fast model, so this doubles as the paper's "TTT Linear" ablation with a gate.
- **Causal attention** — 2-layer transformer block stack (67K params), KV-cached at inference.
- **No-context MLP** — per-step baseline (17K params).

**Metric**: R² of action prediction on held-out episodes, over *unsaturated* steps only (`max|a|<0.95`; saturated steps are trivially predictable), bucketed by elapsed context `t` ∈ {8–16, …, 384–512}.

## 4. Results

### 4.1 Accuracy vs context length  →  `results/*.json`, figure `../../assets/robotttt-probe-scaling.png`

R² of action prediction on unsaturated steps, held-out episodes, bucketed by elapsed context (models trained at T=512 unless noted; 2500 iters; noctx 800):

| context bucket | TTT-MLP | delta rule | attention | no-context | TTT trained@64 |
|---|---|---|---|---|---|
| 8–16 (first trial) | 0.850 | 0.787 | 0.941 | 0.939 | 0.840 |
| 16–32 (first trial) | −0.085 | −1.560 | 0.241 | −0.278 | −0.286 |
| 32–64 | 0.895 | 0.896 | 0.924 | 0.842 | 0.896 |
| 64–128 | 0.855 | 0.864 | 0.922 | 0.819 | 0.873 |
| 128–256 | 0.873 | 0.888 | 0.925 | 0.831 | 0.857 |
| 256–384 | 0.877 | 0.859 | 0.908 | 0.798 | 0.674 |
| 384–512 | 0.865 | 0.876 | 0.913 | 0.814 | **0.476** |

Reading: (i) all context models beat no-context from the second trial onward — context is worth +0.05 to +0.10 R²; (ii) everyone plateaus by ~48 steps: after ~1.5 reach trials the latent (g, k) is identified to near-Bayes precision, so this toy task's information content saturates (unlike the paper's real 5-min assemblies); (iii) attention > TTT ≈ delta rule at matched scale — the near-linear system-ID task does not reward the nonlinear fast model; (iv) the TTT model trained at 64-step context collapses beyond its window — the paper's Fig-8 "training context bounds usable horizon" effect in miniature.

### 4.2 Horizon extrapolation (trained@512, evaluated to 1024)  →  `results/eval_long_T1024.json`, right panel of scaling figure

| bucket | TTT-MLP@512 | delta@512 | attention@512 | TTT@64 |
|---|---|---|---|---|
| 384–512 | 0.891 | 0.895 | 0.931 | 0.488 |
| 512–640 | 0.884 | 0.885 | 0.932 | −0.005 |
| 640–768 | 0.821 | 0.840 | 0.920 | −1.688 |
| 768–896 | 0.788 | 0.848 | 0.922 | −1.812 |
| 896–1024 | **0.726** | **0.851** | **0.922** | **−2.939** |

The gated delta rule is perfectly stable at 2× the training window; TTT erodes gradually (−0.17 R²); the 64-trained TTT diverges. Mechanism: TTT's fast weights take *ungated* SGD steps, so ||W_t − W₀||_F keeps growing (measured 0.27 → 0.50 relative drift over 512 steps) and only horizons seen in training are reliably calibrated; the delta rule's built-in gate/decay keeps its state bounded. Our attention baseline has *no positional embedding*, hence extrapolates perfectly — the paper notes RoPE extension into unseen positions as one of its own beyond-window degradation factors.

### 4.3 TTT inner-loop diagnostics  →  `results/ttt_diag.json`, figure `../../assets/robotttt-probe-ttt-inner.png`

Layer-0 self-supervised loss ||f_W(k_t) − v_t||² measured *before* the update at each step: **0.315 (t=8) → 0.094 (t=64) → 0.050 (t=256)** → 0.076 (t=512). The fast model genuinely becomes a better model of the stream as context accumulates — direct evidence that in-context learning happens in weight space. Spikes align with phase boundaries (teleports).

### 4.4 Per-step cost vs context length  →  `results/latency_d256.json`, figure `../../assets/robotttt-probe-latency.png`

Autoregressive decoding, batch=1, single CPU thread, d=256, 2 layers, element-wise-min over 3 reps:

| window (steps) | TTT-MLP | delta rule | attention (KV cache) |
|---|---|---|---|
| 0–64 | 0.314 ms | 0.150 ms | 0.123 ms |
| 256–512 | 0.307 | 0.144 | 0.164 |
| 512–1024 | 0.307 | 0.145 | 0.213 |
| 1024–1536 | 0.305 | 0.144 | 0.320 |
| 1536–2048 | 0.306 | 0.144 | 0.482 |

TTT and delta rule are exactly flat; KV-cached attention grows ~linearly (3.9× from first to last window) and overtakes TTT's per-step cost at t≈1000 despite TTT's 2.1× higher constant (the inner gradient step is real work). Cumulative rollout cost: O(T) vs O(T²).

## 5. Probe-design pitfalls we hit (worth knowing)

1. **Metric collapse by action saturation**: with a converging controller, actions decay to ~0; per-position NMSE then explodes or trivializes. Fixed by multi-phase episodes + unsaturated-step masking.
2. **Local-extrapolation shortcut**: with `dt=0.05`, `a_t ≈ (1−0.05k̄)·a_{t−1}` regardless of the latent task; a *no-context* MLP reached R²≈0.95, drowning the in-context-identification signal. Fixed by `dt=0.15`, which makes the per-step contraction factor strongly task-dependent.
3. Delta-rule instability without key L2-normalization (‖k‖²β ≫ 2 diverges); fixed with F.normalize on q,k (as GDN does).

## 6. Assessment of the paper's claims

**Supported by our probe (mechanism level):**
- *Constant per-step inference cost in context length* — verified exactly (§4.4); TTT's constant factor is ~2× a plain linear-recurrent state, and attention overtakes TTT per-step beyond ~1K steps even at toy scale.
- *Fast weights do in-context learning* — the inner self-supervised loss falls 4× over the rollout (§4.3); accuracy follows context (§4.1).
- *Training context bounds usable horizon* — the T64→eval-512/1024 ablation (§4.1–4.2) reproduces, in controlled form, the paper's own observation that sub-1K-trained models degrade when fast weights update far beyond the training window. Their 30K-step pretraining "gradually increasing context to 8K" is what makes 8K deployment coherent.

**Qualified or not reproduced at toy scale:**
- *Steady gains with no saturation* — in our task every model plateaus after ~1.5 reach trials: once the latent task is identified to Bayes precision, more context adds nothing. The paper's unsaturated 128→8K curve therefore says something real about the *data*: dense, repetitive, partially-observed visuomotor streams keep paying off far longer than a clean synthetic control family. The mechanism enables the gains; it doesn't create them.
- *Nonlinear fast model ≫ linear recurrent state* — on our near-linear system-ID task the gated delta rule ties TTT-MLP within the window (0.876 vs 0.865 @384–512) and beats it on extrapolation (flat vs −0.17 R² at 2× horizon). The paper's 27% MLP-over-linear gap and its GDN gap must come from nonlinearity/structure our toy lacks; our probe does however surface a *stability* argument the paper underplays: ungated SGD on fast weights drifts outside the training window (§4.2), while the gated delta state stays bounded. RoboTTT's long-horizon reliability is bought with long-context *training*, not with an intrinsically stable recurrence.

**On the paper's own evidence:** real-robot only (no sim), single embodiment (YAM bimanual), three tasks from one lab's data mixture; the headline one-shot-human-video result is 6/10 on one task (Circuit) with the prompt held fixed; scaling-curve evaluations "predate the DAgger training" so the curve is clean of that confound; GDN baseline is param/placement-matched from FLA, which is a fair comparison but only one recurrent family; evaluation is 10–20 trials per cell with rubric scores, so ±1 trial swings are visible in their tables. The 87%/63%/57% headline numbers are real but rest on small-N real-robot evals. No code release as of 2026-07-18, so the DiT-integration details (exact η parametrization, whether TTT runs per-register-token, chunking) cannot be verified against an implementation.

## 7. How to rerun

```bash
cd code/    # this directory
python -m venv .venv && . .venv/bin/activate   # or: uv venv
pip install -r requirements.txt                 # CPU torch is fine
# trainings (each saves results/<name>.json + <name>.pt):
python train.py --mixer ttt   --T 512 --iters 2500 --batch 32 --out results/ttt_T512.json
python train.py --mixer delta --T 512 --iters 2500 --batch 32 --out results/delta_T512.json
python train.py --mixer attn  --T 512 --iters 2500 --batch 32 --out results/attn_T512.json
python train.py --mixer noctx --T 64  --iters 800 --d 128      --out results/noctx.json
python train.py --mixer ttt   --T 64  --iters 1200             --out results/ttt_T64.json   # training-context ablation
python latency.py --d 256 --T 2048 --reps 3 --out results/latency_d256.json
python eval_extra.py --ckpt results/ttt_T512.pt --out results/ttt_diag.json   # TTT inner-loss / drift diagnostics
python eval_long.py    # 2x-horizon extrapolation (uses saved .pt checkpoints)
python plot_results.py <out_dir>   # writes robotttt-probe-scaling.png / -latency.png / -ttt-inner.png
```

Hardware used: CPU only (16-core x86, torch 2.13.0+cpu). Full suite ≈ 2–3 h wall-clock if run sequentially; parallelize at will.
