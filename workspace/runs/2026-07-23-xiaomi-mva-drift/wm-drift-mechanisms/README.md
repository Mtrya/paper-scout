# WM Drift Mechanisms — cross-paper thread

**Date:** 2026-07-23 · **Thread of run:** `runs/2026-07-23-xiaomi-mva-drift`
**Question:** how do three papers in this week's pool attack long-horizon drift in interactive/autoregressive video world models — what exactly does each diagnose as the cause, and what does it actually fix?

**Sources (durable local caches):**
- `papers/world-models/abot-world0-2607.19191.md` — ABot-World-0 (pre-existing pdftotext dump)
- `papers/world-models/alayaworld-2607.18367.md` — AlayaWorld full technical report (fetched from arXiv HTML this run)
- `papers/world-models/self-gradient-forcing-2607.20368.md` — Self Gradient Forcing (fetched from arXiv HTML this run)
- Evidence artifacts: `code/wm_drift_evidence.json`, `code/artifact_verification.log`; figure `../assets/wm_drift_taxonomy.png`

---

## 1. The taxonomy: four diagnoses, four fixes

The three papers do **not** offer competing answers to the same question. Each diagnoses a *different* failure link in the autoregressive chain, and all three sit on top of a shared baseline (Self Forcing) that answers a fourth, earlier diagnosis. The working hypotheses brought into this thread were all **confirmed**, with refinements noted below.

### Layer 0 — Exposure bias → self-rollout training (the shared baseline)
- **Diagnosis:** teacher forcing trains on ground-truth context; inference conditions on self-generated context. Train/inference distribution mismatch.
- **Fix:** Self Forcing (Huang et al. 2025) — train the student on its own rollouts, scored by DMD against a bidirectional teacher.
- All three papers inherit this paradigm (ABot's LongForcing is a DMD-on-self-rollout stage; AlayaWorld uses self-forcing++ in distillation; SGF is literally a patch to Self Forcing). Exposure bias alone is *not* what any of the three papers claims to solve — it is the solved substrate they build on.

### Layer 1 — Context-quality gap → corrupted-history training (AlayaWorld, 2607.18367)
- **Diagnosis (refined vs hypothesis):** at inference the model must condition on its *own imperfect past* — blur, illumination/color shift, noise, and model-specific residuals. A model trained on clean history has never learned to *recover* from a corrupted prefix, so errors compound. This is not a distribution-mismatch story (the context *is* self-generated at inference); it is a robustness story: the model must function given degraded memory tokens.
- **Fix (two concrete mechanisms, applied to temporal-memory, spatial-memory, and nearby-frame tokens; the sink is kept clean):**
  1. **Helios-style drift simulation** — latent-space corruptions mimicking observed rollout artifacts: additive noise, down/up-sampling blur, saturation shift.
  2. **Error bank** (StableVideoInfinity-style) — the model's *own* reconstruction residuals δ = ẑ₀ − z₀, bucketed by chunk length and noise level, replayed additively into context (and target). Warm-up schedule: Helios only until the bank fills, then error bank takes priority. This closes the gap between synthetic corruption and the failure modes the model actually produces.
- Supporting cast: bounded visual context (persistent sink frame at RoPE position 0, compressed 6-frame temporal history, GEN3C-style geometry-aligned spatial memory reprojected into the target view, most-recent-frame I2V conditioning) keeps per-chunk compute constant; Next Forcing auxiliary head reinforces chunk continuity; discrete 4-step distillation (DMD + self-forcing++ + consistency) preserves the memory stack.
- **Not in the paper but in the shipped code:** `--ttc` "Pathwise Test-Time Correction" re-anchors each chunk to the first frame at inference — an inference-time anti-drift knob (see `code/artifact_verification.log` §1).

### Layer 2 — Memory-writing gradient gap → Self Gradient Forcing (SGF, 2607.20368)
- **Diagnosis (confirmed precisely):** Self Forcing exposes the model to self-generated history, but the historical KV cache is *frozen rollout state*. Future losses supervise how later noisy tokens **read** cached history, never how earlier generated latents are **written** into K/V at the clean context timestep (t_ctx=0). Because the same DiT parameters are shared across timesteps, DMD updates at noisy steps can silently degrade the unsupervised cache-writing path — the "historical context-gradient gap."
- **Fix:** two-pass training. Pass 1: ordinary no-gradient serial rollout; record context latents and noisy latents at a sampled denoising exit step. Pass 2: parallel reconstruction of that exit-step computation — recorded context is a stop-gradient *input*, but gradients flow through the re-computed context K/V formation and future-to-context attention. This restores memory-writing supervision without serial BPTT (direct differentiable cache OOMs; SGF costs 79→87 GB peak, +13% wall-clock). Recovery fidelity of Pass 2 is verified empirically (cosine 0.9999, RelL2 ≈ 1.8× bf16 epsilon).
- SGF is **training-only**: inference is byte-identical to Self Forcing.

### Layer 3 — Teacher-horizon mismatch → LongForcing (ABot-World-0, 2607.19191)
- **Diagnosis (confirmed; paper's own framing is sharper):** drift is a *distribution-shift* problem (Sec. 6): errors accumulate in the visual context and push the rollout beyond the temporal region covered by teacher supervision. Short-horizon teachers provide no distribution-matching signal for long-horizon self-rollout contexts.
- **Fix:** a final DMD stage where the student performs **long** self-rollouts and is scored by an **extended-horizon teacher** — distribution-level correction over a longer temporal range, no frame-level trajectory matching. Sits at the end of a progressive pipeline: bidirectional teacher → teacher forcing (causal adaptation on GT history) → causal ODE distillation (few-step) → LongForcing.
- Notable nuance: ABot explicitly rejects over-anchoring (sink/reference frames "delay" drift but lock the model to its initial observation) — LongForcing is positioned as the alternative that keeps the model free to imagine new content while staying in-distribution.

---

## 2. Composability verdict

**Mostly orthogonal, partially overlapping — and explicitly designed to be composable.**

- **SGF vs corrupted-history training — orthogonal.** They act on different sides of the memory interface. Corrupted-history training fixes the *input* side: given a degraded context, produce a good next chunk. SGF fixes the *encoding* side: given any context, write it into K/V in a way that remains maximally useful to future chunks. SGF's own paper claims orthogonality to forcing improvements and cache designs (Sec. 2, App. F) and suggests combining with long-rollout exposure and retrieval. Nothing in corrupted-history training supervises the memory write; nothing in SGF teaches recovery from corruption.
- **LongForcing vs corrupted-history training — overlapping in effect, different in mechanism.** LongForcing's long self-rollout DMD *implicitly* trains on imperfect self-generated context (the rollout context is genuinely corrupted by the student's own errors), so it partially covers context-quality robustness. But it is a distribution-level output correction that requires an extended-horizon teacher score model; corrupted-history training is teacher-free, applicable in the SFT stage, and explicitly teaches *recovery* (including from its own residual bank and from corrupted spatial-memory tokens, which a pure video-DMD teacher does not specifically target). Neither subsumes the other; the cheap robustness training plausibly *reduces the corruption level* that LongForcing's teacher must correct.
- **SGF vs LongForcing — orthogonal and stackable.** LongForcing extends *how far* teacher supervision reaches; SGF extends *what* the supervision trains (the memory write). Both are DMD-stage techniques on self-rollouts. SGF even names "long-rollout exposure" (Self-Forcing++/Rolling Forcing family, which includes LongForcing's approach) as a natural combination partner (App. F).
- **Could SGF be bolted onto ABot's or AlayaWorld's causal students?**
  - **ABot — yes, cleanly.** ABot's student is a chunk-wise causal DiT with a bounded sink+rolling-eviction KV cache — exactly the geometry SGF evaluates (SGF's chunk-wise setting is sink 3 / FIFO 6 / chunk 3). Adding SGF to the LongForcing DMD stage means: keep Pass-1 rollout as-is, add the Pass-2 context-gradient reconstruction. No architectural blocker; low-bit quantization is inference-only.
  - **AlayaWorld — partially.** AlayaWorld does not use a pure latent KV cache: its context is an in-context clean prefix (sink + learned compressed-history embedding H_φ + depth-warped spatial memory + recent frame) under full self-attention. SGF-style gradients could flow through the DiT-side re-encoding of the clean prefix tokens, but two memory components sit outside the SGF picture: the neural H_φ compressor and the non-differentiable spatial-memory pipeline (Depth-Anything-3, unprojection, z-buffering, splatting). So SGF would supervise only part of "how history is written." Feasible as a gradient-boundary change in the distillation stage; not a drop-in.
  - Framework compatibility is high in both cases: SGF is built precisely for few-step DMD students with sampled exit steps, which both systems use.

---

## 3. Evidence quality — who actually measures drift?

Ranking (strongest → weakest) with the reasoning:

1. **ABot-World-0 — the only drift *curves*.** Fig. 10 plots four artifact metrics (HPSv3, high-saturation pixel ratio, perceptual blur, patch-repeat ratio) **vs time over 60 s**, LongForcing vs an adapted Causal-Forcing baseline; divergence grows in the second half — a genuine accumulation measurement. Plus WorldRoamBench interactive benchmark numbers vs Genie 3 / HappyOyster / LingBot-World / HY-World 1.5. **Caveats:** the curve comparison is against its *own* adapted baseline, not a published method; curves stop at 60 s; the headline hour-/day-scale rollouts are qualitative sampled checkpoints with no quantitative identity or controllability decay metric.
2. **Self Gradient Forcing — best-controlled, but endpoints only.** Paired SGF vs Self Forcing under matched initialization, prompts, seeds, and inference geometry, at 5/60/240 s, across frame-wise and chunk-wise and three initializations; VBench-Long consistency metrics (subject, background, flickering, imaging) + >1,900-judgment GSB human study (all positive). Honest about the dynamic-degree confound (SF's scene jumps inflate apparent motion). **Caveats:** no metric-vs-time curves within a rollout; 240 s = 4 min, so "minutes-long" means *single-digit* minutes; the setting is text-to-video extrapolation, **not interactive** — no action control, so controllability decay is untested.
3. **AlayaWorld — best benchmark coverage, weakest drift attribution.** iWorld-Bench gives drift-sensitive endpoint metrics (brightness consistency 0.9492, color-temperature 0.9379, sharpness retention 0.8361 — large margins) and loop-closure memory metrics (Memory Symmetry 0.8871) vs six open models; plus WorldMark/World Model Arena Elo. **Caveats:** per-clip horizon is never stated; no drift curves; no ablation isolating corrupted-history training from the memory architecture or distillation; the "stable long-horizon" claim (Fig. 8) is qualitative; the repo's own recommendation to enable `--ttc` for ~1-minute rollouts suggests residual appearance drift at that scale.

**Common gap across all three:** nobody publishes identity-consistency or controllability decay measured at 10–60 min horizons. The field's quantitative drift evidence currently tops out at 60 s (ABot, curves) and 240 s (SGF, endpoints).

## 4. Artifact availability (verified today, 2026-07-23)

The brief's assumption ("AlayaWorld promises, SGF promises, ABot silent") is **outdated — all three have shipped code and weights**:

| | ABot-World-0 | AlayaWorld | SGF |
|---|---|---|---|
| Code | `amap-cvlab/ABot-World` (883★, Apache-2.0) — verified | `AlayaLab/AlayaWorld` (646★) — verified; inference only, training code on roadmap | `zhuang2002/Self_Gradient_Forcing` (65★, Apache-2.0) — verified; **full training + inference**, released same day as paper |
| Weights | HF `acvlab/ABot-World-0-5B-LF` (992 dl, Apache-2.0; also ModelScope) — verified | HF `AlayaLab/AlayaWorld` (196 dl, 33.6 GB, **LTX-2 non-commercial license**) — verified | HF `JunhaoZhuang/Self_Gradient_Forcing` (61 dl, Apache-2.0) — verified |
| Missing | bidirectional teacher; 500-h dataset ("very soon") | training code/data; third-party deps gated (Gemma-3-12B) | nothing critical |

## 5. Deployment reality — what can an outsider actually run?

- **ABot-World-0** is the only one engineered for consumer hardware and the easiest to run: 5B student (Wan2.2-TI2V-5B base), 720p 16 FPS, 1.2 s action-to-first-frame, ≤19.3 GiB VRAM on a single RTX 5090 — i.e., any 24 GB card plausibly works. The stack: LightVAE decoder, FP8/low-bit DiT (LightX2V), SageAttention2, Fast-RoPE Triton kernels, bounded+quantized KV cache, FramePack-style module swapping. Weights + gradio demo + hosted playground all live.
- **AlayaWorld** is a datacenter-GPU proposition despite the "real-time" framing: ~13–15B DiT (weights alone 33.6 GB) *plus* a gated Gemma-3-12B text encoder *plus* Depth-Anything-3-Giant for the spatial memory. No inference hardware/VRAM figure anywhere in the report; multi-GPU Ulysses is supported. Runnable by an outsider with ≥80 GB (or multi-GPU), under a non-commercial license.
- **SGF** is a training recipe, not a system: inference = vanilla Self Forcing on Wan2.1-1.3B-scale models (single consumer GPU feasible), but *training* needs ~87 GB. An outsider can download the released checkpoints and generate 240 s videos today; reproducing training needs one H100-class card.

**Outsider-runnability ranking:** ABot (desktop, today) > SGF inference (desktop) > AlayaWorld (single 80 GB+ or multi-GPU) > SGF training (H100-class).

## 6. One-paragraph synthesis for the report

This week's pool contains a rare natural experiment: three papers that each name a *different* root cause for long-horizon drift, and the causes stack rather than compete. Self Forcing already fixed *where* the context comes from (exposure bias). AlayaWorld fixes *robustness to degraded context* by training on corrupted histories and its own residual bank. SGF fixes *how context is written into memory* by reopening the gradient path into clean-timestep K/V formation. ABot's LongForcing fixes *how far teacher supervision reaches*, matching the student's long-rollout distribution against an extended-horizon teacher. A system combining all three is architecturally straightforward for KV-cache students (ABot-style) and partially applicable to explicit-memory systems (AlayaWorld-style, whose neural-rendering memory lies outside SGF's gradient picture). Evidence quality lags the mechanism design: only ABot shows drift-vs-time curves (60 s), only SGF is cleanly controlled (to 240 s), and no paper quantifies identity or controllability decay at the 10–60 min horizons all three implicitly claim.
