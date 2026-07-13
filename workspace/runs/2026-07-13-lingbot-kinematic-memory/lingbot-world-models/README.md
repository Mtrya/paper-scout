# LingBot-Video + LingBot-World 2.0 — Joint Research Memo

**Date:** 2026-07-13  
**Papers:**
- `papers/world-models/lingbot-video-moe-2607.07675.md` (arXiv:2607.07675)
- `papers/world-models/lingbot-world-infinity-2607.07534.md` (arXiv:2607.07534)

**Code inspected:**
- `code/lingbot-video/` (cloned from https://github.com/robbyant/lingbot-video)
- `code/lingbot-world-v2/` (cloned from https://github.com/robbyant/lingbot-world-v2)
- HF model cards / configs copied to `code/lingbot-world-models-probes/`
- Parameter-count probe: `code/lingbot-world-models-probes/param_counts.py`

---

## 1. Executive Summary

These are companion releases from the LingBot / RobbyAnt team, both pitched as open, embodied-AI-oriented video generation systems, but they are **not the same codebase and not the same model**.

- **LingBot-Video** is a *general video foundation model* — a diffusion transformer (DiT) with sparse Mixture-of-Experts (MoE) routing, trained for text-to-image/video (T2I/T2V) and image-to-video (TI2V). It is packaged as a Diffusers-compatible model, uses a Qwen3-VL text encoder and Wan2.1-VAE, and is released in dense (1.3 B) and MoE (30 B total / ~2.9 B active) variants plus a 480p→1080p refiner.
- **LingBot-World 2.0 (Infinity)** is an *interactive causal world model*. It is built on the Wan2.2 codebase, uses a T5-XXL text encoder and Wan2.1-VAE, and is released as a dense 14 B causal-fast distilled model (the causal-pretrain and 1.3 B variants are still TODO). Its focus is long-horizon, action-conditioned autoregressive generation at 720p / 60 fps, with a “Director-Pilot” agentic harness.

The two share authors, a similar data-profiling vocabulary, and a focus on embodied footage, but the architectures, conditioning, and training objectives diverge. **LingBot-World 2.0 is not a direct descendant of LingBot-Video** in the released artifacts; it is a Wan2.2-style dense causal transformer.

---

## 2. Core Mechanisms

### 2.1 LingBot-Video — single-stream MoE video diffusion

**Architecture (Sec. 2)**
- **Single-stream DiT.** Visual latent tokens and condition tokens are projected to the same hidden dimension and concatenated into one sequence, with no separate condition stream. A 3D multi-modal RoPE places text tokens at temporal coordinates `(i,0,0)` and visual tokens at `(L+1+f, h, w)` so attention stays single-stream.
- **QK-Norm + adaLN-single.** Queries/keys are RMSNorm-ed per head; a shared timestep embedding plus per-layer learnable modulation tables produce scale/shift/gate for attention and FFN.
- **Sparse MoE FFN (Sec. 2.2).** Only the FFN branch is sparse. Each sparse layer has:
  - `N_s` shared experts (always active).
  - `N_r` routed experts; each token is sent to top-`K_r` of them.
  - Sigmoid router with DeepSeek-style group-limited routing (`N_g` groups, top-`K_g` groups scored by sum of top-2 affinities).
  - Auxiliary-loss-free load balancing via online correction bias `b_j` updated by `b_j ← b_j − η sign(n_j − n̄)`.
  - Sequence-wise auxiliary balance loss added to the diffusion loss.
- **Cascaded refiner (Sec. 2.5).** Base model generates 480p; a second-stage refiner upscales to 1080p with a conditional rectified-flow trajectory starting from a degraded low-res latent (`τ ∈ [0.85,0.95]`).

**Public config (MoE 30B-A3B)** — `code/lingbot-world-models-probes/lingbot-video-moe-30b-a3b-transformer-config.json`:

| Quantity | Value |
|---|---|
| hidden size | 2048 |
| depth | 48 |
| attention heads | 16 (head dim 128 = 32+48+48) |
| patch size | (1,2,2) |
| routed experts `N_r` | 128 |
| top-`K_r` | 8 |
| groups `N_g` / top groups `K_g` | 4 / 2 |
| shared experts `N_s` | 1 |
| routed expert intermediate | 768 |
| dense intermediate (for dense layers) | 6144 |
| route scaling factor | 2.5 |

Analytic parameter count from the probe: **≈30.1 B total, ≈2.89 B active per token**; the dense 1.3 B variant is **≈1.36 B total** (24 layers, no experts).

**Data engine (Sec. 3)**
- Data Profiling Engine annotates every sample along structural, semantic, motion, camera, and quality axes using VLMs + specialized scorers (HPSv3, OmniAID, LocoTrack, TransNetV2, PaddleOCR).
- A World-Knowledge Topological Graph (50 k leaf concepts, 1 k intermediates, 25 top-level groups) plus an action tree drives distribution-aware sampling and loss-based re-weighting.
- Dense structured JSON captions for images, videos, robot VLA videos, and egocentric videos; a two-stage Caption Rewriter (Qwen3.6-27B expand → LoRA map) converts short prompts to this schema at inference.
- Five-stage curriculum: 192p images → 192p image+video (with >70 k hours of embodiment footage: manipulation, navigation, egocentric) → 480p re-filter → 480p challenge-focused rebalancing → 1080p refiner subset.

**Training objectives (Sec. 5)**
- Base pre-training: rectified-flow / flow-matching velocity loss on the unified T2I/T2V/TI2V task.
- MoE load balancing: sequence-wise auxiliary loss + online bias correction.
- Post-training RL (Sec. 5.2): Group Relative Policy Optimization (GRPO) with six de-coupled reward models:
  1. **Vision Quality** (HPSv3-derived).
  2. **Text-Video Alignment** (temporal VQA with Qwen3.6-27B on parsed entities/actions).
  3. **Dynamic Degree** (VLM-assigned motion score 1–5).
  4. **Motion Coherence** (real-world playback speed, anti slow-motion).
  5. **Human-Motion Consistency** (fine-tuned VLM-MoE for topology/facial/hand/limb artifacts).
  6. **Physical Plausibility** (motion causality, object permanence/non-penetration, material realism, task completion).
- Training uses Flash-GRPO-style single-step exploration, Coefficients-Preserving Sampling (CPS), and timestep-balanced gradient re-weighting.
- Negative-aware finetuning with real videos as positives and model generations as negatives (DiffusionNFT-style forward-process loss) plus a KL regularizer against a frozen reference.
- Distillation (Sec. 5.4): improved Distribution Matching Distillation (DMD2) for few-step generation.
- Action-to-Video adapter (Sec. 5.3): adds an `ActionEmbedder` and finetunes on GR-1/Fourier trajectories for 8 k steps, batch size 64, lr 1e-5.

### 2.2 LingBot-World 2.0 — causal interactive world model

**Formulation (Sec. 3.1)**
Factorizes the world as a causal generative process:

```
p(x_1:T | a_1:T) = ∏_t p(x_t | x_<t, a_≤t)
```

where `a_t` includes camera poses and textual prompts.

**Pre-training (Sec. 3.2)**
- **Actions:** camera poses encoded as per-pixel Plücker embeddings, injected via AdaLN; chunk-wise text prompts injected via cross-attention.
- **MoBA attention mask.** Self-attention mixes a *teacher-forcing* causal block (each noisy frame attends only to itself and clean past frames) with a *bidirectional* block at the bottom-right. The idea is to avoid the model over-relying on the clean context and degrading when it must generate future frames. A matching cross-attention mask uses a background prompt + lower-triangular chunk prompts for the causal part and a global prompt for the bidirectional part.
- **Objective:** conditional flow matching on `x_t^i = (1−t)x_i + tε`, target velocity `ε − x_i`, conditioned on `x_<i`, `p_≤i`, `a_≤i`.

**Post-training / distillation (Sec. 3.3)**
- **Consistency distillation:** student `G_θ` learns to predict a trajectory-invariant target by matching adjacent points on the teacher PF-ODE, using an EMA target network.
- **Distribution Matching Distillation (DMD):** minimizes KL between noised student distribution and noised data distribution, with real/fake score models. Crucially, DMD is applied over **long self-rollouts** rather than only teacher-forced states, which is meant to suppress compounding drift.

**Deployment (Sec. 4)**
- Causal-fast inference uses KV-cache chunk generation; supports `local_attn_size` and `sink_size` for long-context windowing (the released example uses `--local_attn_size 18 --sink_size 6`).
- Spatio-temporal refiner after VAE decode, compiled to TensorRT.
- **Director-Pilot harness:** a VLM (Director) reasons about scene state and proposes event cards; the video generator (Pilot) renders the consequences. Two modes: direct semantic interaction and SAM-assisted object interaction.

**Public config (causal-fast 14 B)** — `code/lingbot-world-models-probes/lingbot-world-v2-14b-causal-fast-config.json`:

| Quantity | Value |
|---|---|
| dim | 5120 |
| num layers | 40 |
| num heads | 40 (head dim 128) |
| FFN dim | 13824 |
| in dim | 36 (latent + mask + Plücker channels) |
| out dim | 16 |
| text length | 512 (T5-XXL) |
| sample steps (fast) | 4 steps, no CFG |
| sample steps (pretrain) | 40 steps, CFG |

Analytic count: **≈12.97 B DiT parameters**; with T5-XXL and VAE the marketed “14 B” refers to the DiT.

---

## 3. Relationship Between the Two Models

| Aspect | LingBot-Video | LingBot-World 2.0 |
|---|---|---|
| **Codebase** | Custom Diffusers pipeline (`lingbot_video/`) | Wan2.2-derived (`wan/`) |
| **Text encoder** | Qwen3-VL-4B | T5-XXL (umt5-xxl) |
| **VAE** | Wan2.1-VAE | Wan2.1-VAE |
| **Architecture** | Single-stream MoE DiT | Wan-style dense DiT + causal cross/self-attention |
| **Conditioning** | Text / image / structured JSON captions | Camera Plücker + chunk-wise prompts |
| **Generation mode** | Diffusion, non-causal | Causal autoregressive, chunk-by-chunk |
| **Released sizes** | Dense 1.3 B, MoE 30B-A3B | 14 B causal-fast only (pretrain & 1.3 B TODO) |
| **License** | Apache-2.0 | CC BY-NC-SA 4.0 (non-commercial) |

**Do they share code?** Not in the public repositories. The two repos have disjoint top-level code and different dependency stacks. LingBot-World-v2 explicitly acknowledges Wan2.2, not LingBot-Video, as its base.

**Is LingBot-World built on LingBot-Video?** There is **no evidence in the papers or code** that World 2.0 is initialized from or finetuned on LingBot-Video weights. The world model is a dense Wan2.2-style transformer with causal modifications; LingBot-Video is a non-causal MoE diffusion model. They are sibling efforts.

**What transfers?** The *concepts* are shared: a multi-axis data profiler, emphasis on robot/egocentric/navigation footage, structured captions, and a cascaded/refinement pipeline. The exact implementations differ.

---

## 4. Key Quantitative Findings

### 4.1 Model sizes (from public configs + analytic count)

| Model | DiT params (analytic) | Marketing label | Active params/token (MoE) |
|---|---|---|---|
| LingBot-Video-Dense | 1.36 B | 1.3 B | — |
| LingBot-Video-MoE | 30.07 B | 30 B | 2.89 B |
| LingBot-World-v2-14B | 12.97 B | 14 B | — |

### 4.2 LingBot-Video benchmarks

**Internal benchmark (Fig. 15)** — LingBot-Video is reported SOTA among open-source models on TI2V (general quality + embodied domain); on T2V it is second in general quality but first in embodied domain.

**RBench (Tab. 1 / Fig. 16a)** — average score:

| Model | Avg | Manip. | Spatial | Multi-entity | Long-hor. | Reasoning | Single arm | Dual arm | Quadruped | Humanoid |
|---|---|---|---|---|---|---|---|---|---|---|
| **LingBot-Video** | **0.620** | 0.578 | 0.643 | 0.444 | 0.634 | 0.505 | 0.636 | 0.639 | 0.758 | 0.689 |
| Cosmos3 Super | 0.581 | 0.487 | 0.642 | 0.444 | 0.591 | 0.395 | 0.615 | 0.623 | 0.739 | 0.691 |
| Wan 2.2 A14B | 0.507 | 0.381 | 0.454 | 0.373 | 0.501 | 0.330 | 0.608 | 0.582 | 0.690 | 0.648 |
| HunyuanVideo 1.5 | 0.460 | 0.442 | 0.316 | 0.312 | 0.438 | 0.364 | 0.513 | 0.526 | 0.634 | 0.595 |
| LongCat-Video | 0.437 | 0.372 | 0.310 | 0.220 | 0.384 | 0.186 | 0.586 | 0.576 | 0.681 | 0.621 |
| Wan 2.6 (closed) | 0.607 | 0.546 | 0.656 | 0.479 | 0.514 | 0.531 | 0.666 | 0.681 | 0.723 | 0.667 |
| Seedance 1.5 pro (closed) | 0.584 | 0.577 | 0.495 | 0.484 | 0.570 | 0.470 | 0.648 | 0.641 | 0.680 | 0.692 |

**Physics-IQ Verified I2V (Fig. 16b)**
- LingBot-Video: **40.4**
- Cosmos3: 39.5
- HunyuanVideo 1.5: 33.4
- Wan 2.2 A14B: 32.2

**MoE efficiency (Fig. 7, 1M-token sequence)** — speed-up of MoE 30B-A3B vs dense baselines:
- vs Dense 3B (active-param matched): 0.97×
- vs Dense 6B: 1.50×
- vs Dense 14B: 2.59×
- vs Dense 30B: 3.18×

**RL infrastructure (Sec. 4.2.2)** — end-to-end MFU 43.9% on the 30B MoE during GRPO post-training; 20 s full-param sync; 50 ms exchange of GB-scale intermediate states.

### 4.3 LingBot-World 2.0 claims

The paper is **almost entirely qualitative** on numbers:
- “Over an hour of continuous generation without quality loss” (Fig. 10).
- Real-time 720p at 60 fps for the distilled model.
- Comparison table (Tab. 1) labels itself the only model with “Hours (Infinite)” generation duration plus high dynamic degree, semantic interaction, real-time, and open-source.
- No standard VBench, GenEval, or Physics-IQ numbers are reported for the world model.

---

## 5. Code Inspection Findings

### 5.1 What is actually released

**LingBot-Video repo**
- Inference-only: Diffusers + optional SGLang backends, single/multi-GPU scripts, refiner scripts.
- Prompt rewriter inference code (`rewriter/`) and auto-negative prompt generator.
- No training scripts, no data profiling implementation, no reward models.
- The MoE execution path supports grouped GEMM, SGLang Triton, and SGLang FP8 backends, with numerous environment switches (`LINGBOT_MOE_*`).
- FP32 modules list: `time_embedder`, `time_modulation`, `scale_shift_table`, norms, router.

**LingBot-World-v2 repo**
- Inference-only causal generation (`generate.py`, `wan/image2video.py`).
- Supports two modes: `causal_fast` (distilled, 4 steps, no CFG, KV cache) and `causal_pretrain` (40 steps, CFG), but only `causal_fast` weights are released.
- Built on Wan2.2 modules: `WanModel`, `WanModelFast`, `WanModelCausal`, T5-XXL encoder, Wan2.1-VAE.
- Camera control is implemented via Plücker embeddings injected through small cam-scale/shift MLPs inside each block (`cam_injector_layer*`, `cam_scale_layer`, `cam_shift_layer`).
- No Director/Pilot agentic harness code in the repo; that part of the system is not open-sourced. The deployment stack is also not released (README: “We do NOT plan to release our deployment code”).

### 5.2 Key code evidence

- `code/lingbot-world-models-probes/lingbot-video-moe-30b-a3b-transformer-config.json` confirms 48 layers, 128 routed experts, top-8 routing, 1 shared expert, sigmoid router, group-limited routing (4 groups / top-2).
- `code/lingbot-world-models-probes/lingbot-video-dense-1.3b-transformer-config.json` confirms the dense baseline is 24 layers with the same hidden size (2048) and no experts.
- `code/lingbot-world-models-probes/lingbot-world-v2-14b-causal-fast-config.json` confirms `_class_name: WanXModel`, diffusers 0.30.0, dim 5120, 40 layers, 40 heads, FFN 13824 — a dense Wan-style architecture, not an MoE.
- `code/lingbot-video/lingbot_video/transformer_lingbot_video.py` implements the MoE block with `LingBotVideoSparseMoeBlock`, shared experts, sigmoid `LingBotVideoRouter`, group-limited top-k, and explicit load-balancing bias buffer `e_score_correction_bias`.
- `code/lingbot-world-v2/wan/modules/model_fast.py` implements `CausalWanSelfAttention` with KV-cache update, local attention eviction, and sink tokens; no MoE.

### 5.3 Parameter-count probe

`code/lingbot-world-models-probes/param_counts.py` analytically counts parameters from the public configs (no weights downloaded). Output:

```
lingbot-video-dense-1.3b-transformer-config.json
  h=2048, layers=24, experts=0, top_k=8
  attention: 0.403 B
  ffn:       0.906 B
  misc:      0.048 B
  total:     1.357 B

lingbot-video-moe-30b-a3b-transformer-config.json
  h=2048, layers=48, experts=128, top_k=8
  attention: 0.805 B
  ffn:       29.218 B
  misc:      0.048 B
  total:     30.071 B
  active/tok:2.892 B

lingbot-world-v2-14b-causal-fast-config.json
  h=5120, layers=40, experts=0, top_k=1
  attention: 4.194 B
  ffn:       8.493 B
  misc:      0.285 B
  total:     12.973 B
```

---

## 6. What Is Genuinely New vs. Incremental

### Genuinely new / distinctive
1. **Open MoE video diffusion at 30 B scale.** Most open video diffusion models are dense; LingBot-Video ships a sparse 30 B model with public weights, Diffusers packaging, and SGLang/FP8 serving code.
2. **Embodied-aware pre-training mixture.** The claim of >70 k hours of robot manipulation + navigation + egocentric footage blended with web video, plus a structured caption schema that covers VLA/egocentric actions, is unusual for a general video foundation model.
3. **Multi-reward RL post-training for physical plausibility.** The six specialized reward models (especially the physical-plausibility and human-motion-consistency evaluators) go beyond generic aesthetic/text-alignment rewards.
4. **Causal world model with MoBA mask.** The explicit hybrid teacher-forcing + bidirectional attention mask to mitigate context over-reliance is a novel training detail.
5. **Long-horizon real-time world model claim.** Hour-level stability at 720p/60 fps, if reproducible, is beyond the minutes-level horizon typical of prior open world models.
6. **Director-Pilot agentic harness.** The architectural choice to pair a VLM “director” with a video-generator “pilot” for open-ended interaction is a system-level contribution.

### Incremental / borrowed
- **MoE design** is essentially DeepSeekMoE (fine-grained experts, shared experts, group-limited routing, aux-loss-free balancing) ported to video diffusion.
- **Data profiling** follows prior large-scale video curation pipelines (e.g., InternVid, Panda-70m, Wan2.1).
- **Rectified flow / consistency distillation / DMD** are existing techniques; the world model applies them with a causal wrapper.
- **Architecture** of World 2.0 is explicitly Wan2.2 + causal attention; LingBot-Video uses Wan2.1-VAE and a DiT design close to other flow-matching video transformers.
- **Action-conditioned video generation** for robotics has been explored by GR-1, DreamDojo, Ctrl-World, etc.; LingBot-Video-A2V is a post-trained adapter on GR-1 data.

---

## 7. Open Questions, Limitations, and Blockers

### Limitations acknowledged by the authors
- **Long-term memory.** World 2.0 admits that regions leaving the context window are regenerated rather than recalled; persistence is visual, not identity-based.
- **Identity/style drift** over very long rollouts.
- **Imperfect physics** — no explicit geometry/collision model; objects can intersect or pass through each other.
- **Compute.** Real-time 720p/60 fps still requires substantial hardware; the example command uses 8-GPU distributed inference.

### Critical gaps from inspection
1. **No training code or data.** Neither repo releases training scripts, data pipelines, or reward-model weights. Reproducing the models from scratch is impossible.
2. **World model lacks numeric benchmarks.** The hour-long stability claim is qualitative; there are no VBench, EvalCrafter, Physics-IQ, or task-success metrics for World 2.0. The comparison table is self-authored.
3. **Causal-pretrain and 1.3 B world models are not released.** Only the distilled 14 B causal-fast checkpoint is available; the teacher and lightweight variants are TODO.
4. **Agentic harness is not in the repo.** The Director-Pilot system, SAM loop, event-card UI, and deployment stack are described but not open-sourced.
5. **Compute budget undisclosed.** Neither paper reports total GPU-hours, FLOPs, training data size in tokens or videos, or exact cluster configuration. The scaling-law curves are early-stage and not converged.
6. **Dense-vs-MoE comparisons are partial.** The “active-parameter comparable” Dense 1.3 B has half the depth (24 vs 48) of the MoE 13 B/30 B models; the inference speed comparisons mix active-parameter and total-parameter baselines without full architectural ablations.
7. **License mismatch.** Video model is Apache-2.0; World 2.0 is CC BY-NC-SA 4.0 (non-commercial). Any plan to combine them must respect the stricter world-model license.
8. **Reward hacking risk.** The six reward models are not released; it is hard to verify whether the RL stage overfits to them. The negative-aware real-video finetuning is a mitigation but details are sparse.

### Blockers for follow-up work
- **Cannot run the world model without heavy GPUs.** The 14 B model + T5-XXL + VAE at 720p requires multi-GPU memory even with FSDP/offloading.
- **No training data identifiers.** The exact embodiment datasets and their proportions are not listed; one cannot replicate the data mixture.
- **No evaluation protocol.** There is no published evaluation script or benchmark prompt set for the world model, making independent verification difficult.

---

## 8. Bottom Line

LingBot-Video is the stronger *empirical* contribution: it ships a large open MoE video diffusion model, reports public benchmark numbers (RBench, Physics-IQ), and releases a usable inference stack. Its main novelty is scale + embodiment-aware data/rewards rather than a fundamentally new architecture.

LingBot-World 2.0 is the stronger *system* vision — a real-time, long-horizon causal world simulator with an agentic harness — but it currently rests on qualitative demonstrations and a single released checkpoint. The two projects share a team and a data-centric philosophy, but they are architecturally distinct, and World 2.0 is not simply LingBot-Video turned causal.

**Recommended follow-up:** try to run LingBot-World-v2 causal-fast on a small test case to verify memory/runtime requirements and stability over a few hundred frames; evaluate LingBot-Video on the public Physics-IQ / RBench splits to sanity-check the reported numbers; and inspect whether the unreleased causal-pretrain world model would be necessary for any robotics planning use case.
