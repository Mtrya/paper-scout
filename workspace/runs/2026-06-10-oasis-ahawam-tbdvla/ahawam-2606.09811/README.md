# Deep Investigation: AHA-WAM (arXiv:2606.09811)

**Paper:** AHA-WAM: Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing  
**Authors:** Jisong Cai, Long Ling, Shiwei Chu, et al. (Shanghai Jiao Tong University, Shanghai AI Laboratory, Baidu AI Cloud, HKU)  
**Date investigated:** 2026-06-10  
**Thread directory:** `runs/2026-06-10-oasis-ahawam-tbdvla/ahawam-2606.09811/`

---

## 1. What Was Attempted

This thread investigates whether AHA-WAM's central claim holds: that **decoupling a slow video-DiT world planner from a fast action-DiT executor** preserves (or improves) manipulation performance while substantially increasing closed-loop control frequency. The investigation focused on four mechanisms:

1. **Dual-DiT architecture with layerwise joint attention** — how world and action branches interact.
2. **Observation-Guided Video-Context Routing (OVCR)** — how cached planner context stays aligned with the latest observation without rerunning the video DiT.
3. **Horizon-adaptive offset training** — how the action expert learns to consume planner context under arbitrary phase shifts.
4. **Rolling K/V memory** — how the video planner preserves temporal context across refreshes.
5. **Inference optimization path** — TensorRT, CUDA Graphs, and ODE distillation.

**External-signal work:**
- The AHA-WAM project page (https://serene-sivy.github.io/aha-wam/) was inspected; **no public code repository is linked**.
- Fast-WAM (the closest baseline, arXiv:2603.16666, GitHub: `yuantianyuan01/FastWAM`) was cloned and its model definition, MoT coupling, and inference path were read to establish the architectural baseline that AHA-WAM extends.
- Because AHA-WAM code is unavailable, compact PyTorch reconstructions of OVCR, layerwise joint attention, horizon-adaptive offset training, and rolling K/V memory were written from the paper's equations and architecture description. These serve as evidence of what the method implies mechanistically.

---

## 2. Method Breakdown

### 2.1 Dual-DiT Planner–Executor Architecture (Eq. 3)

AHA-WAM starts from the same backbone as Fast-WAM: a **Wan2.2-5B video DiT** initialized from pretrained weights and a lightweight **action DiT** (~1.02B params, `d_a = 1024`, 30 layers). The critical departure is temporal decoupling:

| Branch | Horizon | Frequency | Role |
|--------|---------|-----------|------|
| Video DiT (planner) | `h_v = 64` frames | Low | Long-horizon world plan; produces reusable layerwise K/V context |
| Action DiT (executor) | `h_a = 16` actions | High | Short closed-loop action chunks; queries planner context |

**Layerwise joint attention (Eq. 3):**

```
H̄_t^{a,ℓ} = Attn( Q_t^{a,ℓ}, [K_t^{a,ℓ} ; K̃_t^{p,ℓ}], [V_t^{a,ℓ} ; Ṽ_t^{p,ℓ}] )
```

At each layer ℓ, the action branch's queries attend to a concatenation of its own keys/values and the **OVCR-adapted planner keys/values**. This preserves the WAM-style interaction (action generation conditioned on visual dynamics) while keeping the expensive video DiT off the per-update critical path.

**Training objective (Eqs. 4–7):** Joint flow matching over action chunks and future video latents:

```
L = L_a + λ L_v
L_FM(y) = E[ || f_θ(y_ρ, ρ, O_t^v, s_t, l) - (ε - y) ||² ]
```

The video branch is trained with a **fully causal mask** to learn forward scene dynamics. During inference, the future-video prediction path is removed entirely; only the planner context production remains.

> **Evidence preserved:** `reconstruction/layerwise_joint_attention.py` — a runnable PyTorch module implementing Eq. 3 with multi-head attention and an `AsyncDualDiTBlock` that consumes adapted planner K/V.

---

### 2.2 Observation-Guided Video-Context Routing (Eqs. 8–10)

OVCR is the mechanism that prevents stale planner context from degrading performance during asynchronous execution. It operates in four steps:

**Step 1 — Construct routing queries from the latest observation (Eq. 8):**

```
Z_t^q = Attn( B, f_v(X_t^v), f_v(X_t^v) )
```

- `B ∈ R^{Q × d}`: `Q = 32` learnable base queries (Table 4).
- `X_t^v`: current visual observation tokens (VAE-encoded).
- `f_v`: lightweight visual projection.
- The routing queries are **compact observation-conditioned slots** (32 vectors) rather than dense visual tokens.

**Step 2 — Read planner features (Eq. 9, first half):**

```
R_t^ℓ = Attn( Z_t^{q,ℓ}, K_{τ(t)}^{p,ℓ}, V_{τ(t)}^{p,ℓ} )
```

The routing queries attend to the **cached planner K/V** from the latest refresh τ(t), retrieving chunk-relevant information.

**Step 3 — Predict residual K/V updates (Eq. 9, second half):**

```
(ΔK_t^{p,ℓ}, ΔV_t^{p,ℓ}) = g_ψ^ℓ( R_t^ℓ, Z_t^{q,ℓ} )
```

`g_ψ^ℓ` is a lightweight layerwise router (we reconstruct it as an MLP in the code).

**Step 4 — Gated residual update (Eq. 10):**

```
K̃_t^{p,ℓ} = K_{τ(t)}^{p,ℓ} + α_t^ℓ ΔK_t^{p,ℓ}
Ṽ_t^{p,ℓ} = V_{τ(t)}^{p,ℓ} + α_t^ℓ ΔV_t^{p,ℓ}
```

`α_t^ℓ` is a learned sigmoid gate that controls how much the cached context is updated. The adapted context is then consumed by Eq. 3.

**Why this design is clever:** Visual feedback enters the high-frequency action branch **indirectly** — through a small set of routing queries that edit latent K/V states — rather than by concatenating dense visual tokens into the action sequence. This keeps the action DiT lightweight and fast.

> **Evidence preserved:** `reconstruction/ovcr.py` — runnable `ObservationGuidedVideoContextRouter` and `OVCRCoupling` implementing Eqs. 8–10 with sanity checks.

---

### 2.3 Horizon-Adaptive Offset Training (Eqs. 11–12)

If training always aligns the action chunk to the start of the planner window, the action DiT overfits to a single phase. During asynchronous deployment, the executor reuses planner context at intermediate phases, causing a mismatch.

**Training procedure:**

```
δ ~ Uniform{0, 1, ..., h_a - 1}          (Eq. 11)
L_a = E_δ [ L_FM( A_τ^δ ) ]               (Eq. 12)
```

For each training segment starting at τ, the video planner models `[τ, τ + h_v)`. The action-chunk grid is randomly shifted by `δ ∈ [0, 16)` inside this window. Because the phase relationship is periodic with action chunk size, sampling over `[0, h_a)` covers all chunk-level phases encountered when one planner context is reused across multiple executor updates.

**Intuition:** The action expert learns that the same planner K/V cache must support action chunks starting at offset 0, 5, 12, etc., within the 64-frame video horizon. OVCR then handles observation-conditioned context selection at each specific phase.

> **Evidence preserved:** `reconstruction/horizon_adaptive_training.py` — `HorizonAdaptiveOffsetSampler` and `horizon_adaptive_action_loss` with alignment logic and sanity checks.

---

### 2.4 Rolling Planner Memory (Eq. 13)

Because the video DiT is low-frequency, it needs to remember what happened before the current refresh. AHA-WAM maintains a fixed-size FIFO rolling K/V memory:

```
M_τ^ℓ = FIFO( M_{τ-1}^ℓ ∪ { (K_τ^{p,ℓ}, V_τ^{p,ℓ}) } )   (Eq. 13)
```

- Window size: **up to 6 historical observation frames** (Table 4).
- This memory is **internal to the video planner**; the action DiT never sees it directly.
- At the next refresh, the video DiT attends to this memory when producing new planner context, extending its temporal receptive field.

**Why this matters:** In long-horizon tasks, completed subgoals or displaced objects may no longer be visible in the current frame. Rolling memory lets the planner retain that context without forcing the action branch to process longer histories.

> **Evidence preserved:** `reconstruction/rolling_kv_memory.py` — `RollingPlannerMemory` with FIFO update, concatenated history retrieval, and attention integration.

---

### 2.5 Inference Latency Optimization Path

AHA-WAM's latency improvements come from **three stacked sources**:

| Stage | Optimization | Action latency | Frequency | Source |
|-------|-------------|----------------|-----------|--------|
| Baseline | PyTorch eager (both branches coupled) | ~415.77 ms | ~2.4 Hz | Appendix D, Table 8 |
| + CUDA accel | TensorRT + CUDA Graphs + hoisting | 41.37 ms | 24.17 Hz | Appendix D, Table 8 |
| + ODE distill | 10-step → 2-step action sampler | 17.56 ms | 56.95 Hz | Appendix D, Table 11 |

**CUDA acceleration details (Table 8):**
1. **Graph-level static capture:** Action DiT, memory/context modules, and VAE encoder compiled to TensorRT engines; denoising loop replayed with CUDA Graphs.
2. **Selective `torch.compile`:** Applied to the video-DiT prefill path (default mode; `reduce-overhead` trades chunk latency for prefill latency).
3. **Hot-path redundancy elimination:** Chunk-level computations (condition embeddings, positional encodings, planner-context references) hoisted outside the 10-step denoising loop; repeated host-to-device copies skipped.

**ODE distillation details (Appendix D):**
- Teacher: 16-step denoising trajectory with anchors `{0, 1, 2, 4, 8, 12, 16}`.
- Student: 2-step sampler, trained to regress the teacher's final action output from sampled intermediate states.
- Sampling is **biased toward noisier states** because accurate high-noise prediction is the bottleneck for aggressive step reduction.
- Video DiT is **frozen** during distillation; the student preserves the same planner-context and OVCR interface.

**Ablations (Table 11):**
| Denoising steps | Latency | Frequency | Drop from 10-step |
|-----------------|---------|-----------|-------------------|
| 1 | 14.67 ms | 68.3 Hz | −64.6% |
| 2 | 17.56 ms | 56.9 Hz | −57.5% |
| 4 | 23.45 ms | 42.6 Hz | −43.3% |
| 10 | 41.37 ms | 24.2 Hz | baseline |

The 2-step Flash variant is the sweet spot: only +3 ms over 1-step but substantially more stable.

---

## 3. External Signals: What We Found

### 3.1 AHA-WAM Code Availability

**No public repository exists.** The project page (https://serene-sivy.github.io/aha-wam/) contains:
- Abstract, method summary, and animated figures
- RoboTwin and real-world results tables
- Supplementary videos
- BibTeX citation

There is **no "Code" link**, no GitHub organization, and no arXiv source bundle with training scripts. A targeted GitHub search for `"AHA-WAM"`, `"Observation-Guided Video-Context Routing"`, and author handles (`serene-sivy`) returned no relevant repositories.

**Implication:** All claims about training dynamics, hyperparameter sensitivity, and reproducibility must be taken from the paper alone. We cannot verify the OVCR implementation details (e.g., whether the gate α is per-query, per-layer, or shared), the exact rolling-memory attention pattern, or the distillation data pipeline.

### 3.2 Fast-WAM Baseline Inspection

Fast-WAM (the direct predecessor) **is open-source** (`code/fast-wam-2603.16666/`). Key findings from reading its model code:

- **Fast-WAM's `infer_action`** runs a single-frame video prefill for **every action chunk**, caching K/V and reusing them during action denoising. This is essentially a synchronous, single-horizon version of AHA-WAM's action path.
- **Attention mask:** Action tokens attend only to **first-frame video tokens** (`mask[action, :first_frame_tokens] = True`). No long-horizon planner context exists.
- **Latency:** 190 ms per chunk because the video prefill (even for one frame through a 5B DiT) is on the critical path every time.

AHA-WAM's speedup over Fast-WAM therefore comes from two factors:
1. **Amortization:** Running the 5B video DiT less often (low-frequency planner).
2. **System optimization:** TensorRT/CUDA Graphs on top of the amortized baseline.

The naive speedup from amortization alone is `h_v / h_a = 4×` in the best case; the measured 4.59× suggests the system optimizations contribute an additional ~15%.

> **Evidence preserved:** `external/fast-wam-comparison.md` — detailed architectural comparison with code snippets from the cloned Fast-WAM repository.

---

## 4. Critical Evaluation

### 4.1 Does Asynchronous Execution Preserve Performance?

**The ablations are telling (Table 2):**

| Variant | Clean | Randomized | Avg | Δ vs Fast-WAM |
|---------|-------|------------|-----|---------------|
| Fast-WAM | 91.88 | 91.78 | 91.83 | — |
| Naive-Async | 88.64 | 88.56 | **88.60** | **−3.23** |
| + KV Memory | 91.40 | 90.62 | 91.01 | −0.82 |
| + OVCR | 91.52 | 91.42 | **91.47** | **−0.36** |
| **Full AHA-WAM** | 93.40 | 92.20 | **92.80** | **+0.97** |

**Interpretation:**
- **Naive-Async fails badly.** Simply decoupling the branches and reusing stale planner context drops performance by 3.2 points. This confirms that amortization alone is insufficient.
- **Rolling K/V memory recovers most of the gap** (+2.41 points). Persistent planner states stabilize context across refreshes, but the effect is modest because RoboTwin tasks are short-to-medium horizon.
- **OVCR is the more direct remedy** (+0.46 points over memory alone, +2.87 over naive). Observation-conditioned adaptation directly addresses the stale-context problem.
- **The full model exceeds Fast-WAM** by 0.97 points. The authors attribute this to the long-horizon planner context enabling better scene understanding; an alternative interpretation is that OVCR acts as a beneficial regularizer.

**Open question:** The full model outperforms Fast-WAM, but the ablation shows that memory+OVCR together only match Fast-WAM (91.47 vs 91.83). The remaining +1.33 points to reach 92.80 are unexplained. Is there an interaction effect between asynchronous training and the full joint objective? A missing ablation row (e.g., "+ offset training without OVCR") would clarify this.

### 4.2 Real-World Results: Strong but Limited

**Setup:** 4 tasks on AgileX Piper bimanual arms, ~120 episodes each, pre-trained on RoboCOIN subset (24,600 trajectories, ~165 hours).

**Results:**
- AHA-WAM: 78.33% original-setting success
- Fast-WAM: 68.33%
- π0.5 (generalist VLA): 76.67%

**Strengths:**
- Outperforms WAM baselines decisively.
- Matches π0.5 without π0.5-scale generalist pretraining.
- Under generalization shifts (lighting, backgrounds), AHA-WAM is second only to π0.5 and achieves the highest progress score (35.00).

**Limitations:**
- **Only 4 tasks.** Real-world coverage is narrow: rigid placement, deformable manipulation, multi-object organization, tool use. No long-horizon compositional tasks, no mobile manipulation, no contact-rich assembly.
- **Pretraining on RoboCOIN.** The "no robot-data pretraining" claim in the abstract refers to RoboTwin training, but real-world deployment uses RoboCOIN pretraining. This is disclosed (Section 4.4) but easy to miss.
- **Small sample size:** ~120 episodes per task is modest by modern standards (e.g., π0.5 trains on tens of thousands of episodes). The gap to π0.5 might widen with more data.

### 4.3 Distillation Tradeoff

**AHA-WAM-Flash:** 2-step ODE-distilled sampler.

| Metric | AHA-WAM | AHA-WAM-Flash | Δ |
|--------|---------|---------------|---|
| RoboTwin avg | 92.80% | 90.20% | **−2.60** |
| Latency | 41.37 ms | 17.56 ms | −57.5% |
| Frequency | 24.17 Hz | 56.95 Hz | +2.35× |

The tradeoff is **steep but acceptable** for deployment: losing 2.6% success for 2.35× speedup is a reasonable Pareto frontier point. However, the paper does not report real-world Flash results — only simulation. It is unclear whether the 2-step sampler remains stable on physical hardware where action jitter has real consequences.

**Another open question:** The distillation teacher uses 16 steps (Table 5), but the baseline AHA-WAM uses 10 steps at inference (Section 4.1). Why distill from 16 rather than 10? A 10→2 distillation might incur a smaller accuracy drop.

### 4.4 The "No Robot-Data Pretraining" Claim

AHA-WAM achieves 92.80% on RoboTwin **without robot-data pretraining**, outperforming LingBot-VA (92.20%), which does use embodied pretraining. This is a genuine strength — the architecture itself is data-efficient.

However, the real-world experiments **do use RoboCOIN pretraining** (24,600 trajectories). The claim should be read as: "AHA-WAM reaches SOTA-level simulation performance without pretraining, and real-world deployment benefits from moderate pretraining."

### 4.5 Hardware-Specific Optimizations

All latency numbers are measured on an **NVIDIA RTX 5090D** — a consumer GPU not widely available. The optimizations (TensorRT 10.16, PyTorch 2.12, cu130) are cutting-edge and may not transfer cleanly to other hardware (e.g., A100, H100, or edge Jetson devices). The 10.82× speedup over Fast-WAM is therefore **hardware-and-software-configuration-specific**.

Fast-WAM's reported latency (190 ms) is from its paper/hardware; the AHA-WAM paper measures Fast-WAM on the same RTX 5090D for fair comparison in Table 3, but the actual numbers in Table 8 show AHA-WAM's PyTorch eager baseline is 415 ms — suggesting Fast-WAM's 190 ms may already include some optimization, or that AHA-WAM's eager path has additional overhead (OVCR, rolling memory).

---

## 5. What the Result Means for the Report

### Report-ready takeaways

1. **AHA-WAM is a principled architectural advance.** The temporal decoupling of world planning and action execution is well-motivated, and the ablations show that naive async fails while OVCR + memory recovers and exceeds synchronous baselines.

2. **OVCR is the core technical contribution.** It transforms planner-context reuse from static caching into observation-conditioned retrieval and adaptation. The design — routing queries, attention read, gated residual K/V update — is elegant and keeps visual feedback out of the action DiT's hot path.

3. **Speedup is real but layered.** The 4.59× over Fast-WAM comes from amortization (running 5B video DiT less often) plus system optimization (TensorRT/CUDA Graphs). The Flash variant's 10.82× adds ODE distillation on top. These are complementary and all preserve the same planner–executor interface.

4. **Evidence gaps exist.** No public code means we cannot independently verify training dynamics, OVCR implementation details, or distillation stability. The real-world evaluation is limited to 4 tasks with moderate data scale. The distillation tradeoff is only characterized in simulation.

5. **Fast-WAM context matters.** Reading Fast-WAM's code reveals that AHA-WAM generalizes an existing cached-inference pattern (single-frame prefill + K/V reuse) to a long-horizon, observation-adapted, phase-robust regime. The performance gain is therefore not just "make it faster" but "make it faster *and* give the action branch access to richer context."

### Recommended report emphasis
- Lead with the **temporal asymmetry insight** (world planning is inherently slower-scale than action execution).
- Explain OVCR with the reconstructed code as concrete evidence.
- Highlight the **ablation story**: naive async fails → memory helps → OVCR is decisive → full model exceeds baseline.
- Flag the **real-world limitations** (4 tasks, RoboCOIN pretraining) and the **code availability blocker**.
- Compare explicitly with Fast-WAM's architecture using the preserved code inspection.

---

## 6. Evidence Inventory

| File | What it is | How to rerun |
|------|-----------|--------------|
| `reconstruction/ovcr.py` | PyTorch reconstruction of Eqs. 8–10 | `python reconstruction/ovcr.py` |
| `reconstruction/layerwise_joint_attention.py` | PyTorch reconstruction of Eq. 3 | `python reconstruction/layerwise_joint_attention.py` |
| `reconstruction/horizon_adaptive_training.py` | PyTorch reconstruction of Eqs. 11–12 | `python reconstruction/horizon_adaptive_training.py` |
| `reconstruction/rolling_kv_memory.py` | PyTorch reconstruction of Eq. 13 | `python reconstruction/rolling_kv_memory.py` |
| `external/fast-wam-comparison.md` | Baseline code inspection notes | Read alongside `code/fast-wam-2603.16666/` |
| `assets/` | Symlinks to MinerU-extracted paper figures | Copied from `assets/aha-wam-2606.09811/` |

---

## 7. Key Findings and Uncertainties

### Findings
- AHA-WAM's dual-DiT async design is mechanically sound and well-supported by ablations.
- OVCR is the critical component: without it, async execution drops 3.2 points; with it, the gap closes to 0.36 and the full model surpasses the synchronous baseline.
- The inference optimization stack (TensorRT + CUDA Graphs + ODE distillation) is impressively engineered, yielding a 10.82× speedup for the Flash variant.
- Real-world robustness is promising but evaluated on a narrow task set.

### Uncertainties
- **No open code.** We cannot verify the exact OVCR gate mechanism, the rolling-memory attention pattern, or the distillation training code. The reconstructions are faithful to the paper's equations but may diverge from the authors' implementation.
- **Missing ablation:** What is the performance of "+ horizon-adaptive offset training only" (without OVCR or memory)? This would isolate whether offset training alone provides the remaining +1.33 points.
- **Distillation details:** Why distill from 16-step teacher when baseline inference uses 10 steps? What is the real-world stability of the 2-step sampler?
- **Generalization:** Does the asynchronous interface remain robust on tasks where the video planner horizon `h_v=64` is much shorter than the task duration? The paper notes this as future work.
- **Hardware portability:** Will the TensorRT/CUDA Graph optimizations transfer to datacenter GPUs or edge devices?

---

## 8. Blockers

**None for the investigation itself.** The paper is complete and the reconstructions are runnable. However:

- **Primary blocker for deeper verification:** No public AHA-WAM code repository. Independent reproduction of the exact training loop, OVCR internals, and distillation pipeline is impossible without author release.
- **Secondary blocker:** Real-world evaluation data (RoboCOIN subset, task-specific demonstrations) is not publicly released, so independent real-world replication is also blocked.
