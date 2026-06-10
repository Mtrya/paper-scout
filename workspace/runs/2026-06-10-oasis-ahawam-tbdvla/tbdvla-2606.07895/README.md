# TBD-VLA Deep Investigation

**Paper:** TBD-VLA: Temporal Block Diffusion Vision Language Action Model  
**arXiv:** 2606.07895  
**Authors:** Sung-Wook Lee, Xuhui Kang, Yen-Ling Kuo (University of Virginia)  
**Investigation date:** 2026-06-10  
**Thread:** `runs/2026-06-10-oasis-ahawam-tbdvla/tbdvla-2606.07895/`

---

## What was attempted

1. **Read the full paper** including appendices (training details, hyperparameters, real-world evaluation protocol).
2. **Search for official code.** The project page (`https://tbd-vla.github.io/`) does not link to a public repository. A GitHub org `TBD-VLA` exists but appears empty/private. The paper states training uses the LeRobot framework [39].
3. **Triangulate with open-source neighbors.** We cloned **UD-VLA** (`OpenHelix-Team/UD-VLA`, ICLR 2026) to `code/ud-vla-reference/` because it implements blockwise attention masks (`build_blockwise_attn_mask`, `build_blockwise_causal_attn_mask`) and discrete diffusion training with `--use_blockwise_attn_mask` and `--mask_image` flags. TBD-VLA cites UD-VLA as a related baseline and shares the same underlying technique lineage (block diffusion from Arriola et al. [8] and Fast-dLLM v2 [9]).
4. **Write compact PyTorch reconstructions** of the three mechanisms the paper leaves semi-formal:
   - `reconstruction_training.py` — doubled-layout attention mask + block diffusion training loop
   - `reconstruction_inference.py` — expectation sampling decoder + Real-Time Chunking (RTC) logic
5. **Compare with related discrete-diffusion VLAs** and critically evaluate empirical claims.

---

## Evidence preserved and how to rerun it

| Artifact | Location | What it is |
|----------|----------|------------|
| Full paper markdown | `papers/vla/tbd-vla-2606.07895.md` | MinerU-extracted text & tables |
| Paper figures | `assets/tbd-vla-2606.07895/` | 28 images extracted by MinerU |
| External reference code | `code/ud-vla-reference/` | UD-VLA repo (blockwise attn implementation) |
| Training reconstruction | `runs/.../tbdvla-2606.07895/reconstruction_training.py` | Doubled-layout mask + forward process |
| Inference reconstruction | `runs/.../tbdvla-2606.07895/reconstruction_inference.py` | Expectation sampling + RTC in-painting |

### Rerunning the reconstructions

```bash
cd runs/2026-06-10-oasis-ahawam-tbdvla/tbdvla-2606.07895
python reconstruction_training.py   # sanity-checks attention mask shapes & no-leak property
python reconstruction_inference.py  # sanity-checks expectation sampling value range
```

No model weights are required; the scripts are self-contained sketches.

---

## What TBD-VLA does — mechanism reconstruction

### 1. Block diffusion forward / reverse process (Eq. 1)

Action chunk $A_t$ of length $H_p \cdot D_a$ is partitioned into $K = H_p/m$ temporal blocks ($m=4$ in all experiments). During training, each clean block $x_k^0$ is independently noised into $x_k^t$ by masking each token with probability $t_{k,i} \sim \mathcal{U}(0,1)$. The reverse process predicts the clean block conditioned on $z_k$, where $z_0$ is an all-MASK anchor block and $z_{k>0}$ contains the clean preceding blocks. The loss is cross-entropy averaged **only over masked positions**.

**Key detail:** The paper does *not* use a scheduled diffusion timestep $t$ shared across the block. Each token gets its own independent masking probability. This is exactly the "simple and effective masked diffusion" setup of Sahoo et al. [13] (MDLM-style), not a D3PM categorical transition matrix.

### 2. Doubled-layout attention mask and training efficiency

The training bottleneck in naive block diffusion is that each block $k$ needs a separate forward pass (you must feed the prefix + clean blocks $0..k-1$ + noised block $k$ to predict clean block $k$). For $K=4$ blocks, that is 4 passes per sample.

TBD-VLA uses the **doubled-layout trick** from Block Diffusion [8] / Fast-dLLM v2 [9]:

- The input sequence is laid out as:  
  `[prefix] [clean_0] [noised_0] [clean_1] [noised_1] ... [clean_{K-1}] [noised_{K-1}]`
- **RoPE positions are shared** between `clean_k` and `noised_k` (they represent the same temporal slot).
- The custom attention mask enforces:
  - `noised_k` can attend to prefix + clean blocks $<k$
  - `clean_k` can attend to prefix + clean blocks $<k$ + `noised_k`
  - standard causal mask on the prefix

This packs all $K$ block-level training signals into **a single forward pass**, which is where the training speedup comes from. Our `reconstruction_training.py` builds this mask explicitly and verifies that `noised_1` cannot see `clean_1` (no information leakage) while `clean_1` can see `noised_1` (as required for denoising).

### 3. Temporal-level token shift

Discrete diffusion is naturally a *self-reconstruction* objective (predict the clean token from a noised version of itself). Autoregressive VLMs are pretrained on *next-token prediction*. TBD-VLA bridges the two by shifting at the **temporal block level**: the logits for the current action block are generated from the *prior* block's position in the sequence. In the doubled layout, this means `clean_k` is placed *after* `noised_{k-1}` (and after the clean previous blocks), so the model must autoregressively "step forward" one block at a time. This preserves the causal inductive bias of the Qwen3-VL backbone.

### 4. Real-Time Chunking (RTC) — hard in-painting

RTC is the paper's most practically interesting idea. During closed-loop control, inference latency means the robot has no new actions to execute for $d$ timesteps. TBD-VLA handles this by **freezing the tail** of the previously generated action chunk and re-using it as an in-painting prefix for the next generation cycle.

Specifically, each cycle decodes $H_a + d$ timesteps: the first $d$ are the frozen tail, the remaining $H_a$ are newly generated. Because the model was trained with masked block diffusion (it routinely sees partial action context and must complete the block), the in-painting is natively aligned with the training objective. This is conceptually clean and distinct from simply re-planning from scratch.

Our `reconstruction_inference.py` sketches this by converting the committed tail to tokens, appending them to `previous_blocks`, and decoding the remainder with the standard block diffusion decoder.

### 5. Expectation sampling vs argmax

After the discrete tokens are decoded, they must be mapped back to continuous actions. TBD-VLA proposes **expectation sampling**:

$$a_{t+h,j} = \sum_{x \in \mathcal{V}} p_{\theta,h,j}(x) \, c_j(x)$$

where $c_j(x)$ is the bin center for action dimension $j$. This uses the full predicted distribution rather than the single most-likely bin. The ablation (Table 5) shows this is worth **+7.1 points** on SimplerEnv Google Robot compared to argmax. That is a large gain for a seemingly small decoding change, suggesting the token distributions are genuinely multimodal and the argmax throws away useful uncertainty information.

---

## External signals: what supports or challenges the claims

### Code availability

- **TBD-VLA:** No public repository as of 2026-06-10. The project page has no "Code" link. The paper says "We use the LeRobot framework [39] for TBD-VLA training and policy deployment." We preserved UD-VLA as the closest open-source implementation of blockwise discrete diffusion for VLA.
- **UD-VLA** (`OpenHelix-Team/UD-VLA`): Open source, implements `build_blockwise_attn_mask`, joint image-action denoising (`--with_i_ia True`, `--mask_image True`), and trains on CALVIN / LIBERO / SimplerEnv. Its attention mask code confirms the feasibility of the doubled-layout trick.
- **Fast-dVLA** (Song et al., arXiv 2603.25661): Discusses block diffusion for KV-cache reuse in discrete diffusion VLAs. Their Figure 2(c) explicitly illustrates the block-wise decoding behavior that TBD-VLA exploits.
- **Discrete Diffusion VLA** (Liang et al., ICLR 2026): Claims it will open-source but no repository found yet. Uses fully parallel decoding (no temporal AR), achieving 0.069s latency vs TBD-VLA's 0.117s (Table 1). TBD-VLA trades some raw speed for temporal structure.

### The SOTA claim on LIBERO (97.7% average)

**Support:** TBD-VLA does achieve the highest reported average on LIBERO among the listed baselines (Table 2). The margin over the previous best ($\pi_{0.5}$ at 96.9%) is 0.8 points — real but modest.

**Challenges / nuances:**
1. **Model size asymmetry.** TBD-VLA is 2B parameters vs. $\pi_{0.5}$ (3B), OpenVLA (7B), Discrete Diffusion VLA (7B). The smaller model winning suggests the architecture is efficient, but it also means the comparison is not apples-to-apples in capacity.
2. **SimplerEnv Widow-X:** TBD-VLA scores 66.8%, which is **second** to UniVLA (69.8%). It is not universally SOTA.
3. **SimplerEnv Google Robot:** TBD-VLA does lead at 91.0% (VM) / 86.3% (VA), but this is a fine-tuning-heavy benchmark where data mixture and fine-tuning steps matter as much as architecture.

### The RTC benefit: +20.9 pp at latency L=4

**Support:** Figure 4 and Table 11 show that without RTC, TBD-VLA collapses from 97.7% to 72.3% at $L=4$; with RTC it stays at 93.2%. The absolute gain is large and consistent across suites.

**Critical reading:**
1. **Is the baseline fair?** The paper compares TBD-VLA w/ RTC against $\pi_{0.5}$ w/ RTC. $\pi_{0.5}$ is a continuous flow-matching model. RTC for flow-matching policies (Black et al., NeurIPS 2026 [10]) is essentially action-replay of the tail; it does not involve in-painting because flow matching is not trained for partial-context completion. TBD-VLA's training-inference alignment *is* a genuine architectural advantage here — but the comparison tells us more about "diffusion models handle in-painting better than flow models" than about an absolute upper bound.
2. **The +20.9 pp is vs. TBD-VLA *without* RTC**, not vs. $\pi_{0.5}$. The main text says "TBD-VLA with RTC retains 93.2% success rate, which is 3.4% higher than $\pi_{0.5}$ with RTC." So the architecture-specific RTC boost is +20.9 pp, while the cross-method RTC boost is only +3.4 pp. The paper's phrasing could easily be misread as TBD-VLA beating the baseline by 20.9 pp, which is not what the cross-method comparison shows.
3. **RTC is only as good as the latency estimate.** The real-world protocol (Appendix C.3) rounds a measured 0.119s latency to 2 timesteps at 15 FPS. If the real latency varies, the hard in-painting could mis-align.

### Real-world results: only 3 tasks, 50 demos each, FR3 robot

**Support:** TBD-VLA beats $\pi_{0.5}$ (67.1% vs. 50.0% average across perturbations). The tasks are challenging (long-horizon, dexterity, liquid transfer). RTC improves TBD-VLA by +7.08 pp in this setting.

**Challenges:**
1. **Sample size is tiny.** 50 demos per task is standard for tabletop manipulation but means the fine-tuned policy is highly data-limited. The pre-training on 160K episodes likely dominates performance.
2. **Only 3 tasks.** Generalization claims beyond these tasks are unsupported.
3. **Camera viewpoint failure mode.** The qualitative results (Figure 9, Appendix D) show TBD-VLA fails completely on "Transfer the Liquid" under a modified camera viewpoint. The authors attribute this to "under-representation of similar types of tasks in the pre-training dataset." This is honest but also highlights that the architecture is not magic — data coverage matters.

### The block size tradeoff: m=4

Table 5 is the most architecturally informative ablation:

| Configuration | Success Rate | Inference Time | Forward Passes |
|---------------|-------------|----------------|----------------|
| m=1 (fully AR) | 84.6 (-4.1) | 0.223s | 16 |
| m=16 (full diffusion) | 84.0 (-4.7) | 0.061s | 2 |
| m=4, n_d=1 | 85.7 (-3.0) | 0.060s | 2 |
| **m=4, n_d=2, Expectation** | **88.7** | **0.086s** | **4** |

**Interpretation:**
- $m=1$ recovers standard autoregressive decoding. It is **slower and worse** than the chosen config, confirming that parallel decoding within blocks helps both speed and quality.
- $m=16$ (one big diffusion block, no temporal AR) is the fastest but the **worst performing**. This confirms that purely parallel decoding without temporal structure hurts policy quality.
- $n_d=1$ (single denoising step per block) degrades performance by 3.0 points. With $n_d=2$ the model has enough iterative refinement to recover.
- The chosen $m=4$ sits at a sweet spot: it gets most of the speed benefit of parallel decoding while preserving enough temporal autoregression to model dependencies across time.

**Open question:** The ablation is only on SimplerEnv Google Robot. Does the same $m=4$ sweet spot hold for LIBERO-Long or real-world tasks? The paper does not vary $m$ on those benchmarks.

### LIBERO-Plus robustness: the pre-training confound

The paper claims 83.5% average on LIBERO-Plus, "outperforming the second best method by 15.1%." The project page shows the breakdown:

| Model | Avg |
|-------|-----|
| TBD-VLA (w/ Pre-train) | **83.5** |
| RIPT-VLA | 68.4 |
| OpenVLA-OFT | 67.9 |
| TBD-VLA (w/o Pre-train) | 66.2 |

**Critical finding:** TBD-VLA **without pre-training scores 66.2%**, which is *below* RIPT-VLA (68.4%) and OpenVLA-OFT (67.9%). The massive margin comes almost entirely from large-scale pre-training (160K episodes, 32M samples), not from the block-diffusion architecture per se. The architecture enables efficient training and RTC, but the raw robustness numbers are driven by data scale. The paper does not clearly disentangle architecture vs. data contributions on LIBERO-Plus.

---

## Key findings for the report

1. **Genuinely new:** TBD-VLA is the first discrete-diffusion VLA to combine *temporal* block autoregression with intra-block parallel decoding. Prior work (Discrete Diffusion VLA, UD-VLA, LLaDA-VLA) uses either fully parallel decoding across the whole action chunk or joint image-action denoising, but not temporal blocks. The $m=4$ sweet spot is a concrete empirical contribution.

2. **RTC is the strongest practical claim:** Hard in-painting aligns training and inference in a way that flow-matching baselines cannot easily replicate. The +20.9 pp self-boost is large; the cross-method boost (+3.4 pp vs. $\pi_{0.5}$ w/ RTC) is more modest but still meaningful. RTC is the clearest "buildable" idea from this paper.

3. **Expectation sampling matters more than expected:** +7.1 pp from replacing argmax with a weighted average over bin centers is a striking result. It suggests the discrete token distributions retain useful uncertainty that standard decoding discards.

4. **Doubled-layout attention is a training-engineering win, not a conceptual novelty:** The trick comes from Arriola et al. [8] and Wu et al. [9]. TBD-VLA's contribution is applying it to VLA action blocks with shared RoPE positions. The reconstruction in `reconstruction_training.py` shows the mask is straightforward to implement once the layout is understood.

5. **SOTA claims need careful framing:** LIBERO numbers are strong but the margin is small (0.8 pp). LIBERO-Plus robustness is dominated by pre-training scale. Real-world evaluation is minimal (3 tasks, 50 demos). The paper is honest about limitations but the headline numbers could be misread as pure architecture wins.

6. **Reproducibility gap:** No code released as of investigation date. The paper says it uses LeRobot, but without configs or model definitions, independent replication is hard. UD-VLA and Fast-dVLA provide the closest public reference implementations.

---

## Uncertainties and follow-up questions

- **What is the exact RoPE sharing mechanism?** The paper says clean and noised blocks "share the same RoPE positions." In practice this likely means the position IDs for `clean_k` and `noised_k` are identical, while the causal layout enforces temporal ordering. We reconstructed this as identical position offsets, but the precise indexing in Qwen3-VL's multimodal RoPE is not specified.
- **Does RTC work under variable latency?** The real-world experiments assume a fixed latency compensation of 2 timesteps. In deployment, GPU scheduling jitter could make the actual latency 1.5 or 2.5 steps. How robust is hard in-painting to this mismatch?
- **How does $m$ interact with action dimension?** The paper uses $D_a=7$ (6-DOF + gripper). For higher-DoF platforms (e.g. bimanual, dexterous hands), $m=4$ might need to scale with $D_a$ to keep the total tokens per block manageable.
- **Comparison with BlockVLA (arXiv 2605.13382):** A very recent concurrent work also applies block diffusion to VLA but via *fine-tuning* autoregressive VLAs into block-diffusion decoders. TBD-VLA trains from scratch (or pre-trains) with the block objective. Which approach is more sample-efficient? The literature has not converged.

---

## Method checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Read full paper + appendices | ✅ | `papers/vla/tbd-vla-2606.07895.md` (364 lines) |
| Search for official code | ✅ | No public repo found; GitHub org empty |
| Preserve external code if found | ✅ | `code/ud-vla-reference/` (UD-VLA blockwise attn) |
| Reconstruct core mechanism | ✅ | `reconstruction_training.py`, `reconstruction_inference.py` |
| Compare related work | ✅ | Discrete Diffusion VLA, UD-VLA, Fast-dVLA, LLaDA-VLA triangulated |
| Critical evaluation | ✅ | See sections above on SOTA, RTC, real-world, block size, pre-training confound |
| Write README | ✅ | This file |
