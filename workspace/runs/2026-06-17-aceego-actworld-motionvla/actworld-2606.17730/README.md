# ActWorld — Deep-Dive Notes (arXiv 2606.17730)

**Paper:** *ActWorld: From Explorable to Interactive World Model via Action-Aware Memory*  
**Authors:** Zhexiao Xiong, Yizhi Song, Hao Kang, Qing Yan, Liming Jiang, Jenson Yang, Zhoujie Fu, Stathi Fotiadis, Angtian Wang, Zichuan Liu, Bo Liu, Yiding Yang, Xin Lu, Nathan Jacobs (ByteDance / WashU)  
**Source Markdown:** `papers/world-models/actworld-2606.17730.md`  
**Project page:** https://interactwm.github.io/ActWorld  

---

## Research question

Can a single real-time, chunk-autoregressive video world model unify **flexible navigation** (WASD / mouse viewpoint control) with **mid-rollout object interaction** (pick up, open, pour, attach, etc.)? And if existing navigation-centric world models fail at this, is the bottleneck architecture, data, or memory?

ActWorld’s answer: the gap is mainly **data + memory**. Navigation-centric datasets lack dense interaction supervision, and recency-based history compression discards the sparse contact/manipulation frames that causally determine future object states — an *action-forgetting* pathology.

---

## Core mechanism

1. **100K interaction-dense dataset**  
   - 55K first-person + 45K third-person synthetic videos, 40 action verbs.
   - Each 33-frame chunk annotated offline by a VLM (GPT-5.4) with chain-of-thought reasoning, producing:
     - a per-chunk caption,
     - an interaction flag `y_int`,
     - a phase label `y_ph ∈ {approaching, reaching, contact, manipulating, completing, post-action}`,
     - a video-level action class `a`.

2. **Dual-branch camera conditioning**  
   - **Plücker-ray FiLM:** continuous per-pixel camera-motion modulation, applied only to current-chunk tokens.
   - **Symbolic text-camera branch:** 81 discrete (keyboard, mouse) command templates encoded by frozen UMT5 and concatenated with the chunk caption; dropout prevents shortcutting.

3. **Hierarchical action-aware memory** (the key novelty)
   - **EAFR — Event-Aware Frame Re-assignment:** replaces time-based short/mid/long buckets with an importance score
     ```
     w_k = λ_φ φ(y_int, y_ph) + λ_r exp(-(t-k)/τ)
     ```
     so contact/manipulating/completing frames can migrate into the fine-grained bucket even after long navigation gaps.
   - **ACHA — Action-Conditioned History Amplification:** the per-head history-key scaler is modulated by a learned action-class embedding, sharpening attention onto causally relevant history.
   - **Persistent action-aware memory bank:** a FIFO bank of up to 16 tokens prepended to the DiT self-attention input, containing:
     - *event tokens* fired at phase transitions (Enter-Manip, Enter-Complete, Release),
     - *object tokens* from DINOv3 patch features of the touched region.
     Contact/manipulating/completing tokens are pinned; the bank survives the latent buffer’s eviction horizon.

4. **Real-time distillation**  
   - Initializes from a Helios 14B chunk-autoregressive checkpoint.
   - Multi-resolution flow matching + DMD-style GAN distillation reduces sampling to 3 steps per 33-frame chunk.

---

## Key results

Evaluated on the authors’ new **I-Bench** benchmark (300 prompts, 30 sequences of 10 prompts each, mixing 40 action verbs with camera primitives).

| Axis | Headline | Context |
|------|----------|---------|
| **VLM-AJ success rate** | **57.8%** | More than doubles the next best baseline (Yume 1.5: 20.1%; LingBot-World: 19.9%). Mean IF 2.557 vs. 1.638 for Yume. |
| **VBench overall** | Best or near-best on SC/BC/MS/AQ/IQ/DD/TF/OC/i2v-S/i2v-B | Quality and consistency do not collapse when interaction is added. |
| **KMF joint accuracy** | **20.62%** | Beats Matrix-Game 3 (20.00%) on full key+mouse following; strong mouse accuracy (43.67%). |
| **User study** | 1st on action following (4.05 / 5), key/mouse following (3.69), overall quality (3.92) | Largest margin is on action following. |
| **Ablation** | `+EventMem` gives the biggest single jump | Subject consistency 0.844 → 0.871; Level-3 success 54.0% → 57.8%. |

Figures preserved locally:
- `../assets/actworld_fig1_teaser.jpg` — navigation + interaction rollouts under WASD/arrow control
- `../assets/actworld_fig2_pipeline.jpg` — full pipeline (EAFR + persistent bank)
- `../assets/actworld_fig3_rollouts.jpg` — qualitative rollouts
- `../assets/actworld_fig4_comparison.jpg` — long-horizon interaction comparison vs. baselines
- `../assets/actworld_fig5_data_pipeline.jpg` — offline data-generation pipeline
- `../assets/actworld_fig6_more_results.jpg` — additional results

---

## Close neighbors and what is new

| Neighbor | Relationship to ActWorld |
|----------|--------------------------|
| **DreamX-World 1.0** (2606.16993, previous run) | Real-time Wan2.2-based interactive world model with camera/projector conditioning and inference code release. Strong on navigation and event prompting, but its public snapshot only implements a local rolling KV cache + sink tokens; it does not target mid-rollout object manipulation with explicit action/phase labels. ActWorld goes further on the interaction axis and provides a clear memory mechanism for it. |
| **HY-WorldPlay / HY-World 1.5** | Tencent’s streaming diffusion transformer with dual keyboard+mouse action representation and reconstituted context memory for geometric consistency. Open weights and code, real-time at 720p. Action vocabulary is camera/keyboard; object-level manipulation is not the focus. ActWorld’s 40-verb interaction taxonomy and action-aware memory are the differentiators. |
| **LingBot-World** | Open-source general-domain world model, camera-pose control, long horizon, high dynamic degree. Again navigation-centric; no dense object-interaction labels or memory routing. |
| **PAN** (cited baseline) | Language-conditioned manipulation world model, but offline / prompt-to-video, not real-time interactive. |
| **Matrix-Game 3.0** | Real-time streaming world model with long-horizon memory, but interactions are largely game-specific (block breaking). |

The genuinely new part is not “generate video from actions” — that exists. It is the **joint support for fine-grained object interaction + camera navigation in one real-time model**, enabled by a dataset with per-chunk interaction phases and a memory design that routes compression by interaction importance rather than time.

---

## Self-contained probe

`code/probe_memory.py` simulates the action-forgetting idea on a synthetic long rollout with no model weights or downloads.

What it does:
- Builds a 50-chunk trajectory: long navigation → interaction event (`approaching → contact → manipulating → completing`) → long navigation → revisit.
- Compares three history strategies:
  1. **Recency baseline** — fine bucket holds the most recent chunks.
  2. **EAFR** — fine bucket holds the highest-importance chunks (contact/manipulating/completing).
  3. **EAFR + persistent memory bank** — adds pinned event/object tokens that survive the gap.
- Measures how well each strategy can reconstruct the object state at the revisit chunk.

Run it:
```bash
cd runs/2026-06-17-aceego-actworld-motionvla/actworld-2606.17730/code
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python probe_memory.py
```

Outputs (saved to `../assets/`):
- `probe_trajectory.png` — phase strip, object state, EAFR weights, and bucket assignments for one rollout.
- `probe_gap_sweep.png` — prediction MSE vs. navigation gap length, averaged over many random seeds.
- `probe_metrics.json` — numeric snapshot.

Example single-trajectory predictions (true object state at revisit = 1.0):
```
recency     : 0.290   # forgets the causal event
eafr        : 0.987   # keeps contact/manipulating frames in fine memory
eafr+bank   : 0.970   # pinned event/object tokens survive the gap
```

---

## Limitations and blockers

- **No public artifacts yet.** The project page has placeholder `#` links for arXiv, code, and GitHub as of this scrape. Only the PDF (`static/pdfs/actworld.pdf`) and comparison videos are hosted.
- **Proprietary dataset.** The 100K interaction videos and GPT-5.4 annotations are not released; reproduction requires rebuilding both the data engine and the annotation pipeline.
- **Self-designed benchmark.** I-Bench is author-constructed and judged by a VLM, so the absolute numbers are best read as a relative signal among the evaluated systems.
- **Small bank capacity.** The persistent memory bank is only 16 tokens. It is unclear how it scales to multi-object scenes or long interaction chains with many event transitions.
- **DINOv3 overhead.** Online object-anchor extraction requires VAE-decoding each generated chunk and running DINOv3; gated to interaction chunks, but still adds ~15% average cost.
- **Label noise.** Phase labels come from a VLM; mismatches between the video-level action label and observed motion are flagged but not corrected automatically.
- **Real-world coverage.** Only 400 hours of real data, mostly walking/game footage; the bulk of interaction supervision is synthetic.
- **Open sim-to-real question.** The paper does not test whether learned object dynamics transfer to unseen real objects or physics.

---

## What this run learned

1. **Action-forgetting is a concrete failure mode, not just a capacity problem.** The diagnosis in §3.3 is precise: recency-based history compression degrades the very frames that determine object state, and simply enlarging the buffer does not fix it because those frames stay coarsely patchified.

2. **The fix is two-headed.** Better data (dense per-chunk phase labels) and better memory (EAFR + ACHA + persistent bank) are both necessary. The ablation shows each component helps, but the persistent bank gives the largest single gain, confirming that symbolic event/object anchors are the heavy lifter for long-horizon object identity.

3. **Real-time unified navigation + interaction is now empirically feasible.** ActWorld’s numbers on I-Bench are a clear step beyond navigation-centric baselines without sacrificing viewpoint control. That makes AI-generated interactive gameplay the nearest downstream target, with embodied planning and content creation behind it.

4. **The reproducibility gap is wide.** As with DreamX-World, the strongest world-model papers this cycle ship demos and project pages but not the full training stack. Until code/weights/data appear, the headline numbers are a strong signal, not a verifiable recipe.

---

## Files in this thread

```
runs/2026-06-17-aceego-actworld-motionvla/
├── actworld-2606.17730/
│   ├── README.md                  # this file
│   ├── code/
│   │   ├── probe_memory.py        # synthetic action-aware-memory probe
│   │   └── requirements.txt       # numpy + matplotlib
│   └── patches/                   # reserved for future diffs
└── assets/
    ├── actworld_fig1_teaser.jpg
    ├── actworld_fig2_pipeline.jpg
    ├── actworld_fig3_rollouts.jpg
    ├── actworld_fig4_comparison.jpg
    ├── actworld_fig5_data_pipeline.jpg
    ├── actworld_fig6_more_results.jpg
    ├── probe_trajectory.png
    ├── probe_gap_sweep.png
    └── probe_metrics.json
```
