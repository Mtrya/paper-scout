# Deep-dive: APT (2606.12366)

**Paper:** *APT: Action Expert Pretraining Improves Instruction Generalization of Vision-Language-Action Policies*  
**Authors:** Kechun Xu, Zhenjie Zhu, Anzhe Chen, Rong Xiong, Yue Wang  
**Repo:** https://github.com/xukechun/APT  
**Thread:** `runs/2026-06-16-gam-apt-dreamx/apt-2606.12366/`

## Research questions

1. How is the action expert actually implemented — gated fusion, two-stage layer expansion, attention masks, FiLM?
2. Is the Bayesian factorization a faithful implementation or a post-hoc motivation?  How do Stage 1 / Stage 2 manifest in code?
3. Can we instantiate the model without checkpoints, count parameters per stage, and trace a forward shape?
4. How does APT differ from π0.5 and BayesVLA?
5. What are the limitations and reproduction obstacles?

## Methodology

1. Cloned `xukechun/APT` into `code/APT/`.
2. Read the paper markdown, repo README, and the core model files:
   - `apt/action_expert.py` — diffusion action expert, masks, gated fusion
   - `apt/vla.py` — VLM + action expert wrapper, checkpoint expansion logic
   - `apt/vlm.py`, `apt/encoders/qwen3_vl.py` — Qwen3-VL bridge
   - `apt/train.py`, `apt/configs.py` — two-stage training recipe
3. Wrote two lightweight probes:
   - `code/inspect_apt_model.py` — instantiates `ActionExpert` for Stage 0 and Stage 1, counts parameters, and traces a dummy forward pass through `HybridAttentionLayers`.
   - `code/extract_action_expert_snippets.py` — pulls the key code blocks into `action_expert_snippets.md`.
4. Generated two report-facing figures in `runs/2026-06-16-gam-apt-dreamx/assets/`:
   - `apt_param_table.png`
   - `apt_architecture.png`
5. Compared with π0.5 / Knowledge Insulation and BayesVLA using the paper text and external signals.

## Key findings

### 1. Action-expert implementation

APT’s action expert is a **diffusion transformer** with block-wise causal self-attention over concatenated vision-language-action tokens.

- **Action tokens** = action history + proprioception + noisy future chunk (`a^hist`, `s^prop`, `a^noisy`).
- **FiLM conditioning:** the diffusion timestep is injected in every `AttentionBlock` via `AdaRMSNorm` on the action side (`SplitNorm`).
- **Positional encoding:** Stage-0 layers use **PRoPE** (camera-aware projective PE); Stage-1 inserts new layers with **mRoPE** for interleaved vision/language/action tokens.
- **Layer-wise gated fusion:** at every layer the VLM highway feature is merged with the action-expert VL stream by a learnable sigmoid gate:
  ```
  vl = vl * σ(w_i) + vl_highway_i * (1 - σ(w_i))
  ```
  This is exactly the paper’s Eq. 3, implemented in `HybridAttentionLayers.forward`.

### 2. The Bayesian factorization is a training recipe, not a separate inference objective

The paper writes

```
π(a | v, ℓ) ∝ π^p(a | v) · L(ℓ | v, a)
```

In the code this decomposition is realized structurally:

- **Stage 0 (`train_stage=0`)** — only the first `N/2` attention layers are built; language tokens are completely masked out of the attention (`prepare_attention_mask` returns a dilated mask where vision/action queries cannot attend to language keys). The VLM is frozen. The model is trained as a pure vision-action prior.
- **Stage 1 (`train_stage=1`)** — the network expands to `N` interleaved layers: even indices are new mRoPE layers, odd indices are the inherited PRoPE layers. The language mask is removed, and the full model is trained jointly.

So the factorization is **not** implemented as an energy-based posterior or two separate inference networks; it is a **two-stage initialization and masking recipe** inside one network. This is similar in spirit to BayesVLA, but differs in a crucial detail: **APT does not freeze the prior in Stage 2**; it jointly optimizes all `N` layers.

Checkpoint expansion is explicit in `VLA.load_from_pretrain(..., load_from_va=True)`: each Stage-0 layer `i` is copied into the **odd** Stage-1 layer `2*i+1`, while the even-index mRoPE layers keep their random initialization.

### 3. Parameter count / shape probe

Using `ActionExpert(hdim=768, num_heads=12, num_diffusion_layers=20)` — the base config from the paper:

| Component | Stage 0 params | Stage 1 params | Delta |
|---|---:|---:|---:|
| **Total** | **110.45 M** | **193.10 M** | **+74.8%** |
| Context encoder | 24.79 M | 24.79 M | — |
| Diffusion head | 85.66 M | 168.31 M | +96.5% |
| Traj-context attention | 82.67 M | 165.32 M | +100% |
| Layer gates | 0.02 M | 0.02 M | — |

The entire growth comes from doubling the attention blocks (`N/2 → N`).  The action expert is still tiny next to the Qwen3-VL-2B backbone.

A dummy forward pass through `HybridAttentionLayers` (Stage 1, `hdim=192`, `num_layers=8`) confirmed:

- input/output shape `(B, L_vl+L_a, C)`
- active attention layers = 8
- gate shape `(num_layers, hdim)` = `(8, 192)`
- even layers use rope, odd layers use PRoPE

See `inspect_apt_model_summary.json` for the raw numbers.

### 4. Comparison with neighbors

| | **π0.5** | **BayesVLA** | **APT** |
|---|---|---|---|
| Core remedy | Knowledge Insulation (stop gradient from action expert to VLM) + discrete FAST-token co-training | Bayesian factorization; **freeze** the VA prior in Stage 2 | Bayesian factorization; **pretrain** the action expert as VA prior, then **jointly** finetune prior + likelihood |
| Action expert init | Randomly initialized in post-training | Randomly initialized prior, then frozen likelihood adapter | Pretrained as VA prior; expanded inside the same stack |
| Language injection | Through standard VLM attention | Language-aware tokens refine a frozen prior | Layer-wise gated fusion + interleaved mRoPE layers |
| VLM training | Co-trained on VL reasoning data; backbone shielded by stop-gradient | Frozen or limited fine-tuning | Can be frozen, LoRA, or fully fine-tuned; a good prior makes joint VLM tuning helpful rather than harmful |
| Architecture | π-style | Pre/post-contact + foundation models | π-style, GR00T-style, or the proposed gated-fusion stack |

**What is genuinely new?**

- The **action-expert pretraining** perspective: the prior is not an auxiliary loss or a frozen adapter, but a *structured initialization* of the continuous action expert itself.
- The **gated fusion + interleaved layer expansion** design, which lets the model keep the prior and add language conditioning without a hard freeze.
- The empirical claim that **Knowledge Insulation is not necessary** if the action expert is pretrained — joint VLM tuning then helps rather than hurts.

**What is a training recipe?**

Much of the benefit comes from the two-stage data/masking schedule (vision-action pretraining → full VLA finetuning) rather than from a new probabilistic objective. The Bayesian equation motivates the recipe but is not itself a new loss.

### 5. Limitations and reproduction obstacles

From the paper:

- No explicit long-horizon memory → struggles on multi-step progress tracking.
- Evaluation is tabletop manipulation only; locomotion / mobile manipulation are unexplored.
- Sub-task termination failures remain (e.g., continued pushing after grasp, skipping “close the box”).

From the repo:

- **Compute:** the base model uses Qwen3-VL-2B-Instruct + a 768-dim diffusion action expert. Pretraining runs at batch size 256 for 100k iterations per stage.
- **Data:** the released configs expect DROID, AgiBotWorld-Alpha, InternData-A1, InternVLA-M1, LIBERO, and ALOHA HDF5 datasets. Paths are resolved via `data_utils/data_loc.py`, which must be edited for a new machine.
- **Checkpoints:** pretrained weights are released on Hugging Face (`KechunXu1/apt_models`), so the main blocker is obtaining the datasets and GPU resources rather than missing model weights.
- **Dependencies:** `xformers` / `deepspeed` are optional; DDP training works without them.

## Files produced

```
runs/2026-06-16-gam-apt-dreamx/apt-2606.12366/
├── README.md                           # this file
├── code/
│   ├── inspect_apt_model.py            # parameter counting + shape probe
│   ├── extract_action_expert_snippets.py
│   ├── generate_figures.py
│   ├── inspect_apt_model_summary.json
│   └── action_expert_snippets.md       # curated code excerpts
runs/2026-06-16-gam-apt-dreamx/assets/
├── apt_param_table.png
└── apt_architecture.png
```

`code/APT/` (the cloned upstream repo) is at the workspace root as requested.

## How to re-run the probes

Create a small virtualenv and install the three packages the probe actually uses:

```bash
python3.10 -m venv runs/2026-06-16-gam-apt-dreamx/apt-2606.12366/code/.venv
source runs/2026-06-16-gam-apt-dreamx/apt-2606.12366/code/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install einops diffusers matplotlib
```

Then:

```bash
python runs/2026-06-16-gam-apt-dreamx/apt-2606.12366/code/inspect_apt_model.py
python runs/2026-06-16-gam-apt-dreamx/apt-2606.12366/code/extract_action_expert_snippets.py
python runs/2026-06-16-gam-apt-dreamx/apt-2606.12366/code/generate_figures.py
```

The full training stack requires the rest of `code/APT/requirements.txt`, the Qwen3-VL weights, and the datasets listed above.

## Report takeaways

- APT’s central move is to **pretrain the continuous action expert as a vision-action prior** before any language signal is introduced. The code faithfully implements this through masking, layer expansion, and gated fusion.
- The Bayesian factorization is best understood as a **principled training recipe**: Stage 0 learns `π^p(a|v)`; Stage 1 learns a language-conditioned likelihood while keeping the prior in the same weights.
- Unlike BayesVLA, APT **does not freeze the prior** in Stage 2; unlike π0.5, it does not rely on Knowledge Insulation or discrete-token co-training. The evidence in the paper suggests that a pretrained prior is enough to make joint VLM/action-expert training beneficial.
- The architecture is modular enough to apply to π-style and GR00T-style stacks, but the core novelty is the **initialization and fusion recipe**, not a new generative objective.
- Reproduction is feasible because code and checkpoints are public, but it remains data- and compute-heavy (multi-embodiment pretraining, Qwen3-VL-2B, diffusion rollout).
