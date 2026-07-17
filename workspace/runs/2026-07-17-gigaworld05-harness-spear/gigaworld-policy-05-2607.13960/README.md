# Deep-dive: GigaWorld-Policy-0.5 — the action-centered WAM, verified in code and weights

**Thread id:** `gigaworld-policy-05-2607.13960`
**Paper:** GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch (arXiv 2607.13960, GigaAI + Tsinghua)
**Code:** https://github.com/open-gigaai/giga-world-policy (single repo covers v0 and v0.5; README already rebranded to 0.5)
**Weights:** https://huggingface.co/open-gigaai/Giga-World-Policy-0.5 (transformer only, fp32, 3 shards, 24.086 GB; needs Wan2.2-TI2V-5B VAE/T5)
**Project page:** https://open-gigaai.github.io/giga-world-policy/
**Local clone (scratch):** `code/giga-world-policy/` (shallow, 7.2 MB of code; LFS skipped)

## What I attempted

Planned as a read-and-probe; it became a code-and-weights verification because **both shipped on 2026-07-16**, one day after the arXiv posting. I cloned the repo, read the MoT model/trainer/inference implementation end-to-end, pulled the released checkpoint's safetensors *headers* via HTTP range requests (no 24 GB download) to validate parameter counts to the byte, reconstructed the causal mask in numpy and proved the inference-time claim exact, built a FLOP model that reproduces the paper's latency ratios, and checked the neighborhood (FastWAM arXiv 2603.16666, π0.5, GigaWorld-Policy v1 config in the same repo).

## Key findings (evidence-backed)

### 1. The action-centered causal mask — exact mechanism, verified in code

`world_action_model/models/transformer_wa_casual_mot.py :: build_mot_attention_mask()`. Sequence layout `[state | action | ref | future]` (action stream = state+action → action expert; visual stream = ref+future → visual expert). Attention allow-matrix:

| query \ key | state | action | ref | future |
|---|---|---|---|---|
| state  | ✓ | ✗ | ✓ | ✗ |
| action | ✓ | ✓ | ✓ | **✗** |
| ref    | ✓ | ✗ | ✓ | ✗ |
| future | ✓ | ✓ | ✓ | ✓ (all) |

Same semantics as GigaWorld-Policy-0's dense model (`transformer_wa_casual.py`, interleaved layout). `code/probe_mask_reconstruction.py` reimplements it and shows numerically that **deleting the future block leaves every other token's attention output bit-identical (diff ~2e-16)**: the future stream is a pure information sink — it reads action tokens (so the future-video loss backprops into action representations = the "dense supervision"), and nothing else reads it. That is the precise reason future-video generation is *exactly* optional at inference, not approximately. Counterfactual: if action→future attention leaked, dropping future would shift action outputs (probe shows 0.26 max change).

Token budget (from config + Wan2.2 VAE: 16× spatial / 4× temporal, z=48): 320×384 composite frame → 20×24 latent → patch (1,2,2) → **120 tokens/latent frame**. Training samples 5 RGB frames at t=[0,12,24,36,48] → 1 ref + 1 future latent frame (the 4 future frames at Δ=12 collapse into one latent frame). Full training sequence = 1 state + 120 ref + 48 action + 120 future = **289 tokens**. Action horizon 48, action dim 16, 1 state token.

### 2. MoT: what is shared vs split, validated against the checkpoint itself

Per MoT block (`WanMoTTransformerBlock`): **self-attention is joint** — each expert computes its own Q/K/V, then Q/K/V are concatenated across streams and one masked attention runs over both. **Everything else is per-expert**: QKV/out projections, cross-attention to text (each expert has its *own* T5 projection: 4096→3072 visual, 4096→1024 action), FFN, adaLN tables. Subtle detail: the action expert's attention projects to the *same* inner dim as the visual expert (24 heads × 128 = 3072) so K/V concatenate — it is lightweight via width (1024) and FFN (4096 vs 14336), not via head geometry.

Parameter counts pulled from the released safetensors headers (1664 tensors, all F32, total **6.0216 B params = 24.086 GB — exact match** with HF `total_size`): per visual block 163.6 M (attn 37.8 + cross 37.8 + ffn 88.1), per action block 33.6 M (12.6 + 12.6 + 8.4), ×30 layers → **visual expert 4.91 B (81.5%), action expert 1.01 B (16.7%)**, embeddings ~103 M. During action-only inference the per-step active params are just the 1.0 B action expert (the visual expert runs once per chunk on the 121-token prefix, then its K/V is cached). Init: visual expert from GigaWorld-1 (Wan2.2-TI2V-5B lineage); action expert copy-initialized from visual weights with leading-slice truncation (`_copy_param_sliced`).

### 3. FLOP model reproduces the latency story

`code/probe_flops_params.py` (per 48-action chunk, 10 flow-matching steps, no CFG — `num_inference_steps=10`, `guidance_scale=0.0` in `scripts/inference_openloop.py`):

| scenario | TFLOP/chunk | paper latency (4090 / A100) |
|---|---|---|
| GWP-0.5 action-only (deployed) | 2.18 | 110 / 189 ms (+C++: 85 / 140 ms) |
| GWP-0 dense action-only | 5.93 (**2.72×**) | 293 / 360 ms (**2.66× / 1.90×**) |
| GWP-0.5 full joint WAM | 24.85 (**11.4×**) | — (Motus: 3231 ms A100 = 17×) |

The 2.72× model prediction vs 2.66× measured on the 4090 is a strong consistency check. Fairness caveats on Table 4: (a) GWP-0.5 numbers are with KV cache + `torch.compile(reduce-overhead)`; the text says FastWAM is "under the same torch-compiled setting" but π0.5's measurement conditions are unspecified; (b) the headline "85 ms, 23% faster than π0.5, 53% faster than FastWAM" compares the **C++ runtime** of GWP-0.5 against **Python** baseline numbers — the C++ row has no baseline counterparts; (c) FastWAM's own paper reports 190 ms on its own setup, so cross-paper latency is not apples-to-apples; (d) Motus has no 4090 number. The 85 ms chunk latency ≈ 11.8 Hz full-chunk replan rate (replan default 30 of 48 steps in eval).

### 4. AutoResearch is Karpathy's loop, and the paper's use is genuine but shallow

`[karpathy2026autoresearch]` = Karpathy's `autoresearch` repo (March 2026, ~66 k stars): a coding agent iteratively edits a training script, runs bounded pilots, keeps changes that improve the metric ("observe → propose → pilot → keep/discard → extend"). GigaWorld's instantiation: 1K-step pilot sweep on pick-the-fruit (930 episodes; 300 train / 30 val), then promotion to long runs. Figure 7 shows real loop semantics, not a plain grid: a gray "observe (baseline)" bar at **lr=4.3e-5** (this resolves Table 5's odd 4.316e-5 — it's the *inherited baseline*, not an agent-invented value), agent-proposed candidates lr ∈ {3e-5, 6e-5, 8e-5}, bs ∈ {12, 8}, frames=64 (abandoned: "stopped early, slow steps"), then extension 10k→30k→50k with keep/discard (eval action MSE 5.07e-4 → **1.13e-4 @30k, kept** → 1.73e-4 @50k, discarded as regression). What it found: **lr 6e-5** — best train action loss 0.252476 and eval action MSE 0.409764, while 3e-5 had the best *visual* loss (0.172330) but worse action MSE (0.416832) — a genuine multi-objective tradeoff the agent resolved by prioritizing action quality. Batch size 16 kept. The shipped finetune config uses exactly lr=6e-5, constant schedule, bs=16, 50k max steps — the agent's recipe is what got released. One unresolved discrepancy: pilot eval action MSE (~0.41, Table 5 / Fig 7 Panel A) and long-run eval action MSE (~1e-4, Panel B) differ by ~4000×; the paper never explains the rescaling (different normalization or eval protocol between the pilot and long-run harnesses — flagged, not fatal). Honest read: real agent loop with keep/discard semantics, but the demonstrated search is a handful of single-factor probes around a strong baseline — closer to disciplined pilot sweeping than deep autonomous research. It also never mutates code or architecture, only hyperparameters, and the multi-objective conflict is resolved by a hard-coded preference (action over visual), not by Pareto reasoning.

### 5. Training recipe confirmed in code

Flow matching with per-modality shifted timesteps (paper Eq. 4): `visual_flow_shift=2.0`, `action_flow_shift=5.0` (GWP-0 used a single shift 5.0 for both — an under-appreciated 0.5 change). Per-token timesteps (`expand_timesteps=True`): state/ref tokens get τ=0 (clean condition), action/future get their own τ. Loss = `visual_loss + action_loss` **summed 1:1** (giga-train `sum(losses.values())`); visual loss masked to future frames only. Target = noise − data (rectified-flow convention). Optimizer CAME8Bit, EMA, bf16. Pretraining: 2K hours robot data with mixed AC-WM + WAM batches; post-training on target-robot demos. Mixed AC-WM ablation (Fig. 6) claims mixed pretraining reaches 0.85 vs 0.75 — **but the figure's legend is almost certainly swapped**: the blue "w/o AC-WM" curve hits 85% while orange "w/ AC-WM" hits 75%, contradicting the caption text. Flagged as a paper erratum; the claim matches the text, not the printed legend.

### 6. What's missing / release gaps

- **No C++ runtime in the repo** — the 85 ms deployment stack is described but not shipped; only a Python server/client (`scripts/inference_openloop.py`).
- **No GigaWorld-Policy v1 weights** on HF (only 0.5), though the repo keeps a v0 training config.
- **Config drift**: repo finetune config + inference script use `model_action_dim=32`/`action_dim=32`, but the released checkpoint is strictly 16-channel (`action_encoder.in_proj.weight [2,16,128]` from headers) — users must align these or loading/running breaks. Also the arXiv v1 HTML has a broken bibliography (all citations render as `[key]`; e.g. FastWAM = arXiv 2603.16666 had to be identified externally). Single squashed git commit.

## How to rerun

```bash
cd runs/2026-07-17-gigaworld05-harness-spear/gigaworld-policy-05-2607.13960/code
python3 probe_mask_reconstruction.py   # mask block diagram + exactness proof
python3 probe_flops_params.py          # param accounting + per-chunk FLOP model
python3 probe_autoresearch.py          # Table 5 x Fig 7 dissection of the AutoResearch loop
```

Note: two independent probe implementations of the mask exactness check and the FLOP model were written in this packet and produce identical numbers (mask diff ~2e-16; per-chunk 2.18 TFLOP action-only / 5.93 dense / 24.85 joint) — the accounting is cross-validated, not single-sourced.

Outputs saved in `code/probe_outputs/`. Header validation: HTTP range GET of the first ~80 KB of each shard of `open-gigaai/Giga-World-Policy-0.5` (scripts/ephemera in `.tmp/shard*_header.json`). Repo clone: `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/open-gigaai/giga-world-policy code/giga-world-policy`.

## Figures saved (in `runs/2026-07-17-gigaworld05-harness-spear/assets/`)

- `fig2_architecture_mot.png` — paper **Figure 2** (MoT architecture + train/inference paths). Report-worthy.
- `fig7_autoresearch.png` — paper **Figure 7** (AutoResearch pilot sweep + keep/discard progression). Report-worthy.
- `fig6_acwm_ablation.png` — paper **Figure 6** (AC-WM ablation; note the likely-swapped legend). Useful as an erratum exhibit.
- `fig1_latency_sr_radar.png` — paper **Figure 1** (latency-vs-SR scatter; optional 4th).

## What it means for the report

The strongest framing: "train with future video, infer action-only" is now confirmed as a *reproducible, open-weight* recipe, and GigaWorld-Policy-0.5 is its most deployment-serious instantiation — but its architecture is convergent evolution, not invention. The lineage, now pinned to arXiv IDs: **FastWAM (2603.16666)** and **GigaWorld-Policy v1 (2603.17240)** are near-concurrent (March 2026) and both established training-time-video / inference-time-action-only; v1 additionally introduced the action-centered mask with action-conditioned future prediction, the 3-view composite input, sparse strided future frames, and curriculum pretraining (Wan web-video init; 6000 GPU-hours, AdamW 1e-4 cosine), reporting 0.36 s/inference — exactly the 360 ms A100 that Table 4 re-measures. FastWAM (2603.16666) already had MoT + shared attention + structured mask + no-future-at-inference; GigaWorld-Policy v1 already had the action-centered mask. The sharpest mask-level contrast with FastWAM: FastWAM's future video tokens attend only to the first-frame anchor and themselves (video co-training is *not* action-conditioned, and its first-frame tokens attend to nothing else), whereas GWP-0.5's future tokens read the full context *including all 48 action tokens* — the future loss is literally an action-conditioned world-modeling loss, which is also exactly what the AC-WM pretraining stage trains in isolation. FastWAM's own evidence supports the shared thesis though: removing video co-training drops its real-world towel-folding SR to 10%, i.e. the training-time video objective, not test-time imagination, carries the value. What is *genuinely new* here: (1) future tokens attend to *action* tokens inside the mask (action-conditioned future prediction baked into the joint net — FastWAM's video branch is not action-conditioned), plus mixed AC-WM+WAM pretraining that doubles down on action↔dynamics coupling; (2) a validated lightweight action expert (1.0 B active of 6.0 B total) whose FLOP advantage we could verify end-to-end; (3) deployment engineering (prefix KV cache, torch.compile, C++ runtime claim) with honest, reproducible Python-side numbers; (4) AutoResearch as a shipped-recipe producer, warts and all. Caveats for the report: C++ latency has no baseline counterparts, SR deltas (0.85 vs 0.80) sit within 10-trial noise, the Fig. 6 legend appears swapped, and the v1 paper's broken bibliography makes its citations untraceable from the PDF alone.
