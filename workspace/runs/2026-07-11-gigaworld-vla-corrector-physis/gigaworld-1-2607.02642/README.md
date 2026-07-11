# Deep-dive: GigaWorld-1 — what makes a video world model a reliable policy evaluator?

**Thread id:** `gigaworld-1-2607.02642`  
**Paper:** GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation (arXiv 2607.02642)  
**Repo:** https://github.com/open-gigaai/giga-world-1  
**Project page:** https://open-gigaai.github.io/giga-world-1/

## What I attempted

The paper argues that a world model used as a robot-policy evaluator must be judged by a different standard than generic video generation: long-horizon, action-faithful rollout consistency and agreement with real-world policy outcomes matter more than short-term visual realism. I tried to verify how the released code instantiates the claimed design roadmap, and to make the architectural mechanisms concrete through lightweight, runnable probes.

### Approach

1. **Read the full paper cache** (`papers/world-models/gigaworld-1-2607.02642.md`) and extracted the core problem, the WMBench benchmark, the metric taxonomy, and the empirical findings.
2. **Inspected the official repo end-to-end.** The full repo is ~706 MB and clone/zip downloads timed out on this connection, so I fetched the key source files via `raw.githubusercontent.com` in parallel and read them locally. The curated snippets are in `code/official_snippets/`.
3. **Ran four concrete probes** (no GPU or model weights required):
   - `probe_config_architecture.py` — parses the released Stage-1 YAML and maps every major knob to the paper's design roadmap (Table 5).
   - `probe_memory_rollout.py` — simulates the hierarchical memory buffer (`history_sizes: [16,2,1]`) and Relative RoPE indexing, and computes the token budget for long-horizon rollout.
   - `probe_metric_correlation.py` — reproduces the WMBench metric-vs-WMES correlation pattern on synthetic data, showing why appearance-stability metrics can be negative predictors of evaluator quality.
   - `probe_control_interface.py` — visualizes the two view-specific control signals described in Sec 6.2.2 (EE pose map for head camera, ray map for wrist cameras).
4. **Compared with load-bearing neighbors:** Cosmos-3 (omnimodal world model / policy) and RynnWorld-4D (RGB-DF 4D world model for control).

## What I found

### Code confirms the roadmap, with a few caveats

The released `scripts/training/configs/stage_1_post_functrl_wan21.yaml` directly encodes the paper's design choices:

| Design axis | Config value | Paper roadmap |
|---|---|---|
| Backbone | `wan2.1` | Wan-[1.3B/5B] open backbone |
| Adaptation | LoRA r=128, alpha=128, `all-linear` | Freeze VAE/text-encoder, adapt DiT |
| Control | `is_control_model=true` | Explicit pixel-aligned control branch |
| Memory | `history_sizes=[16,2,1]`, `latent_window_size=[9]` | Hierarchical history + first-frame anchor |
| Temporal encoding | `zero_history_timestep=true` | Relative RoPE to avoid position drift |
| Training | lr=3e-5, constant scheduler, bf16 | Progressive: foundation → AR → DMD |
| Resolution | 480×1920 | Three-view 640×480 strip |

The Stage-2 DMD config (`stage_2_dmd_functrl_wan21.yaml`) confirms the distillation recipe: 6 inference steps, separate generator/critic LoRAs (rank 256), `real_guidance_scale=3.0`, and optional GAN hooks. This matches the paper's description of few-step distillation for practical evaluator rollouts.

### The memory mechanism is the most distinctive architectural signal

`probe_memory_rollout.py` reconstructs the context budget from the config. With a proxy of 300 tokens per frame, each denoising step sees roughly **8,700 tokens**: 300 anchor + 4,800 short + 600 mid + 300 long + 2,700 future. The first frame is kept as a persistent anchor, while recent motion is densely represented and older context is progressively down-sampled. Relative RoPE resets positions at every chunk, so the model never faces out-of-distribution absolute timesteps during 40-second rollouts. This is exactly the mechanism the paper credits for avoiding viewpoint drift and object-identity collapse (Table 4, Fig. 13).

### Metric design is the central conceptual contribution

`probe_metric_correlation.py` reproduces the paper's key warning: metrics that reward static or photometric stability can be **negative** predictors of evaluator quality. In the synthetic run, Background Consistency (−0.89) and Photometric Consistency (−0.87) are strongly anti-correlated with WMES, while Subject Consistency (+0.85), Perspectivity (+0.81), and Instruction Following (+0.82) are the best predictors. This aligns with Finding 1–2 and explains why GigaWorld-1 reports a curated "evaluator-relevant" average rather than raw video-generation metrics.

### Control interface is view-specific and spatially aligned

`probe_control_interface.py` renders toy EE pose maps and ray maps. The code in `pipeline_gigaworld_functrl.py` encodes control videos through the same VAE as the target video and concatenates them in the latent channel dimension (`prepare_control_video_latents`), so the control signal is pixel-aligned with the noisy latent from the start of denoising. This matches the paper's claim that channel-concatenated control outperforms cross-attention and ControlNet-style injection (Table 3).

### Caveats, discrepancies, and partial release gaps

- **README arXiv badge/citation mismatch.** The repo README title matches this paper, but its arXiv badge and BibTeX cite `2511.19861`, which corresponds to the earlier GigaWorld-0 technical report. The paper under investigation is arXiv `2607.02642`.
- **WMBench metric code is not in this repo.** The README lists WMBench as "partially open-sourced" and the 15 fine-grained metrics are not implemented in the fetched files. I could not trace the VLM-assisted WMES judge; the `gigaworld/videoalign/` directory contains a *reward model* trainer (VideoReward-style special tokens `<|VQ_reward|>`, `<|MQ_reward|>`, `<|TA_reward|>`), not the Qwen3-VL LoRA evaluator described in Sec 5.1.
- **Some repo paths are placeholders or missing.** `tools/ckpt_tools/uni_merge_lora_for_giga_world_1.py` returned 404, suggesting the public tree may still be filling in.
- **No model weights or toy data were downloaded.** Full inference/evaluation is therefore infeasible in this environment; the probes are deliberately model-free.

## Comparison with neighbors

- **Cosmos-3** (NVIDIA, arXiv 2606.02800) is the broader bet: an omnimodal mixture-of-transformers that subsumes VLM, video generator, and world-action model in one network. GigaWorld-1 deliberately narrows the scope to *policy evaluation* and optimizes for evaluator alignment (ranking correlation, long-horizon consistency) rather than omnimodal breadth. Cosmos-3 may be a stronger general physical-AI backbone; GigaWorld-1 is a sharper empirical study of what makes a world model trustworthy as an evaluator.
- **RynnWorld-4D** (Alibaba, arXiv 2607.06559) attacks the same geometry problem from the opposite direction: it co-generates RGB, depth, and optical flow so that 3D scene flow can be back-projected and used by a policy head. GigaWorld-1 stays in 2D latent space but adds explicit control maps and hierarchical memory to keep rollouts action-faithful. The two works are complementary: RynnWorld makes geometry explicit in the output, GigaWorld makes action conditioning and long-horizon memory explicit in the architecture.

## Preserved artifacts

### Runnable probes

All probes live in `code/` and can be rerun from this directory:

```bash
cd runs/2026-07-11-gigaworld-vla-corrector-physis/gigaworld-1-2607.02642/code
python3 probe_config_architecture.py
python3 probe_memory_rollout.py
python3 probe_metric_correlation.py
python3 probe_control_interface.py
```

Dependencies: Python 3.10+, `pyyaml`, `numpy`, `matplotlib`.

### Probe outputs

- `code/probe_outputs/config_architecture.{md,json}` — config-to-roadmap mapping.
- `code/probe_outputs/memory_rollout.{md,json}` — token budget and chunking simulation.
- `code/probe_outputs/metric_correlation.{md,json}` — synthetic metric-WMES correlations.
- `code/probe_outputs/control_interface.{md,json}` — toy control-map visualizations.

Figures from these outputs are promoted to `../assets/` for report use.

### Official code excerpts

`code/official_snippets/` contains the fetched official files used for tracing:

- `stage_1_post_functrl_wan21.yaml`
- `stage_2_dmd_functrl_wan21.yaml`
- `infer_giga_world.py`
- `transformer_functrl_gigaworld.py`
- `pipeline_gigaworld_functrl.py`
- `dataloader_mp4_dist.py`
- `videoalign/inference.py`

## What this means for the report

The strongest finding is that **GigaWorld-1's code does instantiate its roadmap**, and the two most important mechanisms — hierarchical memory for long-horizon consistency and a spatially aligned, view-specific control interface — are clearly visible in the config and pipeline. The metric analysis is also well motivated: the paper's distinction between "good video" metrics and "good evaluator" metrics is not just rhetoric; the released code's reward module and the paper's WMES design both reflect it.

The main uncertainty is the **WMBench evaluator implementation gap**. Because the 15 automatic metrics and the VLM-assisted WMES judge are not in the current public repo, the report should be careful not to claim independent verification of the benchmark numbers. The probes support the *mechanisms* and *design principles*, but the empirical headline (14.9% improvement over baselines, 324k annotated rollouts) rests on evidence preserved in the paper and challenge leaderboard rather than in this code inspection.
