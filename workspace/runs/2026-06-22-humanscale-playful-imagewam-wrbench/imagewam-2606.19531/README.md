# ImageWAM (2606.19531) — Deep-dive thread

**Paper:** *ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?*  
**Authors:** Yuyang Zhang, Wenyao Zhang, et al.  
**Code:** https://github.com/yuyangalin/ImageWAM  
**Project page:** https://zhangwenyao1.github.io/ImageWAM/

## Research question

Video-generation world action models (WAMs) reason about the future by denoising dense spatio-temporal tokens. ImageWAM asks whether that is necessary. Its hypothesis is that an **image-editing backbone** already provides the right prior for manipulation: it is source-grounded, instruction-guided, and change-centric. The key question for this thread was: *what exactly is the architectural difference, and where do the reported efficiency gains come from?*

## What ImageWAM does

ImageWAM repurposes pretrained image-editing models (FLUX.2, OmniGen2, Ovis-U1) as policy backbones. At a high level:

1. **Inputs:** current observation image + language instruction.
2. **Editing backbone:** runs a single image-editing forward step, producing layer-wise key/value caches. The backbone is trained to predict a *single* future endpoint frame, but at inference the endpoint is **not decoded**.
3. **Action expert:** a slim flow-matching DiT (ActionDiT) attends to those editing caches through a Mixture-of-Transformers (MoT) interface and denoises an action chunk.

The contrast with video-WAMs is the intermediate representation:

| Model | Intermediate | What is materialized at inference |
|---|---|---|
| FastWAM-IDM | future video | dense multi-frame video tokens, denoised and decoded |
| FastWAM-1step | current-context cache | current-frame tokens only (no future frames) |
| **ImageWAM** | **editing cache** | text + current-image prefix only; no target frame decoded |

## Code inspected

The official repository was cloned to `code/imagewam-2606.19531/` (ignored lab bench). The most important files for understanding the mechanism are:

- `src/imagewam/models/backbones/imagewam.py` — the main model class; defines `infer_action_flux2`, `infer_action_omnigen2`, `infer_action_ovis_u1`, and the MoT attention masks.
- `src/imagewam/models/backbones/mot.py` — MoT implementation. Key methods: `prefill_flux2_video_cache`, `forward_flux2_action_with_video_cache`, `prefill_video_cache`, `_mixed_attention`.
- `src/imagewam/models/backbones/action_dit_flux2.py` — slim FLUX.2-compatible action expert.
- `src/imagewam/models/backbones/action_dit_omnigen2.py` — OmniGen2-style action expert.
- `src/imagewam/models/backbones/action_dit_yak.py` — Ovis-U1/Yak-style action expert.
- `src/imagewam/models/backbones/flux2_video_expert.py` — wraps FLUX.2 for image-editing; shows how ref/target image tokens are built.
- `src/imagewam/models/backbones/omnigen2_video_expert.py` — OmniGen2 wrapper; `prefix_only=True` path is the one used at inference.
- `configs/model/imagewam_flux2_klein_4b_base.yaml` — ActionDiT config for the recommended 4B variant.

### Key architectural findings from the code

1. **Single editing forward step.** In `infer_action_flux2` (imagewam.py:3515), the video timestep is set to zero, the target latent is empty (`target_len=0`), and only the reference image + text are fed through the editing backbone. `mot.prefill_flux2_video_cache` stores the resulting K/V caches.
2. **No future-frame attention.** The FLUX.2 attention mask (`_build_mot_attention_mask_flux2`, imagewam.py:2072) lets action tokens attend only to the stable text+ref prefix and to themselves; they never attend to noisy target/frame tokens.
3. **ActionDiT is much smaller than the editing backbone.** The FLUX.2 4B variant uses a 642M ActionDiT (paper Appendix 5.1) with `hidden_dim=1024`, 5 double + 20 single layers, and 24 heads × 128 head_dim.
4. **Training still co-trains the editing objective.** The loss is `lambda_video * L_video + lambda_action * L_action` (configs/model/imagewam_flux2_klein_4b_base.yaml), so the editing backbone learns to predict the endpoint frame, which keeps the caches transformation-aware.

## The probe

`code/probe_imagewam.py` inspects the released configs and makes the architecture/efficiency argument concrete without downloading any checkpoints.

### What it does

1. Loads the model configs for all three variants.
2. Analytically counts ActionDiT parameters and cross-checks against the paper's reported sizes.
3. Computes the visual-token budget for the paper's input resolutions.
4. Compares ImageWAM's prefix-only token count to a video-WAM that denoises 8 or 16 future frames.
5. Prints a concrete FLUX.2 attention-mask slice showing that action tokens attend only to the text+ref-image prefix, never to future/target image tokens.
6. Verifies the Table 5 efficiency claim with the paper's own latency/FLOPs numbers.

### How to rerun

From the workspace root:

```bash
python runs/2026-06-22-humanscale-playful-imagewam-wrbench/imagewam-2606.19531/code/probe_imagewam.py
```

Requirements: Python ≥3.9 and `pyyaml`.

### Probe output highlights

```text
FLUX.2 4B
  estimated ActionDiT params: 642.0M
  paper-reported ActionDiT params: 642M
  estimate / paper: 100.01%

FLUX.2 9B
  estimated ActionDiT params: 952.4M
  paper-reported ActionDiT params: 952M
  estimate / paper: 100.04%

Resolution libero_2cam: 224x448
  tokens per frame: 392
  ImageWAM inference prefix tokens (text + current image): 469
  Video-WAM (16 frames): 6272 tokens  -> ImageWAM uses 7.48% as many visual tokens

FLUX.2 prefix-only attention mask (T=text, R=ref image, A=action):
   T  T  T  T  R  R  R  R  R  R  A  A  A
   1  1  1  1  1  1  1  1  1  1  .  .  .  T
   ...
   1  1  1  1  1  1  1  1  1  1  1  1  1  A
  Notice: action rows attend to T+R but never to a target-image column.

Efficiency claim (Table 5)
  Latency: 263 / 1081 = 0.243 (~1/4.1)
  FLOPs:   9.72 / 63.65 = 0.153 (~1/6.5)
```

The FLUX.2 ActionDiT parameter counts match the paper almost exactly. OmniGen2 and Ovis-U1 estimates are rougher because their transformer blocks have vendor-specific FFN and normalization conventions not fully captured by the simple formula, but they are in the correct ballpark.

## Takeaway

ImageWAM's efficiency claim is structurally sound. The code shows that inference does not instantiate future video tokens or decode an edited target frame; it materializes only a text + current-image prefix, extracts KV caches from one editing forward step, and runs a modest action expert. The token-count reduction (~7–15% of a 16-frame video-WAM) and the measured FLOPs/latency reduction (~1/6 FLOPs, ~1/4 latency vs FastWAM-IDM) are consistent with that design.

The deeper point is that the *kind* of pretraining matters. Image editing is trained to map "source image + instruction → transformed image," which is a closer match to manipulation than open-ended video generation. ImageWAM leverages that by using the editing backbone's internal transformation-aware features as the world-action context, not by treating the edited image as a goal state.

## Limitations / open questions

- The probe does not load released weights (FLUX.2 4B/9B checkpoints are several GB) and does not run policy rollouts. It verifies architecture and efficiency arithmetic, not task success.
- The exact mechanism by which editing caches concentrate attention on task-relevant regions (Figure 4) is asserted by the paper and supported by the attention-mask design, but this thread did not reproduce the attention visualizations.
- Real-world gains (84.5% vs FastWAM's 79.0%) are impressive, but the real-world dataset is small (4 tasks, ~100 demos each); broader embodiment transfer remains to be tested.
