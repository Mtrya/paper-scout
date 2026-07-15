# Thread: Xiaomi-Robotics-U0 (arXiv 2607.11643)

## What this thread investigated

Xiaomi-Robotics-U0 is a large autoregressive world foundation model that claims to unify foundation image/video generation with robot-centric generation in a single continual-training paradigm. The paper's central argument is that embodied generation should be treated as a natural extension of foundation generation, not as a separate post-training stage that forgets visual priors. We investigated whether the released artifacts — code, weights, and inference scripts — support that claim and how the method is actually implemented.

## External signals preserved

We cloned the official repository and inspected the architecture, task configs, tokenizer, and FlashAR acceleration path. The repository is public at:

- GitHub: `https://github.com/XiaomiRobotics/Xiaomi-Robotics-U0` (commit `777c62007a92a0a848060b4c3889fb8f1b00d74b`)
- HuggingFace collection: `https://huggingface.co/collections/XiaomiRobotics/xiaomi-robotics-u0`
- Released checkpoints:
  - `XiaomiRobotics/Xiaomi-Robotics-U0` (AR engine)
  - `XiaomiRobotics/Xiaomi-Robotics-U0-FlashAR` (FlashAR engine)
  - `BAAI/Emu3.5-VisionTokenizer` (IBQ visual tokenizer)
  - `XiaomiRobotics/Xiaomi-Robotics-U0-Video` is listed as **coming soon**.

We also cloned the upstream FlashAR reference implementation at `https://github.com/lxazjk/Emu3.5-FlashAR` (commit `0132543572c29ab8096bd339ba390213b6d6e426`) to compare with Xiaomi's FlashAR+ extension.

## Core mechanism, made concrete

### Model

The released model config (`XiaomiRobotics/Xiaomi-Robotics-U0/config.json`) is:

```json
{
  "vocab_size": 282926,
  "hidden_size": 5120,
  "intermediate_size": 25600,
  "num_hidden_layers": 64,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "max_position_embeddings": 16384,
  "qkv_bias": true,
  "vq_type": "ibq"
}
```

A rough parameter count puts this at ~34B dense parameters (embed + head + 64 transformer layers with GQA and SwiGLU MLP), close to the paper's "38B" claim and consistent with the project page. It is therefore a materially larger model than the repository's default `UNISConfig`, which documents an 8B-style setting (`hidden_size=4096`, `num_hidden_layers=32`).

The architecture is a standard decoder-only Transformer (the code is adapted from the Emu3.5 / LLaMA modeling stack) with no task-specific generation heads: all five tasks — T2I, X2I, embodied scene generation, embodied transfer, and embodied video generation — are cast as next-token prediction over a unified discrete vocabulary.

### Tokenization

Images are tokenized by the IBQ (Index-Propagated Binary Quantization) visual tokenizer from Emu3.5 at a 16×16 spatial compression ratio. The text vocabulary is extended with the IBQ codebook, and images are framed by special tokens (`<|extra_100|>` ... `<|extra_101|>` in the released configs, analogous to BOI/EOI). Multi-view scenes are generated as a single horizontally concatenated image grid.

### Training paradigm

The paper describes two continual-training stages:

1. **Single-step training** jointly optimizes T2I, X2I, embodied scene generation, and embodied transfer. The intent is to preserve the base model's general image-generation capability while aligning it with robot-centric multi-view synthesis.
2. **Sequential training** adds image-text interleaved subtask-subgoal sequences and multi-FPS embodied manipulation videos (1, 3, and 5 FPS) for long-horizon prediction.

The code only exposes inference; training data recipes and training scripts are **not released**. We therefore cannot verify the exact data mixture, reweighting, or training schedule from code alone.

### FlashAR+ acceleration

The standard AR engine decodes image tokens one at a time in raster order. The FlashAR engine adds:

- A **vertical prediction branch** that shares the backbone up to an intermediate layer, then runs cloned decoder layers with a step-causal attention mask.
- **Horizontal and vertical prediction heads** that predict the right-neighbor and bottom-neighbor token respectively.
- A **learnable H/V fusion gate** that mixes the two predictions at each interior grid position; boundary positions use the available single-direction prediction.
- An **anti-diagonal decoding schedule**: position `(r, c)` is generated at step `r + c`, so all positions on the same diagonal are sampled in parallel.

The loss in `xr_u0_flashar/model.py` is:

```python
loss = loss_fuse + 0.05 * loss_h + 0.05 * loss_v + 0.2 * loss_distill
```

matching the paper's Equation (5). The `loss_distill` term distills the vertical branch against the standard AR teacher, which is important because the vertical branch is trained from a model that was originally optimized only for horizontal raster prediction.

For a 32×32 token grid, FlashAR reduces serial decoding steps from 1024 to 63. The paper reports that FlashAR eager is ~27× faster than AR eager at 1024×1024 on a single H20, and FlashAR+vLLM reaches 5.44 s/image, an additional ~3× over FlashAR eager.

The upstream FlashAR repo and Xiaomi's `xr_u0_flashar/model.py` are nearly identical for the core model wrapper; Xiaomi's extension is primarily the integration with vLLM (`xr_u0_flashar/vllm/`) and support for multi-image X2I conditioning.

### Structured scene control

The released task configs (`configs/tasks/scene_gen.py`, `configs/tasks/transfer.py`) implement exactly the five disentangled dimensions described in the paper:

- **Workspace**
- **Task objects**
- **Irrelevant objects**
- **Lighting**
- **Background**

Embodied transfer additionally conditions on monocular inverse-depth maps extracted with Depth Anything 3 (`depth-anything/DA3-LARGE-1.1`), converting RGB references into depth grids before generation. The system prompt explicitly tells the model that brighter pixels are closer and darker pixels are farther.

## How to rerun the preserved probes

All probes are self-contained Python scripts in this thread's `code/` directory.

```bash
cd runs/2026-07-15-xiaomi-egosteer-abot/xiaomi-robotics-u0-2607.11643

# Fetch released model config and estimate parameter count.
python3 code/inspect_config.py

# Print the FlashAR anti-diagonal schedule for a 32x32 token grid.
python3 code/diagonal_schedule.py

# Print the structured system prompts and example task templates.
python3 code/task_prompt_probe.py
```

The first probe downloads `config.json` from the public HuggingFace repo and requires internet access; the other two depend only on the local clone of `XiaomiRobotics/Xiaomi-Robotics-U0` under `code/xiaomi-robotics-u0`.

## What the code confirms and what it does not

**Confirmed by the released artifacts:**

- The model is real, weights are public, and inference code is provided for AR eager, AR+vLLM, FlashAR eager, and FlashAR+vLLM.
- The architecture is a single autoregressive decoder with unified image/text tokens.
- The IBQ tokenizer is the Emu3.5 vision tokenizer.
- FlashAR+ is implemented as described: anti-diagonal schedule, H/V heads, gated fusion, auxiliary losses, and AR distillation.
- The five structured scene dimensions are present in the actual prompts.
- Depth-conditioned transfer uses DA3 for RGB-to-depth preprocessing.

**Not confirmed / open questions:**

- The video-generation checkpoint is **not released** ("coming soon"), so the strongest claim — long-horizon embodied video generation and the World Arena results — cannot be reproduced from the public artifacts.
- Training code, data pipeline, and exact data mixture are not released. We cannot verify the 9.5M single-step / 2.6M sequential sample counts, the HDBSCAN subtask decomposition, the VLM annotation prompts, or the sample-level reweighting scheme.
- The real-world policy augmentation experiments (π₀.₅ on three bimanual tasks) rely on the not-yet-released video capability and proprietary MiBot data, so they are not independently reproducible.
- The comparison against GPT-Image-2.0 in the paper uses a small internal benchmark (300 transfer samples, 400 scene-generation samples) that is not public.
- The paper says the base model is "EMU3.5 ... built upon Qwen-3-32B," but the released config has 64 layers / 5120 hidden size, which does not match either the published Emu3.5-34B config or a literal Qwen-3-32B config. The relationship between the released U0 weights and the named base model is therefore approximate; U0 is likely a reconfigured or further-scaled continual-training descendant rather than a direct Emu3.5 checkpoint.

## Implications for the report

Xiaomi-Robotics-U0 is best understood as a **scaling + unification bet**: take a large autoregressive image generator, keep training it on robot data without forking it into a narrow world model, and add a FlashAR-derived acceleration path so that high-resolution embodied generation is practical. The released code and weights substantiate the single-step generation story (scene gen, transfer, T2I, X2I) and the FlashAR speed-up. The sequential/video story is plausible but still artifactually thin because the video checkpoint and training pipeline are missing.

The honest takeaway for the report is:

- The single-task results are credible and the codebase is clean enough that practitioners can run the model.
- The real-world policy numbers are a promising signal but should be framed as **proprietary, not yet reproducible** until the video checkpoint and data pipeline are released.
- The "38B" parameter claim is in the right ballpark given the released config.
- The comparison with GPT-Image-2.0 is informative as a directional indicator but not a public benchmark.

## Files in this thread

- `README.md` — this file.
- `code/inspect_config.py` — fetch and summarize the released model config.
- `code/diagonal_schedule.py` — visualize FlashAR anti-diagonal decoding.
- `code/task_prompt_probe.py` — extract structured prompt templates from the released configs.
