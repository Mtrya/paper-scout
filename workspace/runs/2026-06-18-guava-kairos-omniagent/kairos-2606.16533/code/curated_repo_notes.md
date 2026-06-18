# Curated repo notes — kairos-sensenova

Official repo cloned to `workspace/code/kairos-sensenova` (https://github.com/kairos-agi/kairos-sensenova).
These are the key implementation facts pulled from the code for the deep-dive.

## Model definition

File: `kairos/modules/dits/kairos_dit.py`

Registered class: `KairosDiT`.

### 4B robot distilled config (`kairos/configs/kairos_4b_config_DMD.py`)

```python
dit_config = {
    "dit_type" : 'KairosDiT',
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 16,
    "dim": 2560,
    "ffn_dim": 10240,
    "freq_dim": 256,
    "text_dim": 3584,
    "out_dim": 16,
    "num_heads": 20,
    "num_layers": 32,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
    "dilated_lengths": [1, 1, 4, 1],
    "use_seq_parallel": True,
    "use_tp_in_getaeddeltanet": True,
    "use_tp_in_self_attn": True,
    "attend_k0": False,
}
```

Other released configs (e.g. `kairos_4b_config.py`) use `dilated_lengths: [1, 1, 6, 1]`,
no sequence/tensor parallelism, and an explicit TeaCache threshold.

### Block construction

From `KairosDiT.__init__` (lines 1027–1048):

```python
use_linear_attns = [(i + 1) % 4 == 0 for i in range(num_layers)]
# ... every 4th layer is a GatedDeltaNet (GLA) layer
gateddeltanet_layer_indexs = [-1 for _ in range(num_layers)]
gidx = 0
for i, vi in enumerate(use_linear_attns):
    if vi:
        gateddeltanet_layer_indexs[i] = gidx
        gidx += 1

for i in range(num_layers):
    _block = DiTBlock(
        has_image_input, dim, num_heads, ffn_dim, eps,
        use_linear_attn=(i + 1) % 4 == 0,
        dilated_length=dilated_lengths[i % 4],
        ...
    )
```

So the interleaving is **3 SWA/DSWA layers followed by 1 GLA layer**, matching the
paper’s Figure 5 (2× SWA, 1× Dilated SWA, 1× GLA per group of 4).

### Sliding-window attention (`SelfAttention`)

* Uses RMSNorm on Q/K, RoPE on Q/K (1-D or 3-D `rope_apply_for3d`),
  then Flash/Sage attention with a window.
* `dilated_length` reshapes the sequence `(B, F*L, D) -> (B*d, F/d*L, D)`
  before attention, giving dilated sliding-window attention (DSWA).
* `window_size` is scaled by `L = h*w` (tokens per frame), so the window covers
  a fixed number of neighbouring frames.
* Optional `attend_k0` mixes the first-frame key back into the local window,
  used for image-conditioning.

### Gated Linear Attention (`GatedDeltaNet`)

File: `kairos/third_party/fla/layers/gated_deltanet.py`

Kairos wraps the `fla.layers.GatedDeltaNet` class with `mode='chunk'` for training
and `mode='fused_recurrent'` for short inference sequences.

Key hyperparameters from `DiTBlock`:

```python
self.gated_delta = GatedDeltaNet(
    hidden_size=dim,        # 2560
    num_heads=num_heads,    # 20
    mode='chunk',
    use_gate=True,
    norm_eps=eps,
    layer_idx=gateddeltanet_layer_idx
)
```

The layer implements the delta-rule update described in the paper:
`S_t = alpha_t * S_{t-1} + beta_t * (v_t - v_t^old) k_t^T`,
plus a 1-D short convolution on Q/K/V and an output gate.

## Inference / deployment co-design

Files: `kairos/pipelines/kairos_embodied_pipeline.py`,
`kairos/pipelines/kairos_embodied_pipeline_dmd.py`,
`kairos/apis/kairos_embodied_api.py`,
`kairos/configs/kairos_4b_config_DMD.py`.

### Two pipeline variants

* `KairosEmbodiedPipeline` — standard flow-matching sampler with
  `exponential_shift=True`.
* `KairosEmbodiedPipeline_DMD` — distilled model pipeline with a
  `DMDFlowMatchScheduler` and `selected_sampling_time = [1000, 800, 500, 100]`.

### TeaCache

Both pipelines include a `WanVideoUnit_TeaCache` unit.  The `TeaCache` class
(lines 905–954 of `kairos_embodied_pipeline.py`) skips DiT blocks when the
relative L1 change of the modulated input is below a threshold, reusing the
previous residual.  Coefficients are hard-coded for several Wan2.1 model IDs;
the Kairos configs reuse the Wan2.1-T2V-1.3B coefficients.

### Parallelism knobs

From the DMD config:

```python
"use_seq_parallel": True,
"use_tp_in_getaeddeltanet": True,
"use_tp_in_self_attn": True,
```

`DiTBlock` splits the GLA state across TP ranks and the SWA heads across ranks.
The inference script (`examples/inference.py`) builds TP/CFG groups via
`parallel_state`.

### Memory / efficiency tricks visible in the code

* BFloat16 inference by default.
* FFN chunked forward pass with `chunk_size=2310` inside `DiTBlock`.
* VAE tiled decoding (`tile_size=(30, 52)`, `tile_stride=(15, 26)`).
* `TemporalTiler_BCTHW` for long-sequence sliding-window generation.
* `vram_management_enabled` offloads inactive sub-models to CPU in
  `BasePipeline.load_models_to_device`.
* Attention backend is Flash-Attention 2/3 if available, otherwise
  SageAttention for the local window.

## What is *not* in the repo

* No training scripts, data loaders, or dataset configs.
* No explicit implementation of the Cross-Embodiment Data Curriculum (CEDC)
  stages described in the paper.
* No ActionDiT training code; the released configs only cover the video
  generation / distilled-robot DiT.
* No model weights (those are downloaded separately from HuggingFace / ModelScope).
