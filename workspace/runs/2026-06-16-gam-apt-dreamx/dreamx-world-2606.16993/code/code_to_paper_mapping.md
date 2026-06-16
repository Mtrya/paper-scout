# DreamX-World: code-to-paper mapping

## 3.1 Camera-Aware Training / E-PRoPE

**Claim:** Lightweight PRoPE self-attention branch on spatially reduced tokens  
**Status:** implemented

| File | Lines | Note | Verification |
|------|-------|------|--------------|
| `models/prope_utils.py` | 61-100 | prope_qkv: projective transform of q/k/v | ok (253 lines) |
| `models/wan_transformer3d.py` | 190-270 | PropeSelfAttention (dim -> dim/4 attention) | ok (1281 lines) |
| `models/wan_transformer3d.py` | 432-487 | WanAttentionBlock: y + cam_self_attn branch | ok (1281 lines) |
| `wan/modules/camera_prope.py` | 61-117 | causal PRoPE q/k/v transform | ok (303 lines) |
| `wan/modules/causal_camera_model_2_2_prope_infinity.py` | 233-370 | CausalPropeSelfAttention with KV cache | ok (735 lines) |
| `wan/modules/causal_camera_model_2_2_prope_infinity.py` | 405-445 | CausalWanAttentionBlock camera branch | ok (735 lines) |

## 3.2 Memory-Conditioned Scene Persistence

**Claim:** Geometry-based retrieval of non-local memory frames + residual recycling  
**Status:** partially implemented (cache only, no geometry retrieval)

| File | Lines | Note | Verification |
|------|-------|------|--------------|
| `wan/modules/causal_camera_model_2_2_prope_infinity.py` | 58-230 | CausalWanSelfAttention: rolling KV cache + sink tokens | ok (735 lines) |
| `pipeline/pipeline_causal_camera.py` | 47-254 | CausalCameraInferencePipeline: chunk-wise KV cache inference | ok (282 lines) |

## 3.3 Event Instruction Tuning

**Claim:** Composable multi-entity event control via text conditioning  
**Status:** not explicitly implemented (no event parser / structured event module)

| File | Lines | Note | Verification |
|------|-------|------|--------------|
| `configs/dreamx/eval.json` | 1 | Example captions contain structured event-like descriptions | ok (206 lines) |
| `pipeline/pipeline_dreamxworld.py` | 192-235 | Text prompt encoding (events enter through prompt only) | ok (843 lines) |

## 3.4 Autoregressive Long Video Generation and Distillation

**Claim:** Causal forcing + DMD + long-rollout training for few-step AR generation  
**Status:** inference-only; training/distillation code not released

| File | Lines | Note | Verification |
|------|-------|------|--------------|
| `inference_ar_forcing.py` | 250-353 | AR inference entry point | ok (356 lines) |
| `pipeline/pipeline_causal_camera.py` | 47-254 | CausalCameraInferencePipeline: block-wise denoising, few-step schedule | ok (282 lines) |
| `configs/dreamx-ar/causal_camera_forcing_5b.yaml` | 1 | denoising_step_list = [1000,750,500,250] (few-step DMD sampling) | ok (28 lines) |
| `wan/modules/causal_camera_model_2_2_prope_infinity.py` | 482-661 | CausalWanModel forward with KV cache and Block-Relativistic RoPE | ok (735 lines) |

## 3.5 Reinforcement Learning

**Claim:** Post-DMD RL with camera-control and video-quality rewards  
**Status:** not implemented in released code

| File | Lines | Note | Verification |
|------|-------|------|--------------|
| — | — | no released implementation | — |

## 4.1 Autoregressive Streaming Inference

**Claim:** Chunk-by-chunk generation with rolling KV cache and chunk-relative cameras  
**Status:** implemented

| File | Lines | Note | Verification |
|------|-------|------|--------------|
| `pipeline/pipeline_causal_camera.py` | 47-254 | block-wise inference with kv_cache1 / crossattn_cache | ok (282 lines) |
| `inference_ar_forcing.py` | 74-133 | cam_params_to_prope_dict: chunk_relative option | ok (356 lines) |
| `utils/trajectory_processor.py` | 557-622 | chunk-relative Plücker / pose computation | ok (958 lines) |

## 4.2 Inference Acceleration

**Claim:** Quantized attention/FFN, sequence parallelism, VAE pruning, async pipeline  
**Status:** mostly implemented (Sage/Flash, FP8, SP, VAE cache); async pipeline is implicit in script

| File | Lines | Note | Verification |
|------|-------|------|--------------|
| `models/attention_utils.py` | 152-211 | attention dispatcher: SageAttention / FlashAttention | ok (211 lines) |
| `utils/fp8_optimization.py` | 1 | FP8 weight conversion helpers | ok (64 lines) |
| `wan/modules/vae_2_2.py` | 1 | Wan2.2 VAE with cached_decode / causal convs | ok (1088 lines) |
| `dist/fuser.py` | 1 | xfuser sequence-parallel setup | ok (67 lines) |
| `dist/wan_xfuser.py` | 1 | USP/ring attention sharding | ok (310 lines) |

## 2 Data

**Claim:** UE rendering, real-world/game data, filtering, captioning  
**Status:** not implemented in released code

| File | Lines | Note | Verification |
|------|-------|------|--------------|
| `configs/dreamx/eval.json` | 1 | Evaluation prompts/camera actions only | ok (206 lines) |

