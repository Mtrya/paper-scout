#!/usr/bin/env python3
"""
Map DreamX-World paper contributions to concrete files/lines in the released
code snapshot.  The mapping is evidence-backed: each entry stores the file and
the line range we observed in the snapshot, and the script verifies that the
file exists and that the line range is valid.
"""
import os
from pathlib import Path


MAPPING = [
    {
        "paper_section": "3.1 Camera-Aware Training / E-PRoPE",
        "claim": "Lightweight PRoPE self-attention branch on spatially reduced tokens",
        "files": [
            ("models/prope_utils.py", 61, 100, "prope_qkv: projective transform of q/k/v"),
            ("models/wan_transformer3d.py", 190, 270, "PropeSelfAttention (dim -> dim/4 attention)"),
            ("models/wan_transformer3d.py", 432, 487, "WanAttentionBlock: y + cam_self_attn branch"),
            ("wan/modules/camera_prope.py", 61, 117, "causal PRoPE q/k/v transform"),
            ("wan/modules/causal_camera_model_2_2_prope_infinity.py", 233, 370, "CausalPropeSelfAttention with KV cache"),
            ("wan/modules/causal_camera_model_2_2_prope_infinity.py", 405, 445, "CausalWanAttentionBlock camera branch"),
        ],
        "status": "implemented",
    },
    {
        "paper_section": "3.2 Memory-Conditioned Scene Persistence",
        "claim": "Geometry-based retrieval of non-local memory frames + residual recycling",
        "files": [
            ("wan/modules/causal_camera_model_2_2_prope_infinity.py", 58, 230, "CausalWanSelfAttention: rolling KV cache + sink tokens"),
            ("pipeline/pipeline_causal_camera.py", 47, 254, "CausalCameraInferencePipeline: chunk-wise KV cache inference"),
        ],
        "status": "partially implemented (cache only, no geometry retrieval)",
    },
    {
        "paper_section": "3.3 Event Instruction Tuning",
        "claim": "Composable multi-entity event control via text conditioning",
        "files": [
            ("configs/dreamx/eval.json", 1, None, "Example captions contain structured event-like descriptions"),
            ("pipeline/pipeline_dreamxworld.py", 192, 235, "Text prompt encoding (events enter through prompt only)"),
        ],
        "status": "not explicitly implemented (no event parser / structured event module)",
    },
    {
        "paper_section": "3.4 Autoregressive Long Video Generation and Distillation",
        "claim": "Causal forcing + DMD + long-rollout training for few-step AR generation",
        "files": [
            ("inference_ar_forcing.py", 250, 353, "AR inference entry point"),
            ("pipeline/pipeline_causal_camera.py", 47, 254, "CausalCameraInferencePipeline: block-wise denoising, few-step schedule"),
            ("configs/dreamx-ar/causal_camera_forcing_5b.yaml", 1, None, "denoising_step_list = [1000,750,500,250] (few-step DMD sampling)"),
            ("wan/modules/causal_camera_model_2_2_prope_infinity.py", 482, 661, "CausalWanModel forward with KV cache and Block-Relativistic RoPE"),
        ],
        "status": "inference-only; training/distillation code not released",
    },
    {
        "paper_section": "3.5 Reinforcement Learning",
        "claim": "Post-DMD RL with camera-control and video-quality rewards",
        "files": [],
        "status": "not implemented in released code",
    },
    {
        "paper_section": "4.1 Autoregressive Streaming Inference",
        "claim": "Chunk-by-chunk generation with rolling KV cache and chunk-relative cameras",
        "files": [
            ("pipeline/pipeline_causal_camera.py", 47, 254, "block-wise inference with kv_cache1 / crossattn_cache"),
            ("inference_ar_forcing.py", 74, 133, "cam_params_to_prope_dict: chunk_relative option"),
            ("utils/trajectory_processor.py", 557, 622, "chunk-relative Plücker / pose computation"),
        ],
        "status": "implemented",
    },
    {
        "paper_section": "4.2 Inference Acceleration",
        "claim": "Quantized attention/FFN, sequence parallelism, VAE pruning, async pipeline",
        "files": [
            ("models/attention_utils.py", 152, 211, "attention dispatcher: SageAttention / FlashAttention"),
            ("utils/fp8_optimization.py", 1, None, "FP8 weight conversion helpers"),
            ("wan/modules/vae_2_2.py", 1, None, "Wan2.2 VAE with cached_decode / causal convs"),
            ("dist/fuser.py", 1, None, "xfuser sequence-parallel setup"),
            ("dist/wan_xfuser.py", 1, None, "USP/ring attention sharding"),
        ],
        "status": "mostly implemented (Sage/Flash, FP8, SP, VAE cache); async pipeline is implicit in script",
    },
    {
        "paper_section": "2 Data",
        "claim": "UE rendering, real-world/game data, filtering, captioning",
        "files": [
            ("configs/dreamx/eval.json", 1, None, "Evaluation prompts/camera actions only"),
        ],
        "status": "not implemented in released code",
    },
]


def verify(src_dir, path, start, end):
    full = src_dir / path
    if not full.exists():
        return "FILE MISSING"
    n_lines = sum(1 for _ in open(full, "r", encoding="utf-8", errors="ignore"))
    if start is None:
        start = 1
    if end is None:
        end = n_lines
    if start > n_lines or end > n_lines:
        return f"OUT OF RANGE (file has {n_lines} lines)"
    return f"ok ({n_lines} lines)"


def main():
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / "src"

    out_path = script_dir / "code_to_paper_mapping.md"
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("# DreamX-World: code-to-paper mapping\n\n")
        for entry in MAPPING:
            sec = entry["paper_section"]
            claim = entry["claim"]
            status = entry["status"]
            out.write(f"## {sec}\n\n")
            out.write(f"**Claim:** {claim}  \n")
            out.write(f"**Status:** {status}\n\n")
            out.write("| File | Lines | Note | Verification |\n")
            out.write("|------|-------|------|--------------|\n")
            if not entry["files"]:
                out.write("| — | — | no released implementation | — |\n")
            for path, start, end, note in entry["files"]:
                v = verify(src_dir, path, start, end)
                line_range = f"{start}" if end is None or start == end else f"{start}-{end}"
                out.write(f"| `{path}` | {line_range} | {note} | {v} |\n")
            out.write("\n")

    print(f"Wrote mapping to {out_path}")
    print()
    with open(out_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
