#!/usr/bin/env python3
"""
Probe: parse GigaWorld-1 Stage-1 config and map it to the paper's design roadmap.

This script does not require model weights or GPUs. It reads the released YAML
config and extracts the architectural choices that instantiate the claims in
Table 5 of the paper (backbone, data, control, memory, training).
"""

import json
import sys
from pathlib import Path

import yaml


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_design_choices(cfg: dict) -> dict:
    mc = cfg.get("model_config", {})
    tc = cfg.get("training_config", {})
    dc = cfg.get("data_config", {})
    vc = cfg.get("validation_config", {})

    return {
        "backbone": {
            "model_type": mc.get("model_type"),
            "pretrained": mc.get("pretrained_model_name_or_path"),
            "lora_rank": mc.get("lora_rank"),
            "lora_alpha": mc.get("lora_alpha"),
            "lora_target": mc.get("lora_layers"),
            "is_control_model": mc.get("is_control_model"),
        },
        "data": {
            "instance_data_root": dc.get("instance_data_root"),
            "single_resolution": (dc.get("single_height"), dc.get("single_width")),
            "caption_dropout": dc.get("caption_dropout_p"),
        },
        "control": {
            # The config does not spell out EE-pose vs ray-map; those live in the
            # data preprocessing (plucker/sketch). is_control_model=True is the
            # switch that enables the unified control branch in the transformer.
            "is_control_model": mc.get("is_control_model"),
        },
        "memory": {
            "history_sizes": tc.get("history_sizes"),
            "latent_window_size": tc.get("latent_window_size"),
            "has_multi_term_memory_patch": tc.get("has_multi_term_memory_patch"),
            "zero_history_timestep": tc.get("zero_history_timestep"),
            "guidance_cross_attn": tc.get("guidance_cross_attn"),
            "restrict_self_attn": tc.get("restrict_self_attn"),
        },
        "training": {
            "learning_rate": tc.get("learning_rate"),
            "lr_scheduler": tc.get("lr_scheduler"),
            "mixed_precision": tc.get("mixed_precision"),
            "gradient_checkpointing": tc.get("gradient_checkpointing"),
            "max_train_steps": tc.get("max_train_steps"),
            "history_amplification": tc.get("is_amplify_history"),
            "history_scale_mode": tc.get("history_scale_mode"),
            "train_lora_patch_embedding": tc.get("is_train_lora_patch_embedding"),
            "train_full_multi_term_memory_patch": tc.get("is_train_full_multi_term_memory_patchg"),
        },
        "validation": {
            "validation_height": vc.get("validation_height"),
            "validation_width": vc.get("validation_width"),
            "validation_max_num_frames": vc.get("validation_max_num_frames"),
            "num_inference_steps": vc.get("num_inference_steps"),
            "guidance_scale": vc.get("validation_guidance_scale"),
            "use_dynamic_shifting": vc.get("use_dynamic_shifting"),
            "time_shift_type": vc.get("time_shift_type"),
        },
    }


def roadmap_table(choices: dict) -> str:
    rows = [
        ("Backbone", f"{choices['backbone']['model_type']}", "Wan-[1.3B/5B] open backbone"),
        ("Parameter-efficient adaptation", f"LoRA r={choices['backbone']['lora_rank']}, alpha={choices['backbone']['lora_alpha']}", "Keep VAE/text-encoder frozen, adapt DiT"),
        ("Control interface", f"is_control_model={choices['control']['is_control_model']}", "Explicit pixel-aligned control (EE pose map + ray map)"),
        ("Long-horizon memory", f"history_sizes={choices['memory']['history_sizes']}, latent_window={choices['memory']['latent_window_size']}", "Hierarchical history + first-frame anchor"),
        ("Temporal encoding", f"zero_history_timestep={choices['memory']['zero_history_timestep']}", "Relative RoPE to avoid position drift"),
        ("Training stage", f"lr={choices['training']['learning_rate']}, scheduler={choices['training']['lr_scheduler']}", "Progressive multi-stage: foundation -> AR -> DMD"),
        ("Resolution", "x".join(map(str, choices["data"]["single_resolution"])), "Three-view 480x1920 (640x480 per view)"),
    ]

    lines = ["| Design axis | Config value | Paper roadmap (Table 5) |", "|---|---|---|"]
    for axis, value, roadmap in rows:
        lines.append(f"| {axis} | {value} | {roadmap} |")
    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path(__file__).parent / "official_snippets" / "stage_1_post_functrl_wan21.yaml"

    cfg = load_config(config_path)
    choices = extract_design_choices(cfg)

    out_dir = Path(__file__).parent / "probe_outputs"
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "config_architecture.json"
    with open(json_path, "w") as f:
        json.dump(choices, f, indent=2)

    md_path = out_dir / "config_architecture.md"
    with open(md_path, "w") as f:
        f.write("# GigaWorld-1 Config → Design Roadmap Mapping\n\n")
        f.write(f"Source config: `{config_path}`\n\n")
        f.write(roadmap_table(choices))
        f.write("\n\n## Raw extracted choices\n\n")
        f.write("```json\n")
        f.write(json.dumps(choices, indent=2))
        f.write("\n```\n")

    print(f"Wrote {json_path} and {md_path}")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
