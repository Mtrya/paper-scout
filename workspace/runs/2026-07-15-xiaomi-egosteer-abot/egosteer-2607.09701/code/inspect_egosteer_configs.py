#!/usr/bin/env python3
"""Probe: inspect EgoSteer configs and verify key architectural claims.

Reads the Hydra config tree in the official EgoSteer repository and prints the
parameters that correspond to the paper's architectural claims. This script is
read-only: it does not download weights or run training.

Usage (from workspace root):
    python runs/2026-07-15-xiaomi-egosteer-abot/egosteer-2607.09701/code/inspect_egosteer_configs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The official repo is cloned under code/ by the deep-dive agent.
REPO_ROOT = Path(__file__).resolve().parents[4] / "code" / "egosteer-2607.09701"
CONFIG_ROOT = REPO_ROOT / "src" / "config"


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc
    with open(path, "r") as f:
        return yaml.safe_load(f)


def require(key_path: list[str], d: dict):
    node = d
    for k in key_path:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(" -> ".join(key_path))
        node = node[k]
    return node


def main() -> int:
    if not CONFIG_ROOT.exists():
        print(f"ERROR: expected config tree at {CONFIG_ROOT}", file=sys.stderr)
        return 1

    model_cfg = load_yaml(CONFIG_ROOT / "model" / "qwen3_vl_2b.yaml")
    train_cfg = load_yaml(CONFIG_ROOT / "training" / "default.yaml")
    data_cfg = load_yaml(CONFIG_ROOT / "data" / "unified_wds.yaml")
    wm_cfg = load_yaml(CONFIG_ROOT / "world_model" / "frozen_regression.yaml")
    exp_cfg = load_yaml(CONFIG_ROOT / "experiment" / "egosteer_qwen3_vl.yaml")

    report = {
        "repo_root": str(REPO_ROOT),
        "config_files": {
            "model": "src/config/model/qwen3_vl_2b.yaml",
            "training": "src/config/training/default.yaml",
            "data": "src/config/data/unified_wds.yaml",
            "world_model": "src/config/world_model/frozen_regression.yaml",
            "experiment": "src/config/experiment/egosteer_qwen3_vl.yaml",
        },
        "backbone": {
            "model_name_or_path": require(["pretrained", "model_name_or_path"], model_cfg),
            "vlm_hidden_size": require(["dims", "vlm_hidden_size"], model_cfg),
            "text_attn_implementation": require(["pretrained", "text_attn_implementation"], model_cfg),
        },
        "unified_action_space": {
            "state_dim": require(["dims", "state_dim"], model_cfg),
            "action_dim": require(["dims", "action_dim"], model_cfg),
            "state_history_frames": require(["data", "shape_meta", "obs", "state", "horizon"], data_cfg),
            "action_horizon": require(["data", "shape_meta", "action", "horizon"], data_cfg),
            "lowdim_layout_note": "world-frame wrist_state(18) + hand_state(30) + wrist_action(18) + hand_action(30)",
        },
        "action_expert": {
            "hidden_size": require(["dims", "action_hidden_size"], model_cfg),
            "intermediate_size": require(["policy", "flow_expert", "intermediate_size"], model_cfg),
            "num_layers": require(["policy", "flow_expert", "num_layers"], model_cfg),
            "num_heads": require(["policy", "flow_expert", "num_heads"], model_cfg),
            "use_adaln": require(["policy", "flow_expert", "use_adaln"], model_cfg),
            "use_kv_projection": require(["policy", "flow_expert", "use_kv_projection"], model_cfg),
        },
        "world_model": {
            "enabled": require(["world_model", "enabled"], wm_cfg),
            "frozen_teacher": require(["world_model", "frozen_teacher", "model_name_or_path"], wm_cfg),
            "teacher_image_size": require(["world_model", "teacher_image_size"], wm_cfg),
            "teacher_patch_size": require(["world_model", "teacher_patch_size"], wm_cfg),
            "upsample_factor": require(["world_model", "upsample_factor"], wm_cfg),
            "num_expert_layers": require(["world_model", "world_model_expert", "num_layers"], wm_cfg),
            "expert_hidden_size": require(["world_model", "world_model_expert", "hidden_size"], wm_cfg),
            "expert_intermediate_size": require(["world_model", "world_model_expert", "intermediate_size"], wm_cfg),
            "action_conditioning": require(["world_model", "action_conditioning"], wm_cfg),
            "motion_conditioning": require(["world_model", "motion_conditioning"], wm_cfg),
            "future_frame_horizon": require(["world_model", "future_frame", "horizon"], wm_cfg),
            "future_frame_stride": require(["world_model", "future_frame", "stride"], wm_cfg),
        },
        "training_time_rtc": {
            "rtc_enabled_default": require(["policy", "rtc_config", "enabled"], model_cfg),
            "delay_strategy": require(["policy", "rtc_config", "delay_strategy"], model_cfg),
            "max_delay": require(["policy", "rtc_config", "max_delay"], model_cfg),
            "note": "disabled by default in base config; intended to be enabled for real-robot post-training",
        },
        "losses": {
            "ce_loss_weight": require(["policy", "loss_config", "ce_loss_weight"], model_cfg),
            "flow_loss_weight": require(["policy", "loss_config", "flow_loss_weight"], model_cfg),
            "wm_loss_weight": require(["policy", "loss_config", "wm_loss_weight"], model_cfg),
        },
        "training": {
            "action_expert_lr": require(["optimizer", "action", "lr"], train_cfg),
            "vlm_lr": require(["optimizer", "vlm", "lr"], train_cfg),
            "world_model_lr": require(["optimizer", "world_model", "lr"], train_cfg),
            "ar_action_heads_lr": require(["optimizer", "ar_action_heads", "lr"], train_cfg),
            "flow_sampling": require(["flow", "sampling"], train_cfg),
            "flow_alpha": require(["flow", "alpha"], train_cfg),
            "flow_beta": require(["flow", "beta"], train_cfg),
            "num_inference_steps": require(["flow", "num_inference_steps"], train_cfg),
            "num_parallel_t": require(["flow", "num_parallel_t"], train_cfg),
        },
        "data": {
            "target_image_size": require(["data", "target_image_size"], data_cfg),
            "video_base_fps": require(["data", "video_base_fps"], data_cfg),
            "shuffle_buffer": require(["dataset", "vla_dataset", "shuffle_buffer"], data_cfg),
            "keep_ratio": require(["dataset", "vla_dataset", "keep_ratio"], data_cfg),
        },
        "experiment": {
            "name": require(["name"], exp_cfg),
            "workspace": require(["_target_"], exp_cfg),
        },
    }

    print(json.dumps(report, indent=2))

    # A few sanity checks against the paper's stated numbers.
    checks = []
    ae = report["action_expert"]
    checks.append(("action expert hidden size == 1024", ae["hidden_size"] == 1024))
    checks.append(("action expert layers == 14", ae["num_layers"] == 14))
    checks.append(("action expert intermediate size == 2816", ae["intermediate_size"] == 2816))
    checks.append(("action expert AdaLN enabled", ae["use_adaln"] is True))

    wm = report["world_model"]
    checks.append(("WM expert layers == 4", wm["num_expert_layers"] == 4))
    checks.append(("WM expert hidden size == 1024", wm["expert_hidden_size"] == 1024))
    checks.append(("WM teacher is DINOv3 ViT-L/16", "dinov3-vitl16" in wm["frozen_teacher"]))
    checks.append(("WM future horizon == 1", wm["future_frame_horizon"] == 1))

    us = report["unified_action_space"]
    checks.append(("state/action dim == 48", us["state_dim"] == 48 and us["action_dim"] == 48))
    checks.append(("action horizon == 32", us["action_horizon"] == 32))
    checks.append(("state history == 6 frames", us["state_history_frames"] == 6))

    losses = report["losses"]
    checks.append(("CE weight == 0.05", abs(losses["ce_loss_weight"] - 0.05) < 1e-6))
    checks.append(("flow weight == 1.0", abs(losses["flow_loss_weight"] - 1.0) < 1e-6))
    checks.append(("WM weight == 1.0", abs(losses["wm_loss_weight"] - 1.0) < 1e-6))

    print("\n--- sanity checks ---")
    failed = 0
    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {desc}")
        if not ok:
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
