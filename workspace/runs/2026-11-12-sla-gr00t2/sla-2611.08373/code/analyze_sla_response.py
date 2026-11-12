#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from openpi.policies import liberoplus_policy, piper_policy, policy_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config

from sla_counterfactual import (
    load_libero_observations,
    load_real_observations,
    perturb,
    preserve_state_mask,
)


def mean_rmse(delta):
    return float(np.sqrt(np.mean(delta ** 2, axis=(1, 2))).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["real", "libero"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-assets", type=Path, required=True)
    parser.add_argument("--norm-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.run_dir / "summary.json").read_text())
    frames = summary["frames"]
    active_dims = summary["action_channels_compared"]
    if args.mode == "real":
        observations = load_real_observations(args.data, summary["episode"], frames)
        grippers = [6, 13]
    else:
        observations = load_libero_observations(args.data, summary["episode"], frames)
        grippers = [6]

    preserve_state_mask(piper_policy.PiperInputs)
    preserve_state_mask(liberoplus_policy.LiberoPlusInputs)
    config_name = "train_real_world_handover_vegetable" if args.mode == "real" else "train_libero_plus"
    config = _config.get_config(config_name)
    norm_stats = _checkpoints.load_norm_stats(args.norm_assets, args.norm_id)
    policy = policy_config.create_trained_policy(config, args.checkpoint, norm_stats=norm_stats)

    state_token_differences = []
    image_different_pairs = 0
    for index, observation in enumerate(observations):
        partner = observations[(index + max(1, len(observations) // 3)) % len(observations)]
        plain = policy._input_transform(copy.deepcopy(observation))
        shuffled = policy._input_transform(perturb(observation, "state_shuffle", partner))
        state_token_differences.append(int(np.count_nonzero(plain["tokenized_prompt"] != shuffled["tokenized_prompt"])))
        if args.mode == "real":
            changed = any(not np.array_equal(observation["images"][key], partner["images"][key]) for key in observation["images"])
        else:
            changed = any(not np.array_equal(observation[key], partner[key]) for key in ["observation/image", "observation/wrist_image"])
        image_different_pairs += int(changed)

    actions = np.load(args.run_dir / "actions.npz")
    original = actions["original"][..., :active_dims]
    horizon = original.shape[1]
    edge = max(1, horizon // 5)
    non_grippers = [index for index in range(active_dims) if index not in grippers]
    result = {
        "samples": len(observations),
        "frames_source": "summary.json",
        "input_pair_checks": {
            "state_token_different_pairs": int(sum(count > 0 for count in state_token_differences)),
            "state_token_differences_min": min(state_token_differences),
            "state_token_differences_median": float(np.median(state_token_differences)),
            "state_token_differences_max": max(state_token_differences),
            "image_different_pairs": image_different_pairs,
        },
        "response_structure": {
            "edge_definition": f"first and last {edge} steps of each {horizon}-step action chunk",
            "original_action_rms_mean": float(np.sqrt(np.mean(original ** 2, axis=(1, 2))).mean()),
            "gripper_channels_zero_based": grippers,
            "conditions": {},
        },
    }
    for condition in ["state_zero", "state_shuffle", "state_dropout", "image_zero", "image_shuffle"]:
        delta = actions[condition][..., :active_dims] - original
        result["response_structure"]["conditions"][condition] = {
            "early_rmse_mean": mean_rmse(delta[:, :edge]),
            "late_rmse_mean": mean_rmse(delta[:, -edge:]),
            "gripper_rmse_mean": mean_rmse(delta[..., grippers]),
            "other_channels_rmse_mean": mean_rmse(delta[..., non_grippers]),
        }
    (args.run_dir / "response_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
