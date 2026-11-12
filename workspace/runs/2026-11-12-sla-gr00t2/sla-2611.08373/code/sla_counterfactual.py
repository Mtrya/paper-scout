#!/usr/bin/env python3
import argparse
import copy
import csv
import io
import json
from pathlib import Path

import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from openpi.models import model as _model
from openpi.policies import liberoplus_policy, piper_policy, policy_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


def preserve_state_mask(cls):
    original = cls.__call__

    def wrapped(self, data):
        result = original(self, data)
        if "state_mask" in data:
            result["state_mask"] = data["state_mask"]
        return result

    cls.__call__ = wrapped


def decode_video(path, selected):
    import os
    import re

    selected = sorted(set(int(x) for x in selected))
    binary = os.environ.get("FFMPEG_BINARY", "ffmpeg")
    probe = subprocess.run([binary, "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b", probe.stderr)
    if not match:
        raise RuntimeError(probe.stderr)
    width, height = map(int, match.groups())
    expression = "+".join(f"eq(n\\,{index})" for index in selected)
    process = subprocess.run(
        [binary, "-i", str(path), "-vf", f"select={expression}", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    )
    frame_size = width * height * 3
    if len(process.stdout) != len(selected) * frame_size:
        raise RuntimeError(f"decoded {len(process.stdout) / frame_size} frames, expected {len(selected)}")
    return {
        index: np.frombuffer(process.stdout[offset * frame_size:(offset + 1) * frame_size], dtype=np.uint8).reshape(height, width, 3).copy()
        for offset, index in enumerate(selected)
    }


def load_real_observations(root, episode, selected):
    prefix = root / f"handover_vege{episode}"
    table = pq.read_table(prefix / f"data/chunk-000/episode_{episode - 1:06d}.parquet")
    rows = table.to_pydict()
    tasks = [json.loads(line) for line in (prefix / "meta/tasks.jsonl").read_text().splitlines()]
    task_by_index = {r["task_index"]: r["task"] for r in tasks}
    views = {
        "cam_high": "right_front",
        "cam_left_wrist": "left_wrist",
        "cam_right_wrist": "right_wrist",
    }
    decoded = {
        target: decode_video(prefix / f"videos/chunk-000/observation.images.{source}/episode_{episode - 1:06d}.mp4", selected)
        for target, source in views.items()
    }
    observations = []
    for frame in selected:
        observations.append({
            "images": {key: decoded[key][frame] for key in views},
            "joint_state": np.asarray(rows["observation.state"][frame], dtype=np.float32),
            "prompt": task_by_index[rows["task_index"][frame]],
        })
    return observations


def load_libero_observations(root, episode, selected):
    path = root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    table = pq.read_table(path)
    rows = table.to_pydict()
    tasks = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines()]
    task_by_index = {r["task_index"]: r["task"] for r in tasks}

    def image(value):
        if isinstance(value, dict):
            value = value.get("bytes", value.get("path"))
        if isinstance(value, (bytes, bytearray)):
            return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
        return np.asarray(value)

    observations = []
    for frame in selected:
        observations.append({
            "observation/image": image(rows["image"][frame]),
            "observation/wrist_image": image(rows["wrist_image"][frame]),
            "observation/state": np.asarray(rows["state"][frame], dtype=np.float32),
            "prompt": task_by_index[rows["task_index"][frame]],
        })
    return observations


def perturb(observation, condition, partner):
    result = copy.deepcopy(observation)
    if condition == "original":
        return result
    if condition == "state_zero":
        key = "joint_state" if "joint_state" in result else "observation/state" if "observation/state" in result else "state"
        result[key] = np.zeros_like(result[key])
    elif condition == "state_shuffle":
        key = "joint_state" if "joint_state" in result else "observation/state" if "observation/state" in result else "state"
        result[key] = partner[key].copy()
    elif condition == "state_dropout":
        result["state_mask"] = np.False_
    elif condition == "image_zero":
        if "images" in result:
            result["images"] = {key: np.zeros_like(value) for key, value in result["images"].items()}
        else:
            result["observation/image"] = np.zeros_like(result["observation/image"])
            result["observation/wrist_image"] = np.zeros_like(result["observation/wrist_image"])
    elif condition == "image_shuffle":
        if "images" in result:
            result["images"] = {key: value.copy() for key, value in partner["images"].items()}
        else:
            result["observation/image"] = partner["observation/image"].copy()
            result["observation/wrist_image"] = partner["observation/wrist_image"].copy()
    else:
        raise ValueError(condition)
    return result


def bootstrap_interval(values, seed=20261112):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def make_figure(rows, actions, conditions, active_dims, output):
    labels = {
        "state_zero": "State = 0",
        "state_shuffle": "State shuffled",
        "state_dropout": "State tokens removed",
        "image_zero": "Images = black",
        "image_shuffle": "Images shuffled",
    }
    colors = {
        "state_zero": "#8172B3",
        "state_shuffle": "#55A868",
        "state_dropout": "#4C72B0",
        "image_zero": "#DD8452",
        "image_shuffle": "#C44E52",
    }
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
    fig = plt.figure(figsize=(12, 8.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.4, wspace=0.25)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    x = np.arange(len(conditions))
    means = [np.mean([float(r["relative_rmse"]) for r in rows if r["condition"] == c]) for c in conditions]
    errors = []
    for c in conditions:
        values = [float(r["relative_rmse"]) for r in rows if r["condition"] == c]
        lo, hi = bootstrap_interval(values)
        errors.append([means[len(errors)] - lo, hi - means[len(errors)]])
    axes[0].bar(x, means, yerr=np.asarray(errors).T, capsize=4, color=[colors[c] for c in conditions])
    axes[0].set_xticks(x, [labels[c].replace(" ", "\n") for c in conditions])
    axes[0].set_ylabel("Relative action RMSE")
    axes[0].set_title("Paired counterfactual response")
    original = actions["original"]
    for condition in conditions:
        delta = np.sqrt(np.mean((actions[condition][..., :active_dims] - original[..., :active_dims]) ** 2, axis=(0, 2)))
        axes[1].plot(np.arange(original.shape[1]), delta, color=colors[condition], label=labels[condition])
    axes[1].set(xlabel="Action-chunk timestep", ylabel="Action RMSE", title="Where the chunk changes")
    axes[1].legend(frameon=False, fontsize=9)
    width = 0.16
    for index, condition in enumerate(conditions):
        delta = np.sqrt(np.mean((actions[condition][..., :active_dims] - original[..., :active_dims]) ** 2, axis=(0, 1)))
        axes[2].bar(np.arange(active_dims) + (index - 2) * width, delta, width=width, color=colors[condition], label=labels[condition])
    axes[2].set(xlabel="Action channel", ylabel="Action RMSE", title="Which channels respond")
    axes[2].set_xticks(np.arange(active_dims))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["real", "libero"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-assets", type=Path, required=True)
    parser.add_argument("--norm-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    preserve_state_mask(piper_policy.PiperInputs)
    preserve_state_mask(liberoplus_policy.LiberoPlusInputs)
    config_name = "train_real_world_handover_vegetable" if args.mode == "real" else "train_libero_plus"
    config = _config.get_config(config_name)
    norm_stats = _checkpoints.load_norm_stats(args.norm_assets, args.norm_id)
    policy = policy_config.create_trained_policy(config, args.checkpoint, norm_stats=norm_stats)
    policy._output_transform = lambda data: data
    rng = np.random.default_rng(args.seed)
    frame_count = pq.read_metadata(
        args.data / (
            f"handover_vege{args.episode}/data/chunk-000/episode_{args.episode - 1:06d}.parquet"
            if args.mode == "real"
            else f"data/chunk-{args.episode // 1000:03d}/episode_{args.episode:06d}.parquet"
        )
    ).num_rows
    candidates = np.arange(0, frame_count - 1, max(1, frame_count // 256))
    selected = np.sort(rng.choice(candidates, size=min(args.num_samples, len(candidates)), replace=False))
    if args.mode == "real":
        observations = load_real_observations(args.data, args.episode, selected)
        active_dims = 14
    else:
        observations = load_libero_observations(args.data, args.episode, selected)
        active_dims = 7
    conditions = ["state_zero", "state_shuffle", "state_dropout", "image_zero", "image_shuffle"]
    action_horizon = config.model.action_horizon
    noise = rng.standard_normal((action_horizon, config.model.action_dim), dtype=np.float32)
    outputs = {"original": []}
    outputs.update({condition: [] for condition in conditions})
    rows = []
    plain = policy._input_transform(copy.deepcopy(observations[0]))
    dropped = policy._input_transform(perturb(observations[0], "state_dropout", observations[0]))
    tokenization_check = {
        "prompt_tokens": int(plain["tokenized_prompt_mask"].sum()),
        "prompt_tokens_after_state_dropout": int(dropped["tokenized_prompt_mask"].sum()),
    }
    tokenization_check["removed_tokens"] = tokenization_check["prompt_tokens"] - tokenization_check["prompt_tokens_after_state_dropout"]
    if tokenization_check["removed_tokens"] <= 0:
        raise RuntimeError(f"state dropout did not remove prompt tokens: {tokenization_check}")
    (args.output / "tokenization_check.json").write_text(json.dumps(tokenization_check, indent=2) + "\n")
    print(f"state dropout tokenization: {tokenization_check}", flush=True)
    for index, observation in enumerate(observations):
        partner = observations[(index + max(1, len(observations) // 3)) % len(observations)]
        sample_actions = {}
        for condition in ["original", *conditions]:
            result = policy.infer(perturb(observation, condition, partner), noise=noise)
            action = np.asarray(result["actions"], dtype=np.float32)
            outputs[condition].append(action)
            sample_actions[condition] = action
        original = sample_actions["original"][..., :active_dims]
        original_rms = float(np.sqrt(np.mean(original ** 2)))
        for condition in conditions:
            changed = sample_actions[condition][..., :active_dims]
            delta = changed - original
            rmse = float(np.sqrt(np.mean(delta ** 2)))
            rows.append({
                "sample": index,
                "frame": int(selected[index]),
                "condition": condition,
                "rmse": rmse,
                "relative_rmse": rmse / (original_rms + 1e-6),
                "original_rms": original_rms,
                "max_abs_delta": float(np.abs(delta).max()),
                "cosine": float(np.sum(changed * original) / (np.linalg.norm(changed) * np.linalg.norm(original) + 1e-8)),
            })
        if index == 0:
            print(f"qualification sample: {len(rows)} paired rows", flush=True)
            (args.output / "qualification.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"completed {index + 1}/{len(observations)}", flush=True)
    arrays = {key: np.asarray(value) for key, value in outputs.items()}
    np.savez_compressed(args.output / "actions.npz", **arrays)
    with (args.output / "paired_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "config": config_name,
        "episode": args.episode,
        "samples": len(observations),
        "frames": selected.tolist(),
        "seed": args.seed,
        "action_horizon": action_horizon,
        "action_channels_compared": active_dims,
        "metric_space": "normalized model actions before unnormalization and absolute-action reconstruction",
        "conditions": {},
    }
    for condition in conditions:
        selected_rows = [r for r in rows if r["condition"] == condition]
        values = [r["relative_rmse"] for r in selected_rows]
        lo, hi = bootstrap_interval(values)
        summary["conditions"][condition] = {
            "relative_rmse_mean": float(np.mean(values)),
            "relative_rmse_median": float(np.median(values)),
            "relative_rmse_bootstrap95": [lo, hi],
            "rmse_mean": float(np.mean([r["rmse"] for r in selected_rows])),
            "cosine_mean": float(np.mean([r["cosine"] for r in selected_rows])),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figure(rows, arrays, conditions, active_dims, args.output / "counterfactual.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
