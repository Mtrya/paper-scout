"""
icwm_data_construction.py

Self-contained probe that demonstrates how In-Context World Modeling (ICWM)
builds training samples and how the test-time probing protocol is executed.

Reference paper: "In-Context World Modeling for Robotic Control" (arXiv:2606.26025)
Key details used:
    - N = 5 task-agnostic interaction clips are prepended as context.
    - Each clip is a tuple (o_i^s, a_i, o_i^e): start image, action, end image.
    - At test time the robot first performs ~20 random workspace probes,
      then samples N=5 triplets from the recorded transitions as the prefix.
    - The VLA backbone (Qwen2.5-VL-3B in the paper) is unchanged;
      only the input sequence is extended.

This script does NOT train a model. It preserves the exact data layout and
shows where the interaction context is inserted in the VLA input sequence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class InteractionClip:
    """One task-agnostic probing transition."""

    start_image: np.ndarray  # (H, W, 3) uint8
    action: np.ndarray       # continuous action vector
    end_image: np.ndarray    # (H, W, 3) uint8

    def to_tokens_stub(self) -> list[str]:
        """Stub tokenization: real code would use FAST / action tokenizer."""
        # In the paper actions are tokenized with FAST (DCT + BPE).
        # Here we just emit a placeholder string per action dimension.
        action_tokens = [f"<a{int(v):+04d}>" for v in np.round(self.action * 100)]
        return ["<img_start>"] + action_tokens + ["<img_end>"]


@dataclass
class ICWMTrainingSample:
    """One training sample: context clips + task query (image, instruction, target action)."""

    context_clips: list[InteractionClip]
    task_image: np.ndarray
    instruction: str
    target_action: np.ndarray

    def describe_sequence(self) -> dict[str, Any]:
        """Return a human-readable description of the multimodal sequence."""
        context_tokens: list[str] = []
        for i, clip in enumerate(self.context_clips):
            context_tokens.append(f"[clip_{i}:img_s]")
            context_tokens.extend([f"a{int(v):+04d}" for v in np.round(clip.action * 100)])
            context_tokens.append(f"[clip_{i}:img_e]")

        task_tokens = [f"[task_img]", f"\"{self.instruction}\"", "=>"]
        task_tokens.extend([f"a{int(v):+04d}" for v in np.round(self.target_action * 100)])

        return {
            "num_context_clips": len(self.context_clips),
            "context_token_summary": " ".join(context_tokens),
            "task_token_summary": " ".join(task_tokens),
            "full_sequence_summary": " ".join(context_tokens + task_tokens),
        }


def sample_interaction_clip(
    trajectory: dict[str, np.ndarray], rng: np.random.Generator
) -> InteractionClip:
    """Sample one (o_i^s, a_i, o_i^e) transition from a trajectory.

    trajectory keys (LIBERO-like):
        - "images": (T, H, W, 3) uint8
        - "actions": (T, action_dim) float32
    """
    T = trajectory["actions"].shape[0]
    assert T >= 2, "trajectory must contain at least one transition"
    i = rng.integers(0, T - 1)
    return InteractionClip(
        start_image=trajectory["images"][i],
        action=trajectory["actions"][i],
        end_image=trajectory["images"][i + 1],
    )


def build_icwm_training_sample(
    task_trajectory: dict[str, np.ndarray],
    interaction_pool: list[dict[str, np.ndarray]],
    instruction: str,
    n_context_clips: int = 5,
    rng: np.random.Generator | None = None,
) -> ICWMTrainingSample:
    """Build one ICWM training sample.

    During training, the paper prepends N randomly sampled interaction clips
    (collected across all training trajectories and viewpoints) to each task
    sample. The clips are task-agnostic; they only need to expose the current
    system dynamics.
    """
    rng = rng or np.random.default_rng()
    context_clips = [
        sample_interaction_clip(pool_traj, rng)
        for pool_traj in rng.choice(interaction_pool, size=n_context_clips, replace=True)
    ]
    # The task query uses the final frame and the expert action at that step.
    t_query = task_trajectory["actions"].shape[0] - 1
    return ICWMTrainingSample(
        context_clips=context_clips,
        task_image=task_trajectory["images"][t_query],
        instruction=instruction,
        target_action=task_trajectory["actions"][t_query],
    )


def active_probing_protocol(
    env_step_fn: callable,
    initial_obs: np.ndarray,
    n_probes: int = 20,
    action_dim: int = 7,
    rng: np.random.Generator | None = None,
) -> list[InteractionClip]:
    """Execute the ICWM test-time active probing phase.

    The robot performs N task-agnostic random workspace movements, records
    each (o_i^s, a_i, o_i^e) transition, and returns the pool. At inference
    time 5 clips are sampled from this pool and prepended as context.

    env_step_fn(action) -> next_obs (stub; in reality calls the robot/simulator).
    """
    rng = rng or np.random.default_rng()
    clips: list[InteractionClip] = []
    obs = initial_obs
    for _ in range(n_probes):
        action = rng.uniform(-1, 1, size=action_dim).astype(np.float32)
        next_obs = env_step_fn(obs, action)
        clips.append(InteractionClip(start_image=obs, action=action, end_image=next_obs))
        obs = next_obs
    return clips


def make_stub_trajectory(T: int = 30, action_dim: int = 7, image_size: int = 128) -> dict[str, np.ndarray]:
    """Create a fake trajectory for offline inspection."""
    rng = np.random.default_rng(0)
    return {
        "images": rng.integers(0, 256, size=(T, image_size, image_size, 3), dtype=np.uint8),
        "actions": rng.uniform(-1, 1, size=(T, action_dim)).astype(np.float32),
    }


def main() -> None:
    rng = np.random.default_rng(42)

    # 1) Build a small interaction pool (normally collected across all training
    #    trajectories and viewpoints in LIBERO).
    interaction_pool = [make_stub_trajectory(T=rng.integers(20, 50)) for _ in range(10)]

    # 2) Build a training sample.
    task_traj = make_stub_trajectory(T=25)
    sample = build_icwm_training_sample(
        task_trajectory=task_traj,
        interaction_pool=interaction_pool,
        instruction="pick up the black bowl and place it on the plate",
        n_context_clips=5,
        rng=rng,
    )

    desc = sample.describe_sequence()
    print("=" * 70)
    print("ICWM training sample sequence layout")
    print("=" * 70)
    print(json.dumps(desc, indent=2))

    # 3) Demonstrate test-time active probing.
    def stub_env_step(obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        # In reality this sends the action to the robot/simulator and returns
        # the next camera observation under the current configuration psi.
        return np.random.default_rng().integers(0, 255, size=obs.shape, dtype=np.uint8)

    probe_clips = active_probing_protocol(
        env_step_fn=stub_env_step,
        initial_obs=task_traj["images"][0],
        n_probes=20,
        action_dim=7,
        rng=rng,
    )
    print("\n" + "=" * 70)
    print("Test-time probing phase")
    print("=" * 70)
    print(f"Recorded {len(probe_clips)} probing transitions.")
    print(f"First probe action (rounded): {np.round(probe_clips[0].action, 3)}")
    print("At each inference step, N=5 of these clips are sampled as context.")

    # 4) Persist a tiny visual artifact.
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "icwm_sequence_summary.json").write_text(json.dumps(desc, indent=2))
    Image.fromarray(sample.task_image).save(out_dir / "task_query_image.png")
    print(f"\nSaved summary to {out_dir}")


if __name__ == "__main__":
    main()
