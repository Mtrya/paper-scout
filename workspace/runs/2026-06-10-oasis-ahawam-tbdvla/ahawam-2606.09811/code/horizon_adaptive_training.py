"""
PyTorch reconstruction of horizon-adaptive offset training (Eqs. 11-12).

During training, instead of always aligning the first action chunk to the
planner start, AHA-WAM samples a random offset δ ∈ [0, h_a) and shifts the
action-chunk grid by δ within the planner horizon [τ, τ + h_v).

This teaches the action DiT to consume long-horizon planner context under
variable action-start offsets, matching the phase shifts induced by
asynchronous streaming deployment.
"""

import random
from typing import Dict, Tuple

import torch
import torch.nn as nn


class HorizonAdaptiveOffsetSampler:
    """
    Training-time sampler for action-chunk offsets within the video horizon.

    Args:
        video_horizon: h_v, number of video frames in the planner window.
        action_horizon: h_a, number of actions per chunk.
        action_frames_per_step: How many action steps correspond to one video
            frame. In AHA-WAM the ratio is 8 (Table 4: "Video/action frequency
            ratio = 8").
    """

    def __init__(
        self,
        video_horizon: int = 64,
        action_horizon: int = 16,
        action_frames_per_step: int = 1,
    ):
        self.video_horizon = video_horizon
        self.action_horizon = action_horizon
        self.action_frames_per_step = action_frames_per_step

    def sample_offset(self) -> int:
        """
        Eq. 11: δ ~ Uniform{0, 1, ..., h_a - 1}

        Returns:
            Integer offset in [0, action_horizon).
        """
        return random.randint(0, self.action_horizon - 1)

    def align_action_to_planner(
        self,
        action_sequence: torch.Tensor,       # [B, T_total, action_dim]
        planner_start: int,
        offset: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Extract an action chunk aligned with the planner window at the given
        phase offset.

        Args:
            action_sequence: Full action trajectory.
            planner_start: Start timestep τ of the video planner window.
            offset: Sampled offset δ.

        Returns:
            action_chunk: [B, h_a, action_dim] aligned chunk.
            chunk_start: Absolute start index of the chunk in the trajectory.
        """
        B, T_total, action_dim = action_sequence.shape
        chunk_start = planner_start + offset

        # Ensure the chunk fits inside the available trajectory.
        if chunk_start + self.action_horizon > T_total:
            # In practice the dataloader pads or truncates; here we clamp.
            chunk_start = max(0, T_total - self.action_horizon)

        action_chunk = action_sequence[:, chunk_start : chunk_start + self.action_horizon, :]

        # Pad if necessary (should not happen with proper data pipeline).
        if action_chunk.shape[1] < self.action_horizon:
            pad = torch.zeros(
                B,
                self.action_horizon - action_chunk.shape[1],
                action_dim,
                dtype=action_chunk.dtype,
                device=action_chunk.device,
            )
            action_chunk = torch.cat([action_chunk, pad], dim=1)

        return action_chunk, chunk_start


def horizon_adaptive_action_loss(
    model: nn.Module,
    video_latents: torch.Tensor,       # [B, C, h_v, H, W]  planner input
    action_sequence: torch.Tensor,     # [B, T_total, action_dim]
    language_context: torch.Tensor,    # [B, L, d]
    context_mask: torch.Tensor,        # [B, L]
    sampler: HorizonAdaptiveOffsetSampler,
    flow_matching_loss_fn: callable,
    num_offsets_per_sample: int = 1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Eq. 12: L_a = E_δ [ L_FM(A_τ^δ) ]

    For each training sample, sample one or more offsets and average the
    flow-matching loss over offset-aligned action chunks.

    Args:
        model: The AHA-WAM model (or a training wrapper around it).
        video_latents: Future video latents over the planner horizon h_v.
        action_sequence: Full action trajectory.
        language_context: Text/language embeddings.
        context_mask: Mask for language context.
        sampler: HorizonAdaptiveOffsetSampler instance.
        flow_matching_loss_fn: Callable that computes L_FM for a given action
            chunk under the current model state.
        num_offsets_per_sample: How many offsets to sample per batch item.
            Paper does not specify; typical is 1 per forward pass.

    Returns:
        loss: Scalar averaged over offsets.
        metrics: Dict with auxiliary info.
    """
    losses = []
    offsets = []

    for _ in range(num_offsets_per_sample):
        offset = sampler.sample_offset()
        offsets.append(offset)

        # Align action chunk to planner window starting at τ=0 for simplicity.
        action_chunk, _ = sampler.align_action_to_planner(
            action_sequence=action_sequence,
            planner_start=0,
            offset=offset,
        )

        # Compute flow-matching loss for this offset-aligned chunk.
        # The model internally routes planner context through OVCR before
        # action denoising.
        loss_chunk = flow_matching_loss_fn(
            action_chunk=action_chunk,
            video_latents=video_latents,
            context=language_context,
            context_mask=context_mask,
        )
        losses.append(loss_chunk)

    loss = torch.stack(losses).mean()
    metrics = {
        "offset_mean": sum(offsets) / len(offsets),
        "offset_min": min(offsets),
        "offset_max": max(offsets),
    }
    return loss, metrics


if __name__ == "__main__":
    sampler = HorizonAdaptiveOffsetSampler(video_horizon=64, action_horizon=16)

    # Verify that offsets cover the expected range.
    offsets = [sampler.sample_offset() for _ in range(1000)]
    assert min(offsets) == 0
    assert max(offsets) == 15
    print(f"Offset distribution over 1000 samples: mean={sum(offsets)/len(offsets):.2f}")

    # Verify chunk alignment.
    action_seq = torch.randn(2, 200, 14)  # [B, T, action_dim]
    chunk, start = sampler.align_action_to_planner(action_seq, planner_start=0, offset=5)
    assert chunk.shape == (2, 16, 14)
    assert start == 5
    print("Horizon-adaptive offset training reconstruction sanity check passed.")
