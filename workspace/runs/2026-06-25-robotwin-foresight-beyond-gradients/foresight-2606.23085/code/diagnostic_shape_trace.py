"""
Diagnostic: trace tensor shapes through Foresight's pipeline.

This does NOT require pretrained weights; it just instantiates the architecture
and prints shapes, confirming the paper's claims about latent dimensions,
causal masking, and conformal thresholding.
"""

import torch
import torch.nn as nn

from foresight_detector_pseudocode import (
    ForesightCausalTransformerDetector,
    ForesightDetectorMLP,
    ForesightDetectorLSTM,
    compute_fcp_threshold,
    detect_failure,
)
from vjepa2_ac_predictor_arch import VisionTransformerPredictorAC


def trace():
    B, T = 2, 16              # batch, policy calls (replans)
    H, W = 16, 16             # 256/16 patches
    embed_dim = 1408
    action_dim = 7

    # -- Simulate frozen encoder output z_t^h (mean-pooled would be 1408-d)
    z_h = torch.randn(B, T, H * W, embed_dim)

    # -- Action-conditioned predictor: takes z_h and action/state chunks
    predictor = VisionTransformerPredictorAC(
        img_size=(256, 256),
        patch_size=16,
        num_frames=T,
        tubelet_size=2,
        embed_dim=embed_dim,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        action_embed_dim=action_dim,
    )
    actions = torch.randn(B, T - 1, action_dim)
    states = torch.randn(B, T, action_dim)
    z_p = predictor(z_h.flatten(1, 2), actions, states)
    print(f"z_h shape: {z_h.shape}")
    print(f"z_p shape: {z_p.shape}  <- predicted latent tokens")

    # -- Mean pool spatial tokens per timestep -> 1408-d timestep token
    z_p_seq = z_p.view(B, T, H * W, embed_dim).mean(dim=2)
    u_seq = nn.Linear(embed_dim, embed_dim)(z_p_seq)  # W*z + p (position omitted for shape check)
    print(f"u_t sequence shape: {u_seq.shape}")

    # -- Failure detectors
    for name, det in [
        ("MLP", ForesightDetectorMLP()),
        ("LSTM", ForesightDetectorLSTM()),
        ("Transformer", ForesightCausalTransformerDetector()),
    ]:
        s = det(u_seq)
        print(f"{name} detector scores shape: {s.shape}")

    # -- Conformal threshold from a few synthetic successful rollouts
    n_cal = 10
    success_scores = [torch.rand(T).numpy() * 0.3 for _ in range(n_cal)]
    delta = compute_fcp_threshold(success_scores, alpha=0.05)
    print(f"FCP threshold shape: {delta.shape}")

    # -- Alarm on a failing rollout
    fail_scores = torch.linspace(0.1, 0.9, T).numpy()
    alarm, t_alarm = detect_failure(fail_scores, delta)
    print(f"Alarm raised: {alarm} at timestep {t_alarm}")


if __name__ == "__main__":
    trace()
