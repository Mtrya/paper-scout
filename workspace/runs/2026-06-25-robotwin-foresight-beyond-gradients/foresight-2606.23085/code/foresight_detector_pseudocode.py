"""
Foresight failure detector -- reconstructed from the paper (Section 4.2-4.4)
because the official Foresight code repository is not publicly available.

This file is a faithful, minimal reconstruction of the architecture described
in the paper and Appendix 7, intended to make the method concrete and traceable.
"""

import torch
import torch.nn as nn
import numpy as np


class ForesightCausalTransformerDetector(nn.Module):
    """
    Paper spec (Appendix 7):
      input_dim = 1408 (mean-pooled V-JEPA 2-AC predicted latent)
      2 layers, hidden_dim=256, 4 attention heads, ff_dim=1024=4*256
      dropout=0.1, L2 lambda=1e-2, Adam LR=1e-4, 300 epochs
    """

    def __init__(self, input_dim=1408, hidden_dim=256, num_layers=2,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,      # pre-norm
        )
        # Causal mask: position i attends only to positions <= i
        self.register_buffer(
            'causal_mask',
            nn.Transformer.generate_square_subsequent_mask(4096)  # max horizon
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, u_seq):
        """
        Args:
            u_seq: [B, T, input_dim] sequence of timestep tokens
                   u_t = W*z_t^p + positional_encoding(t)
        Returns:
            s: [B, T] per-timestep failure scores in [0, 1]
        """
        x = self.input_proj(u_seq)
        T = x.size(1)
        x = self.encoder(x, mask=self.causal_mask[:T, :T], is_causal=True)
        s = self.head(x).squeeze(-1)
        return s


class ForesightDetectorMLP(nn.Module):
    """Baseline used in ablations; treats each timestep independently."""
    def __init__(self, input_dim=1408, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, u_seq):
        return self.net(u_seq).squeeze(-1)


class ForesightDetectorLSTM(nn.Module):
    """Baseline used in ablations."""
    def __init__(self, input_dim=1408, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, u_seq):
        out, _ = self.lstm(u_seq)
        return self.head(out).squeeze(-1)


def compute_fcp_threshold(success_score_trajectories, alpha=0.05):
    """
    Functional conformal prediction threshold (Appendix 9).

    Args:
        success_score_trajectories: list of arrays, each [T_i] failure scores
                                    for one successful calibration rollout.
        alpha: desired miscoverage / false-positive rate.
    Returns:
        delta: [T_max] time-varying threshold.
    """
    T_max = max(len(s) for s in success_score_trajectories)

    # Pad trajectories to common length for mean/std computation
    def pad(s):
        arr = np.zeros(T_max)
        arr[:len(s)] = s
        return arr

    S = np.stack([pad(s) for s in success_score_trajectories])  # [n, T_max]
    mu = S.mean(axis=0)
    sigma = S.std(axis=0) + 1e-8

    # Nonconformity score: supremum over time of normalized deviation from mean
    R = np.max((S - mu) / sigma, axis=1)
    q_hat = np.quantile(R, 1 - alpha)

    delta = mu + q_hat * sigma
    return delta


def detect_failure(score_trajectory, threshold):
    """Alarm fires at first timestep where s_t >= delta_t."""
    T = min(len(score_trajectory), len(threshold))
    for t in range(T):
        if score_trajectory[t] >= threshold[t]:
            return True, t
    return False, T


# Training objective (Section 4.3):
#   Each timestep inherits the rollout-level binary label y in {0, 1}.
#   BCE loss with early-detection weighting:
#       loss = sum_t w_t * BCE(s_t, y)
#   where w_t is larger for earlier timesteps to encourage failure alarms
#   before task termination.
