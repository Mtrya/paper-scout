"""Bidirectional conditional denoiser: a faithful miniature of arXiv:2608.00675.

One network learns p(x_{t+cd} | x_t, x_{t-cd}, cd) with cd in {+1, -1} via
conditional noise prediction; deterministic DDIM sampling makes the
forward-then-backward round trip well defined.

State space is the raw 3-D Lorenz state (no VAE -- the mechanism under study
does not depend on the autoencoder).
"""
import math
import numpy as np
import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden):
        super().__init__()
        self.net = nn.Linear(cond_dim, 2 * hidden)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, h, c):
        scale, shift = self.net(c).chunk(2, dim=-1)
        return h * (1 + scale) + shift


class BidirectionalDenoiser(nn.Module):
    """eps_theta(x_noisy, k, x_ctx_pair, cd)."""

    def __init__(self, state_dim=3, hidden=256, depth=4, cond_dim=128):
        super().__init__()
        self.state_dim = state_dim
        in_dim = state_dim + 2 * state_dim  # noisy target + context pair
        self.inp = nn.Linear(in_dim, hidden)
        self.k_emb = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.t_emb = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.cd_emb = nn.Embedding(2, cond_dim)
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)))
            self.films.append(FiLM(cond_dim, hidden))
        self.out = nn.Linear(hidden, state_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_noisy, k, t_phys, cd, ctx):
        """x_noisy [B,3], k [B] diffusion step in [0,1], t_phys [B] normalized
        simulation-time index, cd [B] in {0,1} (0=forward, 1=backward),
        ctx [B,6] the conditioning pair."""
        c = self.k_emb(k.unsqueeze(-1)) + self.t_emb(t_phys.unsqueeze(-1)) + self.cd_emb(cd)
        h = self.inp(torch.cat([x_noisy, ctx], dim=-1))
        for block, film in zip(self.blocks, self.films):
            h = h + film(block(h), c)
        return self.out(h)


class DDIM:
    """Deterministic DDIM (eta=0) over a linear-beta schedule."""

    def __init__(self, n_train_steps=1000, n_infer_steps=50, beta0=1e-4, beta1=0.02, device="cpu"):
        self.n_train = n_train_steps
        self.n_infer = n_infer_steps
        betas = torch.linspace(beta0, beta1, n_train_steps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.device = device
        # inference sub-sequence of timesteps (train-grid indices), ascending
        self.infer_ts = torch.linspace(0, n_train_steps - 1, n_infer_steps, device=device).long()

    def add_noise(self, x0, t_idx, eps):
        ac = self.alphas_cumprod[t_idx].unsqueeze(-1)
        return torch.sqrt(ac) * x0 + torch.sqrt(1 - ac) * eps

    def denoise_step(self, model, x, step_i, t_phys, cd, ctx):
        """One DDIM update from infer step i to i-1 (i from N-1 down to 1)."""
        t_cur = self.infer_ts[step_i]
        t_prev = self.infer_ts[step_i - 1]
        B = x.shape[0]
        k = (t_cur.float() / (self.n_train - 1)).expand(B)
        eps = model(x, k, t_phys, cd, ctx)
        ac_cur = self.alphas_cumprod[t_cur]
        ac_prev = self.alphas_cumprod[t_prev]
        x0_pred = (x - torch.sqrt(1 - ac_cur) * eps) / torch.sqrt(ac_cur)
        return torch.sqrt(ac_prev) * x0_pred + torch.sqrt(1 - ac_prev) * eps

    @torch.no_grad()
    def transition(self, model, x_target_seed, ctx, t_phys, cd, batch=4096):
        """Sample x_{t+cd} ~ p_theta(. | pair) deterministically from pure noise.

        x_target_seed: noise generator state handled outside; here we draw the
        terminal noise from a fixed torch.Generator for determinism.
        """
        B = ctx.shape[0]
        x = x_target_seed
        outs = []
        for s in range(0, B, batch):
            xs = x[s:s + batch]
            for i in range(self.n_infer - 1, 0, -1):
                xs = self.denoise_step(model, xs, i, t_phys[s:s + batch], cd[s:s + batch], ctx[s:s + batch])
            outs.append(xs)
        return torch.cat(outs)


@torch.no_grad()
def rollout(model, ddim, seed_pair, n_steps, direction, t_start, n_traj_total, gen):
    """Autoregressive rollout. seed_pair [B,2,3] chronological (x_{t-1}, x_t).

    direction +1: predicts x_{t+1..t+n}; returns tensor [B, n_steps, 3].
    direction -1: predicts x_{t-2..t-n-1} backwards; returns [B, n_steps, 3]
    in rollout order (first element = one step beyond the pair in the
    backward direction).
    t_phys values are normalized physical-time indices of the pair's latest
    index; each step advances/retreats by 1/n_traj_total.
    """
    B = seed_pair.shape[0]
    cd = torch.full((B,), 0 if direction > 0 else 1, dtype=torch.long, device=seed_pair.device)
    cur = seed_pair.clone()  # [B,2,3] chronological
    outs = []
    for i in range(n_steps):
        t_phys = torch.full((B,), (t_start + direction * i) / n_traj_total,
                            dtype=torch.float32, device=seed_pair.device)
        noise = torch.randn(cur.shape[0], cur.shape[-1], generator=gen, device="cpu").to(seed_pair.device)
        nxt = ddim.transition(model, noise, cur.reshape(B, -1), t_phys, cd)
        outs.append(nxt)
        if direction > 0:
            cur = torch.stack([cur[:, 1], nxt], dim=1)
        else:
            cur = torch.stack([nxt, cur[:, 0]], dim=1)
    return torch.stack(outs, dim=1)
