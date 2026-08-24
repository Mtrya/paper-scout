"""Closed-loop evaluation.

Rollout protocol: at each step the policy sees the image rendered from the TRUE
camera yaw, predicts a delta (base frame for the 'base' head; camera frame for the
'cam' head). For 'cam', execution synthesizes the base-frame delta with
R_bc(yaw_est), where yaw_est = yaw + eps(t).

Error structures (marginal N(0, sigma), exact match across structures):
- 'iid':    eps resampled every step
- 'static': eps fixed for the whole episode (misaligned/knocked camera)
- 'ar':     eps_{t+1} = rho*eps_t + N(0, sigma*sqrt(1-rho^2))  (filtered/drift)

`query_k` controls how often the policy is re-queried: with k>1 the same command
is executed open-loop for k steps (mimicking a low-rate VLA controller).
"""
from __future__ import annotations

import numpy as np
import torch

import blob_world as bw
from models import Policy


def _error_seq(rng: np.random.Generator, struct: str, sigma_deg: float,
               rho: float | None, n: int) -> np.ndarray:
    sigma = np.deg2rad(sigma_deg)
    if struct == 'iid':
        return rng.normal(0.0, sigma, n)
    if struct == 'static':
        return np.full(n, rng.normal(0.0, sigma))
    if struct == 'ar':
        seq = np.empty(n)
        e = rng.normal(0.0, sigma)
        sd_eta = sigma * np.sqrt(max(1.0 - rho * rho, 0.0))
        for t in range(n):
            seq[t] = e
            e = rho * e + rng.normal(0.0, sd_eta)
        return seq
    raise ValueError(struct)


@torch.no_grad()
def rollout(model: Policy, head: str, yaw: float, rng: np.random.Generator,
            struct: str | None = None, sigma_deg: float = 0.0,
            rho: float | None = None, query_k: int = 1,
            max_steps: int = bw.MAX_STEPS) -> tuple[float, bool, int]:
    """Run one closed-loop episode. Returns (final_dist, success, steps_taken)."""
    model.eval()
    ee, tgt = bw.sample_state(rng)
    p = ee.copy()
    if struct is not None and sigma_deg > 0:
        eps = _error_seq(rng, struct, sigma_deg, rho, max_steps)
    else:
        eps = np.zeros(max_steps)
    cmd = None  # camera-frame command between queries
    last_query = -query_k
    for t in range(max_steps):
        if t - last_query >= query_k:
            img = bw.render(p, tgt, yaw)
            x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            cmd = model(x).squeeze(0).numpy()
            last_query = t
        if head == 'cam':
            R = bw.rz(yaw + float(eps[t])) @ bw._R0
            d_b = R @ cmd
        else:
            d_b = cmd
        p = bw.step_world(p, d_b, rng)
        if np.linalg.norm(p - tgt) < bw.SUCCESS_THRESH:
            return np.linalg.norm(p - tgt), True, t + 1
    return np.linalg.norm(p - tgt), False, max_steps


def _stats(bools: np.ndarray, floats: np.ndarray) -> dict:
    n = len(bools)
    p = float(np.mean(bools))
    se_p = float(np.sqrt(max(p * (1 - p), 1e-12) / n))
    return {
        'n': n,
        'success_rate': p,
        'success_se': se_p,
        'mean_final_dist': float(np.mean(floats)),
        'final_dist_se': float(np.std(floats, ddof=1) / np.sqrt(n)),
    }


def eval_yaw_sweep(model: Policy, head: str, yaws: np.ndarray, n_eps: int,
                   seed: int) -> dict:
    """Closed-loop success/final-dist vs yaw (GT extrinsics at execution)."""
    out = {}
    for yaw in yaws:
        rng = np.random.default_rng(seed + 10000 + int(round(yaw * 100)))
        su, fd = [], []
        for _ in range(n_eps):
            d, ok, _ = rollout(model, head, float(yaw), rng)
            su.append(ok)
            fd.append(d)
        out[float(yaw)] = _stats(np.array(su), np.array(fd))
    return out


def eval_error_structures(model: Policy, head: str, amps_deg: list[float],
                          n_eps: int, seed: int, yaw: float = 0.0,
                          query_k: int = 1) -> dict:
    """Success/final-dist vs error amplitude for each temporal structure."""
    structs = [('iid', None), ('static', None), ('ar', 0.5), ('ar', 0.9)]
    out = {}
    for struct, rho in structs:
        key = struct if rho is None else f'{struct}{rho:.1f}'
        out[key] = {}
        for amp in amps_deg:
            rng = np.random.default_rng(seed + int(round(amp * 10)))
            su, fd = [], []
            for _ in range(n_eps):
                d, ok, _ = rollout(model, head, yaw, rng, struct=struct,
                                   sigma_deg=amp, rho=rho, query_k=query_k)
                su.append(ok)
                fd.append(d)
            out[key][amp] = _stats(np.array(su), np.array(fd))
    return out
