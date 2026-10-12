"""2D rectified-flow RL toy for the DiffusionART propositions (2609.12572).

Symbol map (paper -> toy), noise at t=1, data at t=0:

  x_t = (1-t) x_0 + t eps,   eps ~ N(0, I),   u_t = x_0 - eps
  v_theta(x_t, t, c)          denoising-direction velocity field (integrator x <- x + h v)
  v_rvs^old                   the field the roller actually integrates (= model / EMA copy)
  v_fwd^old(s_t)              ideal forward velocity of the *own endpoint* distribution
  e = v_fwd^old - v_rvs^old   forward-reverse consistency gap
  r = 1/2 [1 + clip((R - Rbar_c) / (2 * adv_clip * std_c), -1, 1)]
  q = E[r | s_t],  g = Cov(r, u_t | s_t),  h = u_t - v_old,  d = v_theta - v_old

Every endpoint distribution here is the *sampled* one (rollout output); forward
quantities are estimated by Nadaraya-Watson kernel regression on forward-noised
rollout endpoints, with fresh independent noise (this is Q_old).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

# the machine hosts several concurrent runs; a tiny 2D net is fastest single-threaded
torch.set_num_threads(int(os.environ.get("TOY_THREADS", "1")))

DTYPE = torch.float32
N_PROMPTS = 2
CLUSTER_STD = 0.5
ADV_CLIP = 5.0
# (preferred mode, disfavoured mode) per prompt. Deliberately asymmetric: a
# reflection-symmetric two-mode world would cancel any statewise residual force
# exactly in aggregation, which is not the regime the paper is about.
MODE_CENTERS = {0: ((2.2, 0.4), (-1.0, -0.8)), 1: ((-0.6, 2.4), (1.2, -1.4))}


def mode_centers(c: int) -> torch.Tensor:
    return torch.tensor(MODE_CENTERS[c], dtype=DTYPE)


def prompt_tensor(rows: int, c: int) -> torch.Tensor:
    return torch.full((rows,), c, dtype=torch.long)


def reward(x0: torch.Tensor, c: int) -> torch.Tensor:
    """R(x_0, c) = -||x_0 - m_hi(c)||, maximised at the preferred mode."""
    return -(x0 - mode_centers(c)[0]).norm(dim=-1)


def target_endpoint_std(c: int) -> float:
    """Std of the reward-optimal endpoint distribution N(m_hi, CLUSTER_STD^2 I)."""
    del c
    return CLUSTER_STD


# --------------------------------------------------------------------------------------
# analytic reference: exact flow-matching field of the *pretraining* mixture
# --------------------------------------------------------------------------------------
def mixture_sample(c: int, rows: int, generator: torch.Generator) -> torch.Tensor:
    centers = mode_centers(c)
    pick = torch.randint(0, 2, (rows,), generator=generator)
    return centers[pick] + CLUSTER_STD * torch.randn(rows, 2, generator=generator)


def mixture_batch(c_vec: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """50/50 mixture endpoints for a vector of prompts."""
    rows = c_vec.shape[0]
    centers = torch.stack([mode_centers(0), mode_centers(1)])[c_vec]  # [B,2,2]
    pick = torch.randint(0, 2, (rows,), generator=generator)
    chosen = centers[torch.arange(rows), pick]
    return chosen + CLUSTER_STD * torch.randn(rows, 2, generator=generator)


def analytic_field(xt: torch.Tensor, t: torch.Tensor, c: int) -> torch.Tensor:
    """u*(x_t,t,c) = E[x_0 - eps | x_t, t, c] for x_0 ~ 0.5 N(m_hi) + 0.5 N(m_lo)."""
    centers = mode_centers(c)
    s2 = CLUSTER_STD**2
    a = (1.0 - t)[:, None]  # coefficient on x_0, [N,1]
    var = a.square() * s2 + t[:, None].square()
    diff = xt[:, None, :] - a[:, :, None] * centers[None]  # [N,2,2]
    log_post = -0.5 * diff.square().sum(-1) / var  # equal component weights
    post = torch.softmax(log_post, dim=1)
    mu = centers[None] + (a * s2 / var)[:, :, None] * diff
    x0_hat = (post[:, :, None] * mu).sum(1)
    eps_hat = (post[:, :, None] * ((xt[:, None, :] - a[:, :, None] * mu) / t[:, None, None])).sum(1)
    tiny = t < 1e-6
    eps_hat = torch.where(tiny[:, None], xt, eps_hat)
    return x0_hat - eps_hat


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class VelocityNet(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 3):
        super().__init__()
        dims = [2 + 3 + N_PROMPTS] + [hidden] * depth
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        layers += [nn.Linear(dims[-1], 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        feats = torch.cat(
            [
                x,
                torch.stack([t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], -1),
                torch.nn.functional.one_hot(c, N_PROMPTS).to(x.dtype),
            ],
            -1,
        )
        return self.net(feats)


def pretrain(
    steps: int = 4000,
    batch: int = 1024,
    lr: float = 3e-3,
    hidden: int = 128,
    depth: int = 3,
    seed: int = 0,
    verbose: bool = False,
) -> VelocityNet:
    """Flow matching on the analytic mixture -> the 'pretrained' rollout policy."""
    torch.manual_seed(seed)
    net = VelocityNet(hidden, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    gen = torch.Generator().manual_seed(seed + 1)
    for step in range(steps):
        c = torch.randint(0, N_PROMPTS, (batch,), generator=gen)
        x0 = mixture_batch(c, gen)
        eps = torch.randn(batch, 2, generator=gen)
        t = torch.rand(batch, generator=gen).clamp(1e-4, 1 - 1e-4)
        xt = (1 - t)[:, None] * x0 + t[:, None] * eps
        loss = (net(xt, t, c) - (x0 - eps)).square().sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and step % 1000 == 0:
            print(f"  pretrain {step}: fm mse {loss.item():.4f}")
    net.eval()
    return net


def pretrained(seed: int = 0, **kw) -> VelocityNet:
    return pretrain(seed=seed, **kw)


# --------------------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------------------
@torch.no_grad()
def euler_sample(
    field: VelocityNet,
    c: int,
    count: int,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    x = torch.randn(count, 2, generator=generator)
    h = 1.0 / steps
    cc = prompt_tensor(count, c)
    for i in range(steps):
        t = torch.full((count,), 1.0 - i * h)
        x = x + h * field(x, t, cc)
    return x


@torch.no_grad()
def field_at(field: VelocityNet, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return field(x, t, c)


# --------------------------------------------------------------------------------------
# forward-noising ensemble (Q_old) and Nadaraya-Watson conditional statistics
# --------------------------------------------------------------------------------------
def forward_ensemble(
    endpoints: torch.Tensor,
    rewards: torch.Tensor,
    c: int,
    draws: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Q_old: fresh eps per endpoint at fresh t, giving (x_t, u_t, r)."""
    x0 = endpoints.repeat_interleave(draws, dim=0)
    r = rewards.repeat_interleave(draws, dim=0)
    eps = torch.randn(x0.shape[0], 2, generator=generator)
    t = torch.rand(x0.shape[0], generator=generator).clamp(1e-3, 1.0)
    xt = (1 - t)[:, None] * x0 + t[:, None] * eps
    return {"x0": x0, "r": r, "eps": eps, "t": t, "xt": xt, "u": x0 - eps, "c": c}


def local_weights(
    ensemble: dict[str, torch.Tensor],
    xt_c: torch.Tensor,
    t_c: float,
    bw_x: float,
    bw_t: float,
) -> torch.Tensor:
    t = ensemble["t"]
    d2 = (ensemble["xt"] - xt_c).square().sum(-1) / (2 * bw_x**2) + (t - t_c).square() / (2 * bw_t**2)
    return torch.exp(-d2)


def local_stats(ensemble: dict[str, torch.Tensor], w: torch.Tensor) -> dict[str, torch.Tensor]:
    """Conditional statistics at one state, under kernel weights w."""
    wsum = w.sum().clamp_min(1e-12)
    wp = w / wsum
    r = ensemble["r"]
    u = ensemble["u"]
    q = (wp * r).sum()
    mean_u = (wp[:, None] * u).sum(0)
    g = (wp[:, None] * (r - q)[:, None] * (u - mean_u)).sum(0)
    mean_r_sq = (wp * r.square()).sum()
    var_r = (mean_r_sq - q.square()).clamp_min(0)
    return {"q": q, "mean_u": mean_u, "g": g, "var_r": var_r, "wsum": wsum}


def network_bandwidth(t_c: float, scale: float = 0.45) -> float:
    return max(0.05, scale * t_c)


# --------------------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------------------
def group_weights(rewards: torch.Tensor, clip: float = ADV_CLIP, eps: float = 1e-5) -> torch.Tensor:
    """Official-code advantage semantics: standardise within the prompt group."""
    std = rewards.std(unbiased=True) + eps
    adv = ((rewards - rewards.mean()) / std).clamp(-clip, clip)
    return 0.5 + adv / (2 * clip)


def nft_loss(
    v_pred: torch.Tensor,
    v_old: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """paper Eq. 1: L = beta^2 E || v_theta - v_tgt ||^2, v_tgt = v_old + (2r-1)/beta (u - v_old)."""
    v_tgt = v_old + ((2 * r - 1) / beta)[:, None] * (u - v_old)
    return beta**2 * (v_pred - v_tgt).square().sum(-1).mean()


def nft_branch_losses(
    v_pred: torch.Tensor,
    v_old: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """r-weighted positive / (1-r)-weighted negative branch losses (paper Eq. 19)."""
    d = v_pred - v_old
    h = u - v_old
    pos = (r * (beta * d - h).square().sum(-1)).mean()
    neg = ((1 - r) * (-beta * d - h).square().sum(-1)).mean()
    return pos, neg


def raw_forces(
    ensemble: dict[str, torch.Tensor],
    w: torch.Tensor,
    v_old: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Measured output-space forces at v_theta = v_old under kernel weights w.

    F := -(1/2 beta) grad_v L(s_t).  Branch forces use the r / (1-r) weighted
    branch losses, which is the paper's F_+ = g + q e and F_- = g - (1-q) e.
    """
    wsum = w.sum().clamp_min(1e-12)
    wp = w / wsum
    r = ensemble["r"].detach()
    u = ensemble["u"].detach()

    vp = v_old.clone().detach().requires_grad_(True)
    d = vp - v_old
    h = u - v_old
    total = (wp * ((r * (beta * d - h).square().sum(-1)) + ((1 - r) * (-beta * d - h).square().sum(-1)))).sum()
    grad_total = torch.autograd.grad(total, vp)[0]

    vp2 = v_old.clone().detach().requires_grad_(True)
    d2 = vp2 - v_old
    pos = (wp * (r * (beta * d2 - h).square().sum(-1))).sum()
    grad_pos = torch.autograd.grad(pos, vp2)[0]

    vp3 = v_old.clone().detach().requires_grad_(True)
    d3 = vp3 - v_old
    neg = (wp * ((1 - r) * (-beta * d3 - h).square().sum(-1))).sum()
    grad_neg = torch.autograd.grad(neg, vp3)[0]

    return {
        "F": -grad_total / (2 * beta),
        "F_plus": -grad_pos / (2 * beta),
        "F_minus": -grad_neg / (2 * beta),
    }


def single_sample_force(ensemble: dict[str, torch.Tensor], v_old: torch.Tensor) -> torch.Tensor:
    """(2r-1)(u_t - v_old): the per-sample Monte Carlo estimate of F_NFT."""
    return (2 * ensemble["r"] - 1)[:, None] * (ensemble["u"] - v_old)


def cov_guidance(endpoints: torch.Tensor, rewards: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """paper Eq. 5: G(c) = Cov(R, x_0 | c) / RMS(Cov)."""
    centered_r = rewards - rewards.mean()
    centered_x = endpoints - endpoints.mean(0)
    cov = (centered_r[:, None] * centered_x).mean(0)
    rms = cov.square().mean().sqrt().clamp_min(eps)
    return cov / rms


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def energy_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    def mean_dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.cdist(p, q).mean()

    return float(2 * mean_dist(a, b) - mean_dist(a, a) - mean_dist(b, b))


@torch.no_grad()
def endpoint_metrics(endpoints: torch.Tensor, c: int, reference: torch.Tensor) -> dict[str, float]:
    m_hi = mode_centers(c)[0]
    hi = (endpoints - m_hi).norm(dim=-1)
    lo = (endpoints - mode_centers(c)[1]).norm(dim=-1)
    return {
        "reward": float(reward(endpoints, c).mean()),
        "hi_mode_rate": float((hi < lo).to(DTYPE).mean()),
        "spread": float(endpoints.std(0).norm()),
        "energy_to_target": energy_distance(endpoints, reference),
        "mean_radius_to_hi": float(hi.mean()),
    }


def target_reference(c: int, count: int, generator: torch.Generator) -> torch.Tensor:
    return mode_centers(c)[0] + CLUSTER_STD * torch.randn(count, 2, generator=generator)


def summarise(values: list[float]) -> dict[str, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(t.mean()), "std": float(t.std(unbiased=True)) if t.numel() > 1 else 0.0}
