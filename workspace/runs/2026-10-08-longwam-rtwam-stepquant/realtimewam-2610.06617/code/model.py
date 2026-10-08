"""Flow-matching action expert with an explicit Time MLP + AdaRMS (pi0.5-style).

Timestep Modules = {time_mlp, every block's ada.to_ssg}.  The denoising timestep
enters the network *only* through them (as in pi0.5), so the TS modules are the
sole tau-pathway.

Convention: tau = 0 is noise, tau = 1 is data.
    x_tau = (1 - tau) * eps + tau * a
Two equivalent parametrisations of the same flow-matching objective:
    pred_mode = "v": head predicts the velocity v = a - eps (pi0.5 Eq. 1)
    pred_mode = "x": head predicts the clean action a; v = (a_hat - x) / (1 - tau)
Rollout integrates x_{k+1} = x_k + v(x_k, tau_k) * (1/K) (+ flow-SDE noise).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(h, eps=1e-5):
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)


def sinusoidal(tau, dim):
    half = dim // 2
    freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), half))
    ang = tau.unsqueeze(-1) * freqs * math.pi
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class AdaRMS(nn.Module):
    """Produces (scale, shift, gate) from the conditioning vector c_tau."""

    def __init__(self, d, init_scale=0.0):
        super().__init__()
        self.d = d
        self.to_ssg = nn.Linear(d, 3 * d)
        if init_scale == 0.0:
            nn.init.zeros_(self.to_ssg.weight)
        else:
            nn.init.normal_(self.to_ssg.weight, std=init_scale)
        nn.init.zeros_(self.to_ssg.bias)

    def forward(self, h, c):
        ssg = self.to_ssg(c)
        scale, shift, gate = ssg.chunk(3, dim=-1)
        z = (1.0 + scale) * rms_norm(h) + shift
        return z, gate


class Block(nn.Module):
    def __init__(self, d, hidden, init_scale=0.0, zero_res=False):
        super().__init__()
        self.ada = AdaRMS(d, init_scale)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        if zero_res:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, h, c):
        z, gate = self.ada(h, c)
        return h + gate * self.fc2(F.silu(self.fc1(z)))


class FlowPolicy(nn.Module):
    def __init__(self, obs_dim=10, act_dim=2, chunk=4, d=128, hidden=512,
                 n_blocks=3, t_dim=64, pred_mode="v", init_scale=0.0,
                 zero_res=False, zero_out=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk = chunk
        self.d = d
        self.n_blocks = n_blocks
        self.t_dim = t_dim
        self.pred_mode = pred_mode
        self.in_proj = nn.Linear(obs_dim + chunk * act_dim, d)
        self.time_mlp = nn.Sequential(nn.Linear(t_dim, d), nn.SiLU(),
                                      nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, hidden, init_scale, zero_res)
                                     for _ in range(n_blocks)])
        self.out = nn.Linear(d, chunk * act_dim)
        if zero_out:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        self.steer = None  # dict(layer, k, dir, target, mode, alpha)

    # ---- module bookkeeping -------------------------------------------------
    def ts_module_names(self):
        return ["time_mlp"] + [f"blocks.{i}.ada" for i in range(self.n_blocks)]

    def param_group_of(self, name):
        """Map a state_dict key to 'ts' or 'rest'."""
        for t in self.ts_module_names():
            if name == t or name.startswith(t + "."):
                return "ts"
        return "rest"

    # ---- forward ------------------------------------------------------------
    def cond(self, tau):
        return self.time_mlp(sinusoidal(tau, self.t_dim))

    def steer_delta(self, h):
        s = self.steer
        proj = h @ s["dir"]
        if s["mode"] == "adaptive":
            gap = (s["target"] - proj).clamp_min(0.0)
        else:
            gap = torch.full_like(proj, float(s["target"]))
        return (s["alpha"] * gap).unsqueeze(-1) * s["dir"]

    def velocity(self, x, obs, tau, k=None, hidden_out=None):
        c = self.cond(tau)
        h = self.in_proj(torch.cat([obs, x.flatten(1)], dim=-1))
        if hidden_out is not None:
            hidden_out.append(h)
        for li, blk in enumerate(self.blocks):
            if (self.steer is not None and li == self.steer["layer"]
                    and (self.steer["k"] is None or k == self.steer["k"])):
                h = h + self.steer_delta(h)
            h = blk(h, c)
            if hidden_out is not None:
                hidden_out.append(h)
        raw = self.out(h).view(x.shape[0], self.chunk, self.act_dim)
        if self.pred_mode == "v":
            return raw
        return (raw - x) / (1.0 - tau).clamp_min(1e-3).view(-1, 1, 1)

    def flow_loss(self, obs, act, tau, eps=None):
        if eps is None:
            eps = torch.randn_like(act)
        x_tau = (1 - tau)[:, None, None] * eps + tau[:, None, None] * act
        v = self.velocity(x_tau, obs, tau)
        if self.pred_mode == "v":
            target = act - eps
            return ((v - target) ** 2).mean()
        # x-prediction: raw head is the clean-action estimate; v is derived.
        a_hat = x_tau + (1 - tau)[:, None, None] * v
        return ((a_hat - act) ** 2).mean()

    # ---- sampling -----------------------------------------------------------
    @torch.no_grad()
    def sample(self, obs, K=10, sigma=0.0, gen=None, keep=None, hidden=False):
        B = obs.shape[0]
        x = torch.randn(B, self.chunk, self.act_dim, generator=gen)
        xs = []
        hs = [] if hidden else None
        dt = 1.0 / K
        sq = 0.0
        for k in range(K):
            tau = torch.full((B,), k / K)
            buf = [] if hidden else None
            v = self.velocity(x, obs, tau, k=k, hidden_out=buf)
            if hidden:
                hs.append(torch.stack(buf[:self.n_blocks], dim=1))  # (B, L, d)
            if sigma > 0.0:
                eps = torch.randn(x.shape, generator=gen)
                x_next = x + v * dt + sigma * math.sqrt(dt) * eps
                sq = sq + (sigma * math.sqrt(dt) * eps).pow(2).sum(dim=(1, 2))
            else:
                x_next = x + v * dt
                sq = sq + torch.zeros(B)
            if keep is not None:
                xs.append(x.clone())
            x = x_next
        if keep is not None:
            xs.append(x.clone())
        hstack = torch.stack(hs, dim=1) if hidden else None  # (B, K, L, d)
        return x, xs, sq, hstack

    # ---- steering-aware differentiable log-prob (for PPO) -------------------
    def logprob_from_traj(self, xs, obs, sigma):
        """xs: list of K+1 tensors (B, chunk, act). Returns (logp, sq_total)."""
        K = len(xs) - 1
        dt = 1.0 / K
        B = xs[0].shape[0]
        sq = torch.zeros(B)
        for k in range(K):
            tau = torch.full((B,), k / K)
            v = self.velocity(xs[k], obs, tau, k=k)
            resid = xs[k + 1] - xs[k] - v * dt
            sq = sq + resid.pow(2).sum(dim=(1, 2))
        logp = -sq / (2 * sigma * sigma * dt)
        return logp, sq
