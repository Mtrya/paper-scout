#!/usr/bin/env python3
"""Toy experiment: does piR2's proprio-fast-channel advantage depend on disturbance modality?

Inspired by piR2 (arXiv 2607.26055), whose fast channel carries ONLY proprioception
while vision+language update asynchronously. The paper sweeps inference latency but
never varies *what kind* of disturbance the world throws at the policy.

Conjecture: piR2-style inference (1-NFE staircase schedule, fresh proprio, stale
vision) beats synchronous flow inference when the disturbance is proprio-visible
(impulse on the arm), but the advantage collapses when the disturbance is
vision-only (the target moves) — because then both methods wait on the slow channel.

Toy setup (1D, state-based, CPU):
  point mass x, v; action = acceleration; control tick = 1 step
  target y(t); vision channel observes y with delay d_vis ticks
  disturbance events: ARM impulse (v += j) or TARGET shift (y += s)
  expert: PD controller reacting instantly -> demos teach reactive recovery
  policy: diffusion-forcing flow model, per-position noise level tau in [0,1]
          (tau=1 clean, tau=0 noise), AdaLN-free MLP with tau/obs embeddings
Inference modes:
  sync      : every tick, K Euler steps from noise, conditioning obs d ticks stale,
              commit action 0 (replan-every-tick upper bound on compute) OR commit
              d actions open-loop (realistic). We use the realistic variant.
  staircase : 1 Euler step per tick, front-d actions clamped clean, ramp, noise tail;
              fresh proprio every tick, stale vision with trained delay embedding.
"""
import math
import sys

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

H = 16            # action chunk horizon
K = 4             # full-flow Euler steps (sync mode)
DT = 0.05         # integration step in "seconds" (toy)
D_MAX = 5         # max latency in ticks (matches paper's d_vis_max)
DIM = 128
DEV = "cpu"


# ---------------- environment ----------------
class PointMassEnv:
    """1D point mass tracking a target, with scheduled disturbances."""

    def __init__(self, horizon=200, disturb_p=0.03, arm_impulse=2.0,
                 target_shift=1.0, vis_delay=0):
        self.horizon = horizon
        self.disturb_p = disturb_p
        self.arm_impulse = arm_impulse
        self.target_shift = target_shift
        self.vis_delay = vis_delay
        self.reset()

    def reset(self):
        self.t = 0
        self.x = np.random.uniform(-0.5, 0.5)
        self.v = 0.0
        self.y = np.random.uniform(-1.0, 1.0)
        self.y_hist = [self.y] * (D_MAX + 2)
        self.events = []          # (t, kind) recorded for evaluation
        return self.obs()

    def obs(self):
        y_vis = self.y_hist[-1 - self.vis_delay] if self.vis_delay else self.y
        return np.array([self.x, self.v, y_vis], dtype=np.float32)

    def step(self, a):
        a = float(np.clip(a, -4.0, 4.0))
        # disturbance BEFORE integration so proprio reflects it immediately
        kind = None
        if np.random.rand() < self.disturb_p:
            if np.random.rand() < 0.5:
                self.v += np.random.choice([-1, 1]) * self.arm_impulse * np.random.uniform(0.6, 1.0)
                kind = "arm"
            else:
                self.y += np.random.choice([-1, 1]) * self.target_shift * np.random.uniform(0.6, 1.0)
                kind = "target"
            self.events.append((self.t, kind))
        self.v += a * DT
        self.v *= 0.98
        self.x += self.v * DT
        self.y_hist.append(self.y)
        self.t += 1
        return self.obs()


def expert_action(x, v, y):
    return np.clip(8.0 * (y - x) - 3.0 * v, -4.0, 4.0)


# ---------------- model ----------------
class FlowChunk(nn.Module):
    """Diffusion-forcing flow model: per-position tau, fresh proprio, delayed vision."""

    def __init__(self):
        super().__init__()
        self.act_in = nn.Linear(1, DIM)
        self.tau_emb = nn.Linear(1, DIM)
        self.pos = nn.Parameter(torch.randn(1, H, DIM) * 0.02)
        self.proprio_in = nn.Linear(2, DIM)
        self.vis_in = nn.Linear(1, DIM)
        self.delay_emb = nn.Embedding(D_MAX + 1, DIM)
        nn.init.zeros_(self.delay_emb.weight)   # zero-init, like the paper
        self.trunk = nn.Sequential(
            nn.Linear(DIM, DIM), nn.SiLU(),
            nn.Linear(DIM, DIM), nn.SiLU(),
        )
        self.out = nn.Linear(DIM, 1)

    def forward(self, x_tau, tau, proprio, y_vis, d_vis):
        # x_tau [B,H,1], tau [B,H,1], proprio [B,2], y_vis [B,1], d_vis [B] long
        h = self.act_in(x_tau) + self.tau_emb(tau) + self.pos
        cond = (self.proprio_in(proprio) + self.vis_in(y_vis)
                + self.delay_emb(d_vis))
        h = h + cond.unsqueeze(1)
        h = self.trunk(h)
        return self.out(h)          # velocity field [B,H,1]


# ---------------- training ----------------
def collect_demos(n_eps=400, horizon=60):
    env = PointMassEnv(horizon=horizon, disturb_p=0.05, vis_delay=0)
    data = []
    for _ in range(n_eps):
        env.reset()
        traj = []
        for _ in range(horizon):
            a = expert_action(env.x, env.v, env.y)
            traj.append(([env.x, env.v], env.y, a))
            env.step(a)
        data.append(traj)
    return data


def train(model, data, steps=10000, bs=64, alpha=0.2):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for it in range(steps):
        props, ys, acts = [], [], []
        d_vis = torch.zeros(bs, dtype=torch.long)
        for b in range(bs):
            tr = data[np.random.randint(len(data))]
            i = np.random.randint(D_MAX, len(tr) - H)
            if np.random.rand() < 0.5:
                # stale-observation sample: same trajectory, y from k ticks ago
                k = np.random.randint(1, D_MAX + 1)
                d_vis[b] = k
                props.append(tr[i][0])
                ys.append(tr[i - k][1])
                acts.append([tr[i + m][2] for m in range(H)])
            else:
                props.append(tr[i][0])
                ys.append(tr[i][1])
                acts.append([tr[i + m][2] for m in range(H)])
        proprio = torch.tensor(np.array(props), dtype=torch.float32)
        y_vis = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(1)
        a = torch.tensor(np.array(acts), dtype=torch.float32).unsqueeze(-1)  # [B,H,1]
        eps = torch.randn_like(a)
        if np.random.rand() < alpha:
            tau = torch.rand(bs, 1, 1).expand(-1, H, -1).clone()  # standard flow
        else:
            tau = torch.rand(bs, H, 1)                            # diffusion forcing
        x_tau = (1 - tau) * eps + tau * a
        pred = model(x_tau, tau, proprio, y_vis, d_vis)
        loss = ((pred - (a - eps)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 2500 == 0:
            print(f"  train {it+1}/{steps} loss={loss.item():.4f}")
    return model


# ---------------- inference ----------------
@torch.no_grad()
def run_policy(model, mode, delay, horizon=150, disturb_p=0.06, seed=None):
    if seed is not None:
        np.random.seed(seed)
    env = PointMassEnv(horizon=horizon, disturb_p=disturb_p, vis_delay=0)
    env.reset()
    d_vis_t = torch.tensor([min(delay, D_MAX)])
    err_series = []

    def full_denoise(yv_val):
        x_tau = torch.randn(1, H, 1)
        for k in range(K):
            tk = torch.full((1, H, 1), k / K)
            prop = torch.tensor(np.array([env.x, env.v], dtype=np.float32)).unsqueeze(0)
            yv = torch.tensor([[yv_val]], dtype=torch.float32)
            vfield = model(x_tau, tk, prop, yv, torch.zeros(1, dtype=torch.long))
            x_tau = x_tau + vfield * (1.0 / K)
        return x_tau

    # warm-start (paper: episode init via standard-flow full-chunk inference)
    clean = full_denoise(env.y).detach()
    # tick-start staircase profile: front slot sits at tau=1-1/K and gets its FINAL
    # denoise step under THIS tick's fresh conditioning, then is emitted immediately
    # (last-minute finalization; no frozen clean slots, no 1-tick commit lag)
    tau = (1.0 - 1.0 / K - torch.arange(H).float().div(K)).clamp(0, 1).view(1, H, 1)
    # re-noise the clean chunk to the staircase levels so (buf, tau) is on-distribution
    buf = (1 - tau) * torch.randn(1, H, 1) + tau * clean
    for t in range(horizon):
        fresh = np.array([env.x, env.v], dtype=np.float32)
        if mode == "sync":
            # stale obs by `delay`, K Euler steps, commit `delay` actions open loop
            stale = env.y_hist[-1 - delay] if delay else env.y
            x_tau = torch.randn(1, H, 1)
            for k in range(K):
                tk = torch.full((1, H, 1), k / K)
                prop = torch.tensor(fresh).unsqueeze(0)
                yv = torch.tensor([[stale]], dtype=torch.float32)
                vfield = model(x_tau, tk, prop, yv, d_vis_t)
                x_tau = x_tau + vfield * (1.0 / K)
            plan = x_tau[0, :, 0].numpy()
            n_commit = max(delay, 1)
            for a in plan[:n_commit]:
                if env.t >= horizon:
                    break
                env.step(a)
                err_series.append(abs(env.x - env.y))
        else:  # staircase: uniform 1-slot Euler step, emit clean front, slide
            stale = env.y_hist[-1 - delay] if delay else env.y
            prop = torch.tensor(fresh).unsqueeze(0)
            yv = torch.tensor([[stale]], dtype=torch.float32)
            vfield = model(buf, tau, prop, yv, d_vis_t)
            # only the front-K slots advance, one denoise level per tick: every slot's
            # K steps happen in the K ticks right before execution, each under the
            # freshest conditioning (back slots wait as pure noise at tau=0)
            active = torch.zeros(1, H, 1)
            active[:, :K] = 1.0
            active = active * (tau < 1).float()
            buf = buf + vfield * (1.0 / K) * active
            tau = tau + (1.0 / K) * active
            a = float(buf[0, 0, 0])
            env.step(a)
            err_series.append(abs(env.x - env.y))
            # slide buffer, append fresh noise
            buf = torch.cat([buf[:, 1:], torch.randn(1, 1, 1)], dim=1)
            tau = torch.cat([tau[:, 1:], torch.zeros(1, 1, 1)], dim=1)
    # post-disturbance recovery: mean |x-y| in the 12 ticks after each event
    errs_arm, errs_tgt = [], []
    for (te, kind) in env.events:
        w = err_series[te:te + 12]
        if len(w) >= 6:
            (errs_arm if kind == "arm" else errs_tgt).append(float(np.mean(w)))
    return dict(arm=np.mean(errs_arm) if errs_arm else float("nan"),
                tgt=np.mean(errs_tgt) if errs_tgt else float("nan"),
                all=float(np.mean(err_series)))


def main():
    print("collecting demos...")
    data = collect_demos()
    model = FlowChunk()
    print("training...")
    train(model, data)
    print("evaluating...")
    rows = []
    for mode in ("sync", "staircase"):
        for delay in (1, 3, 5):
            arms, tgts, alls = [], [], []
            for seed in (1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901):
                r = run_policy(model, mode, delay, seed=seed)
                arms.append(r["arm"]); tgts.append(r["tgt"]); alls.append(r["all"])
            arm, tgt, all_ = map(np.mean, (arms, tgts, alls))
            sarm, stgt = np.std(arms), np.std(tgts)
            rows.append((mode, delay, arm, tgt, all_))
            print(f"{mode:10s} d={delay}  err@arm-dist={arm:.3f}±{sarm:.3f}  "
                  f"err@target-dist={tgt:.3f}±{stgt:.3f}  err_all={all_:.3f}")
    import json
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
