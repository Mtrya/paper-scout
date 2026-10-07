"""PPO fine-tuning of the flow policy with flow-SDE exploration (piRL-style),
plus the matched-budget BC control branches used for C1/C2.

RL trains the action expert only at the K discrete denoising timesteps used
during rollouts; the BC control branches start from the same checkpoint with the
same number of optimizer updates and the same data, differing only in the tau
sampling (continuous vs. the same K discrete timesteps).
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from data import make_demos
from env import PushEnv
from model import FlowPolicy
from rollout import run_episodes
from train_bc import sample_tau, train_bc_steps


class ValueNet(nn.Module):
    def __init__(self, obs_dim=10, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


@torch.no_grad()
def collect(policy, env, vf, gen, K=10, sigma=0.25, exec_h=2, n_decisions=30,
            dist_lo=0.6, dist_hi=1.0):
    obs = env.reset(gen, dist_lo, dist_hi)
    n = env.n
    O, X, SQ, R, D, V = [], [], [], [], [], []
    for t in range(n_decisions):
        chunk, xs, sq, _ = policy.sample(obs, K=K, sigma=sigma, gen=gen, keep=True)
        v = vf(obs)
        O.append(obs.clone())
        X.append(torch.stack(xs, 1))          # (n, K+1, C, A)
        SQ.append(sq.clone())
        V.append(v.clone())
        rew = torch.zeros(n)
        done = torch.zeros(n, dtype=torch.bool)
        for h in range(exec_h):
            obs, r, done, succ, dist = env.step(chunk[:, h])
            rew += r
        R.append(rew)
        D.append(done.clone())
        mask = done
        if bool(mask.any()):
            env.reset(gen, dist_lo, dist_hi, mask=mask)
            obs = env.obs()
    obs_last = obs.clone()
    with torch.no_grad():
        v_last = vf(obs_last)
    return (torch.stack(O), torch.stack(X), torch.stack(SQ), torch.stack(R),
            torch.stack(D), torch.stack(V), v_last)


def compute_gae(rew, val, done, v_last, gamma=0.97, lam=0.95, n_decisions=30):
    """rew/val/done: (T, n). Returns adv, ret."""
    T, n = rew.shape
    adv = torch.zeros_like(rew)
    ret = torch.zeros_like(rew)
    last = torch.zeros(n)
    for t in reversed(range(T)):
        next_v = v_last if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t].float()
        delta = rew[t] + gamma * next_v * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
        ret[t] = last + val[t]
    return adv, ret


def ppo_update(policy, vf, buff, opt_pi, opt_vf, K=10, sigma=0.25, clip=0.2,
               epochs=4, mb=480, vf_coef=0.5, max_grad=1.0):
    O, X, SQ, R, D, V, v_last = buff
    T, n = R.shape
    adv, ret = compute_gae(R, V, D, v_last)
    obs = O.reshape(T * n, -1)
    xs = X.reshape(T * n, X.shape[2], X.shape[3], X.shape[4])
    sq_old = SQ.reshape(-1)
    adv = adv.reshape(-1)
    ret = ret.reshape(-1)
    val_old = V.reshape(-1)
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    N = obs.shape[0]
    dt = 1.0 / K
    denom = 2 * sigma * sigma * dt
    stats = {}
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, mb):
            idx = perm[i:i + mb]
            ob = obs[idx]
            xsb = xs[idx]
            B = ob.shape[0]
            sq_new = torch.zeros(B)
            for k in range(K):
                tau = torch.full((B,), k / K)
                v = policy.velocity(xsb[:, k], ob, tau, k=k)
                resid = xsb[:, k + 1] - xsb[:, k] - v * dt
                sq_new = sq_new + resid.pow(2).sum(dim=(1, 2))
            logp_new = -sq_new / denom
            logp_old = -sq_old[idx] / denom
            ratio = torch.exp((logp_new - logp_old).clamp(-8, 8))
            a = adv_n[idx]
            unclipped = ratio * a
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * a
            pi_loss = -torch.min(unclipped, clipped).mean()
            vpred = vf(ob)
            v_loss = ((vpred - ret[idx]) ** 2).mean()
            loss = pi_loss + vf_coef * v_loss
            opt_pi.zero_grad(set_to_none=True)
            opt_vf.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
            torch.nn.utils.clip_grad_norm_(vf.parameters(), max_grad)
            opt_pi.step()
            opt_vf.step()
            with torch.no_grad():
                stats = {"pi_loss": float(pi_loss), "v_loss": float(v_loss),
                         "ratio_mean": float(ratio.mean()),
                         "ratio_max": float(ratio.max()),
                         "clip_frac": float(((ratio - 1).abs() > clip).float().mean()),
                         "adv_abs": float(adv.abs().mean())}
    return stats


def run_rl(policy, vf, obs_data, act_data, n_iters, K=10, sigma=0.25, lr=3e-4,
           lr_vf=1e-3, exec_h=2, n_envs=64, n_decisions=30, seed=0,
           env_kwargs=None, log_every=10, eval_episodes=0, eval_seed=1000):
    gen = torch.Generator().manual_seed(seed * 31337 + 5)
    env = PushEnv(n_envs=n_envs, seed=seed + 500, **(env_kwargs or {}))
    opt_pi = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_vf = torch.optim.Adam(vf.parameters(), lr=lr_vf)
    log = []
    t0 = time.time()
    for it in range(n_iters):
        buff = collect(policy, env, vf, gen, K=K, sigma=sigma, exec_h=exec_h,
                       n_decisions=n_decisions)
        st = ppo_update(policy, vf, buff, opt_pi, opt_vf, K=K, sigma=sigma)
        st["iter"] = it
        st["ret"] = float(buff[3].sum(0).mean())
        if eval_episodes and ((it + 1) % log_every == 0 or it == 0):
            ev = run_episodes(policy, eval_episodes, eval_seed, K=K,
                              env_kwargs=env_kwargs)
            st["eval_success"] = float(ev["success"].float().mean())
            print(f"  iter {it+1}/{n_iters} ret {st['ret']:.2f} "
                  f"eval_succ {st['eval_success']:.3f} "
                  f"ratio {st['ratio_mean']:.3f} clip {st['clip_frac']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        elif (it + 1) % log_every == 0:
            print(f"  iter {it+1}/{n_iters} ret {st['ret']:.2f} "
                  f"ratio {st['ratio_mean']:.3f} clip {st['clip_frac']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        log.append(st)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="out")
    ap.add_argument("--bc-dir", type=str, default="")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--decisions", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-vf", type=float, default=1e-3)
    ap.add_argument("--demo-noise", type=float, default=0.05)
    ap.add_argument("--n-demos", type=int, default=2500)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--bc-match-steps", type=int, default=0)
    ap.add_argument("--pred-mode", type=str, default="x")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-bc-branches", action="store_true")
    ap.add_argument("--branch-steps-long", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    env_kwargs = {"max_steps": args.max_steps}
    ckdir = args.bc_dir or args.out
    obs, act, succ = make_demos(args.n_demos, args.seed, env_kwargs=env_kwargs,
                                noise=args.demo_noise)
    res = {}

    def load(name, path):
        p = FlowPolicy(pred_mode=args.pred_mode)
        sd = torch.load(path, map_location="cpu")["model"]
        p.load_state_dict(sd)
        return p

    # ---- evaluation baseline of the BC policy ----
    base = load("bc", os.path.join(ckdir, "M_bc.pt"))
    ev = run_episodes(base, args.eval_episodes, 1000 + args.seed, K=args.K,
                      env_kwargs=env_kwargs)
    res["bc_success"] = float(ev["success"].float().mean())
    print(f"seed {args.seed}: BC success {res['bc_success']:.3f}", flush=True)

    # ---- RL ----
    torch.manual_seed(args.seed * 17 + 1)
    policy = load("bc", os.path.join(ckdir, "M_bc.pt"))
    vf = ValueNet()
    rl_log = run_rl(policy, vf, obs, act, args.iterations, K=args.K,
                    sigma=args.sigma, lr=args.lr, lr_vf=args.lr_vf,
                    n_envs=args.n_envs, n_decisions=args.decisions,
                    seed=args.seed, env_kwargs=env_kwargs,
                    log_every=max(1, args.iterations // 6),
                    eval_episodes=args.eval_episodes // 4,
                    eval_seed=2000 + args.seed)
    torch.save({"model": policy.state_dict()}, os.path.join(args.out, "M_rl.pt"))
    ev = run_episodes(policy, args.eval_episodes, 1000 + args.seed, K=args.K,
                      env_kwargs=env_kwargs)
    res["rl_success"] = float(ev["success"].float().mean())
    res["rl_return"] = float(ev["ret"].mean())
    res["rl_steps"] = float(ev["steps"].mean())
    print(f"seed {args.seed}: RL success {res['rl_success']:.3f}", flush=True)
    res["rl_log"] = rl_log

    # ---- matched-budget BC control branches (C1/C2) ----
    if not args.skip_bc_branches:
        n_upd = args.bc_match_steps or (args.iterations * 4)
        for anchor in ["mid", "bc"]:
            path = os.path.join(ckdir, f"M_{anchor}.pt")
            if not os.path.exists(path):
                continue
            for tmode, tag in [("continuous", "cont"), ("discrete", "disc")]:
                p = load(anchor, path)
                losses = train_bc_steps(p, obs, act, n_upd, batch=256, lr=5e-4,
                                        lr_end=1e-4, tau_mode=tmode, K=args.K,
                                        seed=args.seed + (0 if tag == "cont" else 999))
                torch.save({"model": p.state_dict()},
                           os.path.join(args.out, f"M_{anchor}_{tag}.pt"))
                print(f"  branch {anchor}/{tag}: final loss {sum(losses[-50:])/50:.5f}",
                      flush=True)
        res["branch_updates"] = n_upd
        res["branch_samples"] = n_upd * 256
        if args.branch_steps_long:
            for tmode, tag in [("continuous", "cont"), ("discrete", "disc")]:
                p = load("mid", os.path.join(ckdir, "M_mid.pt"))
                losses = train_bc_steps(p, obs, act, args.branch_steps_long,
                                        batch=256, lr=1e-3, lr_end=1e-4,
                                        tau_mode=tmode, K=args.K, seed=args.seed + 55)
                torch.save({"model": p.state_dict()},
                           os.path.join(args.out, f"M_mid_{tag}_long.pt"))
                print(f"  branch mid/{tag}_long: final loss "
                      f"{sum(losses[-50:])/50:.5f}", flush=True)
            res["branch_long_updates"] = args.branch_steps_long
            res["branch_long_samples"] = args.branch_steps_long * 256
    res["rl_transitions"] = args.iterations * args.n_envs * args.decisions
    res["rl_optimizer_updates"] = args.iterations * 4
    res["sigma"] = args.sigma
    res["K"] = args.K
    res["seed"] = args.seed
    with open(os.path.join(args.out, "rl_metrics.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
