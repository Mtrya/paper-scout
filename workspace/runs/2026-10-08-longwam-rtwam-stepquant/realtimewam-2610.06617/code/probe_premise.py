"""Premise probe: why does one-step generation not lose to the 10-step teacher here?

The paper's premise is that the multi-step teacher is the accuracy ceiling and a
one-step student must be distilled up to it.  On this bench the ordering is
inverted, and this probe tests the mechanism: the teacher's 10-step ODE endpoint
differs from its own one-step endpoint mainly by *dispersion* (flow sampling
noise + coarse Euler error), and dispersion is what costs task success.  If
injecting Gaussian noise of that measured size into the one-step action
reproduces the 10-step success rate, the inversion is a property of the task's
action distribution (unimodal, mean-friendly) rather than of the distillation.
"""
import argparse
import json
import os

import torch

from data import expert_success_rate
from distill import endpoint, make_diag_set, teacher_rollout, EVAL_SEED
from env import PushEnv
from model import FlowPolicy


class MultiSample:
    """Average of m independent K-step teacher rollouts: variance-reduced sample.

    m=1 is the plain multi-step teacher; m->inf converges to E_eps[a_T] which is
    the one-step endpoint prediction's target.  Sweeping m separates "the
    multi-step endpoint is a noisy sample of a mean-friendly action
    distribution" from "the multi-step integration itself is wrong".
    """

    def __init__(self, policy, m, K=10):
        self.policy = policy
        self.m = m
        self.K = K
        self.n_blocks = policy.n_blocks
        self.d = policy.d

    @torch.no_grad()
    def sample(self, obs, K=None, sigma=0.0, gen=None, hidden=False):
        B = obs.shape[0]
        acc = 0.0
        for _ in range(self.m):
            x = torch.randn(B, self.policy.chunk, self.policy.act_dim, generator=gen)
            acc = acc + teacher_rollout(self.policy, x, obs,
                                       torch.zeros(B), K=self.K)
        return acc / self.m, None, None, None


class NoisyOneStep:
    """One-step endpoint prediction + fresh Gaussian action noise each decision."""

    def __init__(self, policy, sigma):
        self.policy = policy
        self.sigma = sigma
        self.n_blocks = policy.n_blocks
        self.d = policy.d

    @torch.no_grad()
    def sample(self, obs, K=1, sigma=0.0, gen=None, hidden=False):
        B = obs.shape[0]
        eps = torch.randn(B, self.policy.chunk, self.policy.act_dim, generator=gen)
        a = endpoint(self.policy, eps, obs, torch.zeros(B))
        if self.sigma > 0:
            a = a + self.sigma * torch.randn(a.shape, generator=gen)
        return a, None, None, None


@torch.no_grad()
def success(policy_sampler, n_episodes, seed, K=1, exec_h=2, max_steps=60):
    env = PushEnv(n_envs=min(n_episodes, 150), seed=seed, max_steps=max_steps)
    gen_env = torch.Generator().manual_seed(seed * 7919 + 13)
    gen_pol = torch.Generator().manual_seed(seed * 7919 + 29)
    out = []
    idx, nb = 0, 150
    while idx < n_episodes:
        n = min(nb, n_episodes - idx)
        env.n = n
        obs = env.reset(gen_env)
        active = torch.ones(n, dtype=torch.bool)
        succ = torch.zeros(n, dtype=torch.bool)
        while bool(active.any()):
            chunk = policy_sampler.sample(obs, K=K, sigma=0.0, gen=gen_pol)[0]
            for hh in range(exec_h):
                obs, r, done, s, dist = env.step(chunk[:, hh])
                newly = done & active
                succ = torch.where(newly, s, succ)
                active = active & ~done
        out.append(succ)
        idx += n
    return float(torch.cat(out).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--demo-noise", type=float, default=0.15)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--tang-sign", type=str, default="fixed")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    teacher = FlowPolicy(pred_mode="x")
    teacher.load_state_dict(torch.load(args.teacher, map_location="cpu")["model"])
    teacher.eval()

    obs_d, act_d, eps_d = make_diag_set(teacher, n=512, demo_noise=args.demo_noise,
                                        max_steps=args.max_steps,
                                        tang_sign=args.tang_sign)
    B = obs_d.shape[0]
    z = torch.zeros(B)
    aT = teacher_rollout(teacher, eps_d, obs_d, z, K=10)
    a1 = endpoint(teacher, eps_d, obs_d, z)
    res = {
        "dispersion_1step_vs_10step": float((aT - a1).pow(2).mean()),
        "dispersion_1step_vs_demo": float((endpoint(teacher, eps_d, obs_d, z) - act_d).pow(2).mean()),
        "aT_var": float(aT.var(dim=0).mean()),
        "demo_action_noise": args.demo_noise ** 2,
        "success_sweep": {},
    }
    # reference: 10-step teacher, and one-step at several injected dispersion levels
    res["success_sweep"]["10step"] = success(teacher, args.episodes, EVAL_SEED, K=10,
                                             max_steps=args.max_steps)
    for s in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        res["success_sweep"][f"1step+sigma{s}"] = success(
            NoisyOneStep(teacher, s), args.episodes, EVAL_SEED, K=1,
            max_steps=args.max_steps)
    # variance reduction of the multi-step endpoint itself (m independent rollouts)
    res["success_vs_samples"] = {}
    for m in [1, 2, 4, 8, 16]:
        res["success_vs_samples"][str(m)] = success(
            MultiSample(teacher, m), args.episodes, EVAL_SEED, K=10,
            max_steps=args.max_steps)
    res["success_vs_samples"]["1step"] = res["success_sweep"]["1step+sigma0.0"]
    # expert reference under the same eval protocol (with the demos' action noise)
    res["expert_success_noisy"] = expert_success_rate(
        args.episodes, EVAL_SEED, noise=args.demo_noise,
        env_kwargs={"max_steps": args.max_steps}, tang_sign=args.tang_sign)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
