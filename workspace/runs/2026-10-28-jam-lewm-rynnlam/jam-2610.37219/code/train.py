"""Toy foresight experiment: role x target-quality grid on visual push-block.

Arms:
  action               action-only BC (baseline)
  fs_oracle            + foresight, target = future ground-truth state (normalized)
  fs_random            + foresight, target = random frozen anchor features of future frame
  fs_pixels            + foresight, target = raw future pixel patches (VAE-direct analog)
  fs_const             + foresight, target = fixed random vector (input-independent control)
  fs_current_random    + foresight on CURRENT frame with random anchor (offset-0 control)
  latent_oracle        two-stage: flow-match future oracle state, then retarget to actions
  latent_random        two-stage: flow-match random-anchor pooled features, then actions
  latent_pixels        two-stage: flow-match pooled future pixel patches, then actions

Data: episodes of (states, actions) from the continuous scripted expert.
All rendering happens on the fly on GPU. Idempotent per (arm, seed) result JSON.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from env import PushEnv
from data import expert_action
from model_visual import VisualPolicy, ForesightHead, RandomAnchor, pixel_tokens

FUTURE_T = 8
ARMS = ["action", "fs_oracle", "fs_random", "fs_pixels", "fs_const",
        "fs_current_random", "latent_oracle", "latent_random", "latent_pixels"]


def gen_demos(n_episodes, seed, out, noise=0.15, max_steps=60):
    """Record full state/action sequences; store per-episode lists."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        return
    env = PushEnv(n_envs=150, seed=seed, max_steps=max_steps)
    gen = torch.Generator().manual_seed(seed * 104729 + 7)
    eps = []
    idx = 0
    while idx < n_episodes:
        n = min(150, n_episodes - idx)
        env.n = n
        obs = env.reset(gen)
        states, actions, succ = [env.state.clone()], [], torch.zeros(n, dtype=torch.bool)
        active = torch.ones(n, dtype=torch.bool)
        for _ in range(max_steps):
            if not bool(active.any()):
                break
            a = expert_action(env.state)
            a = (a + noise * torch.randn(a.shape, generator=gen)).clamp(-1, 1)
            actions.append(a.clone())
            obs, r, done, s, dist = env.step(a)
            states.append(env.state.clone())
            newly = done & active
            succ = torch.where(newly, s, succ)
            active = active & ~done
        # states [L+1,n,10], actions [L,n,2]; keep per-env episodes with >=FUTURE_T+2 steps
        for j in range(n):
            Lj = actions[0].shape[0]  # not per-env; use common L (batch stops together)
        L = len(actions)
        S = torch.stack(states)      # [L+1,n,10]
        A = torch.stack(actions)     # [L,n,2]
        for j in range(n):
            if L >= FUTURE_T + 2:
                eps.append({"states": S[:, j], "actions": A[:, j], "success": bool(succ[j])})
        idx += n
        print(f"  demos {idx}/{n_episodes} (kept {len(eps)})", flush=True)
    torch.save(eps, out)


class Renderer:
    def __init__(self, device, size=48):
        self.size = size
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
        self.grid = torch.stack([xs, ys], dim=-1).to(device)

    def __call__(self, states):
        """states [B,10] -> [B,3,48,48]"""
        g = self.grid
        B = states.shape[0]
        hand, block, goal = states[:, 0:2], states[:, 4:6], states[:, 8:10]

        def d2(c):
            d = g.unsqueeze(0) - c[:, None, None, :]
            return (d ** 2).sum(-1)

        hand_m = (d2(hand) < 0.05 ** 2).float()
        block_m = (d2(block) < 0.09 ** 2).float()
        dg = d2(goal) ** 0.5
        goal_m = ((dg < 0.15) & (dg > 0.12)).float()
        img = torch.full((B, 3, self.size, self.size), 0.08, device=states.device)
        img[:, 1] += goal_m * 0.8
        img[:, 2] = torch.where(block_m > 0, torch.full_like(block_m, 0.9), img[:, 2])
        img[:, 0] = torch.where(block_m > 0, torch.full_like(block_m, 0.15), img[:, 0])
        img[:, 1] = torch.where(block_m > 0, torch.full_like(block_m, 0.35), img[:, 1])
        for c in range(3):
            img[:, c] = torch.where(hand_m > 0, torch.ones_like(hand_m), img[:, c])
        return img.clamp(0, 1)


def sample_batch(eps, batch, gen, device, renderer, state_stats):
    """Returns f0, a_chunk, fT_state_norm, fT_img."""
    idx = torch.randint(0, len(eps), (batch,), generator=gen).tolist()
    ts = [int(torch.randint(0, len(eps[i]["actions"]) - FUTURE_T, (1,), generator=gen)) for i in idx]
    s0 = torch.stack([eps[i]["states"][t] for i, t in zip(idx, ts)]).to(device)
    sT = torch.stack([eps[i]["states"][t + FUTURE_T] for i, t in zip(idx, ts)]).to(device)
    chunks = []
    for i, t in zip(idx, ts):
        a = eps[i]["actions"][t:t + 4]
        if a.shape[0] < 4:
            a = torch.cat([a, a[-1:].repeat(4 - a.shape[0], 1)], 0)
        chunks.append(a)
    a_chunk = torch.stack(chunks).to(device)
    f0 = renderer(s0)
    fT = renderer(sT)
    sT_norm = (sT - state_stats["mean"].to(device)) / state_stats["std"].to(device)
    return f0, a_chunk, sT_norm, fT


def flow_loss_action(model, f0, a_chunk, gen, device):
    B = f0.shape[0]
    u = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(device)
    tau = (0.05 + 0.9 * u).clamp(0, 1)
    eps = torch.randn(a_chunk.shape, generator=gen).to(device)
    x = (1 - tau[:, None, None]) * eps + tau[:, None, None] * a_chunk
    a_hat = model(f0, x, tau)
    return F.mse_loss(a_hat, a_chunk), tau, eps


def foresight_loss(arm, model, head, anchor, f0, fT, sT_norm, const_target):
    feat = model.foresight_features(f0)
    if arm == "fs_oracle":
        pred, _ = head(feat.mean(dim=1, keepdim=True))
        target = sT_norm.unsqueeze(1)
    elif arm in ("fs_random", "fs_current_random"):
        src = f0 if arm == "fs_current_random" else fT
        with torch.no_grad():
            target = anchor(src)
        pred, _ = head(feat)
    elif arm == "fs_pixels":
        with torch.no_grad():
            target = pixel_tokens(fT)
        pred, _ = head(feat)
    elif arm == "fs_const":
        target = const_target.expand(feat.shape[0], -1, -1)
        pred, _ = head(feat)
    else:
        raise ValueError(arm)
    return (1 - F.cosine_similarity(pred.float(), target.detach().float(), dim=-1)).mean()


def latent_z(arm, anchor, fT, sT_norm):
    if arm == "latent_const":
        return _const_latent_vec(fT.device).expand(fT.shape[0], -1)
    if arm == "latent_oracle":
        return sT_norm
    if arm == "latent_random":
        with torch.no_grad():
            return anchor(fT).mean(dim=1)
    if arm == "latent_pixels":
        with torch.no_grad():
            return pixel_tokens(fT).mean(dim=1)
    raise ValueError(arm)


def _const_latent_vec(device):
    g = torch.Generator().manual_seed(778)
    return torch.randn(64, generator=g).to(device)


def flow_loss_latent(model, f0, z, gen, device):
    """Single-token flow matching on the future global representation z."""
    B = f0.shape[0]
    u = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(device)
    tau = (0.05 + 0.9 * u).clamp(0, 1)
    eps = torch.randn(z.shape, generator=gen).to(device)
    x = (1 - tau[:, None]) * eps + tau[:, None] * z
    v = model.encode(f0)
    te = model.time_mlp(__import__("model_visual").timestep_embedding(tau, 64))[:, None, :]
    a = model.z_in(x)[:, None, :] + te
    xx = torch.cat([v, a], dim=1)
    nv = v.shape[1]
    mask = torch.zeros(nv + 1, nv + 1, dtype=torch.bool, device=device)
    mask[:nv, nv:] = True
    for b in model.expert:
        xx = b(xx, mask=mask)
    z_hat = model.z_head(xx[:, nv])
    return F.mse_loss(z_hat, z)


@torch.no_grad()
def evaluate(model, seed, device, renderer, n_episodes=200, K=10, max_batch=200):
    model.eval()
    env = PushEnv(n_envs=max_batch, seed=seed + 5000, max_steps=60)
    gen_env = torch.Generator().manual_seed(seed * 7919 + 13)
    gen_pol = torch.Generator().manual_seed(seed * 7919 + 29)
    succ_all, steps_all = [], []
    idx = 0
    while idx < n_episodes:
        n = min(max_batch, n_episodes - idx)
        env.n = n
        env.reset(gen_env)
        active = torch.ones(n, dtype=torch.bool)
        succ = torch.zeros(n, dtype=torch.bool)
        nsteps = torch.zeros(n)
        while bool(active.any()):
            img = renderer(env.state.to(device))
            a = torch.randn(n, 4, 2, device=device)
            for k in range(K):
                tau = torch.full((n,), k / K, device=device)
                a_hat = model(img, a, tau)
                v = (a_hat - a) / (1 - tau[:, None, None])
                a = a + v * (1.0 / K)
            chunk = a.clamp(-1, 1).cpu()
            for hh in range(2):
                obs, r, done, s, dist = env.step(chunk[:, hh])
                nsteps += active.float()
                newly = done & active
                succ = torch.where(newly, s, succ)
                active = active & ~done
        succ_all.append(succ)
        steps_all.append(nsteps)
        idx += n
    model.train()
    return {"success": float(torch.cat(succ_all).float().mean()),
            "steps": float(torch.cat(steps_all).mean())}


@torch.no_grad()
def future_probe(model, eps, device, renderer, state_stats, n_fit=2000, n_val=1000, seed=99):
    """Ridge from frozen current features to future state; returns R^2."""
    gen = torch.Generator().manual_seed(seed)
    model.eval()

    def collect(n, offset):
        f0, _, sT, _ = sample_batch(eps, n, gen, device, renderer, state_stats)
        feat = model.foresight_features(f0).mean(dim=1)
        return feat.float(), sT.float()

    Xf, Yf = collect(n_fit, 0)
    Xv, Yv = collect(n_val, 1)
    Xf = torch.cat([Xf, torch.ones(n_fit, 1, device=device)], 1)
    Xv = torch.cat([Xv, torch.ones(n_val, 1, device=device)], 1)
    lam = 1e-2 * Xf.shape[0]
    W = torch.linalg.solve(Xf.T @ Xf + lam * torch.eye(Xf.shape[1], device=device), Xf.T @ Yf)
    pred = Xv @ W
    ss_res = ((Yv - pred) ** 2).sum(0)
    ss_tot = ((Yv - Yv.mean(0)) ** 2).sum(0)
    r2 = 1 - (ss_res / ss_tot.clamp_min(1e-9)).mean()
    model.train()
    return float(r2)


def run_arm(arm, seed, args):
    out_json = os.path.join(args.out, f"{arm}_seed{seed}.json")
    if os.path.exists(out_json):
        print(f"skip {arm} seed{seed}", flush=True)
        return
    t0 = time.time()
    device = args.device
    torch.manual_seed(seed * 1000 + 7)
    gen = torch.Generator().manual_seed(seed * 31337 + 5)
    eps = torch.load(args.data, weights_only=False)
    state_stats = torch.load(args.stats, weights_only=False)
    renderer = Renderer(device)
    base_arm = arm[:-5] if arm.endswith("_long") else arm
    model = VisualPolicy().to(device)
    anchor = RandomAnchor().to(device) if "random" in arm else None
    const_target = torch.randn(1, 36, 64, generator=torch.Generator().manual_seed(777)).to(device)
    head = None
    if arm.startswith("fs_"):
        tdim = {"fs_oracle": 10, "fs_random": 64, "fs_pixels": 192,
                "fs_const": 64, "fs_current_random": 64}[arm]
        head = ForesightHead(target_dim=tdim).to(device)
    if base_arm.startswith("latent_"):
        zdim = {"latent_oracle": 10, "latent_random": 64, "latent_pixels": 192, "latent_const": 64}[base_arm]
        model.z_in = torch.nn.Linear(zdim, 64).to(device)
        model.z_head = torch.nn.Linear(64, zdim).to(device)
    params = list(model.parameters()) + ([] if head is None else list(head.parameters()))
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    total = (args.steps + args.steps // 2) if arm.endswith("_long") else args.steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-4)
    stage1 = args.steps // 2 if base_arm.startswith("latent_") else 0
    log = {"arm": arm, "seed": seed, "losses": []}
    lam = args.lam
    for step in range(total):
        f0, a_chunk, sT_norm, fT = sample_batch(eps, args.batch, gen, device, renderer, state_stats)
        if base_arm.startswith("latent_") and step < stage1:
            z = latent_z(base_arm, anchor, fT, sT_norm)
            loss = flow_loss_latent(model, f0, z, gen, device)
            kind = "latent"
        else:
            loss_a, _, _ = flow_loss_action(model, f0, a_chunk, gen, device)
            loss = loss_a
            kind = "action"
            if arm.startswith("fs_"):
                loss_f = foresight_loss(arm, model, head, anchor, f0, fT, sT_norm, const_target)
                loss = loss_a + lam * loss_f
                kind = "action+fs"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % args.log_every == 0 or step == 0:
            log["losses"].append({"step": step + 1, "loss": float(loss.detach()), "kind": kind})
            print(f"{arm} s{seed} step {step+1}/{total} {kind} loss {float(loss.detach()):.4f} ({time.time()-t0:.0f}s)", flush=True)
    ev = evaluate(model, seed, device, renderer, n_episodes=args.eval_episodes)
    probe_eps = torch.load(args.probe_data, weights_only=False) if os.path.exists(args.probe_data) else eps
    r2 = future_probe(model, probe_eps, device, renderer, state_stats)
    log.update({"success": ev["success"], "steps_avg": ev["steps"], "probe_r2": r2,
                "wall_s": time.time() - t0})
    with open(out_json, "w") as f:
        json.dump(log, f)
    torch.save(model.state_dict(), os.path.join(args.out, f"{arm}_seed{seed}.pt"))
    print(f"DONE {arm} seed{seed}: success {ev['success']:.3f} probeR2 {r2:.3f} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/demos.pt")
    ap.add_argument("--probe-data", default="data/demos_probe.pt")
    ap.add_argument("--stats", default="data/stats.pt")
    ap.add_argument("--out", default="out")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gen-demos", type=int, default=0)
    args = ap.parse_args()
    if args.gen_demos:
        gen_demos(args.gen_demos, 0, args.data)
        gen_demos(max(500, args.gen_demos // 5), 1, args.probe_data)
        eps = torch.load(args.data, weights_only=False)
        allstates = torch.cat([e["states"] for e in eps[:200]])
        torch.save({"mean": allstates.mean(0), "std": allstates.std(0).clamp_min(1e-3)}, args.stats)
        print(f"saved {len(eps)} episodes", flush=True)
        return
    os.makedirs(args.out, exist_ok=True)
    for arm in args.arms.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            run_arm(arm, seed, args)


if __name__ == "__main__":
    main()
