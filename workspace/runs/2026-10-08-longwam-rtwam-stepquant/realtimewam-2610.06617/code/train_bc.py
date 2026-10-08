"""BC pretraining of the flow policy (continuous or discrete-timestep)."""
import argparse
import json
import os
import time

import torch

from data import make_demos
from model import FlowPolicy
from rollout import run_episodes


def sample_tau(n, mode, K, gen):
    if mode == "continuous":
        u = torch.distributions.Beta(1.5, 1.0).sample((n,))
        return (0.05 + 0.9 * u).clamp(0.0, 1.0)
    if mode == "uniform":
        return torch.rand(n, generator=gen)
    if mode == "discrete":
        k = torch.randint(0, K, (n,), generator=gen)
        return k.float() / K
    raise ValueError(mode)


def train_bc_steps(model, obs, act, n_steps, batch=256, lr=1e-3, lr_end=1e-4,
                   tau_mode="continuous", K=10, seed=0, log_every=0):
    """Optimizer updates on flow-matching BC. Returns (losses, n_updates, n_samples)."""
    gen = torch.Generator().manual_seed(seed * 7717 + 3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps,
                                                       eta_min=lr_end)
    M = obs.shape[0]
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        idx = torch.randint(0, M, (batch,), generator=gen)
        ob, ac = obs[idx], act[idx]
        tau = sample_tau(batch, tau_mode, K, gen)
        loss = model.flow_loss(ob, ac, tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step+1}/{n_steps} loss {sum(losses[-log_every:])/log_every:.5f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="out")
    ap.add_argument("--n-demos", type=int, default=2500)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--demo-noise", type=float, default=0.25)
    ap.add_argument("--tang-sign", type=str, default="fixed",
                    help="'fixed' (single circling direction) or 'random'")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--pred-mode", type=str, default="x")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--init-scale", type=float, default=0.0)
    ap.add_argument("--zero-res", action="store_true")
    ap.add_argument("--zero-out", action="store_true")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    policy = FlowPolicy(pred_mode=args.pred_mode, init_scale=args.init_scale,
                        zero_res=args.zero_res, zero_out=args.zero_out,
                        hidden=args.hidden, n_blocks=args.n_blocks, d=args.d)
    n_par = sum(p.numel() for p in policy.parameters())
    ts_names = policy.ts_module_names()
    ts_par = sum(p.numel() for n, p in policy.named_parameters()
                 if any(n == t or n.startswith(t + ".") for t in ts_names))
    print(f"params {n_par} ts {ts_par} ({100*ts_par/n_par:.2f}%)", flush=True)

    env_kwargs = {"max_steps": args.max_steps}
    obs, act, succ = make_demos(args.n_demos, args.seed, env_kwargs=env_kwargs,
                                noise=args.demo_noise, tang_sign=args.tang_sign)
    print(f"demos {obs.shape[0]} pairs, expert success {float(succ.float().mean()):.3f}",
          flush=True)
    torch.save({"model": policy.state_dict()}, os.path.join(args.out, "M_init.pt"))

    mid = args.steps // 2
    t0 = time.time()
    l1 = train_bc_steps(policy, obs, act, mid, args.batch, args.lr, 1e-4,
                        "continuous", args.K, args.seed, args.log_every)
    torch.save({"model": policy.state_dict()}, os.path.join(args.out, "M_mid.pt"))
    ev_mid = run_episodes(policy, args.eval_episodes, 1000 + args.seed, K=args.K,
                          env_kwargs=env_kwargs)
    print(f"mid  success {float(ev_mid['success'].float().mean()):.3f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    l2 = train_bc_steps(policy, obs, act, args.steps - mid, args.batch, args.lr * 0.5,
                        1e-4, "continuous", args.K, args.seed + 100, args.log_every)
    torch.save({"model": policy.state_dict()}, os.path.join(args.out, "M_bc.pt"))
    ev = run_episodes(policy, args.eval_episodes, 1000 + args.seed, K=args.K,
                      env_kwargs=env_kwargs)
    succ_rate = float(ev["success"].float().mean())
    print(f"bc   success {succ_rate:.3f} ret {float(ev['ret'].mean()):.2f} "
          f"steps {float(ev['steps'].mean()):.1f} ({time.time()-t0:.1f}s)", flush=True)

    with open(os.path.join(args.out, "bc_metrics.json"), "w") as f:
        json.dump({"seed": args.seed, "pred_mode": args.pred_mode,
                   "demo_noise": args.demo_noise, "tang_sign": args.tang_sign,
                   "max_steps": args.max_steps, "n_params": n_par, "ts_params": ts_par,
                   "ts_share": ts_par / n_par, "n_pairs": int(obs.shape[0]),
                   "expert_success": float(succ.float().mean()),
                   "bc_success": succ_rate,
                   "bc_return": float(ev["ret"].mean()),
                   "bc_steps": float(ev["steps"].mean()),
                   "mid_success": float(ev_mid["success"].float().mean()),
                   "loss_first100": sum(l1[:100]) / 100,
                   "loss_last100": sum(l1[-100:]) / 100,
                   "loss_final": sum(l2[-100:]) / 100,
                   "time_s": time.time() - t0}, f, indent=1)


if __name__ == "__main__":
    main()
