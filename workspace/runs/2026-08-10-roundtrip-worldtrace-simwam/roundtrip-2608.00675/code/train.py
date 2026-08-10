"""Train one bidirectional denoiser per Lorenz regime."""
import argparse
import json
import os
import time

import numpy as np
import torch

from systems import gen_dataset, BURN_IN
from model import BidirectionalDenoiser, DDIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n-traj", type=int, default=800)
    ap.add_argument("--n-steps", type=int, default=120)
    ap.add_argument("--train-steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cpu"

    print(f"[r={args.r}] generating {args.n_traj} trajectories ...", flush=True)
    trajs = gen_dataset(args.r, args.n_traj, args.n_steps, rng,
                        burn_in=BURN_IN.get(args.r, 400))  # [N,T,3]
    mean = trajs.reshape(-1, trajs.shape[-1]).mean(0)
    std = trajs.reshape(-1, trajs.shape[-1]).std(0) + 1e-6
    trajs_n = (trajs - mean) / std
    data = torch.tensor(trajs_n, dtype=torch.float32)
    N, T, D = data.shape

    model = BidirectionalDenoiser(state_dim=D)
    ddim = DDIM(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.train_steps)

    t0 = time.time()
    for step in range(args.train_steps):
        ti = torch.randint(2, T - 2, (args.batch,))          # pair's latest index t
        traj_i = torch.randint(0, N, (args.batch,))
        cd = torch.randint(0, 2, (args.batch,))               # 0=forward,1=backward

        x_prev = data[traj_i, ti - 1]                         # x_{t-1}
        x_t = data[traj_i, ti]                                # x_t
        ctx = torch.cat([x_prev, x_t], dim=-1)                # chronological pair
        # forward target x_{t+1}; backward target x_{t-2}
        tgt_idx = torch.where(cd == 0, ti + 1, ti - 2)
        x_tgt = data[traj_i, tgt_idx]

        k_idx = torch.randint(0, ddim.n_train, (args.batch,))
        eps = torch.randn_like(x_tgt)
        x_noisy = ddim.add_noise(x_tgt, k_idx, eps)
        k_frac = k_idx.float() / (ddim.n_train - 1)
        t_phys = ti.float() / T

        pred = model(x_noisy, k_frac, t_phys, cd, ctx)
        loss = torch.mean((pred - eps) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % 1000 == 0 or step == args.train_steps - 1:
            print(f"[r={args.r}] step {step} loss {loss.item():.5f} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "mean": mean, "std": std,
        "r": args.r, "n_traj": args.n_traj, "n_steps": args.n_steps,
    }, os.path.join(args.out, "model.pt"))
    json.dump({"r": args.r, "mean": mean.tolist(), "std": std.tolist(),
               "train_steps": args.train_steps, "final_loss": loss.item()},
              open(os.path.join(args.out, "train_meta.json"), "w"), indent=2)
    print(f"[r={args.r}] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
