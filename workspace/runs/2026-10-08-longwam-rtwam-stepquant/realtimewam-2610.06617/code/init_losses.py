"""Both distillation objectives evaluated at initialization (student = EMA = teacher).

Reports the starting height of the two curves in Fig. 2, on the training
sampling distribution used by distill.py (tau ~ U[0, 0.9], x from the demo data).
"""
import argparse
import json
import os

import torch

from data import make_demos
from distill import DELTA, K_STEP, TAU_MAX, endpoint, teacher_rollout
from model import FlowPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--n-demos", type=int, default=2500)
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
    for p in teacher.parameters():
        p.requires_grad_(False)

    obs, act, _ = make_demos(args.n_demos, args.seed,
                             env_kwargs={"max_steps": args.max_steps},
                             noise=args.demo_noise, tang_sign=args.tang_sign)
    g = torch.Generator().manual_seed(20261008)
    idx = torch.randint(0, obs.shape[0], (args.batch,), generator=g)
    ob, ac = obs[idx], act[idx]
    tau = torch.rand(args.batch, generator=g) * TAU_MAX
    eps = torch.randn(ac.shape, generator=g)
    x = (1 - tau).view(-1, 1, 1) * eps + tau.view(-1, 1, 1) * ac

    f = endpoint(teacher, x, ob, tau)
    x_nb = x + DELTA * teacher.velocity(x, ob, tau)
    tgt = endpoint(teacher, x_nb, ob, tau + DELTA)
    a0T = teacher_rollout(teacher, x, ob, tau, K=K_STEP)
    res = {"teacher": args.teacher, "seed": args.seed, "batch": args.batch,
           "init_loss_cd": float(((f - tgt) ** 2).mean()),
           "init_loss_ta": float((((f - a0T) / (1 - tau).view(-1, 1, 1)) ** 2).mean())}
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
