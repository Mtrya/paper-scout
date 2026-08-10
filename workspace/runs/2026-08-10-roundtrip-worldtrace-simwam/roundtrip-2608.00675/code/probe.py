"""Round-trip blind-spot probe.

For each Lorenz regime (trained model in runs/<r>/):
  1. Teacher-forced one-step error (model-quality control).
  2. Per test trajectory: forward rollout to depth i, true error E_i,
     round-trip consistency C_i, backward residual delta_i (noise floor).
  3. Spearman(C_i, E_i) across trajectories at fixed depth -- the paper's
     headline statistic (their 0.91-0.98 on MHD).
  4. Co-Lipschitz probe: min singular value of the learned one-step backward
     map's Jacobian along rollouts (their Assumption 1, measured).
Outputs results/<r>/metrics.json plus arrays for plotting.
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from systems import gen_dataset, BURN_IN
from model import BidirectionalDenoiser, DDIM, rollout

DEPTHS = [5, 10, 20, 40, 80]
N_TEST = 128
T_PHYS = 120  # matches train trajectory length


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state_dim = ck["mean"].shape[0]
    model = BidirectionalDenoiser(state_dim=state_dim)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["mean"], ck["std"]


def fd_jacobian_svs(model, ddim, pair, t_phys_val, cd_val, eps=1e-3):
    """Finite-difference Jacobian of the one-step map at `pair` [2,3].

    The map is made deterministic by fixing the DDIM terminal noise.
    Returns singular values of the 6x6 Jacobian.
    """
    base = pair.reshape(1, -1)
    dim = base.shape[1] // 2
    gen = torch.Generator().manual_seed(1234)

    def f(p):
        noise = torch.randn(1, dim, generator=gen)
        gen.manual_seed(1234)
        t = torch.tensor([t_phys_val])
        cd = torch.tensor([cd_val])
        return ddim.transition(model, noise, p, t, cd).reshape(-1)

    y0 = f(base)
    J = torch.zeros(dim, 2 * dim)
    for j in range(2 * dim):
        dp = base.clone()
        dp[0, j] += eps
        J[:, j] = (f(dp) - y0) / eps
    return torch.linalg.svdvals(J).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    model, mean, std = load(os.path.join(args.run, "model.pt"))
    ddim = DDIM()
    rng = np.random.default_rng(1000 + int(args.r * 10))

    trajs = gen_dataset(args.r, N_TEST, T_PHYS, rng, burn_in=BURN_IN.get(args.r, 400))
    trajs_n = (trajs - mean) / std
    data = torch.tensor(trajs_n, dtype=torch.float32)
    T0 = 20  # start index (needs t0>=2 for pairs, t0+i<=T)

    # ---- one-step teacher-forced error -------------------------------------
    ti = torch.randint(2, T_PHYS - 2, (512,))
    tr_i = torch.randint(0, N_TEST, (512,))
    pair = torch.stack([data[tr_i, ti - 1], data[tr_i, ti]], dim=1)
    gen = torch.Generator().manual_seed(7)
    pred1 = rollout(model, ddim, pair, 1, +1, ti.float().mean().item(), T_PHYS, gen)[:, 0]
    one_step_fwd = torch.mean((pred1 - data[tr_i, ti + 1]) ** 2).item()
    gen = torch.Generator().manual_seed(7)
    pred1b = rollout(model, ddim, pair, 1, -1, ti.float().mean().item(), T_PHYS, gen)[:, 0]
    one_step_bwd = torch.mean((pred1b - data[tr_i, ti - 2]) ** 2).item()

    # ---- round trips --------------------------------------------------------
    seed_pair = torch.stack([data[:, T0 - 1], data[:, T0]], dim=1)  # [B,2,3]
    gen = torch.Generator().manual_seed(42)
    fwd = rollout(model, ddim, seed_pair, max(DEPTHS), +1, T0, T_PHYS, gen)  # [B,80,3]

    results = {"r": args.r, "one_step_fwd": one_step_fwd, "one_step_bwd": one_step_bwd,
               "depths": {}}
    store = {"E": {}, "C": {}, "delta": {}}

    for i in DEPTHS:
        term_pair = torch.stack([fwd[:, i - 2], fwd[:, i - 1]], dim=1)  # chronological
        gen = torch.Generator().manual_seed(100 + i)
        back = rollout(model, ddim, term_pair, i, -1, T0 + i, T_PHYS, gen)
        ret_pair = torch.stack([back[:, i - 2], back[:, i - 1]], dim=1)  # (x̃_{t0-1}, x̃_{t0})

        true_terminal = data[:, T0 + i]
        E = torch.mean((fwd[:, i - 1] - true_terminal) ** 2, dim=-1)      # [B]
        C = 0.5 * (torch.mean((ret_pair[:, 1] - data[:, T0]) ** 2, dim=-1)
                   + torch.mean((ret_pair[:, 0] - data[:, T0 - 1]) ** 2, dim=-1))

        # noise floor: backward rollout seeded with TRUE terminal pair
        true_term_pair = torch.stack([data[:, T0 + i - 1], data[:, T0 + i]], dim=1)
        gen = torch.Generator().manual_seed(200 + i)
        back_true = rollout(model, ddim, true_term_pair, i, -1, T0 + i, T_PHYS, gen)
        ret_true = torch.stack([back_true[:, i - 2], back_true[:, i - 1]], dim=1)
        delta = 0.5 * (torch.mean((ret_true[:, 1] - data[:, T0]) ** 2, dim=-1)
                       + torch.mean((ret_true[:, 0] - data[:, T0 - 1]) ** 2, dim=-1))

        rho = spearmanr(C.numpy(), E.numpy()).statistic
        results["depths"][str(i)] = {
            "spearman_CE": float(rho),
            "E_mean": float(E.mean()), "E_median": float(E.median()),
            "E_std": float(E.std()),
            "C_mean": float(C.mean()), "C_median": float(C.median()),
            "C_std": float(C.std()),
            "delta_mean": float(delta.mean()), "delta_median": float(delta.median()),
        }
        store["E"][str(i)] = E.numpy()
        store["C"][str(i)] = C.numpy()
        store["delta"][str(i)] = delta.numpy()
        print(f"[r={args.r}] depth {i}: spearman={rho:.3f} "
              f"E_med={E.median():.4f} C_med={C.median():.4f} delta_med={delta.median():.4f}",
              flush=True)

    # ---- co-Lipschitz (mu) along rollouts ----------------------------------
    mus = {}
    for i in [5, 20, 40]:
        sv_mins, sv_maxs = [], []
        for b in range(16):
            pair_b = torch.stack([fwd[b, i - 2], fwd[b, i - 1]], dim=0)
            svs = fd_jacobian_svs(model, ddim, pair_b, (T0 + i) / T_PHYS, cd_val=1)
            sv_mins.append(float(svs.min()))
            sv_maxs.append(float(svs.max()))
        mus[str(i)] = {"mu_min_sv": float(np.mean(sv_mins)),
                       "max_sv": float(np.mean(sv_maxs))}
        print(f"[r={args.r}] depth {i}: backward-step min sv (mu proxy) "
              f"{np.mean(sv_mins):.4f}, max sv {np.mean(sv_maxs):.4f}", flush=True)
    results["mu"] = mus

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    np.savez(os.path.join(args.out, "arrays.npz"),
             **{f"E_{k}": v for k, v in store["E"].items()},
             **{f"C_{k}": v for k, v in store["C"].items()},
             **{f"delta_{k}": v for k, v in store["delta"].items()},
             fwd=fwd.numpy(), data=data.numpy(), T0=T0)
    print(f"[r={args.r}] probe done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
