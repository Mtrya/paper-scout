"""C5: probing episode outcomes from shift-update projections + steering."""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

from env import PushEnv
from rollout import run_episodes
from wutil import load_policy, output_vectors


def shift_dirs(base_pol, rl_pol, K=10):
    """Unit shift-update direction per (block, k). (K, L, d) + raw norms."""
    taus = [k / K for k in range(K)]
    tb = output_vectors(base_pol, taus)   # (T, L, 3, d)
    tn = output_vectors(rl_pol, taus)
    d = tn[:, :, 1] - tb[:, :, 1]         # shift component
    norm = d.norm(dim=-1, keepdim=True)
    return d / norm.clamp_min(1e-8), norm.squeeze(-1)


def features_from_hidden(hidden, dirs):
    """hidden: (n, K, L, d); dirs: (K, L, d) -> (n, K*L)."""
    with torch.no_grad():
        return (hidden * dirs.unsqueeze(0)).sum(-1).reshape(
            hidden.shape[0], -1).detach()


def collect(policy, n, seed, K, env_kwargs, dist_lo=0.6, dist_hi=1.0):
    return run_episodes(policy, n, seed, K=K, record_hidden=True,
                        dist_lo=dist_lo, dist_hi=dist_hi, env_kwargs=env_kwargs,
                        max_batch=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--episodes-hard", type=int, default=400)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--pred-mode", type=str, default="x")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--steer-episodes", type=int, default=200)
    ap.add_argument("--out", type=str, default="analysis/probe.json")
    args = ap.parse_args()

    env_kwargs = {"max_steps": args.max_steps}
    all_res = {}
    for d in args.dirs:
        name = os.path.basename(d.rstrip("/"))
        seed = 1000 + int(name.replace("seed", ""))
        bc = load_policy(os.path.join(d, "M_bc.pt"), args.pred_mode)
        rl = load_policy(os.path.join(d, "M_rl.pt"), args.pred_mode)
        dirs, dnorm = shift_dirs(bc, rl, args.K)      # (K, L, d)
        L = dirs.shape[1]
        res = {"shift_norm_mean": float(dnorm.mean()),
               "shift_norm_per_k": dnorm.mean(-1).tolist()}

        # --- probe data: BC + RL rollouts, mixture of difficulties ---
        parts_x, parts_y, parts_pol = [], [], []
        hiddens = []
        for pi, pol in enumerate([bc, rl]):
            r1 = collect(pol, args.episodes, seed + 100 * pi, args.K, env_kwargs)
            r2 = collect(pol, args.episodes_hard, seed + 100 * pi + 7, args.K,
                         env_kwargs, dist_lo=0.8, dist_hi=1.3)
            for r in (r1, r2):
                hiddens.append(r["hidden"])
                parts_x.append(features_from_hidden(r["hidden"], dirs))
                parts_y.append(r["success"].numpy().astype(int))
                parts_pol.append(np.full(len(r["success"]), pi))
        X = np.concatenate(parts_x, 0)
        y = np.concatenate(parts_y, 0)
        pol_id = np.concatenate(parts_pol, 0)
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(y))
        n_train = int(0.7 * len(y))
        tr, te = perm[:n_train], perm[n_train:]
        res["n_episodes"] = int(len(y))
        res["success_rate_data"] = float(y.mean())

        def fit_auc(Xtr, ytr, Xte, yte):
            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                     class_weight="balanced")
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, p) if len(set(yte.tolist())) > 1 else float("nan")
            f1 = f1_score(yte, (p > 0.5).astype(int))
            return auc, f1, clf

        auc, f1, clf = fit_auc(X[tr], y[tr], X[te], y[te])
        res["probe_all_auc"] = float(auc)
        res["probe_all_f1"] = float(f1)
        # random-label control
        aucs = []
        for r in range(10):
            ysh = rng.permutation(y)
            a, _, _ = fit_auc(X[tr], ysh[tr], X[te], ysh[te])
            aucs.append(a)
        res["probe_random_auc_mean"] = float(np.mean(aucs))
        res["probe_random_auc_std"] = float(np.std(aucs))
        # per-policy probes
        for pi, label in enumerate(["bc", "rl"]):
            m = pol_id == pi
            idx = np.where(m)[0]
            rng2 = np.random.RandomState(1)
            p2 = rng2.permutation(len(idx))
            tr2 = idx[p2[:int(0.7 * len(idx))]]
            te2 = idx[p2[int(0.7 * len(idx)):]]
            if len(set(y[te2].tolist())) > 1:
                a, f, _ = fit_auc(X[tr2], y[tr2], X[te2], y[te2])
                res[f"probe_{label}_auc"] = float(a)
                res[f"probe_{label}_f1"] = float(f)
        # random-direction control: identical features, directions replaced by
        # random unit vectors in R^d.
        g = torch.Generator().manual_seed(seed + 7)
        rdir = torch.randn(dirs.shape, generator=g)
        rdir = rdir / rdir.norm(dim=-1, keepdim=True)
        Xr = np.concatenate([features_from_hidden(h, rdir) for h in hiddens], 0)
        a_rd, _, _ = fit_auc(Xr[tr], y[tr], Xr[te], y[te])
        res["probe_randomdir_auc"] = float(a_rd)
        # single-position best AUC
        best = (0, -1)
        for j in range(X.shape[1]):
            if len(set(y[tr].tolist())) < 2:
                break
            a = fit_auc(X[tr][:, [j]], y[tr], X[te][:, [j]], y[te])[0]
            if a == a and a > best[1]:
                best = (j, a)
        res["probe_best_single_pos"] = int(best[0])
        res["probe_best_single_auc"] = float(best[1])
        res["probe_coef"] = clf.coef_.tolist()

        # --- steering ---
        k_best, l_best = divmod(int(best[0]), L)
        dvec = dirs[k_best, l_best]
        j = int(best[0])
        proj = X[:, j]
        m_succ = float(proj[y == 1].mean())
        random_dir = torch.randn_like(dvec)
        random_dir = random_dir / random_dir.norm()
        steer_res = {"k": k_best, "layer": l_best, "target": m_succ,
                     "dir_fingerprint": [float(x) for x in dvec[:4]]}
        def run_steer(pol, layer, k, direction, target, alpha, mode):
            pol.steer = {"layer": layer, "k": k, "dir": direction,
                         "target": target, "mode": mode, "alpha": alpha}
            r = run_episodes(pol, args.steer_episodes, seed, K=args.K,
                             env_kwargs=env_kwargs)
            pol.steer = None
            return float(r["success"].float().mean())

        scale = float(np.abs(proj).mean())
        for pi, pol in enumerate([rl, bc]):
            label = ["rl", "bc"][pi]
            row = {}
            pol.steer = None
            base = run_steer(pol, 0, 0, dvec, m_succ, 0.0, "adaptive")
            row["alpha0"] = base
            for a in args.alphas:
                row[f"shift_adaptive_a{a}"] = run_steer(pol, l_best, k_best, dvec,
                                                        m_succ, a, "adaptive")
            for a in args.alphas:
                row[f"shift_fixed_a{a}"] = run_steer(pol, l_best, k_best, dvec,
                                                     scale, a, "fixed")
            for a in args.alphas:
                row[f"random_fixed_a{a}"] = run_steer(pol, l_best, k_best,
                                                      random_dir, scale, a, "fixed")
            res[f"steer_{label}"] = row
            print(name, label, {k: round(v, 3) for k, v in row.items()}, flush=True)
        res["steer_info"] = steer_res
        all_res[name] = res
        print(name, "probe auc", res["probe_all_auc"], "rand",
              res["probe_random_auc_mean"], flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
