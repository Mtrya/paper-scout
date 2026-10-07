"""C1/C2/C3: update spectra, module concentration, scale/shift/gate geometry."""
import argparse
import json
import os

import torch

from wutil import (delta_dict, load_policy, output_vectors, spec_stats,
                   summarize_delta, ts_energy_share)

TAU_GRID = torch.linspace(0.0, 0.99, 200)
RL_TAUS = [k / 10 for k in range(10)]


def rl_discrete_directions(policy, dW, K=10, topn=3):
    """For each AdaRMS matrix: |cos| between top right-singular directions of
    dW and the conditioning vector c_tau, normalised per curve."""
    out = {}
    for k, v in dW.items():
        if ".ada." not in k or not k.endswith(".weight"):
            continue
        U, S, Vh = torch.linalg.svd(v, full_matrices=False)
        cg = []
        for t in TAU_GRID:
            tau = torch.full((1,), float(t))
            cg.append(policy.cond(tau)[0])
        c = torch.stack(cg, 0)                      # (T, d)
        cn = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        for i in range(min(topn, Vh.shape[0])):
            vi = Vh[i] / Vh[i].norm().clamp_min(1e-8)
            cos = (cn @ vi).abs()
            out[f"{k}|v{i}"] = {"cos": cos.tolist(),
                                "norm": float(S[i] ** 2 / (S ** 2).sum())}
    return out


def ssg_cosine(base_pol, new_pol, K=10, taus=None):
    """|cos| between AdaRMS output at tau and its change, per block/component."""
    taus = taus if taus is not None else [k / K for k in range(K)]
    tb = output_vectors(base_pol, taus)     # (T, L, 3, d)
    tn = output_vectors(new_pol, taus)
    d = tn - tb
    names = ["scale", "shift", "gate"]
    res = {}
    relmag = {}
    for ci, nm in enumerate(names):
        rel = []
        for ti in range(len(taus)):
            rel.append([float(d[ti, li, ci].norm() / tb[ti, li, ci].norm().clamp_min(1e-8))
                        for li in range(tb.shape[1])])
        relmag[nm] = rel
        cos = []
        for ti in range(len(taus)):
            row = []
            for li in range(tb.shape[1]):
                a = tb[ti, li, ci]
                b = d[ti, li, ci]
                row.append(float(a @ b / (a.norm() * b.norm()).clamp_min(1e-8)))
            cos.append(row)
        res[nm] = cos
    return {"taus": list(taus), "abs_cos": res, "rel_mag": relmag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="seed output dirs (out/seed0 ...)")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--pred-mode", type=str, default="x")
    ap.add_argument("--out", type=str, default="analysis/weights.json")
    args = ap.parse_args()

    keys = ["M_init", "M_mid", "M_bc", "M_rl", "M_mid_cont", "M_mid_disc",
            "M_bc_cont", "M_bc_disc", "M_mid_cont_long", "M_mid_disc_long"]
    all_res = {}
    for d in args.dirs:
        pol = {}
        for k in keys:
            p = os.path.join(d, k + ".pt")
            if os.path.exists(p):
                pol[k] = load_policy(p, args.pred_mode)
        if "M_init" not in pol or "M_bc" not in pol:
            print("skip", d)
            continue
        res = {}
        pairs = {
            "bc_stage": ("M_bc", "M_init"),
            "rl": ("M_rl", "M_bc"),
            "bc_cont": ("M_bc_cont", "M_bc"),
            "bc_disc": ("M_bc_disc", "M_bc"),
            "bc_cont_mid": ("M_mid_cont", "M_mid"),
            "bc_disc_mid": ("M_mid_disc", "M_mid"),
            "bc_cont_long": ("M_mid_cont_long", "M_mid"),
            "bc_disc_long": ("M_mid_disc_long", "M_mid"),
            "bc_stage_mid": ("M_mid", "M_init"),
        }
        dlt = {}
        for tag, (a, b) in pairs.items():
            if a in pol and b in pol:
                dd = delta_dict(pol[a], pol[b])
                dlt[tag] = dd
                st = summarize_delta(dd)
                st["ts_energy_share"] = ts_energy_share(dd)
                res[tag] = st
        # C3: scale/shift/gate geometry (RL vs base, and BC-branch vs base)
        if "rl" in dlt:
            res["ssg_rl_vs_bc"] = ssg_cosine(pol["M_bc"], pol["M_rl"], args.K)
        if "bc_disc" in dlt:
            res["ssg_disc_vs_bc"] = ssg_cosine(pol["M_bc"], pol["M_bc_disc"], args.K)
        if "bc_cont" in dlt:
            res["ssg_cont_vs_bc"] = ssg_cosine(pol["M_bc"], pol["M_bc_cont"], args.K)
        # C1/C2 mechanism: top input directions vs c_tau
        for tag in ["rl", "bc_disc", "bc_cont", "bc_disc_mid", "bc_cont_mid",
                    "bc_disc_long", "bc_cont_long", "bc_stage"]:
            if tag in dlt:
                base_key = ("M_mid" if tag in ("bc_disc_mid", "bc_cont_mid",
                                               "bc_disc_long", "bc_cont_long")
                            else "M_bc")
                res[f"tau_dirs_{tag}"] = rl_discrete_directions(
                    pol[base_key], dlt[tag], args.K)
        all_res[os.path.basename(d.rstrip("/"))] = res
        print("done", d)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
