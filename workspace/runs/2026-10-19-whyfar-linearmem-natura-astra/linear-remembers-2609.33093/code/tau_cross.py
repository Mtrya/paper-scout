"""Cross the state-retention head ranking with STEPQuant's per-head gate half-life."""
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
if not os.path.isdir(RES):
    RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
TAU_PATH = os.path.join(HERE, "..", "results", "probe_a_tau.npy")
if not os.path.exists(TAU_PATH):
    TAU_PATH = os.path.join(HERE, "results", "probe_a_tau.npy")
if not os.path.exists(TAU_PATH):
    TAU_PATH = os.path.join(HERE, "..", "..", "runs", "2026-10-08-longwam-rtwam-stepquant",
                            "stepquant-2609.38169", "code", "results", "probe_a_tau.npy")


def main():
    tau = np.load(TAU_PATH)                     # [24,4], WikiText-2 gate half-life
    flat_tau = tau.ravel()
    tau_rank = (-flat_tau).argsort().argsort() + 1   # 1 = longest half-life
    names = {(20, 2): "L20H2 (paper hub)", (11, 2): "L11H2", (17, 3): "L17H3",
             (7, 2): "L7H2", (14, 2): "L14H2"}
    out = {"tau": tau.tolist(), "tau_rank": tau_rank.reshape(24, 4).tolist(), "splits": {}}
    for split in ["validation", "test"]:
        path = os.path.join(RES, f"state_retention_{split}.json")
        if not os.path.exists(path):
            continue
        ret = json.load(open(path))
        out["splits"][split] = {}
        print(f"=== {split} ===")
        for d, rec in ret.items():
            rel = np.array(rec["rel"])
            rho, p = spearmanr(flat_tau, rel.ravel())
            te = np.array(rec["tau_eff"])
            rho_e, p_e = spearmanr(te.ravel(), rel.ravel())
            order = np.argsort(-rel.ravel())
            top = []
            for idx in order[:8]:
                l, h = divmod(int(idx), 4)
                top.append({"head": f"L{l}H{h}", "rel": float(rel[l, h]),
                            "tau": float(tau[l, h]), "tau_rank": int(tau_rank[idx])})
            heads = {}
            for (l, h), nm in names.items():
                heads[nm] = {"rel": float(rel[l, h]),
                             "rel_rank": int((rel > rel[l, h]).sum() + 1),
                             "tau": float(tau[l, h]),
                             "tau_rank": int(tau_rank[l * 4 + h]),
                             "tau_eff": float(te[l, h])}
            out["splits"][split][d] = {
                "spearman_tau_rel": float(rho), "p": float(p),
                "spearman_tau_eff_rel": float(rho_e), "p_eff": float(p_e),
                "top": top, "named": heads}
            print(f"d={d:>4}: rho(tau,rel)={rho:+.3f} (p={p:.1e}), "
                  f"rho(tau_eff,rel)={rho_e:+.3f}")
            for nm, v in heads.items():
                print(f"    {nm:22s} rel={v['rel']:.4f} (rank {v['rel_rank']:2d}) "
                      f"tau={v['tau']:.2f} (rank {v['tau_rank']:2d}) "
                      f"tau_eff={v['tau_eff']:.1f}")
            print("    top:", ", ".join(
                f"{t['head']}={t['rel']:.3f}(tau#{t['tau_rank']})" for t in top[:5]))
    with open(os.path.join(RES, "tau_cross.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
