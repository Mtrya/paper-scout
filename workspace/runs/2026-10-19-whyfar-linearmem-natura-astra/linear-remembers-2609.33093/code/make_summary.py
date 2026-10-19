"""Build the per-head table and a compact summary of every experiment."""
import json
import os

import numpy as np

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
    tau = np.load(TAU_PATH)
    flat = tau.ravel()
    tau_rank = (-flat).argsort().argsort() + 1
    Jv = np.load(os.path.join(RES, "screen_J_validation.npy"))
    Jt = np.load(os.path.join(RES, "screen_J_test.npy"))
    rv = json.load(open(os.path.join(RES, "state_retention_validation.json")))
    rt = json.load(open(os.path.join(RES, "state_retention_test.json")))
    r512v = np.array(rv["512"]["rel"])
    r512t = np.array(rt["512"]["rel"])
    r0v = np.array(rv["0"]["rel"])
    rank512 = (-r512v.ravel()).argsort().argsort() + 1
    rank512t = (-r512t.ravel()).argsort().argsort() + 1
    lines = ["layer,head,tau,tau_rank,tau_eff_512,rel_d0,rel_d512_val,rank_d512_val,"
             "rank_d512_test,J_val,J_test"]
    for l in range(24):
        for h in range(4):
            i = l * 4 + h
            lines.append(f"{l},{h},{tau[l,h]:.4f},{tau_rank[i]},"
                         f"{rv['512']['tau_eff'][l][h]:.2f},{r0v[l,h]:.5f},"
                         f"{r512v[l,h]:.5f},{rank512[i]},{rank512t[i]},"
                         f"{Jv[l,h]:+.5f},{Jt[l,h]:+.5f}")
    with open(os.path.join(RES, "head_table.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    summary = {
        "baseline": {s: json.load(open(os.path.join(RES, f"baseline_{s}.json")))
                     for s in ["validation", "test"]},
        "screen": {s: json.load(open(os.path.join(RES, f"screen_{s}.json")))
                   for s in ["validation", "test"]},
        "retention": {"validation": rv, "test": rt},
        "write": {h: json.load(open(os.path.join(RES, f"write_validation_{h}.json")))
                  for h in ["L11H2", "L20H2"]},
        "load": {s: json.load(open(os.path.join(RES, f"load_{s}.json")))
                 for s in ["validation", "test"]},
        "distance": json.load(open(os.path.join(RES, "distance_validation.json"))),
        "load_state": json.load(open(os.path.join(RES, "load_state_validation_L11H2.json"))),
        "tau_cross": json.load(open(os.path.join(RES, "tau_cross.json"))),
    }
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f)
    print("wrote head_table.csv and summary.json")
    print("head rows:", len(lines) - 1)


if __name__ == "__main__":
    main()
