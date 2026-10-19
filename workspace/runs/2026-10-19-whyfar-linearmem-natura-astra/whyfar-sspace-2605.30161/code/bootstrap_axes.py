"""Bootstrap the S-Space axes from per-example gradients and re-evaluate readout.

Uses the 2/cat smoke gradient records (8 gradients per axis) to show how much
the readout accuracy depends on the axis estimate.  CPU only.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_DIR, GROUPS  # noqa: E402

CATS = ["left", "right", "above", "below", "far", "close"]
READOUT_SIGN = {"left": -1, "right": 1, "above": 1, "below": -1,
                "far": -1, "close": 1}
GROUPS3 = ["horizontal", "vertical", "distance"]


def main():
    grad = np.load(os.path.join(OUT_DIR, "sspace_smoke", "grad_records.npz"),
                   allow_pickle=True)
    with open(os.path.join(OUT_DIR, "wflu_qwen25", "meta.json")) as f:
        meta = json.load(f)
    hs_q = np.load(os.path.join(OUT_DIR, "wflu_qwen25", "hs_q.npy"))
    hs_r = np.load(os.path.join(OUT_DIR, "wflu_qwen25", "hs_r.npy"))
    layers = [0, 4, 8, 12, 15, 18, 21, 24, 28, 32]
    rng = np.random.default_rng(0)
    out = {}
    for L in [18, 21, 24]:
        li = layers.index(L)
        acc = {g: [] for g in GROUPS3}
        acc_orth = []
        for b in range(200):
            axes = {}
            for g in GROUPS3:
                m = grad["groups"] == g
                idx = np.where(m)[0]
                pick = rng.choice(idx, len(idx), replace=True)
                mu = grad["avecs"][pick, li].mean(0)
                axes[g] = mu / (np.linalg.norm(mu) + 1e-12)
            for g in GROUPS3:
                vals, truth = [], []
                for i, met in enumerate(meta):
                    if met["group"] != g:
                        continue
                    vals.append(float(np.dot(
                        axes[g], hs_q[i, L].astype(np.float64)
                        - hs_r[i, L].astype(np.float64))))
                    truth.append(READOUT_SIGN[met["category"]])
                acc[g].append(float((np.sign(vals) == np.array(truth)).mean()))
            B = np.stack([axes["horizontal"], axes["vertical"]], axis=1)
            vz = axes["distance"]
            vz = vz - B @ (np.linalg.pinv(B) @ vz)
            vz = vz / (np.linalg.norm(vz) + 1e-12)
            vals, truth = [], []
            for i, met in enumerate(meta):
                if met["group"] != "distance":
                    continue
                vals.append(float(np.dot(
                    vz, hs_q[i, L].astype(np.float64)
                    - hs_r[i, L].astype(np.float64))))
                truth.append(READOUT_SIGN[met["category"]])
            acc_orth.append(float((np.sign(vals) == np.array(truth)).mean()))
        out[L] = {g: {"mean": float(np.mean(acc[g])),
                      "ci95": [float(np.percentile(acc[g], 2.5)),
                               float(np.percentile(acc[g], 97.5))]}
                  for g in GROUPS3}
        out[L]["distance_orth"] = {
            "mean": float(np.mean(acc_orth)),
            "ci95": [float(np.percentile(acc_orth, 2.5)),
                     float(np.percentile(acc_orth, 97.5))]}
        print(L, out[L])
    with open(os.path.join(OUT_DIR, "summary", "axis_bootstrap.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
