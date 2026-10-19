"""WFLU-only metrics from the saved arrays (no S-Space axes needed)."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_DIR, axis_coherence, vd_ei_eq3, vd_ei_repo  # noqa: E402

CATS = ["left", "right", "above", "below", "far", "close"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(OUT_DIR, "wflu_qwen25"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "summary"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    delta = np.load(os.path.join(args.dir, "delta.npy"))
    layers = [int(x) for x in np.load(os.path.join(args.dir, "layer_ids.npy"))]
    with open(os.path.join(args.dir, "meta.json")) as f:
        meta = json.load(f)

    rows = []
    for li, L in enumerate(layers):
        d = delta[:, li].astype(np.float64)
        mbc = {c: d[[i for i, m in enumerate(meta) if m["category"] == c]].mean(0)
               for c in CATS}
        per_cat = {c: [d[i] for i, m in enumerate(meta) if m["category"] == c]
                   for c in CATS}
        row = {"layer": L}
        for g in ["horizontal", "vertical", "distance"]:
            row[f"Coh{g[0].upper()}"] = axis_coherence(per_cat, g)[0]
            canon = {"horizontal": "left", "vertical": "above",
                     "distance": "far"}[g]
            opp = {"horizontal": "right", "vertical": "below",
                   "distance": "close"}[g]
            mu = (mbc[canon] - mbc[opp]) / 2
            u = mu / (np.linalg.norm(mu) + 1e-12)
            idx = [i for i, m in enumerate(meta) if m["group"] == g]
            signs = np.array([1 if meta[i]["category"] == canon else -1
                              for i in idx])
            proj = d[idx] @ u * signs
            resid = d[idx] - np.outer(d[idx] @ u, u)
            row[f"dprime_{g[0].upper()}"] = float(
                proj.mean() / (proj.std() + 1e-12))
            row[f"mean_norm_{g[0].upper()}"] = float(np.linalg.norm(mu))
            row[f"resid_rms_{g[0].upper()}"] = float(
                np.sqrt((resid ** 2).sum(1).mean()))
        row["VD_EI_eq3"] = vd_ei_eq3(mbc)
        row["VD_EI_repo"] = vd_ei_repo(mbc)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "wflu_metrics.csv"), index=False)
    cols = ["layer", "CohH", "CohV", "CohD", "VD_EI_eq3", "VD_EI_repo"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
