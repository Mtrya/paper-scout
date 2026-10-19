"""Combine WFLU deltas and S-Space axes on the same Qwen2.5-VL-3B sample set.

Questions answered at every probed layer:
  1. WFLU metrics: Coh_H, Coh_V, Coh_D, VD-EI (paper Eq. 3 and repo variant).
  2. S-Space metrics: per-axis readout accuracy from object-token coordinates.
  3. Do the two coexist?  Project WFLU deltas onto the S-Space basis and ask
     whether the distance component predicts the true relation, and whether
     the low full-vector coherence survives after projection.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    GROUPS, OUT_DIR, axis_coherence, cos, vd_ei_eq3, vd_ei_repo,
)

AXIS_ORDER = ["horizontal", "vertical", "distance"]
# expected sign of the S-Space margin v_g^T (h_q - h_r)
READOUT_SIGN = {"left": -1, "right": 1, "above": 1, "below": -1,
                "far": -1, "close": 1}
# expected sign of v_g^T delta (delta points from original answer to swapped answer)
DELTA_SIGN = {"left": 1, "right": -1, "above": -1, "below": 1,
              "far": 1, "close": -1}


def load_npz(path):
    return np.load(path, allow_pickle=True)


def mean_by_cat(delta, meta, cats):
    out = {}
    for c in cats:
        idx = [i for i, m in enumerate(meta) if m["category"] == c]
        if idx:
            out[c] = delta[idx].astype(np.float64).mean(axis=0)
    return out


def project_basis(delta, axes):
    """Least-squares projection of each delta onto span(axes)."""
    V = axes.T  # [d, 3]
    P = V @ np.linalg.pinv(V)
    return delta @ P.T


def orth_against(v, basis):
    """Remove the span of basis (list of vectors) from v and renormalize."""
    B = np.stack(basis, axis=1)  # [d, k]
    coef = np.linalg.pinv(B) @ v
    vp = v - B @ coef
    return vp / (np.linalg.norm(vp) + 1e-12)


def readout_acc_for_axis(axis, g, hs_q, hs_r, meta):
    vals, truth = [], []
    for i, m in enumerate(meta):
        if m["group"] != g:
            continue
        vals.append(float(np.dot(axis, hs_q[i].astype(np.float64)
                                 - hs_r[i].astype(np.float64))))
        truth.append(READOUT_SIGN[m["category"]])
    vals = np.array(vals)
    truth = np.array(truth)
    return float((np.sign(vals) == truth).mean())


def delta_sign_acc_for_axis(axis, g, delta, meta):
    vals, truth = [], []
    for i, m in enumerate(meta):
        if m["group"] != g:
            continue
        vals.append(float(np.dot(axis, delta[i].astype(np.float64))))
        truth.append(DELTA_SIGN[m["category"]])
    vals = np.array(vals)
    truth = np.array(truth)
    return float((np.sign(vals) == truth).mean())


def metrics_at_layer(L, delta, hs_q, hs_r, meta, axes, axes_layers):
    """axes: [3, d] for this layer (or None)."""
    d = delta.shape[-1]
    cats = ["left", "right", "above", "below", "far", "close"]
    mbc = mean_by_cat(delta, meta, cats)

    out = {"layer": int(L)}
    for g in AXIS_ORDER:
        coh, n = axis_coherence(
            {c: [delta[i] for i, m in enumerate(meta) if m["category"] == c]
             for c in cats}, g)
        out[f"coh_{g[0].upper()}"] = coh
        out[f"n_{g[0].upper()}"] = n
        # centered variant: remove the global mean delta shared by all samples
        dc = delta - delta.mean(axis=0, keepdims=True)
        coh_c, _ = axis_coherence(
            {c: [dc[i] for i, m in enumerate(meta) if m["category"] == c]
             for c in cats}, g)
        out[f"coh_centered_{g[0].upper()}"] = coh_c
    out["vd_ei_eq3"] = vd_ei_eq3(mbc)
    out["vd_ei_repo"] = vd_ei_repo(mbc)

    # cosine of each category mean delta with each S-Space axis
    if axes is not None:
        for ci, c in enumerate(cats):
            for gi, g in enumerate(AXIS_ORDER):
                out[f"cos_delta_{c}_{g[0]}"] = cos(mbc[c], axes[gi])

        # --- S-Space readout from the original prompt's object tokens ---
        for gi, g in enumerate(AXIS_ORDER):
            vals, truth = [], []
            for i, m in enumerate(meta):
                if m["group"] != g:
                    continue
                m_g = float(np.dot(axes[gi], hs_q[i].astype(np.float64)
                                   - hs_r[i].astype(np.float64)))
                vals.append(m_g)
                truth.append(READOUT_SIGN[m["category"]])
            vals = np.array(vals)
            truth = np.array(truth)
            out[f"readout_acc_{g[0]}"] = float((np.sign(vals) == truth).mean())
            out[f"readout_n_{g[0]}"] = int(len(vals))
        accs = [out[f"readout_acc_{g[0]}"] for g in AXIS_ORDER]
        out["readout_acc_overall"] = float(np.mean(accs))

        # --- is the distance axis just a vertical proxy? ---
        vx, vy, vz = axes[0], axes[1], axes[2]
        vz_perp = orth_against(vz, [vx, vy])
        vy_perp = orth_against(vy, [vx, vz])
        out["cos_yz"] = float(np.dot(vy, vz))
        out["cos_xy"] = float(np.dot(vx, vy))
        out["cos_xz"] = float(np.dot(vx, vz))
        out["readout_acc_D_orth"] = readout_acc_for_axis(
            vz_perp, "distance", hs_q, hs_r, meta)
        out["readout_acc_V_orth"] = readout_acc_for_axis(
            vy_perp, "vertical", hs_q, hs_r, meta)
        out["delta_sign_acc_D_orth"] = delta_sign_acc_for_axis(
            vz_perp, "distance", delta, meta)
        out["delta_sign_acc_V_orth"] = delta_sign_acc_for_axis(
            vy_perp, "vertical", delta, meta)

        # --- WFLU delta projected onto S-Space ---
        Pdelta = project_basis(delta.astype(np.float64), axes)
        out["delta_var_in_sspace"] = float(
            (np.sum(Pdelta ** 2, axis=1) / np.sum(delta ** 2, axis=1)).mean())
        V = axes.T
        P = V @ np.linalg.pinv(V)
        for cat in cats:
            mu = mbc[cat]
            pmu = P @ mu
            out[f"mean_proj_frac_{cat}"] = float(
                np.linalg.norm(pmu) / (np.linalg.norm(mu) + 1e-12))
            out[f"mean_cos_sspace_{cat}"] = cos(mu, pmu) if np.linalg.norm(pmu) > 0 else 0.0

        for gi, g in enumerate(AXIS_ORDER):
            a_vals, dc_vals, truth = [], [], []
            for i, m in enumerate(meta):
                if m["group"] != g:
                    continue
                dvec = delta[i].astype(np.float64)
                dc = hs_q[i].astype(np.float64) - hs_r[i].astype(np.float64)
                a_vals.append(float(np.dot(axes[gi], dvec)))
                dc_vals.append(float(np.dot(axes[gi], dc)))
                truth.append(DELTA_SIGN[m["category"]])
            a_vals = np.array(a_vals)
            dc_vals = np.array(dc_vals)
            truth = np.array(truth)
            out[f"delta_sign_acc_{g[0]}"] = float(
                (np.sign(a_vals) == truth).mean())
            out[f"delta_z_{g[0]}"] = float(
                a_vals.mean() / (a_vals.std() + 1e-12))
            # delta points opposite to the coordinate difference
            r = pearsonr(a_vals, -dc_vals).statistic if len(a_vals) > 2 else np.nan
            out[f"delta_vs_coord_r_{g[0]}"] = float(r)
            out[f"delta_vs_coord_r2_{g[0]}"] = float(r ** 2)

            # coherence after projecting deltas onto this single axis
            signs = np.array([DELTA_SIGN[m["category"]] for m in meta
                              if m["group"] == g])
            # sign-correct to the canonical answer-flip direction
            out[f"delta_scalar_coh_{g[0]}"] = float(
                np.mean(a_vals * signs) / (np.std(a_vals) + 1e-12))

        # full-vector vs S-Space-projected coherence
        for g in AXIS_ORDER:
            idx = [i for i, m in enumerate(meta) if m["group"] == g]
            signs = np.array([1 if meta[i]["category"] in
                              ("left", "above", "far") else -1 for i in idx])
            X = Pdelta[idx] * signs[:, None]
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            sim = Xn @ Xn.T
            iu = np.triu_indices(len(idx), k=1)
            out[f"coh_proj_{g[0]}"] = float(sim[iu].mean())

    return out


def auc_from_scores(scores):
    """Kept for compatibility; not used by the current metric set."""
    return float((scores > 0).mean())


def boot_cos_ci(deltas, axis, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    deltas = np.asarray(deltas, dtype=np.float64)
    for _ in range(n_boot):
        idx = rng.integers(0, len(deltas), len(deltas))
        vals.append(cos(deltas[idx].mean(0), axis))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def bootstrap_geometry(delta, meta, axes, L):
    """Bootstrap CIs for cos(mean Δ_category, S-Space axis)."""
    out = {}
    for cat in ["far", "close", "above", "below"]:
        vecs = np.array([delta[i].astype(np.float64) for i, m in enumerate(meta)
                         if m["category"] == cat])
        for gi, g in enumerate(AXIS_ORDER):
            c = cos(vecs.mean(0), axes[gi])
            ci = boot_cos_ci(vecs, axes[gi], seed=sum(ord(ch) for ch in cat + g) % 10000)
            out[f"cos_{cat}_{g[0]}"] = {"value": float(c), "ci95": ci}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wflu-dir", default=os.path.join(OUT_DIR, "wflu_qwen25"))
    ap.add_argument("--sspace-dir", default=os.path.join(OUT_DIR, "sspace_qwen25"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "summary"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    delta = load_npz(os.path.join(args.wflu_dir, "delta.npy"))
    hs_q = load_npz(os.path.join(args.wflu_dir, "hs_q.npy"))
    hs_r = load_npz(os.path.join(args.wflu_dir, "hs_r.npy"))
    wflu_layers = load_npz(os.path.join(args.wflu_dir, "layer_ids.npy"))
    with open(os.path.join(args.wflu_dir, "meta.json")) as f:
        meta = json.load(f)

    axes = load_npz(os.path.join(args.sspace_dir, "axes.npy"))
    ss_layers = [int(x) for x in load_npz(os.path.join(args.sspace_dir, "layer_ids.npy"))]
    axes_by_layer = {L: axes[i] for i, L in enumerate(ss_layers)}

    rows = []
    for li, L in enumerate(wflu_layers):
        L = int(L)
        rows.append(metrics_at_layer(
            L, delta[:, li], hs_q[:, li], hs_r[:, li], meta,
            axes_by_layer.get(L), ss_layers))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "metrics_by_layer.csv"), index=False)

    # pick the S-Space layer by overall readout accuracy (paper criterion)
    ss_df = df[df["readout_acc_overall"].notna()].copy()
    best_L = int(ss_df.loc[ss_df["readout_acc_overall"].idxmax(), "layer"])
    wflu_L = 28
    summary = {
        "n_samples": len(meta),
        "n_by_category": {c: sum(m["category"] == c for m in meta)
                          for c in ["left", "right", "above", "below", "far", "close"]},
        "sspace_layers_probed": ss_layers,
        "best_sspace_layer": best_L,
        "wflu_layer": wflu_L,
        "at_best": df[df.layer == best_L].to_dict(orient="records")[0],
        "at_wflu_layer": df[df.layer == wflu_L].to_dict(orient="records")[0]
        if wflu_L in set(df.layer) else None,
        "bootstrap_geometry_best": bootstrap_geometry(
            delta[:, best_L], meta, axes_by_layer[best_L], best_L),
        "bootstrap_geometry_L28": bootstrap_geometry(
            delta[:, 28], meta, axes_by_layer[28], 28)
        if 28 in axes_by_layer else None,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
