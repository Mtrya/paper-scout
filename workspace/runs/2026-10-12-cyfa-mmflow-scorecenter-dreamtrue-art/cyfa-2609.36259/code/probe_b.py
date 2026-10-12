"""Probe B: wraparound safety of the cyclic memory.

Hypothesis under test (not proven by the paper): the scalar forget gate must erase a write
before the cyclic transport carries it back onto the age-0 slot (model-age difference = m = 127).
From probe A traces we compute, per (layer, head, sequence):
- lambda_T: total accumulated clock over 2048 tokens (cycles = lambda_T / 127)
- for each write i: residual strength when its relative-time profile first re-aligns with
  age-0 (lambda_t - lambda_i >= 127): r_i = beta_i * exp(G_{t_i} - G_i), G = cumsum(g)
- tokens-per-cycle and per-token half-life from the gate series
- aliased mass ratio at the final step: old writes' (age > m/2) remaining strength weighted by
  the transport kernel |kappa(Delta lambda)|, kappa(x) = sin(pi x) / (m sin(pi x / m))

Outputs: results/probe_b_summary.json, results/probe_b_perhead.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TR = HERE / "traces"
M = 127  # active slots


def kappa(x: np.ndarray) -> np.ndarray:
    """Cardinal transport kernel |<P(x) e0, e0>| (paper Eq. 39)."""
    x = np.asarray(x, dtype=np.float64)
    den = M * np.sin(np.pi * x / M)
    num = np.sin(np.pi * x)
    out = np.ones_like(x)
    nz = np.abs(den) > 1e-12
    out[nz] = np.abs(num[nz] / den[nz])
    return out


def analyze_seq(delta: np.ndarray, beta: np.ndarray, g: np.ndarray):
    """delta,beta,g: [T] for one head. Returns per-write residuals at wrap and aggregates."""
    T = len(delta)
    lam = np.cumsum(delta)
    G = np.cumsum(g)
    # wrap time for each i: first t with lam[t] - lam[i] >= M
    # lam strictly increasing (delta > 0); searchsorted on (lam[i] + M)
    targets = lam + M
    idx = np.searchsorted(lam, targets, side="left")
    has_wrap = idx < T
    ii = np.nonzero(has_wrap)[0]
    twrap = idx[ii]
    resid = beta[ii] * np.exp(G[twrap] - G[ii])
    spans = twrap - ii  # tokens to wrap
    # aliased mass at final step
    age = lam[-1] - lam
    strength = beta * np.exp(G[-1] - G)
    old = age > M / 2
    aliased = float((strength[old] * kappa(age[old])).sum())
    fresh = float(beta[-1])
    return {
        "lambda_T": float(lam[-1]),
        "resid_at_wrap": resid,
        "span_to_wrap": spans,
        "aliased_mass": aliased,
        "fresh_beta": fresh,
        "mean_g": float(g.mean()),
    }


def main():
    z = np.load(TR / "probe_a_traces.npz")
    rows = ["class,layer,head,lambda_T_mean,cycles_mean,tokens_per_cycle,half_life_tokens,"
            "decay_per_cycle,beta_median,resid_wrap_median,resid_wrap_p99,resid_wrap_max,"
            "frac_writes_resid_gt_1e-2,aliased_mass_ratio"]
    summary = {}
    for cls in ["natural", "code", "repetitive"]:
        delta = z[f"{cls}_delta"]  # [L,B,T,H]
        beta = z[f"{cls}_beta"]
        g = z[f"{cls}_g"]
        L, B, T, H = delta.shape
        for l in range(L):
            for h in range(H):
                resids, spans = [], []
                lamT, aliased_ratio, half_lives = [], [], []
                for b in range(B):
                    r = analyze_seq(delta[l, b, :, h], beta[l, b, :, h], g[l, b, :, h])
                    resids.append(r["resid_at_wrap"])
                    spans.append(r["span_to_wrap"])
                    lamT.append(r["lambda_T"])
                    ratio = r["aliased_mass"] / max(r["fresh_beta"], 1e-8)
                    aliased_ratio.append(ratio)
                    half_lives.append(np.log(0.5) / r["mean_g"] if r["mean_g"] < 0 else np.inf)
                resids = np.concatenate(resids)
                spans = np.concatenate(spans)
                lamT = np.mean(lamT)
                cycles = lamT / M
                tpc = float(np.mean(spans)) if len(spans) else np.inf
                hl = float(np.mean(half_lives))
                mean_g = float(np.mean([g[l, b, :, h].mean() for b in range(B)]))
                dpc = float(np.exp(mean_g * tpc))  # decay accumulated over one full cycle
                row = [cls, l, h, f"{lamT:.2f}", f"{cycles:.2f}", f"{tpc:.1f}", f"{hl:.1f}",
                       f"{dpc:.3e}",
                       f"{np.median(beta[l, :, :, h]):.4f}",
                       f"{np.median(resids):.3e}" if len(resids) else "nan",
                       f"{np.quantile(resids, 0.99):.3e}" if len(resids) else "nan",
                       f"{resids.max():.3e}" if len(resids) else "nan",
                       f"{(resids > 1e-2).mean():.4f}" if len(resids) else "nan",
                       f"{np.mean(aliased_ratio):.3e}"]
                rows.append(",".join(map(str, row)))
    (RES / "probe_b_perhead.csv").write_text("\n".join(rows))

    # summary per class
    import csv
    from collections import defaultdict
    data = list(csv.DictReader(rows))
    for cls in ["natural", "code", "repetitive"]:
        sub = [r for r in data if r["class"] == cls]
        p99 = np.array([float(r["resid_wrap_p99"]) for r in sub if r["resid_wrap_p99"] != "nan"])
        mx = np.array([float(r["resid_wrap_max"]) for r in sub if r["resid_wrap_max"] != "nan"])
        ali = np.array([float(r["aliased_mass_ratio"]) for r in sub])
        bm = np.array([float(r["beta_median"]) for r in sub if r["resid_wrap_p99"] != "nan"])
        rel = p99 / np.maximum(bm, 1e-8)  # residual at wrap relative to a fresh write's strength
        summary[cls] = {
            "n_heads": len(sub),
            "resid_wrap_p99_median_over_heads": float(np.median(p99)),
            "resid_wrap_p99_max_over_heads": float(p99.max()),
            "resid_wrap_max_over_heads": float(mx.max()),
            "n_heads_p99_gt_1e-2": int((p99 > 1e-2).sum()),
            "n_heads_p99_gt_1e-1": int((p99 > 1e-1).sum()),
            "resid_p99_rel_beta_median": float(np.median(rel)),
            "resid_p99_rel_beta_max_head": float(rel.max()),
            "aliased_mass_ratio_median": float(np.median(ali)),
            "aliased_mass_ratio_max": float(ali.max()),
        }
    (RES / "probe_b_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
