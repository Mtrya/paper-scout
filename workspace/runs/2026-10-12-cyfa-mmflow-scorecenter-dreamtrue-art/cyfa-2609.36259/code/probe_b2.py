"""Probe B2: does cyclic wraparound measurably hurt? Correlate per-token NLL with the
"live aliased mass" at the newest slot: for each (layer, head) at step t, sum over old writes
i (age > m/2) of beta_i * exp(G_t - G_i) * kappa(lambda_t - lambda_i), then correlate
per-token NLL with the across-head mean/max of this quantity (natural + code inputs).

Outputs: results/probe_b2_summary.json, traces/probe_b2_tokens.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cyfa_cpu
from cyfa_cpu import Recorder
from probe_b import kappa, M

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TR = HERE / "traces"


def aliased_mass_series(delta, beta, g):
    """delta,beta,g: [T] -> aliased mass at newest slot per step t."""
    T = len(delta)
    lam = np.cumsum(delta)
    G = np.cumsum(g)
    strength = beta * np.exp(G[-1] - G)  # placeholder; computed per t below
    out = np.zeros(T, dtype=np.float64)
    # s_i(t) = beta_i * exp(G_t - G_i); age_i(t) = lam_t - lam_i
    # vectorized per t via cumulative quantities
    for t in range(T):
        age = lam[t] - lam[: t + 1]
        s = beta[: t + 1] * np.exp(G[t] - G[: t + 1])
        old = age > M / 2
        out[t] = float((s[old] * kappa(age[old])).sum())
    return out


def main():
    natural = np.load(HERE / "data" / "natural_ids.npy")
    code = np.load(HERE / "data" / "code_ids.npy")
    z = np.load(TR / "probe_a_traces.npz")

    model, _ = cyfa_cpu.load_model(op_impl="chunk")
    torch.set_num_threads(16)

    summary = {}
    token_dump = {}
    for cls, arr in (("natural", natural[:16]), ("code", code[:8])):
        ids = torch.tensor(arr, dtype=torch.long)
        nlls, logits = cyfa_cpu.forward_batch_nll(model, ids, want_logits=True)
        lp = torch.log_softmax(logits, dim=-1)
        pt_nll = -lp[:, :-1].gather(2, ids[:, 1:].unsqueeze(2)).squeeze(2).numpy()  # [B,T-1]
        delta = z[f"{cls}_delta"][:, : len(arr)]
        beta = z[f"{cls}_beta"][:, : len(arr)]
        g = z[f"{cls}_g"][:, : len(arr)]
        L, B, T, H = delta.shape
        # aliased mass summed over layers/heads per token
        am = np.zeros((B, T), dtype=np.float64)
        for l in range(L):
            for h in range(H):
                for b in range(B):
                    am[b] += aliased_mass_series(delta[l, b, :, h], beta[l, b, :, h], g[l, b, :, h])
        am = am / (L * H)
        x = am[:, 1:].ravel()          # align: NLL at t predicts token t+1
        y = pt_nll.ravel()
        # spearman via ranks
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        spear = float(np.corrcoef(rx, ry)[0, 1])
        pear = float(np.corrcoef(np.log1p(x), y)[0, 1])
        # binned means
        qs = np.quantile(x, [0, 0.5, 0.9, 0.99, 1.0])
        bins = np.digitize(x, qs[1:-1])
        binned = [float(y[bins == i].mean()) for i in range(len(qs) - 1)]
        binned_x = [float(x[bins == i].mean()) for i in range(len(qs) - 1)]
        summary[cls] = {
            "spearman_nll_vs_aliased_mass": spear,
            "pearson_nll_vs_log1p_aliased_mass": pear,
            "binned_aliased_mass": binned_x,
            "binned_nll": binned,
            "mean_nll": float(y.mean()),
        }
        token_dump[f"{cls}_nll"] = y
        token_dump[f"{cls}_aliased_mass"] = x
        print(cls, json.dumps(summary[cls], indent=2))

    np.savez_compressed(TR / "probe_b2_tokens.npz", **token_dump)
    (RES / "probe_b2_summary.json").write_text(json.dumps(summary, indent=2))


def analyze_offline():
    """Position-controlled re-analysis from the saved per-token dump (no model needed)."""
    z = np.load(TR / "probe_b2_tokens.npz")
    s = json.loads((RES / "probe_b2_summary.json").read_text())
    for cls in ["natural", "code"]:
        nll = z[f"{cls}_nll"]
        am = z[f"{cls}_aliased_mass"]
        B = 16 if cls == "natural" else 8
        nll = nll.reshape(B, -1)
        am = am.reshape(B, -1)
        T = nll.shape[1]
        dec = np.minimum(np.arange(T) * 10 // T, 9)
        rs = []
        for d in range(10):
            m = dec == d
            x, y = am[:, m].ravel(), nll[:, m].ravel()
            rx = np.argsort(np.argsort(x))
            ry = np.argsort(np.argsort(y))
            rs.append(float(np.corrcoef(rx, ry)[0, 1]))
        s[cls]["position_controlled_spearman_per_decile"] = rs
        s[cls]["position_controlled_spearman_mean"] = float(np.mean(rs))
    (RES / "probe_b2_summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({c: s[c]["position_controlled_spearman_mean"] for c in s}, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze-only":
        analyze_offline()
    else:
        main()
