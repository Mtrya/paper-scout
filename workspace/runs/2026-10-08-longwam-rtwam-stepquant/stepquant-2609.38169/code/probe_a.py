"""Probe A: gate half-life vs accumulated INT6 state error (STEPQuant Fig.1b / App. B.5).

For every (layer, head): replay the cached FP32 input stream through
  (i)  the exact FP32 recurrence  S_t = (I - beta k k^T) D S_{t-1} + beta k v^T
  (ii) the same recurrence with per-key-row symmetric INT6 quantization of S
       after every update (scale = absmax/31 clamped to [2^-14, 65504], codes in [-31,31])
identical inputs in both paths (teacher-forced replay, as in the paper).
Unit u = one head (GDN scalar gate per head). D_u = sum_t ||E_t||_F^2 accumulated over
both segments. Half-life tau = ln2 / (-mean(g)). Reports Spearman(tau, D_u), the
longest-quartile share of total error, and per-step relative error curves."""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

torch.set_num_threads(6)
from gdn_model import H, DK, DV, LAYERS, quant_rowwise

HERE = os.path.dirname(os.path.abspath(__file__))
N_SEG = 3

caps = [dict(np.load(os.path.join(HERE, f"caps/seg{i}.npz"))) for i in range(N_SEG)]

# half-life from log-gate g (mean over both segments)
ell = np.zeros((LAYERS, H))
for c in caps:
    for l in range(LAYERS):
        ell[l] += c[f"l{l}_g"].mean(0)  # g is already the log gate (<=0)
ell /= N_SEG
tau = np.log(2.0) / (-ell)

D_u = np.zeros((LAYERS, H))          # cumulative squared Frobenius error (both segments)
final_rel = np.zeros((LAYERS, H))    # relative error at the last step, averaged over segments
T = caps[0]["l0_k"].shape[0]
err_curve = np.zeros((LAYERS, H, T)) # per-step relative error, averaged over segments

for si, c in enumerate(caps):
    for l in range(LAYERS):
        k = torch.from_numpy(c[f"l{l}_k"])      # [T,H,K] post-conv, l2-normalized
        v = torch.from_numpy(c[f"l{l}_v"])
        beta = torch.from_numpy(c[f"l{l}_beta"])  # [T,H]
        g = torch.from_numpy(c[f"l{l}_g"])
        S_ref = torch.zeros(H, DK, DV)
        S_q = torch.zeros(H, DK, DV)
        for t in range(T):
            gt = g[t].exp().view(H, 1, 1)
            kt, vt, bt = k[t], v[t], beta[t].view(H, 1, 1)
            # reference update
            S_ref = S_ref * gt
            dv = vt - (S_ref * kt.unsqueeze(-1)).sum(1)
            S_ref = S_ref + bt * kt.unsqueeze(-1) * dv.unsqueeze(-2)
            # quantized update (same inputs)
            S_q = S_q * gt
            dvq = vt - (S_q * kt.unsqueeze(-1)).sum(1)
            S_q = S_q + bt * kt.unsqueeze(-1) * dvq.unsqueeze(-2)
            S_q = quant_rowwise(S_q, 6)
            E = S_q - S_ref
            D_u[l] += E.pow(2).sum(dim=(1, 2)).numpy()
            err_curve[l, :, t] += (E.norm(dim=(1, 2)) / S_ref.norm(dim=(1, 2)).clamp_min(1e-12)).numpy()
        final_rel[l] += (E.norm(dim=(1, 2)) / S_ref.norm(dim=(1, 2))).numpy()
    print(f"seg{si} done", flush=True)
final_rel /= N_SEG
err_curve /= N_SEG

flat_tau = tau.ravel()
flat_D = D_u.ravel()
rho_D, p_D = spearmanr(flat_tau, flat_D)
rho_F, p_F = spearmanr(flat_tau, final_rel.ravel())
order = np.argsort(flat_tau)
q = len(order) // 4
longest_share = flat_D[order[-q:]].sum() / flat_D.sum()
shortest_share = flat_D[order[:q]].sum() / flat_D.sum()

summary = {
    "n_units": int(flat_tau.size), "n_segments": N_SEG, "T": T,
    "tau_min": float(flat_tau.min()), "tau_median": float(np.median(flat_tau)), "tau_max": float(flat_tau.max()),
    "spearman_tau_vs_cumerr": float(rho_D), "p_value": float(p_D),
    "spearman_tau_vs_final_relerr": float(rho_F), "p_value_final": float(p_F),
    "longest_quarter_error_share": float(longest_share),
    "shortest_quarter_error_share": float(shortest_share),
    "paper_rho": 0.80, "paper_longest_quarter_share_qwen": 0.525,
}
with open(os.path.join(HERE, "probe_a_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# per-head table
import csv
with open(os.path.join(HERE, "probe_a_perhead.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "head", "half_life", "cum_sq_err", "final_rel_err"])
    for l in range(LAYERS):
        for h in range(H):
            w.writerow([l, h, f"{tau[l,h]:.4f}", f"{D_u[l,h]:.6e}", f"{final_rel[l,h]:.6e}"])
np.save(os.path.join(HERE, "probe_a_errcurve.npy"), err_curve)
np.save(os.path.join(HERE, "probe_a_tau.npy"), tau)
print(json.dumps(summary, indent=2))
