"""Probe B: exact-read oracle vs native delta feedback (STEPQuant App. B.1).

Matched single-injection experiment. Both paths follow the cached FP32 input stream.
At step INJ the state is quantized once with per-row INT6 (identical for both paths);
no further quantization afterwards. Readout uses the model's scaled, l2-normalized q.

  native:  S_t = (I - beta k k^T) D S_{t-1} + beta k v^T        (delta feedback on own state)
  oracle:  S_t = D S_{t-1} + beta k (v^T - k^T D S_ref_{t-1})   (exact read inside residual;
           error then propagates only through the gate: E_t = D E_{t-1} + 0)

Metric (paper): sum of squared readout error ||(S_t - S_ref_t)^T q_t||^2 over steps
t > INJ, all heads and segments; report oracle/native ratio (paper: 26.82x INT6, 18.65x INT8).
"""
import json
import os

import numpy as np
import torch

torch.set_num_threads(6)
from gdn_model import H, DK, DV, LAYERS, QSCALE, quant_rowwise

HERE = os.path.dirname(os.path.abspath(__file__))
N_SEG = 3
INJ = 256

caps = [dict(np.load(os.path.join(HERE, f"caps/seg{i}.npz"))) for i in range(N_SEG)]
T = caps[0]["l0_k"].shape[0]

results = {}
for bits in [6, 8]:
    sum_native = np.zeros((LAYERS, H))
    sum_oracle = np.zeros((LAYERS, H))
    # cumulative curves, averaged over units, for the figure
    curve_native = np.zeros(T)
    curve_oracle = np.zeros(T)
    n_units = 0
    for si, c in enumerate(caps):
        for l in range(LAYERS):
            k = torch.from_numpy(c[f"l{l}_k"])
            v = torch.from_numpy(c[f"l{l}_v"])
            beta = torch.from_numpy(c[f"l{l}_beta"])
            g = torch.from_numpy(c[f"l{l}_g"])
            q = torch.from_numpy(c[f"l{l}_q"]) * QSCALE
            S_ref = torch.zeros(H, DK, DV)
            S_nat = None
            S_orc = None
            for t in range(T):
                gt = g[t].exp().view(H, 1, 1)
                kt, vt, bt = k[t], v[t], beta[t].view(H, 1, 1)
                # advance quantized paths first: they may read the PREVIOUS reference state
                if t > INJ:
                    # native path: full delta update on own state
                    S_nat = S_nat * gt
                    dvn = vt - (S_nat * kt.unsqueeze(-1)).sum(1)
                    S_nat = S_nat + bt * kt.unsqueeze(-1) * dvn.unsqueeze(-2)
                    # oracle path: exact reference read inside the delta residual
                    # (S_ref here is still S_ref_{t-1})
                    dvo = vt - (S_ref * gt * kt.unsqueeze(-1)).sum(1)  # k^T D_t S_ref_{t-1}
                    S_orc = S_orc * gt + bt * kt.unsqueeze(-1) * dvo.unsqueeze(-2)
                # reference update
                S_ref = S_ref * gt
                dv = vt - (S_ref * kt.unsqueeze(-1)).sum(1)
                S_ref = S_ref + bt * kt.unsqueeze(-1) * dv.unsqueeze(-2)
                if t == INJ:
                    S_inj = quant_rowwise(S_ref, bits)
                    S_nat = S_inj.clone()
                    S_orc = S_inj.clone()
                    continue  # readout at INJ is pre-quantization: identical, no error
                if t > INJ:
                    e_nat = ((S_nat - S_ref) * q[t].unsqueeze(-1)).sum(1)  # [H, V] readout error
                    e_orc = ((S_orc - S_ref) * q[t].unsqueeze(-1)).sum(1)
                    sn = e_nat.pow(2).sum(-1).numpy()
                    so = e_orc.pow(2).sum(-1).numpy()
                    sum_native[l] += sn
                    sum_oracle[l] += so
                    curve_native[t] += sn.sum()
                    curve_oracle[t] += so.sum()
        print(f"bits={bits} seg{si} done", flush=True)
    n_units = LAYERS * H * N_SEG
    curve_native /= n_units
    curve_oracle /= n_units
    ratio = sum_oracle.sum() / sum_native.sum()
    per_layer_ratio = sum_oracle.sum(1) / np.maximum(sum_native.sum(1), 1e-300)
    results[bits] = {
        "ratio_total": float(ratio),
        "sum_native": float(sum_native.sum()), "sum_oracle": float(sum_oracle.sum()),
        "ratio_per_layer_min": float(per_layer_ratio.min()), "ratio_per_layer_max": float(per_layer_ratio.max()),
    }
    np.save(os.path.join(HERE, f"probe_b_perhead_int{bits}.npy"), np.stack([sum_native, sum_oracle]))
    np.save(os.path.join(HERE, f"probe_b_curves_int{bits}.npy"), np.stack([curve_native, curve_oracle]))
    print(f"INT{bits}: oracle/native cumulative squared readout error ratio = {ratio:.2f}x")

results["inject_step"] = INJ
results["n_units"] = int(LAYERS * H)
results["n_segments"] = N_SEG
results["T"] = int(T)
results["paper_ratio_int6"] = 26.82
results["paper_ratio_int8"] = 18.65
with open(os.path.join(HERE, "probe_b_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
