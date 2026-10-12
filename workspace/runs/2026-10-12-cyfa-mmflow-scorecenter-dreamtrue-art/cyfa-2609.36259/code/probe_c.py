"""Probe C: spectra of the recurrent K/V states of the trained CyFA 400M.

Snapshots the absolute-clock states (hk [d_k=256, m*=128], hv [m*=128, d_v=256]) at token
positions {256, 1024, 2048} on two natural WikiText-2 chunks, plus one contrast run with
the clock increments replaced by per-head means (destroys input-dependent timing; matches
the paper's ablation "w. delta_t^h = sigma(b_delta^h)" style) at t=2048.

Per head: singular values of hk / hv, participation ratio PR = (sum s^2)^2 / sum s^4,
top-8 energy fraction. Layer comparison + clock-vs-mean-clock contrast.

Outputs: results/probe_c_summary.json, results/probe_c_spectra.npz (small)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cyfa_cpu
from cyfa_cpu import Recorder

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)

SNAPSHOT_T = [256, 1024, 2048]
N_CHUNKS = 2


def spectrum_stats(s: np.ndarray):
    s2 = s ** 2
    tot = s2.sum()
    pr = float(tot ** 2 / (s2 ** 2).sum()) if tot > 0 else 0.0
    cum = np.cumsum(s2) / tot if tot > 0 else np.zeros_like(s2)
    top8 = float(cum[7]) if len(cum) >= 8 else float(cum[-1])
    return pr, top8


def svd_states(states, n_layers=24):
    """states: list per layer of (hk [B,H,D,M], hv [B,H,M,D], lam)."""
    L = len(states)
    out = []
    for l in range(L):
        hk, hv, lam = states[l]
        B, H, D, M = hk.shape
        rec = {"layer": l, "lambda_final": lam.tolist()}
        for name, st in (("hk", hk), ("hv", hv)):
            prs, top8s, norms = [], [], []
            spec_sum = None
            for b in range(B):
                for h in range(H):
                    s = np.linalg.svd(st[b, h], compute_uv=False)
                    pr, t8 = spectrum_stats(s)
                    prs.append(pr)
                    top8s.append(t8)
                    norms.append(float(np.linalg.norm(st[b, h])))
                    sn = s / s[0] if s[0] > 0 else s
                    spec_sum = sn if spec_sum is None else spec_sum + sn
            rec[name] = {
                "pr_mean": float(np.mean(prs)), "pr_min": float(np.min(prs)),
                "pr_max": float(np.max(prs)),
                "top8_energy_mean": float(np.mean(top8s)),
                "fro_mean": float(np.mean(norms)),
                "mean_normalized_spectrum": (spec_sum / (B * H)).tolist(),
            }
        out.append(rec)
    return out


def run_with_states(model, ids, delta_fn=None):
    rec = Recorder(n_layers=24, want_state=True, delta_fn=delta_fn)
    nlls = cyfa_cpu.forward_batch_nll(model, ids, recorder=rec)
    return nlls, rec.states


def main():
    natural = np.load(HERE / "data" / "natural_ids.npy")

    model, ckpt = cyfa_cpu.load_model(op_impl="chunk")
    torch.set_num_threads(16)

    out = {"checkpoint": ckpt, "chunks": [0, 1], "snapshot_t": SNAPSHOT_T}
    for T in SNAPSHOT_T:
        ids = torch.tensor(natural[:N_CHUNKS, :T], dtype=torch.long)
        nlls, states = run_with_states(model, ids)
        recs = svd_states(states)
        out[f"t{T}"] = {
            "nll": nlls,
            "layers": [{"layer": r["layer"],
                        "hk_pr_mean": r["hk"]["pr_mean"], "hk_top8": r["hk"]["top8_energy_mean"],
                        "hv_pr_mean": r["hv"]["pr_mean"], "hv_top8": r["hv"]["top8_energy_mean"],
                        "hk_fro": r["hk"]["fro_mean"], "hv_fro": r["hv"]["fro_mean"]}
                       for r in recs],
        }
        if T == 2048:
            np.savez_compressed(RES / "probe_c_spectra.npz",
                                **{f"hk_spec_l{r['layer']}": r["hk"]["mean_normalized_spectrum"] for r in recs},
                                **{f"hv_spec_l{r['layer']}": r["hv"]["mean_normalized_spectrum"] for r in recs})
        print(f"[probe_c] T={T} done; layer23 hk PR={recs[23]['hk']['pr_mean']:.1f} "
              f"hv PR={recs[23]['hv']['pr_mean']:.1f}")

    # contrast: delta -> per-head mean (uses probe A natural stats head means)
    nat_stats = json.loads((RES / "probe_a_cross_class.json").read_text())
    dm = torch.tensor(nat_stats["natural"]["delta_mean_per_head"])  # [L,H]
    def delta_fn(l, d):
        return dm[l].unsqueeze(0).unsqueeze(0).expand_as(d).clone()
    ids = torch.tensor(natural[:N_CHUNKS], dtype=torch.long)
    nlls_m, states_m = run_with_states(model, ids, delta_fn=delta_fn)
    recs_m = svd_states(states_m)
    out["t2048_mean_clock"] = {
        "nll": nlls_m,
        "layers": [{"layer": r["layer"],
                    "hk_pr_mean": r["hk"]["pr_mean"], "hk_top8": r["hk"]["top8_energy_mean"],
                    "hv_pr_mean": r["hv"]["pr_mean"], "hv_top8": r["hv"]["top8_energy_mean"]}
                   for r in recs_m],
    }
    np.savez_compressed(RES / "probe_c_spectra_meanclock.npz",
                        **{f"hk_spec_l{r['layer']}": r["hk"]["mean_normalized_spectrum"] for r in recs_m},
                        **{f"hv_spec_l{r['layer']}": r["hv"]["mean_normalized_spectrum"] for r in recs_m})

    (RES / "probe_c_summary.json").write_text(json.dumps(out, indent=2))
    print("saved results/probe_c_summary.json")


if __name__ == "__main__":
    main()
