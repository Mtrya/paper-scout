"""Probe A: clock behavior of the trained CyFA 400M (paper §4.5 / D.2).

Part 1 (stats): record per-token delta/beta/g over three input classes
(natural=WikiText-2 test chunks 0..15, code, repetitive), classify all 96 heads with the
paper's criteria (near-one: p05(delta)>0.95; input-dependent: p95-p05>=0.1), and decompose
delta variance into token-identity vs context for input-dependent heads.

Part 2 (interventions, paper Table 8 replication): held-out natural chunks 16..23,
baseline vs per-head-mean / shift-257 / delta==1 clocks applied to input-dependent heads.

Outputs: results/probe_a_*.json, results/probe_a_head_stats.csv, traces/probe_a_traces.npz
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import cyfa_cpu
from cyfa_cpu import Recorder

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TR = HERE / "traces"
RES.mkdir(exist_ok=True)
TR.mkdir(exist_ok=True)

N_STAT_NATURAL = 16
N_CODE = 8
N_REP = 12          # 4 per variant
N_HELDOUT = 8       # natural chunks 16..23


def collect(model, ids: torch.Tensor):
    rec = Recorder(n_layers=24)
    t0 = time.time()
    nlls = cyfa_cpu.forward_batch_nll(model, ids, recorder=rec)
    dt = time.time() - t0
    L = 24
    delta = np.stack([t["delta"] for t in rec.traces])  # [L, B, T, H]
    beta = np.stack([t["beta"] for t in rec.traces])
    g = np.stack([t["g"] for t in rec.traces])
    return nlls, delta, beta, g, dt


def classify_heads(delta: np.ndarray):
    """delta [L,B,T,H] -> per (L,H) stats dict using the paper's quantile criteria."""
    L, B, T, H = delta.shape
    flat = delta.transpose(0, 3, 1, 2).reshape(L, H, -1)
    p05 = np.quantile(flat, 0.05, axis=-1)
    p95 = np.quantile(flat, 0.95, axis=-1)
    mean = flat.mean(axis=-1)
    std = flat.std(axis=-1)
    near_one = p05 > 0.95
    input_dep = (p95 - p05) >= 0.1
    return {"p05": p05, "p95": p95, "mean": mean, "std": std,
            "near_one": near_one, "input_dep": input_dep}


def main():
    natural = np.load(HERE / "data" / "natural_ids.npy")
    code = np.load(HERE / "data" / "code_ids.npy")
    rep = np.load(HERE / "data" / "repetitive_ids.npy")

    model, ckpt = cyfa_cpu.load_model(op_impl="chunk")
    torch.set_num_threads(16)

    out = {"checkpoint": ckpt, "n_stat_natural": N_STAT_NATURAL, "n_code": N_CODE, "n_rep": N_REP}

    # ---- stats collection
    sets = {
        "natural": natural[:N_STAT_NATURAL],
        "code": code[:N_CODE],
        "repetitive": rep[:N_REP],
    }
    all_traces = {}
    cache = TR / "probe_a_traces.npz"
    meta_cache = TR / "probe_a_collect_meta.json"
    if cache.exists():
        z = np.load(cache)
        all_traces = {name: (z[f"{name}_delta"], z[f"{name}_beta"], z[f"{name}_g"])
                      for name in sets}
        if meta_cache.exists():
            out.update(json.loads(meta_cache.read_text()))
        print("[collect] loaded cached traces")
    else:
        for name, arr in sets.items():
            ids = torch.tensor(arr, dtype=torch.long)
            nlls, delta, beta, g, dt = collect(model, ids)
            print(f"[collect {name}] B={len(arr)} time={dt:.1f}s meanNLL={np.mean(nlls):.4f}")
            all_traces[name] = (delta, beta, g)
            out[f"{name}_mean_nll"] = float(np.mean(nlls))
            out[f"{name}_time_s"] = dt

        np.savez_compressed(cache,
                            **{f"{k}_{n}": v for k, (d, b, g) in all_traces.items()
                               for n, v in (("delta", d), ("beta", b), ("g", g))})
        meta_cache.write_text(json.dumps(
            {k: v for k, v in out.items() if k.endswith(("_mean_nll", "_time_s"))}, indent=2))

    # ---- head classification on natural (paper comparison)
    stats = classify_heads(all_traces["natural"][0])
    L, H = stats["mean"].shape
    n_one = int(stats["near_one"].sum())
    n_dep = int(stats["input_dep"].sum())
    one_mask = stats["near_one"]
    dep_mask = stats["input_dep"]
    out["head_class"] = {
        "n_near_one": n_one, "n_input_dep": n_dep, "n_other": L * H - n_one - n_dep,
        "near_one_mean": float(stats["mean"][one_mask].mean()) if n_one else None,
        "near_one_std": float(stats["std"][one_mask].mean()) if n_one else None,
        "input_dep_mean": float(stats["mean"][dep_mask].mean()) if n_dep else None,
        "input_dep_std": float(stats["std"][dep_mask].mean()) if n_dep else None,
        "paper": {"n_near_one": 56, "n_input_dep": 40,
                  "near_one_mean": 0.9961, "near_one_std": 0.0065,
                  "input_dep_mean": 0.5118, "input_dep_std": 0.1627},
    }
    rows = ["layer,head,mean,std,p05,p95,near_one,input_dep"]
    for l in range(L):
        for h in range(H):
            rows.append(f"{l},{h},{stats['mean'][l,h]:.6f},{stats['std'][l,h]:.6f},"
                        f"{stats['p05'][l,h]:.6f},{stats['p95'][l,h]:.6f},"
                        f"{int(one_mask[l,h])},{int(dep_mask[l,h])}")
    (RES / "probe_a_head_stats.csv").write_text("\n".join(rows))

    # ---- variance decomposition: token identity vs context (exclude layer 0, paper D.2)
    nat_ids = natural[:N_STAT_NATURAL]                      # [B,T]
    tok_flat = nat_ids.ravel()
    vardec = {}
    delta_nat = all_traces["natural"][0]                    # [L,B,T,H]
    for l in range(1, L):
        for h in range(H):
            if not dep_mask[l, h]:
                continue
            d = delta_nat[l][:, :, h].ravel()
            total_var = d.var()
            if total_var < 1e-12:
                continue
            # within-token-type variance
            order = np.argsort(tok_flat, kind="stable")
            tok_sorted = tok_flat[order]
            d_sorted = d[order]
            _, starts = np.unique(tok_sorted, return_index=True)
            # vectorized: group variances
            sums = np.add.reduceat(d_sorted, starts)
            sums2 = np.add.reduceat(d_sorted * d_sorted, starts)
            counts = np.diff(np.append(starts, len(d_sorted)))
            gvar = sums2 / counts - (sums / counts) ** 2
            within = float((gvar * counts).sum() / counts.sum())
            vardec[f"{l},{h}"] = 1.0 - within / float(total_var)
    if vardec:
        out["token_identity_var_fraction"] = {
            "mean_over_input_dep_heads_excl_layer0": float(np.mean(list(vardec.values()))),
            "n_heads": len(vardec),
            "paper": 0.494,
        }
    out["input_dep_heads"] = sorted([list(map(int, k.split(","))) for k in vardec]) \
        if vardec else [[int(l), int(h)] for l in range(L) for h in range(H) if dep_mask[l, h]]

    # ---- cross-class comparison: per-head mean delta / alpha
    cross = {}
    for name, (d, b, g) in all_traces.items():
        alpha = np.exp(g)
        cross[name] = {
            "delta_mean_per_head": d.mean(axis=(1, 2)).tolist(),   # [L,H] averaged over B,T
            "alpha_mean_per_head": alpha.mean(axis=(1, 2)).tolist(),
            "beta_mean_per_head": b.mean(axis=(1, 2)).tolist(),
        }
    (RES / "probe_a_cross_class.json").write_text(json.dumps(cross))
    # summary: per-class delta mean for the input-dependent heads
    dep_idx = np.argwhere(dep_mask)
    summary = {}
    for name in sets:
        dm = np.array(cross[name]["delta_mean_per_head"])
        summary[name] = float(dm[dep_mask].mean())
    out["input_dep_head_delta_mean_by_class"] = summary

    (RES / "probe_a_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["head_class"], indent=2))
    print("token-identity var fraction:", out.get("token_identity_var_fraction"))

    # ---- interventions on held-out chunks (paper Table 8)
    held = torch.tensor(natural[N_STAT_NATURAL:N_STAT_NATURAL + N_HELDOUT], dtype=torch.long)
    dep_set = set(map(tuple, np.argwhere(dep_mask).tolist()))
    delta_base = all_traces["natural"][0]  # not used for intervention source; need held-out baseline

    rec = Recorder(n_layers=24)
    base_nlls = cyfa_cpu.forward_batch_nll(model, held, recorder=rec)
    held_delta = np.stack([t["delta"] for t in rec.traces])   # [L,B,T,H]
    base_nll = float(np.mean(base_nlls))
    print(f"[heldout] baseline NLL={base_nll:.4f}")

    head_mean = held_delta.mean(axis=(1, 2), keepdims=True)   # [L,1,1,H] per-head mean over B,T

    def run_intervention(kind):
        dep_heads_by_layer = {}
        for (ll, hh) in dep_set:
            dep_heads_by_layer.setdefault(ll, []).append(hh)
        if kind == "mean":
            hm = torch.tensor(head_mean[:, 0, 0, :])          # [L,H]
            def fn(l, d):
                out = d.clone()
                for hh in dep_heads_by_layer.get(l, []):
                    out[:, :, hh] = hm[l, hh]
                return out
        elif kind == "shift":
            src = torch.tensor(held_delta)                    # [L,B,T,H]
            def fn(l, d):
                out = d.clone()
                rolled = torch.roll(src[l], shifts=257, dims=1)   # per-sequence cyclic shift over T
                for hh in dep_heads_by_layer.get(l, []):
                    out[:, :, hh] = rolled[:, :, hh]
                return out
        elif kind == "one":
            def fn(l, d):
                out = d.clone()
                for hh in dep_heads_by_layer.get(l, []):
                    out[:, :, hh] = 1.0
                return out
        else:
            raise ValueError(kind)
        rec2 = Recorder(n_layers=24, delta_fn=fn)
        nlls = cyfa_cpu.forward_batch_nll(model, held, recorder=rec2)
        return float(np.mean(nlls))

    interv = {"baseline": base_nll}
    for kind in ["mean", "shift", "one"]:
        t0 = time.time()
        interv[kind] = run_intervention(kind)
        print(f"[heldout] {kind}: NLL={interv[kind]:.4f} dNLL={interv[kind]-base_nll:+.4f} ({time.time()-t0:.0f}s)")
    out["interventions"] = {
        "baseline_nll": base_nll,
        "mean": interv["mean"], "shift257": interv["shift"], "one": interv["one"],
        "dnll_mean": interv["mean"] - base_nll,
        "dnll_shift257": interv["shift"] - base_nll,
        "dnll_one": interv["one"] - base_nll,
        "n_heldout": N_HELDOUT,
        "paper_table8": {"baseline": 2.6105, "mean": 2.6555, "shift257": 2.7344, "one": 2.7486},
        "note": "paper intervenes on its 40 input-dependent heads; we intervene on our classified set",
    }
    (RES / "probe_a_summary.json").write_text(json.dumps(out, indent=2))
    print("saved results/probe_a_summary.json")


if __name__ == "__main__":
    main()
