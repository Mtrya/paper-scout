"""Shared helpers: checkpoint loading, parameter grouping, update spectra."""
import torch

from model import FlowPolicy


def load_policy(path, pred_mode="x"):
    p = FlowPolicy(pred_mode=pred_mode)
    p.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return p


def param_group(name):
    if ".ada." in name:
        return "AdaRMS"
    if name.startswith("time_mlp"):
        return "TimeMLP"
    return "Rest"


def ts_group(name):
    return param_group(name) != "Rest"


def delta_dict(p_new, p_ref):
    a = p_new.state_dict()
    b = p_ref.state_dict()
    return {k: (a[k].float() - b[k].float()) for k in a}


def spec_stats(dW):
    """Density and effective rank of a single update tensor."""
    out = {}
    flat = dW.abs().flatten()
    n = flat.numel()
    out["n_params"] = int(n)
    out["l2"] = float(dW.pow(2).sum().sqrt())
    out["max_abs"] = float(flat.max())
    for thr in (1e-5, 1e-4):
        out[f"density_{thr:g}"] = float((flat > thr).float().mean())
    for rel in (1e-2, 1e-1):
        out[f"density_rel{rel:g}"] = float((flat > rel * flat.max()).float().mean())
    if dW.dim() == 2 and min(dW.shape) > 1:
        s = torch.linalg.svdvals(dW)
        e = s.pow(2)
        tot = e.sum().clamp_min(1e-30)
        cum = e.cumsum(0) / tot
        r95 = int((cum < 0.95).sum()) + 1
        r99 = int((cum < 0.99).sum()) + 1
        out["r95"] = r95
        out["r99"] = r99
        out["max_rank"] = int(min(dW.shape))
        out["r95_frac"] = r95 / min(dW.shape)
        out["energy"] = float(tot)
        top = min(48, s.numel())
        out["sv_norm"] = (s[:top] / s.sum().clamp_min(1e-30)).tolist()
    return out


def summarize_delta(dW, prefix="", matrices_only=False):
    """Aggregate spectrum stats per parameter-group and per matrix."""
    per_matrix = {}
    groups = {}
    for k, v in dW.items():
        g = param_group(k)
        st = spec_stats(v)
        st["shape"] = list(v.shape)
        per_matrix[k] = st
        groups.setdefault(g, []).append((k, st, v))
    agg = {}
    for g, items in groups.items():
        mats = [(k, s) for k, s, v in items if "r95" in s]
        energy = sum(s.get("energy", s["l2"] ** 2) for k, s, v in items)
        agg[g] = {
            "n_params": sum(s["n_params"] for k, s, v in items),
            "energy": energy,
            "l2": float(sum(s["l2"] ** 2 for k, s, v in items) ** 0.5),
            "density_1e-05": sum(s["density_1e-05"] * s["n_params"] for k, s, v in items)
            / sum(s["n_params"] for k, s, v in items),
            "density_rel0.1": sum(s["density_rel0.1"] * s["n_params"] for k, s, v in items)
            / sum(s["n_params"] for k, s, v in items),
            "density_rel0.01": sum(s["density_rel0.01"] * s["n_params"] for k, s, v in items)
            / sum(s["n_params"] for k, s, v in items),
            "r95_mean": sum(s["r95"] for k, s in mats) / max(1, len(mats)),
            "r95_max": max([s["r95"] for k, s in mats], default=0),
            "r95_frac_mean": sum(s["r95_frac"] for k, s in mats) / max(1, len(mats)),
            "n_matrices": len(mats),
        }
    tot_energy = sum(a["energy"] for a in agg.values()) or 1.0
    for g in agg:
        agg[g]["energy_share"] = agg[g]["energy"] / tot_energy
        agg[g]["param_share"] = agg[g]["n_params"] / sum(a["n_params"] for a in agg.values())
    return {"groups": agg, "per_matrix": per_matrix}


def ts_energy_share(dW):
    e_ts = sum(float((v.float() ** 2).sum()) for k, v in dW.items() if ts_group(k))
    e_all = sum(float((v.float() ** 2).sum()) for k, v in dW.items())
    return e_ts / max(e_all, 1e-30)


def output_vectors(policy, taus):
    """AdaRMS outputs (scale, shift, gate) per block at each tau. (T, L, 3, d)"""
    outs = []
    for t in taus:
        tau = torch.full((1,), float(t))
        c = policy.cond(tau)
        rows = []
        for blk in policy.blocks:
            ssg = blk.ada.to_ssg(c)[0]
            rows.append(ssg.view(3, policy.d))
        outs.append(torch.stack(rows, 0))
    return torch.stack(outs, 0)
