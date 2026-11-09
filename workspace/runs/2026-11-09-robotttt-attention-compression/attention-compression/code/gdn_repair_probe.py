"""Test a causal spectral clamp after fixed κ fails on real QKV.

Purpose: distinguish a removable early spectral instability from an unhelpful
GDN correction. Decision: only a finite clamped path that improves over VaLA
would justify carrying this approximation into an architecture discussion.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from gdn_scale_probe import average_stats, gdn_with_stats, summarize_stats
from qkv_probe import (active_buckets, average_metric_scopes, direct_vala,
                       exact_attention, finish_method, init_method_results,
                       mean_history, real_qkv, update_method)

STATIC_ARMS = [0.0, 1 / 4096, 1 / 2048, 1 / 1024]


def gdn_spectral_clamp(q, k, v):
    batch, heads, length, dim = k.shape
    value_dim = v.shape[-1]
    state = torch.zeros(batch, heads, value_dim, dim, device=q.device, dtype=q.dtype)
    vbar_prev = torch.zeros(batch, heads, value_dim, device=q.device, dtype=q.dtype)
    ksum = torch.zeros_like(k[:, :, 0])
    vsum = torch.zeros_like(v[:, :, 0])
    outputs = []
    stats = {name: [] for name in ("state_fro", "centered_sq", "multiplier", "effective_kappa")}
    for index in range(length):
        t = index + 1
        kt, vt, qt = k[:, :, index], v[:, :, index], q[:, :, index]
        ksum, vsum = ksum + kt, vsum + vt
        kbar, vbar = ksum / t, vsum / t
        centered = kt - kbar
        centered_sq = centered.square().sum(-1)
        kappa = torch.clamp((t - 1) / centered_sq.clamp_min(1e-30), max=1.0)
        c = kappa.unsqueeze(-1) * centered
        residual = vt - vbar_prev - torch.matmul(state, c.unsqueeze(-1)).squeeze(-1)
        state = ((1.0 - 1.0 / t) * state
                 + residual.unsqueeze(-1) * centered.unsqueeze(-2) / t)
        outputs.append(torch.matmul(state, qt.unsqueeze(-1)).squeeze(-1) + vbar)
        stats["state_fro"].append(state.norm(dim=(-2, -1)))
        stats["centered_sq"].append(centered_sq)
        stats["multiplier"].append(1.0 - (1.0 + kappa * centered_sq) / t)
        stats["effective_kappa"].append(kappa)
        vbar_prev = vbar
    return (torch.stack(outputs, dim=2),
            {name: torch.stack(value, dim=2) for name, value in stats.items()})


def summarize(stats, bounds):
    out = summarize_stats(stats, bounds)
    if "effective_kappa" in stats:
        a, b = bounds
        values = stats["effective_kappa"][:, :, a - 1:b]
        out["effective_kappa_mean"] = values.mean().item()
        out["effective_kappa_min"] = values.min().item()
        out["effective_kappa_max"] = values.max().item()
    return out


@torch.no_grad()
def analyze(q, k, v, layer_id):
    q, k, v = q.double(), k.double(), v.double()
    scores, causal, probs, target, readout_k = exact_attention(q, k, v)
    del probs, readout_k
    vala = direct_vala(scores, causal, v)
    mean = mean_history(v)
    arms = [str(x) for x in STATIC_ARMS] + ["spectral_clamp"]
    methods = {name: init_method_results([layer_id]) for name in ["mean", "vala"] + arms}
    stats_by_arm = {name: {} for name in arms}
    qualification = None
    for arm in arms:
        if arm == "spectral_clamp":
            out, stats = gdn_spectral_clamp(q, k, v)
        else:
            out, stats = gdn_with_stats(q, k, v, float(arm))
        if arm == "0.0":
            qualification = (out - vala).abs().max().item()
            if qualification > 2e-9:
                raise RuntimeError(f"κ=0 / VaLA qualification failed: {qualification}")
        for bounds in active_buckets(q.shape[2]):
            a, b = bounds
            sl = slice(a - 1, b)
            update_method(methods[arm], layer_id, out[:, :, sl], target[:, :, sl], bounds)
            stats_by_arm[arm][f"{a}-{b}"] = summarize(stats, bounds)
    for bounds in active_buckets(q.shape[2]):
        a, b = bounds
        sl = slice(a - 1, b)
        update_method(methods["mean"], layer_id, mean[:, :, sl], target[:, :, sl], bounds)
        update_method(methods["vala"], layer_id, vala[:, :, sl], target[:, :, sl], bounds)
    return {"layer": layer_id,
            "methods": {name: finish_method(value)["overall"] for name, value in methods.items()},
            "stats": stats_by_arm, "qualification_vala_max_abs": qualification}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 13, 27])
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.time()
    layer_results = []
    metadata = {}
    for layer_id, q, k, v, metadata in real_qkv(
            args.model_dir, args.text_file, args.layers,
            args.sequences, args.length, args.device):
        print(f"repair probe layer {layer_id}", flush=True)
        layer_results.append(analyze(q, k, v, layer_id))
        del q, k, v
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    arms = [str(x) for x in STATIC_ARMS] + ["spectral_clamp"]
    aggregate = {"methods": {}, "stats": {}}
    for name in ["mean", "vala"] + arms:
        aggregate["methods"][name] = average_metric_scopes(
            [r["methods"][name] for r in layer_results])
    for arm in arms:
        aggregate["stats"][arm] = average_stats([r["stats"][arm] for r in layer_results])
    result = {
        "purpose": "Enforce 0<=m_t<=1 causally with κ_t=min(1,(t-1)/||u_t||²), then test whether stability yields accuracy beyond VaLA.",
        "static_arms": STATIC_ARMS,
        "config": {key: value for key, value in vars(args).items() if key != "out"},
        "metadata": metadata, "aggregate": aggregate, "layers": layer_results,
        "elapsed_s": time.time() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {out}; elapsed={result['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
