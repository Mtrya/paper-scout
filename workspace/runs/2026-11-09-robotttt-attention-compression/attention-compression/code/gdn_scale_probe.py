"""Follow up the GDN recurrence's early spectral instability on real QKV."""
import argparse
import json
import time
from pathlib import Path

import torch

from qkv_probe import (active_buckets, average_metric_scopes, direct_vala,
                       exact_attention, finish_method, init_method_results,
                       mean_history, real_qkv, update_method)

SCALES = [0.0, 1 / 512, 1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8]


def gdn_with_stats(q, k, v, kappa):
    batch, heads, length, dim = k.shape
    value_dim = v.shape[-1]
    state = torch.zeros(batch, heads, value_dim, dim, device=q.device, dtype=q.dtype)
    vbar_prev = torch.zeros(batch, heads, value_dim, device=q.device, dtype=q.dtype)
    ksum = torch.zeros_like(k[:, :, 0])
    vsum = torch.zeros_like(v[:, :, 0])
    outputs = []
    stats = {name: [] for name in ("state_fro", "centered_sq", "multiplier")}
    for index in range(length):
        t = index + 1
        kt, vt, qt = k[:, :, index], v[:, :, index], q[:, :, index]
        ksum, vsum = ksum + kt, vsum + vt
        kbar, vbar = ksum / t, vsum / t
        centered = kt - kbar
        centered_sq = centered.square().sum(-1)
        c = kappa * centered
        residual = vt - vbar_prev - torch.matmul(state, c.unsqueeze(-1)).squeeze(-1)
        state = ((1.0 - 1.0 / t) * state
                 + residual.unsqueeze(-1) * centered.unsqueeze(-2) / t)
        outputs.append(torch.matmul(state, qt.unsqueeze(-1)).squeeze(-1) + vbar)
        stats["state_fro"].append(state.norm(dim=(-2, -1)))
        stats["centered_sq"].append(centered_sq)
        stats["multiplier"].append(1.0 - (1.0 + kappa * centered_sq) / t)
        vbar_prev = vbar
    return (torch.stack(outputs, dim=2),
            {name: torch.stack(value, dim=2) for name, value in stats.items()})


def summarize_stats(stats, bounds):
    a, b = bounds
    sl = slice(a - 1, b)
    state = stats["state_fro"][:, :, sl]
    multiplier = stats["multiplier"][:, :, sl]
    centered = stats["centered_sq"][:, :, sl]
    return {
        "state_fro_mean": state.mean().item(),
        "state_fro_max": state.max().item(),
        "centered_sq_mean": centered.mean().item(),
        "multiplier_mean": multiplier.mean().item(),
        "multiplier_min": multiplier.min().item(),
        "multiplier_max": multiplier.max().item(),
        "noncontractive_fraction": (multiplier.abs() > 1).double().mean().item(),
    }


def average_stats(scopes):
    out = {}
    for bucket in scopes[0]:
        entries = [scope[bucket] for scope in scopes]
        out[bucket] = {}
        for key in entries[0]:
            values = [entry[key] for entry in entries]
            if key == "state_fro_max" or key == "multiplier_max":
                out[bucket][key] = max(values)
            elif key == "multiplier_min":
                out[bucket][key] = min(values)
            else:
                out[bucket][key] = sum(values) / len(values)
        out[bucket]["layer_count"] = len(entries)
    return out


@torch.no_grad()
def analyze(q, k, v, layer_id, scales):
    q, k, v = q.double(), k.double(), v.double()
    scores, causal, probs, target, readout_k = exact_attention(q, k, v)
    del probs, readout_k
    vala = direct_vala(scores, causal, v)
    mean = mean_history(v)
    methods = {"mean": init_method_results([layer_id]),
               "vala": init_method_results([layer_id])}
    methods.update({str(scale): init_method_results([layer_id]) for scale in scales})
    stats_by_scale = {str(scale): {} for scale in scales}
    qualification = None
    for scale in scales:
        out, stats = gdn_with_stats(q, k, v, scale)
        if scale == 0:
            qualification = (out - vala).abs().max().item()
            if qualification > 2e-9:
                raise RuntimeError(f"κ=0 / VaLA qualification failed: {qualification}")
        for bounds in active_buckets(q.shape[2]):
            a, b = bounds
            sl = slice(a - 1, b)
            update_method(methods[str(scale)], layer_id, out[:, :, sl],
                          target[:, :, sl], bounds)
            stats_by_scale[str(scale)][f"{a}-{b}"] = summarize_stats(stats, bounds)
    for bounds in active_buckets(q.shape[2]):
        a, b = bounds
        sl = slice(a - 1, b)
        update_method(methods["mean"], layer_id, mean[:, :, sl], target[:, :, sl], bounds)
        update_method(methods["vala"], layer_id, vala[:, :, sl], target[:, :, sl], bounds)
    centered = k - k.cumsum(dim=2) / torch.arange(
        1, k.shape[2] + 1, device=k.device, dtype=k.dtype).view(1, 1, -1, 1)
    return {
        "layer": layer_id,
        "methods": {name: finish_method(value)["overall"] for name, value in methods.items()},
        "stats": stats_by_scale,
        "qualification_vala_max_abs": qualification,
        "input_stats": {
            "query_norm_mean": q.norm(dim=-1).mean().item(),
            "key_norm_mean": k.norm(dim=-1).mean().item(),
            "centered_key_sq_mean": centered.square().sum(-1).mean().item(),
            "value_norm_mean": v.norm(dim=-1).mean().item(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 13, 27])
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scales", type=float, nargs="+", default=SCALES)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if 0.0 not in args.scales:
        parser.error("scale zero is required")
    started = time.time()
    layer_results = []
    metadata = {}
    for layer_id, q, k, v, metadata in real_qkv(
            args.model_dir, args.text_file, args.layers,
            args.sequences, args.length, args.device):
        print(f"scale sweep layer {layer_id}", flush=True)
        layer_results.append(analyze(q, k, v, layer_id, args.scales))
        del q, k, v
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    aggregate = {"methods": {}, "stats": {}}
    for name in ("mean", "vala"):
        aggregate["methods"][name] = average_metric_scopes(
            [r["methods"][name] for r in layer_results])
    for scale in args.scales:
        aggregate["methods"][str(scale)] = average_metric_scopes(
            [r["methods"][str(scale)] for r in layer_results])
        aggregate["stats"][str(scale)] = average_stats(
            [r["stats"][str(scale)] for r in layer_results])
    result = {
        "purpose": "Test whether κ at the 1/d_head scale removes the GDN recurrence's early spectral instability, and whether stable outputs improve over VaLA.",
        "decision": "Distinguish a normalization failure from an intrinsically unhelpful delta correction; do not interpret inf/nan as task error.",
        "config": {key: value for key, value in vars(args).items() if key != "out"},
        "metadata": metadata,
        "aggregate": aggregate,
        "layers": layer_results,
        "elapsed_s": time.time() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {out}; elapsed={result['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
