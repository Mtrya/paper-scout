"""Separate current-pair self-fitting from the attention-to-TTT readout gap."""
import argparse
import json
import time
from pathlib import Path

import torch

from qkv_probe import (active_buckets, average_metric_scopes, exact_attention,
                       finish_method, init_method_results, real_qkv, update_method)

VARIANTS = ["pre_attention", "post_attention", "pre_query", "post_query"]


@torch.no_grad()
def analyze(q, k, v, layer_id, rho):
    q, k, v = q.double(), k.double(), v.double()
    scores, causal, probs, target, readout_k = exact_attention(q, k, v)
    batch, heads, length, dim = k.shape
    value_dim = v.shape[-1]
    G = torch.zeros(batch, heads, value_dim, dim, device=q.device, dtype=q.dtype)
    H = torch.zeros(batch, heads, dim, dim, device=q.device, dtype=q.dtype)
    eye = torch.eye(dim, device=q.device, dtype=q.dtype).expand(batch, heads, -1, -1)
    outputs = {name: [] for name in VARIANTS}
    self_attention = []
    for index in range(length):
        kt, vt, qt = k[:, :, index], v[:, :, index], q[:, :, index]
        r = readout_k[:, :, index]
        H_before = H
        lam = rho * H_before.diagonal(dim1=-2, dim2=-1).mean(-1)
        if index == 0:
            W_before = torch.zeros_like(G)
        else:
            W_before = torch.linalg.solve(H_before + lam[..., None, None] * eye,
                                          G.transpose(-1, -2)).transpose(-1, -2)
        outputs["pre_attention"].append(torch.matmul(W_before, r.unsqueeze(-1)).squeeze(-1))
        outputs["pre_query"].append(torch.matmul(W_before, qt.unsqueeze(-1)).squeeze(-1))
        G = G + vt.unsqueeze(-1) * kt.unsqueeze(-2)
        H = H + kt.unsqueeze(-1) * kt.unsqueeze(-2)
        lam = rho * H.diagonal(dim1=-2, dim2=-1).mean(-1)
        W_after = torch.linalg.solve(H + lam[..., None, None] * eye,
                                     G.transpose(-1, -2)).transpose(-1, -2)
        outputs["post_attention"].append(torch.matmul(W_after, r.unsqueeze(-1)).squeeze(-1))
        outputs["post_query"].append(torch.matmul(W_after, qt.unsqueeze(-1)).squeeze(-1))
        self_attention.append(probs[:, :, index, index])
    methods = {name: init_method_results([layer_id]) for name in VARIANTS}
    for bounds in active_buckets(length):
        a, b = bounds
        sl = slice(a - 1, b)
        for name in VARIANTS:
            out = torch.stack(outputs[name], dim=2)
            update_method(methods[name], layer_id, out[:, :, sl], target[:, :, sl], bounds)
    diag = torch.stack(self_attention, dim=2)
    diag_result = {}
    for bounds in active_buckets(length):
        a, b = bounds
        diag_result[f"{a}-{b}"] = diag[:, :, a - 1:b].mean().item()
    return {"layer": layer_id,
            "methods": {name: finish_method(value)["overall"] for name, value in methods.items()},
            "exact_self_attention_mean": diag_result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 13, 27])
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--rho", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.time()
    layer_results = []
    metadata = {}
    for layer_id, q, k, v, metadata in real_qkv(
            args.model_dir, args.text_file, args.layers,
            args.sequences, args.length, args.device):
        print(f"pre-update probe layer {layer_id}", flush=True)
        layer_results.append(analyze(q, k, v, layer_id, args.rho))
        del q, k, v
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    aggregate = {name: average_metric_scopes(
        [r["methods"][name] for r in layer_results]) for name in VARIANTS}
    result = {
        "purpose": "Check whether the query-readout loss survives when W_{t-1}, rather than W_t fitted on the current pair, is used.",
        "decision": "Do not use early post-update interpolation as evidence of K-to-V generalization.",
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
