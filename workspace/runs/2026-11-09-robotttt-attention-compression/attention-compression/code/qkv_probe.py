"""Decompose softmax-to-recurrent approximations on real or synthetic QKV."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

BUCKETS = [(1, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 512)]
DEFAULT_KAPPAS = [0.0, 0.25, 0.5, 1.0, 2.0]
DEFAULT_RHOS = [1e-4, 1e-3, 1e-2, 1e-1]


def bucket_name(bounds):
    return f"{bounds[0]}-{bounds[1]}"


def active_buckets(length):
    return [(a, min(b, length)) for a, b in BUCKETS if a <= length]


def new_meter():
    return {"n": 0, "se": 0.0, "target_ss": 0.0, "pred_ss": 0.0,
            "cos": 0.0}


def update_meter(meter, pred, target):
    pred, target = pred.double(), target.double()
    meter["n"] += target.shape[-2] if target.ndim == 3 else target.numel() // target.shape[-1]
    meter["se"] += (pred - target).square().sum().item()
    meter["target_ss"] += target.square().sum().item()
    meter["pred_ss"] += pred.square().sum().item()
    meter["cos"] += F.cosine_similarity(pred, target, dim=-1).sum().item()


def finish_meter(meter):
    if not meter["n"]:
        return None
    denom = max(meter["target_ss"], 1e-30)
    return {
        "count_vectors": meter["n"],
        "nmse": meter["se"] / denom,
        "cosine": meter["cos"] / meter["n"],
        "output_target_rms": math.sqrt(max(meter["pred_ss"], 0.0) / denom),
    }


def combine_meters(meters):
    out = new_meter()
    for meter in meters:
        for key in out:
            out[key] += meter[key]
    return out


def new_scope():
    return {bucket_name(b): new_meter() for b in BUCKETS}


def update_scope(scope, pred, target, bounds):
    update_meter(scope[bucket_name(bounds)], pred, target)


def finish_scope(scope):
    out = {"all": finish_meter(combine_meters(scope.values()))}
    out.update({name: finish_meter(meter) for name, meter in scope.items()
                if meter["n"]})
    return out


def new_diag_scope():
    return {bucket_name(b): {"n": 0, "negative": 0, "above_one": 0,
                             "minimum": float("inf"), "sum": 0.0}
            for b in BUCKETS}


def update_diag(scope, alpha, bounds):
    rec = scope[bucket_name(bounds)]
    rec["n"] += alpha.numel()
    rec["negative"] += (alpha < 0).sum().item()
    rec["above_one"] += (alpha > 1).sum().item()
    rec["minimum"] = min(rec["minimum"], alpha.min().item())
    rec["sum"] += alpha.sum().item()


def finish_diag(scope):
    out = {}
    for name, rec in scope.items():
        if rec["n"]:
            out[name] = {"count": rec["n"],
                         "negative_fraction": rec["negative"] / rec["n"],
                         "above_one_fraction": rec["above_one"] / rec["n"],
                         "minimum": rec["minimum"],
                         "mean": rec["sum"] / rec["n"]}
    return out


def new_geometry_scope():
    return {bucket_name(b): {"n": 0, "cos": 0.0, "projection_residual": 0.0}
            for b in BUCKETS}


def update_geometry(scope, r, q, bounds):
    rec = scope[bucket_name(bounds)]
    r, q = r.double(), q.double()
    q2 = q.square().sum(-1).clamp_min(1e-30)
    lam = (r * q).sum(-1) / q2
    residual = (r - lam.unsqueeze(-1) * q).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
    rec["n"] += q.numel() // q.shape[-1]
    rec["cos"] += F.cosine_similarity(r, q, dim=-1).sum().item()
    rec["projection_residual"] += residual.sum().item()


def finish_geometry(scope):
    out = {}
    for name, rec in scope.items():
        if rec["n"]:
            out[name] = {"count_vectors": rec["n"],
                         "cosine": rec["cos"] / rec["n"],
                         "projection_residual": rec["projection_residual"] / rec["n"]}
    return out


def init_method_results(layer_ids):
    return {"overall": new_scope(), "layers": {str(i): new_scope() for i in layer_ids}}


def update_method(result, layer_id, pred, target, bounds):
    update_scope(result["overall"], pred, target, bounds)
    update_scope(result["layers"][str(layer_id)], pred, target, bounds)


def finish_method(result):
    return {"overall": finish_scope(result["overall"]),
            "layers": {key: finish_scope(value) for key, value in result["layers"].items()}}


def exact_attention(q, k, v):
    length = q.shape[2]
    scores = torch.matmul(q, k.transpose(-1, -2))
    causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    target = torch.matmul(probs, v)
    readout_k = torch.matmul(probs, k)
    return scores, causal, probs, target, readout_k


def direct_vala(scores, causal, v):
    length = v.shape[2]
    steps = torch.arange(1, length + 1, device=v.device, dtype=v.dtype)
    masked_scores = scores.masked_fill(~causal, 0.0)
    score_mean = masked_scores.sum(-1, keepdim=True) / steps.view(1, 1, -1, 1)
    weights = (1.0 + scores - score_mean) / steps.view(1, 1, -1, 1)
    weights = weights.masked_fill(~causal, 0.0)
    return torch.matmul(weights, v)


def mean_history(v):
    steps = torch.arange(1, v.shape[2] + 1, device=v.device, dtype=v.dtype)
    return v.cumsum(dim=2) / steps.view(1, 1, -1, 1)


def gdn_path(q, k, v, kappa):
    batch, heads, length, dim = k.shape
    value_dim = v.shape[-1]
    state = torch.zeros(batch, heads, value_dim, dim, device=q.device, dtype=q.dtype)
    vbar_prev = torch.zeros(batch, heads, value_dim, device=q.device, dtype=q.dtype)
    ksum = torch.zeros_like(k[:, :, 0])
    vsum = torch.zeros_like(v[:, :, 0])
    outputs, alphas = [], []
    for index in range(length):
        t = index + 1
        kt, vt, qt = k[:, :, index], v[:, :, index], q[:, :, index]
        ksum, vsum = ksum + kt, vsum + vt
        kbar, vbar = ksum / t, vsum / t
        centered = kt - kbar
        c = kappa * centered
        residual = vt - vbar_prev - torch.matmul(state, c.unsqueeze(-1)).squeeze(-1)
        state = ((1.0 - 1.0 / t) * state
                 + (residual.unsqueeze(-1) * centered.unsqueeze(-2)) / t)
        out = torch.matmul(state, qt.unsqueeze(-1)).squeeze(-1) + vbar
        outputs.append(out)
        alpha = (1.0 + (qt * centered).sum(-1)) / t
        alphas.append(alpha)
        vbar_prev = vbar
    return torch.stack(outputs, dim=2), torch.stack(alphas, dim=2)


def ridge_paths(q, k, v, probs, readout_k, rho):
    batch, heads, length, dim = k.shape
    value_dim = v.shape[-1]
    G = torch.zeros(batch, heads, value_dim, dim, device=q.device, dtype=q.dtype)
    H = torch.zeros(batch, heads, dim, dim, device=q.device, dtype=q.dtype)
    eye = torch.eye(dim, device=q.device, dtype=q.dtype).expand(batch, heads, -1, -1)
    variants = {"attention_readout": [], "query_readout": [],
                "projected_query_readout": []}
    qualification = []
    check_points = {0, length // 8 - 1, length // 4 - 1, length - 1}
    for index in range(length):
        kt, vt, qt = k[:, :, index], v[:, :, index], q[:, :, index]
        G = G + vt.unsqueeze(-1) * kt.unsqueeze(-2)
        H = H + kt.unsqueeze(-1) * kt.unsqueeze(-2)
        lam = rho * H.diagonal(dim1=-2, dim2=-1).mean(-1)
        W = torch.linalg.solve(H + lam[..., None, None] * eye,
                               G.transpose(-1, -2)).transpose(-1, -2)
        r = readout_k[:, :, index]
        q2 = qt.square().sum(-1).clamp_min(1e-30)
        projection = ((r * qt).sum(-1) / q2).unsqueeze(-1) * qt
        variants["attention_readout"].append(torch.matmul(W, r.unsqueeze(-1)).squeeze(-1))
        variants["query_readout"].append(torch.matmul(W, qt.unsqueeze(-1)).squeeze(-1))
        variants["projected_query_readout"].append(torch.matmul(W, projection.unsqueeze(-1)).squeeze(-1))
        if index in check_points:
            a = probs[:, :, index, :index + 1]
            pred_values = torch.einsum("bhvk,bhsk->bhsv", W, k[:, :, :index + 1])
            rhs = torch.matmul(a.unsqueeze(-2), pred_values).squeeze(-2)
            lhs = variants["attention_readout"][-1]
            qualification.append((lhs - rhs).abs().max().item())
    return ({key: torch.stack(value, dim=2) for key, value in variants.items()},
            max(qualification, default=0.0))


@torch.no_grad()
def analyze_layer(q, k, v, layer_id, kappas, rhos):
    started = time.time()
    q, k, v = q.double(), k.double(), v.double()
    length = q.shape[2]
    scores, causal, probs, target, readout_k = exact_attention(q, k, v)
    mean = mean_history(v)
    vala = direct_vala(scores, causal, v)
    baselines = {name: init_method_results([layer_id])
                 for name in ("mean", "vala")}
    gdn = {str(kappa): init_method_results([layer_id]) for kappa in kappas}
    ridge = {str(rho): {name: init_method_results([layer_id])
                        for name in ("attention_readout", "query_readout",
                                     "projected_query_readout")}
             for rho in rhos}
    diag = new_diag_scope()
    geometry = new_geometry_scope()
    qualification = {"vala_max_abs": {}, "mean_field_max_abs": {}}

    gdn_outputs = {}
    for kappa in kappas:
        out, alpha = gdn_path(q, k, v, kappa)
        gdn_outputs[kappa] = out
        error = (out - vala).abs().max().item() if kappa == 0 else None
        if error is not None:
            qualification["vala_max_abs"][str(kappa)] = error
            if error > 2e-9:
                raise RuntimeError(f"κ=0 / VaLA qualification failed: {error}")
        if kappa == kappas[0]:
            for bounds in active_buckets(length):
                a, b = bounds
                update_diag(diag, alpha[:, :, a - 1:b], bounds)
        for bounds in active_buckets(length):
            a, b = bounds
            update_method(gdn[str(kappa)], layer_id, out[:, :, a - 1:b],
                          target[:, :, a - 1:b], bounds)

    ridge_errors = {}
    for rho in rhos:
        outputs, mean_field_error = ridge_paths(q, k, v, probs, readout_k, rho)
        ridge_errors[rho] = outputs
        qualification["mean_field_max_abs"][str(rho)] = mean_field_error
        if mean_field_error > 2e-9:
            raise RuntimeError(f"mean-field qualification failed: {mean_field_error}")

    for bounds in active_buckets(length):
        a, b = bounds
        sl = slice(a - 1, b)
        update_method(baselines["mean"], layer_id, mean[:, :, sl], target[:, :, sl], bounds)
        update_method(baselines["vala"], layer_id, vala[:, :, sl], target[:, :, sl], bounds)
        update_geometry(geometry, readout_k[:, :, sl], q[:, :, sl], bounds)
        for rho in rhos:
            for name, out in ridge_errors[rho].items():
                update_method(ridge[str(rho)][name], layer_id, out[:, :, sl],
                              target[:, :, sl], bounds)

    result = {
        "layer": layer_id,
        "shape": list(q.shape),
        "baselines": {name: finish_method(value) for name, value in baselines.items()},
        "gdn": {name: finish_method(value) for name, value in gdn.items()},
        "ridge": {rho: {name: finish_method(value) for name, value in variants.items()}
                  for rho, variants in ridge.items()},
        "diagonal_approx": finish_diag(diag),
        "readout_geometry": finish_geometry(geometry),
        "qualification": qualification,
        "elapsed_s": time.time() - started,
    }
    return result


def average_metric_scopes(scopes):
    out = {}
    for bucket in scopes[0]:
        entries = [scope[bucket] for scope in scopes if scope.get(bucket)]
        if not entries:
            continue
        out[bucket] = {
            "count_vectors": sum(x["count_vectors"] for x in entries),
            "nmse": sum(x["nmse"] for x in entries) / len(entries),
            "cosine": sum(x["cosine"] for x in entries) / len(entries),
            "output_target_rms": sum(x["output_target_rms"] for x in entries) / len(entries),
            "layer_count": len(entries),
        }
    return out


def average_diag_scopes(scopes):
    out = {}
    for bucket in scopes[0]:
        entries = [scope[bucket] for scope in scopes if scope.get(bucket)]
        out[bucket] = {
            "count": sum(x["count"] for x in entries),
            "negative_fraction": sum(x["negative_fraction"] for x in entries) / len(entries),
            "above_one_fraction": sum(x["above_one_fraction"] for x in entries) / len(entries),
            "minimum": min(x["minimum"] for x in entries),
            "mean": sum(x["mean"] for x in entries) / len(entries),
            "layer_count": len(entries),
        }
    return out


def average_geometry_scopes(scopes):
    out = {}
    for bucket in scopes[0]:
        entries = [scope[bucket] for scope in scopes if scope.get(bucket)]
        out[bucket] = {
            "count_vectors": sum(x["count_vectors"] for x in entries),
            "cosine": sum(x["cosine"] for x in entries) / len(entries),
            "projection_residual": sum(x["projection_residual"] for x in entries) / len(entries),
            "layer_count": len(entries),
        }
    return out


def aggregate_from_layer_results(layer_results, kappas, rhos):
    aggregate = {"baselines": {}, "gdn": {}, "ridge": {},
                 "diagonal_approx": average_diag_scopes(
                     [r["diagonal_approx"] for r in layer_results]),
                 "readout_geometry": average_geometry_scopes(
                     [r["readout_geometry"] for r in layer_results])}
    for name in ("mean", "vala"):
        aggregate["baselines"][name] = average_metric_scopes(
            [r["baselines"][name]["overall"] for r in layer_results])
    for kappa in kappas:
        aggregate["gdn"][str(kappa)] = average_metric_scopes(
            [r["gdn"][str(kappa)]["overall"] for r in layer_results])
    for rho in rhos:
        aggregate["ridge"][str(rho)] = {}
        for name in ("attention_readout", "query_readout", "projected_query_readout"):
            aggregate["ridge"][str(rho)][name] = average_metric_scopes(
                [r["ridge"][str(rho)][name]["overall"] for r in layer_results])
    return aggregate


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.no_grad()
def real_qkv(model_dir, text_file, layer_ids, sequences, length, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    text = Path(text_file).read_text(encoding="utf-8")
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_offset = int(os.environ.get("QKV_TOKEN_OFFSET", "0"))
    needed = sequences * length
    if token_offset < 0 or len(token_ids) < token_offset + needed:
        raise ValueError(f"need tokens [{token_offset}, {token_offset + needed}), got {len(token_ids)}")
    input_ids = torch.tensor(token_ids[token_offset:token_offset + needed],
                             device=device).view(sequences, length)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    config = model.config
    if config.model_type != "qwen3":
        raise ValueError(f"this RoPE path is qualified for Qwen3, got {config.model_type}")
    captured = {}
    hooks = []
    for layer_id in layer_ids:
        layer = model.model.layers[layer_id]
        def capture(module, args, kwargs, layer_id=layer_id):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            captured[layer_id] = hidden_states.detach()
        hooks.append(layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True))
    with torch.no_grad():
        probe = model.model(input_ids=input_ids, use_cache=False, output_attentions=True)
        for hook in hooks:
            hook.remove()
    metadata = {
        "model_dir": model_dir,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_count_available": len(token_ids),
        "token_offset": token_offset,
        "sequences": sequences,
        "length": length,
        "torch_dtype": "bfloat16",
        "attention_probe_length": length,
        "attention_probe_max_abs_by_layer": {},
    }

    for layer_id in layer_ids:
        layer = model.model.layers[layer_id]
        attn = layer.self_attn
        x = captured[layer_id]
        batch, steps, _ = x.shape
        q = attn.q_proj(x).view(batch, steps, config.num_attention_heads, attn.head_dim)
        k = attn.k_proj(x).view(batch, steps, config.num_key_value_heads, attn.head_dim)
        v = attn.v_proj(x).view(batch, steps, config.num_key_value_heads, attn.head_dim)
        q = attn.q_norm(q).transpose(1, 2)
        k = attn.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        inv_freq = 1.0 / (config.rope_theta ** (
            torch.arange(0, attn.head_dim, 2, device=x.device).float() / attn.head_dim))
        positions = torch.arange(steps, device=x.device).float()
        freqs = torch.outer(positions, inv_freq)
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1).to(q.dtype).view(1, 1, steps, -1)
        sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1).to(q.dtype).view(1, 1, steps, -1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        repeats = config.num_attention_heads // config.num_key_value_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        probe_scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scaling
        probe_mask = torch.ones(steps, steps, device=q.device, dtype=torch.bool).tril()
        probe_probs = torch.softmax(probe_scores.masked_fill(~probe_mask, float("-inf")),
                                    dim=-1, dtype=torch.float32).to(q.dtype)
        attention_error = (probe_probs - probe.attentions[layer_id]).abs().max().item()
        if attention_error > 5e-3:
            raise RuntimeError(f"Qwen attention reconstruction failed at layer {layer_id}: {attention_error}")
        metadata["attention_probe_max_abs_by_layer"][str(layer_id)] = attention_error
        yield layer_id, q.float() * attn.scaling, k.float(), v.float(), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--text-file")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 13, 27])
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--kappas", type=float, nargs="+", default=DEFAULT_KAPPAS)
    parser.add_argument("--rhos", type=float, nargs="+", default=DEFAULT_RHOS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if 0.0 not in args.kappas:
        raise ValueError("κ=0 is required for the VaLA qualification gate")
    torch.manual_seed(args.seed)
    started = time.time()
    layer_results = []
    metadata = {}
    if args.synthetic:
        stream = []
        for layer_id in args.layers:
            gen = torch.Generator(device=args.device).manual_seed(args.seed + layer_id)
            shape = (args.sequences, args.heads, args.length, args.d_head)
            q = torch.randn(shape, generator=gen, device=args.device) / math.sqrt(args.d_head)
            k = torch.randn(shape, generator=gen, device=args.device)
            v = torch.randn(shape, generator=gen, device=args.device)
            stream.append((layer_id, q, k, v, {"synthetic": True}))
    else:
        if not args.model_dir or not args.text_file:
            parser.error("real mode requires --model-dir and --text-file")
        stream = real_qkv(args.model_dir, args.text_file, args.layers,
                          args.sequences, args.length, args.device)
    for layer_id, q, k, v, metadata in stream:
        print(f"analyzing layer {layer_id}: {tuple(q.shape)}", flush=True)
        res = analyze_layer(q, k, v, layer_id, args.kappas, args.rhos)
        layer_results.append(res)
        print(f"layer {layer_id} done in {res['elapsed_s']:.1f}s; "
              f"qualification={res['qualification']}", flush=True)
        del q, k, v
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Aggregate directly from sufficient statistics kept by a second compact pass.
    aggregate = aggregate_from_layer_results(layer_results, args.kappas, args.rhos)
    result = {
        "config": {key: value for key, value in vars(args).items() if key != "out"},
        "metadata": metadata,
        "torch": torch.__version__,
        "bucket_definition": "token positions are one-based and inclusive",
        "metrics": "NMSE=sum squared error/sum squared target; cosine and projection residual are per-vector means",
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
