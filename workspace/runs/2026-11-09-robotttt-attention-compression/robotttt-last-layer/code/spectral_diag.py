"""Measure static-feature last-readout delta updates on held-out episodes."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from models import Policy, fast_mlp_apply
from probe_eval import sample_heldout

BUCKETS = [(1, 64), (65, 128), (129, 256), (257, 512),
           (513, 640), (641, 768), (769, 896), (897, 1024)]


@torch.no_grad()
def summarize(records):
    out = {}
    for a, b in BUCKETS:
        rows = [r for r in records if a <= r["t"] <= b]
        if not rows:
            continue
        out[f"{a}-{b}"] = {
            "count": len(rows),
            "phi_sq_mean": sum(r["phi_sq_mean"] for r in rows) / len(rows),
            "phi_sq_max": max(r["phi_sq_max"] for r in rows),
            "radial_eigenvalue_mean": sum(r["radial_eigenvalue_mean"] for r in rows) / len(rows),
            "radial_eigenvalue_min": min(r["radial_eigenvalue_min"] for r in rows),
            "radial_eigenvalue_max": max(r["radial_eigenvalue_max"] for r in rows),
            "operator_norm_mean": sum(r["operator_norm_mean"] for r in rows) / len(rows),
            "operator_norm_min": min(r["operator_norm_min"] for r in rows),
            "operator_norm_max": max(r["operator_norm_max"] for r in rows),
            "nonpositive_fraction": sum(r["nonpositive_fraction"] for r in rows) / len(rows),
            "amplified_channel_fraction": sum(r["amplified_channel_fraction"] for r in rows) / len(rows),
            "noncontractive_fraction": sum(r["noncontractive_fraction"] for r in rows) / len(rows),
        }
    return out


@torch.no_grad()
def replay(model, tokens):
    model.eval()
    x = model.embed(tokens)
    layers = []
    for layer_index, layer in enumerate(model.layers):
        if layer.fast_update != "last":
            raise ValueError("spectral_diag is specific to the static-feature last-readout arm")
        batch, length, _ = x.shape
        xn = layer.ln(x)
        Q = layer.theta_q(xn)
        K = layer.theta_k(xn)
        V = layer.theta_v(xn)
        W = layer.init_state(batch)
        records = []
        outputs = []
        for index in range(length):
            kt = K[:, index]
            phi = F.gelu(torch.bmm(kt.unsqueeze(1), W["W1"]).squeeze(1) + W["b1"])
            phi64 = phi.detach().double()
            phi_sq = phi64.square().sum(-1)
            eta = layer.eta.detach().double()
            radial = 1.0 - (2.0 / layer.d) * phi_sq.unsqueeze(-1) * eta.view(1, -1)
            operator_norm = torch.maximum(torch.ones_like(phi_sq), radial.abs().amax(dim=-1))
            with torch.enable_grad():
                Wd = {key: value.detach().requires_grad_(True) for key, value in W.items()}
                W, loss = layer._inner_update(Wd, kt, V[:, index], create_graph=False)
                W = {key: value.detach() for key, value in W.items()}
            outputs.append(fast_mlp_apply(W, Q[:, index]))
            records.append({
                "t": index + 1,
                "phi_sq_mean": phi_sq.mean().item(),
                "phi_sq_max": phi_sq.max().item(),
                "radial_eigenvalue_mean": radial.mean().item(),
                "radial_eigenvalue_min": radial.min().item(),
                "radial_eigenvalue_max": radial.max().item(),
                "operator_norm_mean": operator_norm.mean().item(),
                "operator_norm_min": operator_norm.min().item(),
                "operator_norm_max": operator_norm.max().item(),
                "nonpositive_fraction": (radial <= 0).float().mean().item(),
                "amplified_channel_fraction": (radial.abs() > 1.0 + 1e-10).float().mean().item(),
                "noncontractive_fraction": (operator_norm > 1.0 + 1e-10).float().mean().item(),
            })
        layers.append({"layer": layer_index, "buckets": summarize(records),
                       "per_step": records})
        x = x + layer.theta_o(torch.stack(outputs, dim=1))
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    model = Policy(ckpt["mixer"], d=ckpt["args"]["d"], layers=ckpt["args"]["layers"])
    model.load_state_dict(ckpt["state_dict"])
    tokens, _ = sample_heldout(args.batch, T=args.T, batch=args.batch, seed=args.seed)
    layers = replay(model, tokens)
    result = {
        "purpose": "Measure the exact static-feature last-readout delta-update Jacobian on held-out episodes.",
        "ckpt_mixer": ckpt["mixer"], "T": args.T, "batch": args.batch, "seed": args.seed,
        "layers": layers,
        "interpretation": {
            "radial_eigenvalue": "For output channel j, 1-(2 eta_j/d)||phi||^2. Its eigenvector is phi^T; hidden directions orthogonal to phi^T have eigenvalue 1.",
            "operator_norm": "The W2 Jacobian is block diagonal over output columns: J_j=I-(2 eta_j/d)phi^T phi. Its exact spectral norm is max(1,max_j abs(radial_j)) for hidden width above one.",
            "fractions": "nonpositive and amplified_channel fractions average over episodes and output channels; noncontractive_fraction averages the full Jacobian test over episodes.",
            "scope": "This is an instantaneous W2 Jacobian diagnostic, not a global stability proof for the closed-loop policy.",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    for layer in layers:
        print(f"layer {layer['layer']}", {b: {k: round(v, 5) for k, v in vals.items()
              if k in ("operator_norm_max", "radial_eigenvalue_min", "noncontractive_fraction")}
              for b, vals in layer["buckets"].items()})


if __name__ == "__main__":
    main()
