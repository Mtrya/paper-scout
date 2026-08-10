"""Aggregated softmax read-weight probe: naive vs canonical KV summary.

For two compression configs, all 8 KV heads x 16 query positions x 3 layers:
how much softmax mass does the summary slot get, relative to the mass the
full cache puts on the source positions? Naive averaging shrinks the summary
vector (phase cancellation) and the summary goes unread; canonical averaging
keeps it readable.
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rope_probe import TEXT, pair_view, rotate_half_style, rope_angles

MODEL_ID = "Qwen/Qwen3-0.6B"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/results"


def main():
    device = "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    ids = tok(TEXT, return_tensors="pt").input_ids.to(device)
    seq_len = ids.shape[1]
    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True)
    hidden = out.hidden_states
    inv_freq, t_pos = rope_angles(model, seq_len, device)
    P = inv_freq.shape[0]
    cfg = model.config
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // cfg.num_attention_heads
    scale = 1.0 / np.sqrt(head_dim)
    layers = [len(model.model.layers) // 4, len(model.model.layers) // 2,
              3 * len(model.model.layers) // 4]

    results = []
    for (M, stride) in [(8, 4), (32, 1)]:
        Dt = (M - 1) * stride
        base = 32
        kpos = base + np.arange(M) * stride
        ratios_naive, ratios_canon = [], []
        for layer_idx in layers:
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn
            h = layer.input_layernorm(hidden[layer_idx])
            with torch.no_grad():
                q_all = attn.q_proj(h).view(1, seq_len, -1, head_dim)[0]
                k_all = attn.k_proj(h).view(1, seq_len, -1, head_dim)[0]
            for head in range(n_kv):
                k_can_all = k_all[:, head]  # [T, 2P]
                k_rot_all = rotate_half_style(
                    k_can_all, t_pos.unsqueeze(-1) * inv_freq)
                ang_src = t_pos[torch.tensor(kpos)].unsqueeze(-1) * inv_freq
                k_can_grp = k_can_all[torch.tensor(kpos)]
                naive = rotate_half_style(k_can_grp, ang_src).mean(0)
                for q_pos in range(kpos[-1] + 8, min(kpos[-1] + 200, seq_len), 12):
                    virt = q_pos - stride
                    canon = rotate_half_style(
                        k_can_grp.mean(0, keepdim=True),
                        (t_pos[virt] * inv_freq).unsqueeze(0))[0]
                    q_rot = rotate_half_style(
                        q_all[q_pos, head].unsqueeze(0),
                        (t_pos[q_pos] * inv_freq).unsqueeze(0))[0]
                    logits_full = (k_rot_all[: q_pos + 1] @ q_rot) * scale
                    w_full = torch.softmax(logits_full, dim=0)
                    mass_src = float(w_full[torch.tensor(kpos)].sum())
                    if mass_src < 1e-6:
                        continue
                    keep = [p for p in range(q_pos + 1) if p not in set(kpos.tolist())]
                    k_comp = k_rot_all[torch.tensor(keep)]
                    base_logits = (k_comp @ q_rot) * scale
                    for tag, summ, acc in [("naive", naive, ratios_naive),
                                           ("canon", canon, ratios_canon)]:
                        w_c = torch.softmax(
                            torch.cat([base_logits, torch.tensor([float(summ @ q_rot * scale)])]), dim=0)
                        acc.append(float(w_c[-1]) / mass_src)
        rn, rc = np.array(ratios_naive), np.array(ratios_canon)
        results.append({
            "M": M, "stride": stride, "Dt": Dt, "n": len(rn),
            "naive_ratio_median": float(np.median(rn)),
            "naive_ratio_mean": float(rn.mean()),
            "canon_ratio_median": float(np.median(rc)),
            "canon_ratio_mean": float(rc.mean()),
        })
        print(f"M={M} stride={stride} (n={len(rn)}): "
              f"naive mass ratio median={np.median(rn):.3f} mean={rn.mean():.3f} | "
              f"canon median={np.median(rc):.3f} mean={rc.mean():.3f}", flush=True)

    json.dump(results, open(os.path.join(OUT, "rope_softmax.json"), "w"), indent=2)
    print("saved rope_softmax.json")


if __name__ == "__main__":
    main()
