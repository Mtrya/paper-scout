"""WorldTrace mechanism probe on real transformer weights (Qwen3-0.6B).

Verifies, on real K/Q geometry, the two mechanism-level claims of
arXiv:2608.07408 (Addressable Memory for Video World Models):

  Claim 1 (Sec 2.2): naive key averaging in RoPE-rotated space corrupts the
      summary via phase cancellation; the corruption is frequency dependent.
  Claim 2 (Sec 3.3): canonical key averaging (unrotate -> mean -> re-rotate
      to a virtual position) preserves the signal and the attention logits.

We do NOT replicate the video-world-model experiments (no code released);
we verify the RoPE geometry the whole method rests on, using real keys and
queries from a real LLM forward pass on real text.

Measurements:
  A. Per-frequency-pair survival ratio of naive averaging:
     || mean_m R(th_f t_m) K_m^f || / mean_m ||K_m^f ||  vs  th_f * Dt
     (Dt = temporal spread of the averaged group), for real keys.
  B. Attention-logit error of a compressed cache vs the full cache, for
     naive vs canonical summaries, using real queries at various q.
  C. The frequency-resolved view: which RoPE bands carry the cancellation.
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
TEXT = (
    "The world model is not a metaphor but a contract: a system that claims to "
    "simulate must preserve what has been seen, predict what comes next, and "
    "admit when it no longer knows. In interactive generation, the key-value "
    "cache is the only place where yesterday survives. Every frame that leaves "
    "the recent window must either be remembered faithfully or be forgotten "
    "honestly. Rotary positional embeddings give each key a phase, and phases, "
    "unlike meanings, do not average. Two vectors rotated in opposite "
    "directions cancel even when both were loud. The engineer who compresses "
    "memory must therefore choose the space in which the average is taken. "
    "Canonical coordinates, stripped of rotation, are the only space where "
    "content alone remains. Re-rotating the mean to a fresh virtual position "
    "is not a trick but a restatement of what the summary claims to be: a "
    "frame that never existed, placed where the model can still read it. "
    "Addressability is prior to content: a memory that cannot be reached is "
    "indistinguishable from one that was never stored. The sliding window "
    "forgets honestly; the collapsed summary lies. When the agent returns to "
    "a previously visited room, the world must return with it. "
) * 3

OUT = os.path.dirname(os.path.abspath(__file__)) + "/results"


def rope_angles(model, seq_len, device):
    """Return inv_freq [n_pairs] and position indices [seq_len]."""
    rope = model.model.rotary_emb
    inv_freq = rope.inv_freq.to(device=device, dtype=torch.float32)  # [P]
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    return inv_freq, t


def pair_view(x, P):
    """HF/Qwen3 (GPT-NeoX) rotary layout: pairs are (i, i+P), not adjacent.

    x [..., 2P] -> (first, second) each [..., P].
    """
    return x[..., :P], x[..., P:]


def rotate_half_style(x, angles):
    """Rotate x [..., 2P] by angles [..., P] using (i, i+P) pairing."""
    P = angles.shape[-1]
    x1, x2 = pair_view(x, P)
    cos, sin = torch.cos(angles), torch.sin(angles)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def main():
    device = "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    ids = tok(TEXT, return_tensors="pt").input_ids.to(device)
    seq_len = ids.shape[1]
    print(f"seq_len={seq_len}")

    captured = {}

    def hook_qkv(layer_idx):
        def fn(module, args, kwargs, output):
            # self_attn forward in transformers 5.x returns (attn_output, attn_weights)
            pass
        return fn

    # Simpler: run one attention layer manually on hidden states.
    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True)
    hidden = out.hidden_states  # tuple [n_layers+1, 1, T, D]
    layer_idx = len(model.model.layers) // 2
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    h = hidden[layer_idx]  # input to this layer
    h_norm = layer.input_layernorm(h)

    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // n_heads
    with torch.no_grad():
        q = attn.q_proj(h_norm).view(1, seq_len, n_heads, head_dim)
        k = attn.k_proj(h_norm).view(1, seq_len, n_kv, head_dim)
    # canonical (unrotated) q,k; pairs layout in Qwen3: interleaved (x1,x2)
    inv_freq, t_pos = rope_angles(model, seq_len, device)  # [P]
    P = inv_freq.shape[0]

    # attention scale
    scale = 1.0 / np.sqrt(head_dim)

    # ---- experiment sweep ---------------------------------------------------
    # groups: M consecutive frames ending at various absolute positions;
    # spread Dt = M-1 (consecutive) or larger strides.
    results = {"layer": layer_idx, "seq_len": seq_len, "n_kv_heads": n_kv,
               "head_dim": head_dim, "A": [], "B": []}

    head = 0  # analyze one KV head in detail + aggregate over heads
    rng = np.random.default_rng(0)

    def summarize(M, stride, q_pos, base_pos):
        """Naive vs canonical summary of M keys at positions
        base_pos + j*stride (j=0..M-1); query at q_pos."""
        kpos = base_pos + np.arange(M) * stride
        # canonical keys [M, 2P]
        k_can = k[0, torch.tensor(kpos), head]  # [M, head_dim]
        # rotated keys at their true positions
        ang_src = t_pos[torch.tensor(kpos)].unsqueeze(-1) * inv_freq  # [M,P]
        k_rot = rotate_half_style(k_can, ang_src)  # [M, 2P]
        naive = k_rot.mean(0)  # [2P]
        # canonical: mean in canonical space, re-rotate to virtual position
        virt_pos = q_pos - stride  # any in-distribution choice; use q-stride
        k_mean = k_can.mean(0, keepdim=True)  # [1, 2P]
        ang_virt = (t_pos[virt_pos] * inv_freq).unsqueeze(0)  # [1,P]
        canon = rotate_half_style(k_mean, ang_virt)[0]  # [2P]
        # query at q_pos, rotated at q_pos
        q_vec = q[0, q_pos, head].unsqueeze(0)  # [1, 2P]
        ang_q = (t_pos[q_pos] * inv_freq).unsqueeze(0)  # [1,P]
        q_rot = rotate_half_style(q_vec, ang_q)[0]  # [2P]
        # logits: <q, k> * scale
        def logit(kv):
            return float(torch.dot(q_rot, kv) * scale)
        # reference: mean of true logits to each source key
        true_logits = torch.stack([torch.dot(q_rot, k_rot[m]) * scale for m in range(M)])
        ref = float(true_logits.mean())
        # per-frequency-pair survival of naive average:
        # || mean_m (pair_m rotated) || / mean_m ||pair_m||, pairs = (i, i+P)
        def pair_norms(x):  # x [M, 2P] -> [M, P]
            x1, x2 = pair_view(x, P)
            return torch.sqrt(x1 ** 2 + x2 ** 2)
        survival = (pair_norms(naive.unsqueeze(0))[0] / pair_norms(k_rot).mean(0)).numpy()
        # content-only shrinkage (no rotation): canonical mean norm / mean norm
        canon_survival = (pair_norms(k_mean)[0] / pair_norms(k_can).mean(0)).numpy()
        # ---- softmax-mass preservation (App B distribution mismatch) -------
        # full cache: keys at positions 0..q; mass on the M source positions.
        q_rot_full = q_rot  # [2P]
        all_k = rotate_half_style(
            k[0, : q_pos + 1, head],
            t_pos[: q_pos + 1].unsqueeze(-1) * inv_freq)  # [q+1, 2P]
        logits_full = (all_k @ q_rot_full) * scale
        w_full = torch.softmax(logits_full, dim=0)
        mass_src = float(w_full[torch.tensor(kpos)].sum())
        # compressed cache: positions 0..base_pos-1 replaced by ONE summary slot
        # at virtual position; rest verbatim.
        keep = [p for p in range(0, q_pos + 1) if p not in set(kpos.tolist())]
        k_comp = all_k[torch.tensor(keep)]
        for tag, summ in [("naive", naive), ("canon", canon)]:
            logits_c = torch.cat([(k_comp @ q_rot_full) * scale,
                                  torch.tensor([logit(summ)])])
            w_c = torch.softmax(logits_c, dim=0)
            globals()[f"mass_{tag}"] = float(w_c[-1])
        return {
            "logit_ref": ref,
            "logit_naive": logit(naive),
            "logit_canon": logit(canon),
            "survival": survival,
            "canon_survival": canon_survival,
            "mass_src": mass_src,
            "mass_naive": mass_naive,
            "mass_canon": mass_canon,
            "angles_spread": (ang_src[:, :] % (2 * np.pi)).std(0).numpy(),
        }

    for M, stride in [(4, 1), (8, 1), (16, 1), (32, 1), (8, 4), (8, 16), (8, 64)]:
        base = 8
        q_pos = min(base + M * stride + 16, seq_len - 1)
        r = summarize(M, stride, q_pos, base)
        Dt = (M - 1) * stride
        results["A"].append({
            "M": M, "stride": stride, "Dt": Dt, "q_pos": q_pos,
            "logit_ref": r["logit_ref"], "logit_naive": r["logit_naive"],
            "logit_canon": r["logit_canon"],
            "err_naive": abs(r["logit_naive"] - r["logit_ref"]),
            "err_canon": abs(r["logit_canon"] - r["logit_ref"]),
            "mass_src": r["mass_src"], "mass_naive": r["mass_naive"],
            "mass_canon": r["mass_canon"],
            "mass_ratio_naive": r["mass_naive"] / max(r["mass_src"], 1e-12),
            "mass_ratio_canon": r["mass_canon"] / max(r["mass_src"], 1e-12),
        })
        results["B"].append({
            "M": M, "stride": stride, "Dt": Dt,
            "theta": inv_freq.numpy().tolist(),
            "theta_Dt": (inv_freq.numpy() * Dt).tolist(),
            "survival": r["survival"].tolist(),
            "canon_survival": r["canon_survival"].tolist(),
        })
        print(f"M={M} stride={stride} Dt={Dt}: "
              f"mass_src={r['mass_src']:.4f} naive={r['mass_naive']:.4f} "
              f"canon={r['mass_canon']:.4f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "rope_probe.json"), "w"), indent=2)
    print("saved", os.path.join(OUT, "rope_probe.json"))


if __name__ == "__main__":
    main()
