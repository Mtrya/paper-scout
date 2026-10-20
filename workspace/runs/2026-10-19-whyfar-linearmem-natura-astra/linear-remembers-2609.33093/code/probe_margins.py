"""Adjudicate the J_mult magnitude: dump per-source clean margins the packet JSON omitted.

Replicates run_experiments.screen() exactly (offsets 2000+1100i, seed 20260910+i,
K=4, filler 512, target_value=VALUE4[i%4]) on CPU, for both splits, and adds:
  - per-source clean margin m0 (16 floats), mean/var/std/min/max, E[m0^2]
  - per-head decomposition residual E[(m-m0)*m0] (tests E[m*m0] ~= E[m0^2])
  - max-abs-diff against the packet's screen_<split>.json J and J_mult matrices
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import task
from model import GDNModel, LAYERS, H

torch.set_num_threads(os.cpu_count() or 8)
OFFSETS = [2000 + 1100 * i for i in range(16)]
PACKET = os.path.join(HERE, "results")


def margins(logits, srcs, cand):
    di = torch.tensor([task.VALUE4.index(s["donor_value"]) for s in srcs])
    ri = torch.tensor([task.VALUE4.index(s["target_value"]) for s in srcs])
    z = logits[:, cand]
    return z[torch.arange(len(srcs)), di] - z[torch.arange(len(srcs)), ri]


def pad_batch(seqs):
    L = max(len(x) for x in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x)
    return ids


def run_split(model, tok, stream, split):
    srcs = []
    for i, off in enumerate(OFFSETS):
        s = task.make_source(stream, off, 4, tok, target_idx=0, seed=20260910 + i,
                             n_filler=512, target_value=task.VALUE4[i % 4])
        srcs.append(s)
    vt = task.value_token_ids(tok)
    cand = torch.tensor([vt[v] for v in task.VALUE4])
    ids_r = pad_batch([s["ids"] for s in srcs])
    ids_d = pad_batch([s["donor_ids"] for s in srcs])
    cut = srcs[0]["query_key_pos"]
    key = ids_r[:, cut]
    colon = ids_r[:, -1]
    pre_r = model.prefill(ids_r[:, :cut], cut=cut, capture_states=True)
    pre_d = model.prefill(ids_d[:, :cut], cut=cut, capture_states=True)
    _, p_r = model.query_token(key, pre_r, return_pre=True)
    _, p_d = model.query_token(key, pre_d, return_pre=True)
    clean = model.query_token(colon, p_r)
    m0 = margins(clean, srcs, cand).numpy()
    print(f"{split}: m0 mean={m0.mean():+.4f} std={m0.std():.4f} "
          f"min={m0.min():+.4f} max={m0.max():+.4f} E[m0^2]={(m0**2).mean():.4f}",
          flush=True)
    J = np.zeros((LAYERS, H))
    Jm = np.zeros((LAYERS, H))
    resid = np.zeros((LAYERS, H))
    for l in range(LAYERS):
        for h in range(H):
            lg = model.query_token(colon, p_r, swap={"layer": l, "head": h,
                                                     "donor_states": p_d["states"]})
            m = margins(lg, srcs, cand).numpy()
            J[l, h] = (m - m0).mean()
            Jm[l, h] = (m * m0).mean()
            resid[l, h] = ((m - m0) * m0).mean()
        print(f"  {split} layer {l}/24 done", flush=True)
    out = {"split": split,
           "m0": [float(v) for v in m0],
           "m0_mean": float(m0.mean()), "m0_var": float(m0.var()),
           "m0_std": float(m0.std()), "m0_min": float(m0.min()),
           "m0_max": float(m0.max()), "E_m0sq": float((m0 ** 2).mean()),
           "J": J.tolist(), "J_mult": Jm.tolist(),
           "decomp_resid_absmax": float(np.abs(resid).max())}
    pj = json.load(open(os.path.join(PACKET, f"screen_{split}.json")))
    out["packet_clean_margin"] = pj["clean_margin"]
    out["J_maxabsdiff_vs_packet"] = float(np.abs(J - np.array(pj["J"])).max())
    out["Jmult_maxabsdiff_vs_packet"] = float(np.abs(Jm - np.array(pj["J_mult"])).max())
    with open(os.path.join(PACKET, f"probe_{split}.json"), "w") as f:
        json.dump(out, f)
    print(f"{split}: |dJ|max vs packet={out['J_maxabsdiff_vs_packet']:.2e} "
          f"|dJmult|max={out['Jmult_maxabsdiff_vs_packet']:.2e} "
          f"resid|E[(m-m0)m0]|max={out['decomp_resid_absmax']:.4f}", flush=True)


def main():
    tok = task.tokenizer()
    model = GDNModel(device="cpu")
    for split in ["validation", "test"]:
        stream = task.load_stream(split)
        run_split(model, tok, stream, split)


if __name__ == "__main__":
    main()
