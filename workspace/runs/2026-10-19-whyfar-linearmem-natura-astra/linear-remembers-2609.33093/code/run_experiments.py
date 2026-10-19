"""HLAR replication on GDN-340M (m-a-p/340M-20B-GatedDeltaNet-pure-baseline).

Paper protocol: WikiText-103 dev (validation) / confirmation (test), K key-value
bindings x 3 presentations, 512 filler, one-token query. The task format is our own
(" {key}: {value}." bindings, " {key}:" query; the final ":" is the one-token query).

The native LM head of this checkpoint turns out to be at chance on the task (see
README), so every experiment reports both:
  native  - four-choice accuracy / donor margin on the model's own logits
  state   - donor-recipient recurrent-state retention at the query boundary
"""
import argparse
import json
import os

import numpy as np
import torch

import task
from model import GDNModel, LAYERS, H

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
if not os.path.isdir(RES):
    RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
DEV_OFFSETS = [2000 + 1100 * i for i in range(16)]
TEST_OFFSETS = [2000 + 1100 * i for i in range(16)]
FILLER = 512


def srcs_for(stream, offsets, split_tag, K=4, nf=FILLER, fixed_total=None,
             distance=None):
    out = []
    for i, off in enumerate(offsets):
        tv = task.VALUE4[i % 4]
        s = task.make_source(stream, off, K, task.tokenizer(), target_idx=0, seed=20260910 + i,
                             n_filler=nf, fixed_total=fixed_total, distance=distance,
                             target_value=tv)
        s["split"] = split_tag
        out.append(s)
    return out


def pad_batch(seqs, device):
    L = max(len(x) for x in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x, device=device)
    return ids


def cand_ids(tok):
    vt = task.value_token_ids(tok)
    return torch.tensor([vt[v] for v in task.VALUE4])


def margins(logits, srcs, cand, which="donor"):
    """donor - recipient logit margin per source."""
    di = torch.tensor([task.VALUE4.index(s["donor_value"]) for s in srcs], device=logits.device)
    ri = torch.tensor([task.VALUE4.index(s["target_value"]) for s in srcs], device=logits.device)
    z = logits[:, cand]
    return z[torch.arange(len(srcs)), di] - z[torch.arange(len(srcs)), ri]


def acc(logits, srcs, cand):
    pred = logits[:, cand].argmax(1).cpu().numpy()
    tgt = np.array([task.VALUE4.index(s["target_value"]) for s in srcs])
    return float((pred == tgt).mean()), pred, tgt


def baseline(stream, tok, model, split, offsets):
    srcs = srcs_for(stream, offsets, split)
    ids = pad_batch([s["ids"] for s in srcs], model.device)
    L = ids.shape[1]
    logits = model.prefill(ids, logits_pos=L - 1)["logits"]
    cand = cand_ids(tok).to(model.device)
    a, pred, tgt = acc(logits, srcs, cand)
    m = margins(logits, srcs, cand)
    out = {"split": split, "n": len(srcs), "acc": a,
           "pred": [task.VALUE4[i] for i in pred],
           "target": [s["target_value"] for s in srcs],
           "mean_target_logit": float(logits[torch.arange(len(srcs)), cand[
               torch.tensor([task.VALUE4.index(s["target_value"]) for s in srcs], device=model.device)]].mean()),
           "mean_donor_minus_recipient": float(m.mean())}
    print(f"{split}: native acc={a:.3f} donor-recipient margin={m.mean():+.4f}")
    return out, srcs


def screen(stream, tok, model, split="validation"):
    """Paper Eq.6 head screen: per-head pre-query state swap, native LM head."""
    srcs = srcs_for(stream, DEV_OFFSETS, split)
    cand = cand_ids(tok).to(model.device)
    ids_r = pad_batch([s["ids"] for s in srcs], model.device)
    ids_d = pad_batch([s["donor_ids"] for s in srcs], model.device)
    cut = srcs[0]["query_key_pos"]
    key = ids_r[:, cut]
    colon = ids_r[:, -1]
    pre_r = model.prefill(ids_r[:, :cut], cut=cut, capture_states=True)
    pre_d = model.prefill(ids_d[:, :cut], cut=cut, capture_states=True)
    _, p_r = model.query_token(key, pre_r, return_pre=True)
    _, p_d = model.query_token(key, pre_d, return_pre=True)
    clean = model.query_token(colon, p_r)
    m0 = margins(clean, srcs, cand)
    a0, _, _ = acc(clean, srcs, cand)
    J = np.zeros((LAYERS, H))
    J_mult = np.zeros((LAYERS, H))
    flip = np.zeros((LAYERS, H))
    donor_prob = np.zeros((LAYERS, H))
    for l in range(LAYERS):
        for h in range(H):
            lg = model.query_token(colon, p_r, swap={"layer": l, "head": h,
                                                     "donor_states": p_d["states"]})
            m = margins(lg, srcs, cand)
            J[l, h] = float((m - m0).mean())
            J_mult[l, h] = float((m * m0).mean())
            di = torch.tensor([task.VALUE4.index(s["donor_value"]) for s in srcs], device=lg.device)
            a, _, _ = acc(lg, srcs, cand)
            # donor-answer rate
            pred = lg[:, cand].argmax(1)
            flip[l, h] = float((pred == cand[di]).float().mean())
    res = {"split": split, "clean_acc": a0, "clean_margin": float(m0.mean()),
           "J": J.tolist(), "J_mult": J_mult.tolist(), "donor_rate": flip.tolist()}
    np.save(os.path.join(RES, f"screen_J_{split}.npy"), J)
    np.save(os.path.join(RES, f"screen_donor_rate_{split}.npy"), flip)
    with open(os.path.join(RES, f"screen_{split}.json"), "w") as f:
        json.dump(res, f)
    top = np.dstack(np.unravel_index(np.argsort(-J.ravel())[:8], J.shape))[0]
    print(f"screen {split}: clean acc={a0:.3f}; top J heads (layer,head,value):")
    for l, h in top:
        print(f"  L{l}H{h}: J={J[l,h]:+.4f} donor_rate={flip[l,h]:.3f}")
    print(f"  L20H2: J={J[20,2]:+.4f} donor_rate={flip[20,2]:.3f}; "
          f"max|J|={np.abs(J).max():.4f}; std={J.std():.4f}")
    return res


def state_retention(stream, model, distances=(0, 64, 128, 256, 512, 1024), split="validation"):
    """Relative Frobenius donor-recipient state difference per head at query boundary."""
    out = {}
    for d in distances:
        srcs = srcs_for(stream, DEV_OFFSETS, f"dist{d}", distance=d if d > 0 else None,
                        nf=d)
        ids_r = pad_batch([s["ids"] for s in srcs], model.device)
        ids_d = pad_batch([s["donor_ids"] for s in srcs], model.device)
        cut = srcs[0]["query_key_pos"]
        key = ids_r[:, cut]
        pre_r = model.prefill(ids_r[:, :cut], cut=cut, capture_states=True, capture_g=True)
        pre_d = model.prefill(ids_d[:, :cut], cut=cut, capture_states=True)
        _, p_r = model.query_token(key, pre_r, return_pre=True)
        _, p_d = model.query_token(key, pre_d, return_pre=True)
        rel = np.zeros((LAYERS, H))
        for l in range(LAYERS):
            num = (p_d["states"][l] - p_r["states"][l]).flatten(2).norm(dim=2).cpu().numpy()
            den = p_r["states"][l].flatten(2).norm(dim=2).cpu().numpy() + 1e-12
            rel[l] = (num / den).mean(0)
        tau_eff = (np.log(2.0) / (-pre_r["gmean"].cpu().numpy())).astype(float)
        out[str(d)] = {"rel": rel.tolist(), "tau_eff": tau_eff.tolist()}
        top = np.dstack(np.unravel_index(np.argsort(-rel.ravel())[:5], rel.shape))[0]
        print(f"distance {d}: top retention " +
              ", ".join(f"L{l}H{h}={rel[l,h]:.4f}" for l, h in top))
    with open(os.path.join(RES, f"state_retention_{split}.json"), "w") as f:
        json.dump(out, f)
    return out


def write_interventions(stream, tok, model, head, split="validation"):
    """Block / donor / norm-matched-random writes over the fact span, native LM head."""
    srcs = srcs_for(stream, DEV_OFFSETS, split)
    cand = cand_ids(tok).to(model.device)
    ids_r = pad_batch([s["ids"] for s in srcs], model.device)
    ids_d = pad_batch([s["donor_ids"] for s in srcs], model.device)
    l, h = head
    spans = srcs[0]["spans"]
    dw = model.prefill(ids_d, capture_writes={"layer": l, "head": h, "spans": spans})["writes"]
    offs = [0]
    for a, b in spans[:-1]:
        offs.append(offs[-1] + (b - a))
    seg = lambda rep: [(a, b, rep[:, offs[i]:offs[i] + (b - a)]) for i, (a, b) in enumerate(spans)]
    g = torch.Generator(device=model.device).manual_seed(0)
    rnd = torch.randn(dw.shape, generator=g, device=model.device)
    rnd = rnd / rnd.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-9) * \
        dw.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    conds = {
        "clean": None,
        "block": {"layer": l, "head": h, "segments": [(a, b, None) for a, b in spans]},
        "donor": {"layer": l, "head": h, "segments": seg(dw)},
        "random": {"layer": l, "head": h, "segments": seg(rnd)},
    }
    colon = srcs[0]["colon_pos"]
    pre_r = model.prefill(ids_r[:, :colon], cut=colon, capture_states=True)
    pre_d = model.prefill(ids_d[:, :colon], cut=colon, capture_states=True)
    Sr = pre_r["states"][l][:, h]
    Sd = pre_d["states"][l][:, h]
    dS = Sd - Sr
    res = {}
    for name, iv in conds.items():
        lg = model.prefill(ids_r, logits_pos=ids_r.shape[1] - 1, intervene=iv)["logits"]
        a, pred, tgt = acc(lg, srcs, cand)
        m = margins(lg, srcs, cand)
        pre_i = model.prefill(ids_r[:, :colon], cut=colon, capture_states=True, intervene=iv)
        S = pre_i["states"][l][:, h]
        alpha = ((S - Sr) * dS).flatten(1).sum(1) / dS.flatten(1).pow(2).sum(1).clamp_min(1e-12)
        rel = (S - Sr).flatten(1).norm(dim=1) / Sr.flatten(1).norm(dim=1).clamp_min(1e-12)
        dr = (S - Sr).flatten(1).norm(dim=1)
        dd = (S - Sd).flatten(1).norm(dim=1)
        ratio = dr / (dr + dd).clamp_min(1e-12)
        res[name] = {"acc": a, "margin": float(m.mean()),
                     "donor_rate": float((pred == np.array(
                         [task.VALUE4.index(s["donor_value"]) for s in srcs])).mean()),
                     "alpha": float(alpha.mean()), "rel_state": float(rel.mean()),
                     "d_recipient": float(dr.mean()), "d_donor": float(dd.mean()),
                     "closer_ratio": float(ratio.mean())}
        print(f"write {name}: acc={a:.3f} donor-margin={m.mean():+.4f} "
              f"donor-rate={res[name]['donor_rate']:.3f} alpha={alpha.mean():+.3f} "
              f"rel={rel.mean():.4f} d_r={dr.mean():.4f} d_d={dd.mean():.4f} "
              f"ratio={ratio.mean():.3f}")
    res["head"] = [l, h]
    with open(os.path.join(RES, f"write_{split}_L{l}H{h}.json"), "w") as f:
        json.dump(res, f)
    return res


def load_sweep(stream, tok, model, split="validation"):
    res = {}
    for K in [1, 2, 4, 8, 16]:
        total = 64 + 3 * 4 * 4 + FILLER + 2
        srcs = srcs_for(stream, DEV_OFFSETS, split, K=K, fixed_total=total)
        ids = pad_batch([s["ids"] for s in srcs], model.device)
        lg = model.prefill(ids, logits_pos=ids.shape[1] - 1)["logits"]
        cand = cand_ids(tok).to(model.device)
        a, _, _ = acc(lg, srcs, cand)
        m = margins(lg, srcs, cand)
        res[K] = {"acc": a, "margin": float(m.mean()), "len": int(ids.shape[1])}
        print(f"load K={K}: acc={a:.3f} margin={m.mean():+.4f} len={ids.shape[1]}")
    with open(os.path.join(RES, f"load_{split}.json"), "w") as f:
        json.dump(res, f)
    return res


def distance_sweep(stream, tok, model, split="validation"):
    res = {}
    for d in [128, 256, 512, 1024]:
        srcs = srcs_for(stream, DEV_OFFSETS, split, distance=d)
        ids = pad_batch([s["ids"] for s in srcs], model.device)
        lg = model.prefill(ids, logits_pos=ids.shape[1] - 1)["logits"]
        cand = cand_ids(tok).to(model.device)
        a, _, _ = acc(lg, srcs, cand)
        m = margins(lg, srcs, cand)
        res[d] = {"acc": a, "margin": float(m.mean()), "len": int(ids.shape[1])}
        print(f"distance {d}: acc={a:.3f} margin={m.mean():+.4f} len={ids.shape[1]}")
    with open(os.path.join(RES, f"distance_{split}.json"), "w") as f:
        json.dump(res, f)
    return res


def load_state(stream, model, head, split="validation"):
    """Target-specific state retention at fixed sequence length as K grows."""
    l, h = head
    res = {}
    for K in [1, 2, 4, 8, 16]:
        total = 64 + 3 * 4 * 4 + FILLER + 2
        srcs = srcs_for(stream, DEV_OFFSETS, "load", K=K, fixed_total=total)
        ids_r = pad_batch([s["ids"] for s in srcs], model.device)
        ids_d = pad_batch([s["donor_ids"] for s in srcs], model.device)
        cut = srcs[0]["query_key_pos"]
        pr = model.prefill(ids_r[:, :cut], cut=cut, capture_states=True)
        pd_ = model.prefill(ids_d[:, :cut], cut=cut, capture_states=True)
        key = ids_r[:, cut]
        _, pr = model.query_token(key, pr, return_pre=True)
        _, pd_ = model.query_token(key, pd_, return_pre=True)
        Sr, Sd = pr["states"][l][:, h], pd_["states"][l][:, h]
        rel = (Sd - Sr).flatten(1).norm(dim=1) / Sr.flatten(1).norm(dim=1).clamp_min(1e-12)
        res[K] = {"rel_state": float(rel.mean()), "len": int(ids_r.shape[1])}
        print(f"load-state K={K}: rel={rel.mean():.4f} len={ids_r.shape[1]}")
    with open(os.path.join(RES, f"load_state_{split}_L{l}H{h}.json"), "w") as f:
        json.dump(res, f)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["baseline", "screen", "retention", "write", "load", "distance", "load_state"])
    ap.add_argument("--head", default="20,2")
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()
    tok = task.tokenizer()
    stream = task.load_stream(args.split)
    model = GDNModel(device="cuda")
    if args.step == "baseline":
        out, _ = baseline(stream, tok, model, args.split, DEV_OFFSETS)
        with open(os.path.join(RES, f"baseline_{args.split}.json"), "w") as f:
            json.dump(out, f)
    elif args.step == "screen":
        screen(stream, tok, model, args.split)
    elif args.step == "retention":
        state_retention(stream, model, split=args.split)
    elif args.step == "write":
        l, h = [int(x) for x in args.head.split(",")]
        write_interventions(stream, tok, model, (l, h), args.split)
    elif args.step == "load_state":
        l, h = [int(x) for x in args.head.split(",")]
        load_state(stream, model, (l, h), args.split)
    elif args.step == "load":
        load_sweep(stream, tok, model, args.split)
    elif args.step == "distance":
        distance_sweep(stream, tok, model, args.split)


if __name__ == "__main__":
    main()
