"""Train typed Logic Engine: 20k updates, batch 8, AdamW 2e-4/wd 1e-4, clip 1.0, bf16.

Single-transition context, teacher forcing on the 421 next-state tokens.
Sequences are precomputed (int16): prefix(447) + target(421) = 868.
"""
import argparse, json, time
import numpy as np
import torch

from codec import (build_prefix, encode_state_tokens, PREFIX_LEN, SEQ_LEN,
                   N_STATE_TOKENS)
from gen_data import load_split, unpack_fields
from model_typed import LogicEngine, count_params


def build_sequences(train, max_trans=None):
    seqs = np.empty((sum(len(e["fields_t"]) for e in train), SEQ_LEN), np.int16)
    i = 0
    for ep in train:
        for t in range(len(ep["fields_t"])):
            spawns = [int(c) for c in ep["spawns"][t] if c >= 0]
            pref = build_prefix(unpack_fields(ep["fields_t"][t]),
                                ep["actions"][t], spawns)
            tgt = encode_state_tokens(unpack_fields(ep["fields_next"][t]))
            seqs[i, :PREFIX_LEN] = pref
            seqs[i, PREFIX_LEN:] = tgt
            i += 1
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/train.npz")
    ap.add_argument("--out", default="ckpt_typed.pt")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=5000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = load_split(args.data)
    seqs = build_sequences(train)
    print(f"{len(seqs)} transitions, seq len {SEQ_LEN}", flush=True)

    model = LogicEngine().to(dev)
    print(f"params: {count_params(model)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.default_rng(args.seed)
    losses = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.from_numpy(seqs[rng.integers(0, len(seqs), args.batch)].astype(np.int64)).to(dev)
        x, y = idx[:, :-1], idx[:, 1:]
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits[:, PREFIX_LEN - 1:].reshape(-1, logits.shape[-1]).float(),
                y[:, PREFIX_LEN - 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0:
            print(f"step {step:6d} loss {np.mean(losses[-args.log_every:]):.4f} "
                  f"({(time.time()-t0)/step:.3f}s/it)", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save(dict(model=model.state_dict(), args=vars(args), step=step),
                       args.out)
    json.dump(losses, open(args.out.replace(".pt", "_losses.json"), "w"))


if __name__ == "__main__":
    main()
