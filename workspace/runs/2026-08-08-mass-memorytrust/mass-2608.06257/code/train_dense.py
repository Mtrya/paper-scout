"""Train dense-carrier baseline (independent-head CNN, ~2.85M).

Same data, same budget: 20k updates, batch 8, AdamW 2e-4/wd 1e-4, clip 1.0, bf16.
Loss: per-cell cross-entropy + heading CE + survival CE (grids built on the fly).
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F

from gen_data import load_split, unpack_fields
from model_dense import DenseCNN, fields_to_grid, fields_to_target, count_params


def make_batch(eps_trans, dev):
    xs, cs, hs, al = [], [], [], []
    for ep, t in eps_trans:
        spawns = [int(c) for c in ep["spawns"][t] if c >= 0]
        xs.append(fields_to_grid(unpack_fields(ep["fields_t"][t]),
                                 ep["actions"][t], spawns))
        c, h, a = fields_to_target(unpack_fields(ep["fields_next"][t]))
        cs.append(c); hs.append(h); al.append(a)
    return (torch.from_numpy(np.stack(xs)).to(dev),
            torch.from_numpy(np.stack(cs)).to(dev),
            torch.from_numpy(np.stack(hs)).to(dev),
            torch.from_numpy(np.stack(al)).to(dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/train.npz")
    ap.add_argument("--out", default="ckpt_dense.pt")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = load_split(args.data)
    index = [(ep, t) for ep in train for t in range(len(ep["fields_t"]))]
    print(f"{len(index)} transitions", flush=True)

    model = DenseCNN().to(dev)
    print(f"params: {count_params(model)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.default_rng(args.seed)
    losses = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        pick = [index[i] for i in rng.integers(0, len(index), args.batch)]
        x, c, h, a = make_batch(pick, dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            cell, head, alive = model(x)
            loss = F.cross_entropy(cell.float(), c) \
                + 0.1 * F.cross_entropy(head.float().reshape(-1, 4),
                                        h.reshape(-1)) \
                + 0.1 * F.cross_entropy(alive.float().reshape(-1, 2),
                                        a.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0:
            print(f"step {step:6d} loss {np.mean(losses[-args.log_every:]):.4f} "
                  f"({(time.time()-t0)/step:.3f}s/it)", flush=True)
        if step % 5000 == 0 or step == args.steps:
            torch.save(dict(model=model.state_dict(), args=vars(args), step=step),
                       args.out)
    json.dump(losses, open(args.out.replace(".pt", "_losses.json"), "w"))


if __name__ == "__main__":
    main()
