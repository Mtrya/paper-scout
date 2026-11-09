"""Train the paired ttt / ttt_last arms on the toy in-context task.

The arms are identical in architecture, parameter count, initialisation
(copied parameter-for-parameter), optimiser, training episodes (same generator
seed) and held-out evaluation episodes. The only difference is the inner loop:
"ttt" updates W1,b1,W2,b2; "ttt_last" updates only the readout W2, leaving
W1,b1,b2 as sequence-static, outer-trainable slow parameters and never
touching b2.

Implementation note: the inner loop is a sequential per-timestep PyTorch
loop, not the official chunk-parallel TTT kernel, and the fast weights carry
no weight norm.

One call trains both arms, evaluates T=512 and T=1024 R^2 on the same 64
held-out episodes, and writes the inner-loss / drift diagnostics into one
small JSON. Checkpoints (.pt) are written only when --save-ckpt-dir is given
(i.e. on the remote box).
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from models import Policy, TTTLayer, detach_state
from probe_eval import buckets_for, evaluate, inner_diag, sample_heldout
from task import sample_episode_batch


def build_model(mixer, args):
    torch.manual_seed(args.seed)
    return Policy(mixer, d=args.d, layers=args.layers)


def copy_params(model, ref_state, source):
    """Copy the first arm's parameters into the second arm, verify identity."""
    sd = model.state_dict()
    if sorted(sd) != sorted(ref_state):
        raise RuntimeError("parameter sets differ between arms")
    diff = max((sd[k].float() - ref_state[k].float()).abs().max().item()
               for k in sd)
    model.load_state_dict(ref_state)
    return {"identical_param_set": True, "copied_from": source,
            "max_abs_param_diff_before_copy": diff}


def grad_norms(model):
    out = {}
    for li, layer in enumerate(model.layers):
        if not isinstance(layer, TTTLayer):
            continue
        for name in ("W0_1", "b0_1", "W0_2", "b0_2", "eta", "eta_h"):
            p = getattr(layer, name)
            out[f"layer{li}.{name}"] = (None if p.grad is None
                                        else round(p.grad.norm().item(), 8))
    return out


def train_arm(mixer, args):
    model = build_model(mixer, args)
    init_check = {"identical_param_set": True, "copied_from": None,
                  "max_abs_param_diff_before_copy": 0.0}
    if args._ref_state is not None:
        init_check = copy_params(model, args._ref_state, args._ref_source)
        if init_check["max_abs_param_diff_before_copy"] != 0.0:
            raise RuntimeError("same-seed arm initialisation diverged")
    else:
        args._ref_state = {k: v.detach().clone()
                           for k, v in model.state_dict().items()}
        args._ref_source = mixer
    model.to(args.device)

    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    gen = torch.Generator(device=args.device).manual_seed(args.seed + 1)

    hist, first_grads = [], None
    t0 = time.time()
    model.train()
    for it in range(1, args.iters + 1):
        tokens, act = sample_episode_batch(args.batch, args.T,
                                           device=args.device, gen=gen)
        opt.zero_grad(set_to_none=True)
        states, loss_tot = None, 0.0
        nseg = -(-args.T // args.seg)
        for s0 in range(0, args.T, args.seg):
            pred, states = model(tokens[:, s0:s0 + args.seg], states,
                                 create_graph=True)
            loss = F.mse_loss(pred, act[:, s0:s0 + args.seg])
            (loss / nseg).backward()
            states = [detach_state(s) for s in states]
            loss_tot += loss.item() / nseg
        if it == 1:
            first_grads = grad_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % args.log_every == 0 or it == 1:
            hist.append({"iter": it, "loss": round(loss_tot, 6),
                         "elapsed_s": round(time.time() - t0, 2)})
            print(f"[{mixer}] it {it}/{args.iters} loss {loss_tot:.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    train_time = time.time() - t0

    ckpt_path = None
    if args.save_ckpt_dir:
        os.makedirs(args.save_ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_ckpt_dir, f"{mixer}_T{args.T}.pt")
        torch.save({"state_dict": {k: v.cpu() for k, v in
                                   model.state_dict().items()},
                    "args": {k: v for k, v in vars(args).items()
                             if not k.startswith("_")},
                    "mixer": mixer}, ckpt_path)

    return model, {
        "mixer": mixer,
        "params": nparams,
        "inner_update_keys": [layer.fast_keys() for layer in model.layers
                              if isinstance(layer, TTTLayer)],
        "init_check": init_check,
        "grad_norms_iter1": first_grads,
        "train_time_s": round(train_time, 2),
        "history": hist,
        "ckpt": ckpt_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["ttt", "ttt_last"],
                    choices=["ttt", "ttt_last"])
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--eval-seed", type=int, default=999)
    ap.add_argument("--eval-long", type=int, default=1024)
    ap.add_argument("--diag-T", type=int, default=512)
    ap.add_argument("--diag-batch", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-ckpt-dir", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args._ref_state = None
    args._ref_source = None

    held_T = max(args.T, args.eval_long)
    tokens, act = sample_heldout(args.eval_episodes, T=held_T,
                                 batch=args.eval_batch, seed=args.eval_seed,
                                 device=args.device)
    horizons = sorted({args.T, args.eval_long})

    arms = {}
    total0 = time.time()
    for mixer in args.arms:
        model, rec = train_arm(mixer, args)
        t0 = time.time()
        rec["eval"] = evaluate(model, tokens, act, horizons,
                               batch=args.eval_batch)
        rec["eval_time_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        rec["diag"] = inner_diag(model, tokens, act,
                                 T=min(args.diag_T, held_T),
                                 batch=args.diag_batch)
        rec["diag_time_s"] = round(time.time() - t0, 2)
        arms[mixer] = rec
        b = buckets_for(horizons[-1])[-1]
        last_b = f"{b[0]}-{b[1]}"
        print(f"[{mixer}] params={rec['params']} "
              f"R2@{last_b}={rec['eval'][str(horizons[-1])]['r2_by_context_bucket'].get(last_b)} "
              f"train={rec['train_time_s']}s", flush=True)

    if len({r["params"] for r in arms.values()}) > 1:
        raise RuntimeError("parameter counts differ between arms")

    res = {
        "probe": "paired TTT arms: all fast parameters vs readout-only (W2)",
        "config": {k: v for k, v in vars(args).items()
                   if not k.startswith("_") and k != "out"},
        "impl": {
            "mixers": list(arms),
            "inner_loop": "sequential PyTorch per-timestep loop",
            "official_chunk_parallel_kernel": False,
            "weight_norm": False,
            "note": ("Not the official chunk-parallel TTT kernel; no weight "
                     "norm on the fast weights. ttt_last updates only W2, "
                     "keeps W1/b1/b2 as sequence-static slow parameters and "
                     "never updates the readout bias b2."),
        },
        "heldout": {"episodes": args.eval_episodes, "T": held_T,
                    "eval_batch": args.eval_batch, "seed": args.eval_seed,
                    "horizons": horizons,
                    "note": "short horizons are prefixes of the same episodes"},
        "torch": torch.__version__,
        "wall_clock_s": round(time.time() - total0, 2),
        "arms": arms,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
