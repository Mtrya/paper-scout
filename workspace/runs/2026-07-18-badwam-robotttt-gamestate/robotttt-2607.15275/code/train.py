"""Train one probe model and evaluate action-prediction error vs context position.

Recurrent models (ttt, delta) are trained with TBPTT exactly in the spirit of
the paper: the sequence is split into segments, gradients flow within a
segment, and the fast-weight/recurrent state is carried across segments but
detached at the boundary.
"""
import argparse, json, time
import torch
import torch.nn.functional as F

from task import sample_episode_batch
from models import Policy, NoContextPolicy, detach_state

from task import PHASE

# context buckets (total elapsed steps t). Metric positions: steps whose TRUE
# action is unsaturated (max|a| < 0.95) -- the regime where (g, k)
# identification actually matters (saturated steps are trivially predictable
# from a_{t-1} and would drown the signal).
BUCKETS = [(8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 384),
           (384, 512)]
SAT = 0.95


def evaluate(model, mixer, T=512, episodes=64, batch=32, device="cpu", seed=999):
    gen = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    se_sum, act_sum, cnt = 0.0, 0.0, 0.0
    bucket_se = {b: 0.0 for b in BUCKETS}
    bucket_av = {b: 0.0 for b in BUCKETS}
    bucket_n = {b: 0 for b in BUCKETS}
    with torch.no_grad():
        for _ in range(episodes // batch):
            tokens, act = sample_episode_batch(batch, T, device=device, gen=gen)
            pred, _ = model(tokens, create_graph=False)
            se = ((pred - act) ** 2).mean(-1)     # (B, T)
            a2 = (act ** 2).mean(-1)              # (B, T)
            se_sum += se.sum().item(); act_sum += a2.sum().item()
            cnt += se.numel()
            mask = (act.abs().amax(-1) < SAT)     # unsaturated steps only
            for b in BUCKETS:
                m = mask.clone()
                m[:, :b[0]] = False
                m[:, b[1]:] = False
                bucket_se[b] += se[m].sum().item()
                bucket_av[b] += a2[m].sum().item()
                bucket_n[b] += int(m.sum().item())
    r2 = {}
    for b in BUCKETS:
        nmse = bucket_se[b] / max(bucket_av[b], 1e-12)
        r2[f"{b[0]}-{b[1]}"] = 1.0 - nmse
    return {
        "global_nmse": se_sum / max(act_sum, 1e-12),
        "r2_by_context_bucket": r2,
        "bucket_counts": {f"{b[0]}-{b[1]}": bucket_n[b] for b in BUCKETS},
        "metric": "NMSE over unsaturated steps (max|a|<0.95), per context bucket",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", required=True,
                    choices=["ttt", "delta", "attn", "noctx"])
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed + 1)

    if args.mixer == "noctx":
        model = NoContextPolicy(args.d)
    else:
        model = Policy(args.mixer, d=args.d, layers=args.layers)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    t0 = time.time()
    hist = []
    model.train()
    for it in range(1, args.iters + 1):
        tokens, act = sample_episode_batch(args.batch, args.T, gen=gen)
        opt.zero_grad(set_to_none=True)
        if args.mixer in ("ttt", "delta"):
            states, loss_tot = None, 0.0
            nseg = (args.T + args.seg - 1) // args.seg
            for s0 in range(0, args.T, args.seg):
                pred, states = model(tokens[:, s0:s0 + args.seg], states,
                                     create_graph=True)
                loss = F.mse_loss(pred, act[:, s0:s0 + args.seg])
                (loss / nseg).backward()
                states = [detach_state(s) for s in states]
                loss_tot += loss.item() / nseg
        else:
            pred, _ = model(tokens)
            loss_tot = F.mse_loss(pred, act)
            loss_tot.backward()
            loss_tot = loss_tot.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 20 == 0 or it == 1:
            hist.append({"iter": it, "loss": loss_tot})
            print(f"[{args.mixer}] it {it}/{args.iters} loss {loss_tot:.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    res = evaluate(model, args.mixer, T=512)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)},
               args.out.replace(".json", ".pt"))
    res.update({
        "mixer": args.mixer, "T_train": args.T, "iters": args.iters,
        "batch": args.batch, "seg": args.seg, "lr": args.lr, "d": args.d,
        "layers": args.layers, "params": nparams,
        "train_time_s": time.time() - t0, "history": hist,
    })
    with open(args.out, "w") as f:
        json.dump(res, f)
    r2last = res["r2_by_context_bucket"]["384-512"]
    print(f"saved {args.out}; params={nparams}; R2@384-512={r2last:.4f}", flush=True)


if __name__ == "__main__":
    main()
