"""C4: module-replacement experiments (which parameters carry the RL gain)."""
import argparse
import json
import os

import torch

from rollout import run_episodes
from wutil import load_policy, ts_group


def mixed_state(base_pol, src_pol, take_ts):
    """base_pol's weights, with the TS modules (or the non-TS modules) taken
    from src_pol."""
    out = {k: v.clone() for k, v in base_pol.state_dict().items()}
    src = src_pol.state_dict()
    for k in out:
        if ts_group(k) == take_ts:
            out[k] = src[k].clone()
    return out


def low_rank_reconstructions(base_pol, rl_pol, rank):
    """BC + the top-`rank` singular directions of the RL update on TS matrices."""
    out = {k: v.clone() for k, v in base_pol.state_dict().items()}
    rl = rl_pol.state_dict()
    for k in out:
        if not ts_group(k):
            continue
        d = rl[k].float() - out[k].float()
        if d.dim() == 2 and min(d.shape) > 1:
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            r = min(rank, S.numel())
            rec = (U[:, :r] * S[:r]) @ Vh[:r]
            out[k] = out[k].float() + rec
        else:
            out[k] = rl[k].clone()
    return out


def complement_reconstructions(base_pol, rl_pol, rank):
    """RL TS modules with the top-`rank` singular directions of their update
    removed (i.e. base + the remaining directions)."""
    out = {k: v.clone() for k, v in base_pol.state_dict().items()}
    rl = rl_pol.state_dict()
    for k in out:
        if not ts_group(k):
            continue
        d = rl[k].float() - out[k].float()
        if d.dim() == 2 and min(d.shape) > 1:
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            r = min(rank, S.numel())
            rec = (U[:, r:] * S[r:]) @ Vh[r:]
            out[k] = out[k].float() + rec
        else:
            out[k] = out[k].clone()
    return out


def eval_cond(sd, pred_mode, episodes, seed, K, env_kwargs):
    from model import FlowPolicy
    pol = FlowPolicy(pred_mode=pred_mode)
    pol.load_state_dict(sd)
    r = run_episodes(pol, episodes, seed, K=K, env_kwargs=env_kwargs)
    return float(r["success"].float().mean()), float(r["ret"].mean()), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--bc-name", type=str, default="M_bc")
    ap.add_argument("--rl-name", type=str, default="M_rl")
    ap.add_argument("--episodes", type=int, default=220)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--pred-mode", type=str, default="x")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--top-rank", type=int, default=4)
    ap.add_argument("--out", type=str, default="analysis/replace.json")
    args = ap.parse_args()

    env_kwargs = {"max_steps": args.max_steps}
    all_res = {}
    for d in args.dirs:
        bc = load_policy(os.path.join(d, args.bc_name + ".pt"), args.pred_mode)
        rl = load_policy(os.path.join(d, args.rl_name + ".pt"), args.pred_mode)
        seed = 1000 + int(os.path.basename(d.rstrip("/")).replace("seed", "")) * 1
        conds = {
            "bc": bc.state_dict(),
            "rl": rl.state_dict(),
            "ts_only": mixed_state(bc, rl, True),
            "nonts_only": mixed_state(rl, bc, True),
            "ts_top4": low_rank_reconstructions(bc, rl, args.top_rank),
            "ts_wo_top4": complement_reconstructions(bc, rl, args.top_rank),
            "bc_plus_ts_top4_bc": low_rank_reconstructions(
                bc, load_policy(os.path.join(d, "M_bc_disc.pt"), args.pred_mode),
                args.top_rank),
        }
        res = {}
        for name, sd in conds.items():
            if name == "bc_plus_ts_top4_bc" and not os.path.exists(
                    os.path.join(d, "M_bc_disc.pt")):
                continue
            s, ret, r = eval_cond(sd, args.pred_mode, args.episodes, seed,
                                  args.K, env_kwargs)
            res[name] = {"success": s, "ret": ret}
            print(f"{d} {name}: success {s:.3f} ret {ret:.2f}", flush=True)
        all_res[os.path.basename(d.rstrip("/"))] = res
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
