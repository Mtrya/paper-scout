"""CLI entry point: run one arm of the online-TTT comparison.

    python run_arm.py --arm frozen            --out-dir results --tag _v1
    python run_arm.py --arm loop-imitate      --out-dir results
    ...
"""
from __future__ import annotations

import argparse
from dataclasses import fields

import harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=[
        "frozen", "loop-imitate", "rft", "ascent", "rft-settlement",
        "loop-imitate-frozen-gen"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-tasks", type=int, default=160)
    ap.add_argument("--n-phases", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--buffer-tasks", type=int, default=8)
    ap.add_argument("--steps-per-update", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--nll-probe-n", type=int, default=50)
    ap.add_argument("--val-per-family", type=int, default=5)
    ap.add_argument("--max-tokens-per-microbatch", type=int, default=2048)
    ap.add_argument("--max-rows-per-microbatch", type=int, default=6)
    ap.add_argument("--fixed-gen-dir", default="")
    ap.add_argument("--max-seconds", type=float, default=3000.0)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    cfg = harness.Config(
        arm=args.arm, tag=args.tag, seed=args.seed, n_tasks=args.n_tasks,
        n_phases=args.n_phases, eval_every=args.eval_every,
        buffer_tasks=args.buffer_tasks, steps_per_update=args.steps_per_update,
        lr=args.lr, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        max_new_tokens=args.max_new_tokens, grad_clip=args.grad_clip,
        nll_probe_n=args.nll_probe_n, val_per_family=args.val_per_family,
        max_tokens_per_microbatch=args.max_tokens_per_microbatch,
        max_rows_per_microbatch=args.max_rows_per_microbatch,
        out_dir=args.out_dir, fixed_gen_dir=args.fixed_gen_dir,
        max_seconds=args.max_seconds, eval_only=args.eval_only,
    )
    harness.run_arm(cfg)


if __name__ == "__main__":
    main()
