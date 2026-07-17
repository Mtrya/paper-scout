#!/usr/bin/env python3
"""
Probe 3: Dissect the AutoResearch loop evidence in GigaWorld-Policy-0.5.

Data sources (paper v1):
  Table 5  - 1K-step pilot LR sweep (train action/visual loss, eval action MSE)
  Figure 7 - Panel A pilot sweep bars (observe/keep/discard) + Panel B long-run
             progression 10k -> 30k -> 50k with keep/discard decisions.

What we check:
  1. The candidate set is sequential and adaptive, not a grid: a baseline is
     OBSERVED first, then LRs around it, then batch size at fixed best LR,
     then a frames=64 probe that was abandoned early.
  2. The odd 4.316e-5 in Table 5 == the gray "observe (baseline) lr=4.3e-5"
     bar in Fig 7 Panel A (both eval 0.449). It is the INHERITED baseline
     recipe, not an agent-invented value.
  3. Multi-objective tension: 3e-5 minimizes train visual loss (0.172330) but
     6e-5 minimizes train action loss (0.252476) and eval action MSE (0.409764).
     The agent's selection rule (action metric wins) is a genuine judgment call
     the paper makes explicit.
  4. Panel B: the 30K checkpoint selection is itself a keep/discard decision
     (50K regressed 0.000113 -> 0.000173 and was discarded = early stopping
     executed by the agent loop).
"""
import json

# Table 5 (paper): 1K-step pilots, same init/data/optimizer/protocol
pilots = [
    # lr, train_action_loss, train_visual_loss, eval_action_mse, figure7_verdict
    (3.0e-5, 0.257300, 0.172330, 0.416832, "keep"),
    (4.316e-5, 0.256593, 0.172893, 0.449387, "observe (baseline)"),
    (6.0e-5, 0.252476, 0.173318, 0.409764, "keep -> SELECTED"),
    (8.0e-5, 0.261832, 0.175986, 0.461381, "discard"),
]

# Figure 7 Panel A extra candidates (eval_action_mse @1k)
extra_candidates = [
    ("lr=6e-5, bs=12", 0.572, "discard"),
    ("lr=6e-5, bs=8", 0.556, "discard"),
    ("lr=6e-5, frames=64", None, "stopped early (slow steps)"),
]

# Figure 7 Panel B: long run at selected config (lr=6e-5, bs=16)
long_run = [(10_000, 0.000507, None), (30_000, 0.000113, "keep (best ckpt)"), (50_000, 0.000173, "discard: regressed")]


def main():
    print("=" * 78)
    print("AutoResearch pilot sweep (1K steps each) - Table 5 x Fig 7 Panel A")
    print("=" * 78)
    print(f"{'lr':>10s} | {'train_act':>9s} | {'train_vis':>9s} | {'eval_act_mse':>12s} | verdict")
    for lr, ta, tv, em, v in pilots:
        print(f"{lr:>10.3e} | {ta:9.6f} | {tv:9.6f} | {em:12.6f} | {v}")

    best_act = min(pilots, key=lambda p: p[1])
    best_vis = min(pilots, key=lambda p: p[2])
    best_eval = min(pilots, key=lambda p: p[3])
    print(f"\nbest train action loss : lr={best_act[0]:.3e} ({best_act[1]:.6f})")
    print(f"best train visual loss : lr={best_vis[0]:.3e} ({best_vis[2]:.6f})  <- disagrees on objective!")
    print(f"best eval action MSE   : lr={best_eval[0]:.3e} ({best_eval[3]:.6f})  <- selection criterion used")
    print("\nTradeoff: raising LR 3e-5 -> 6e-5 improves action loss "
          f"({best_vis[1]-best_act[1]:+.6f}) and eval MSE ({best_vis[3]-best_act[3]:+.6f})")
    print(f"but degrades visual loss ({best_act[2]-best_vis[2]:+.6f}). The agent (per paper)")
    print("prioritizes action quality because it 'is more directly related to downstream")
    print("policy execution' - an encoded preference, not a Pareto-optimal answer.")

    print("\nIs 4.316e-5 an agent-invented grid point? Cross-check with Fig 7 Panel A:")
    print("  Panel A gray 'observe (baseline)' bar: lr=4.3e-5, bs=16, eval 0.449")
    print("  Table 5 row lr=4.316e-5:              eval 0.449387 -> SAME RUN.")
    print("  => 4.316e-5 is the inherited baseline recipe (likely from GWP-0),")
    print("     observed first; the agent then proposed 3e-5/6e-5/8e-5 around it.")

    print("\nSequential structure (not a grid):")
    seq = ["observe baseline (lr=4.3e-5, bs16)",
           "sweep lr in {3e-5, 6e-5, 8e-5} at bs=16 -> select 6e-5",
           "sweep bs in {12, 8} at lr=6e-5 -> both WORSE (0.572, 0.556), keep bs=16",
           "probe frames=64 -> stopped early (slow steps)",
           "extend best config to 10k/30k/50k -> keep 30k, discard 50k (regression)"]
    for i, s in enumerate(seq, 1):
        print(f"  {i}. {s}")

    print("\nLong-run progression (Fig 7 Panel B), eval_action_mse:")
    for step, mse, verdict in long_run:
        print(f"  {step:>6,d} steps : {mse:.6f}  {verdict or ''}")
    imp = (long_run[0][1] - long_run[1][1]) / long_run[0][1] * 100
    reg = (long_run[2][1] - long_run[1][1]) / long_run[1][1] * 100
    print(f"  10k->30k: {imp:.1f}% improvement; 30k->50k: +{reg:.1f}% regression -> early stop at 30k")
    print("\n  NOTE unit shift: pilot eval MSE ~0.41 vs long-run ~1e-4. The paper does not")
    print("  explain the rescaling (likely different normalization/eval protocol between the")
    print("  1K-step pilot harness and the long-run harness). Worth flagging, not fatal.")

    verdict = {
        "is_grid_search": False,
        "evidence": [
            "candidates depend on prior results (LR sweep -> bs sweep at best LR -> steps)",
            "explicit keep/discard/early-stop decisions, incl. abandoning frames=64 probe",
            "30K checkpoint chosen by regression detection at 50K",
        ],
        "but_shallow": [
            "only 3 effective LR candidates + 2 batch sizes; a human could run this sweep",
            "no code/architecture mutations shown - search stays inside hyperparameter space",
            "multi-objective conflict (visual vs action loss) resolved by a hard-coded preference",
        ],
    }
    with open("probe_outputs/autoresearch_analysis.json", "w") as f:
        json.dump({"pilots": pilots, "extra": extra_candidates, "long_run": long_run, "verdict": verdict}, f, indent=2)
    print("\nwrote probe_outputs/autoresearch_analysis.json")


if __name__ == "__main__":
    main()
