#!/usr/bin/env python3
"""Judge-gate net-gain criterion, computed from REA paper Table 14.

Derivation (ours, this redo): every policy return is a consulted episode.
Let s = first-run success share, f = 1 - s. The judge escalates a fraction
e_f of failures and (falsely) a fraction e_s of successes. Escalated failures
are converted at rate r (rescue); re-run successes are destroyed at rate h.
Net gain per consulted episode (in success probability):

    Delta(s) = f * e_f * r  -  s * e_s * h

Break-even first-run success rate:  s* = e_f*r / (e_f*r + e_s*h).
Optimal judge operating point along its ROC: d(e_s)/d(e_f) = f*r / (s*h).

All inputs are transcribed from REA (arXiv 2610.42053) Table 14 and the
final-verdict confusion matrices (Appendix B.5/B.6); replay variance is
reported separately and excluded from the rate arithmetic.
"""
import json

# --- LIBERO-Pro seeds 1-10 pool (590 episodes, ASPIRE Fix library) ---
pool = {
    "episodes": 590,
    "first_run_success": 0.719,          # REA-M proxy outcome on the same pool
    "failures": 166,                     # first-run failures among the 590
    "escalated_failures": 28 + 117,      # rescued + not rescued
    "rescued": 28,
    "rerun_successes": 198 + 19,         # survived + harmed
    "harmed": 19,
    "paired_wins": 37,                   # vs REA-M (28 rescued + 9 replay variance)
    "paired_losses": 19,
    "replay_variance_wins": 9,
}
# --- matched hard subset (72 episodes) ---
hard = {
    "episodes": 72,
    "first_run_success": 37 / 72,        # REA-M on the matched subset (51.4%)
    "failures": 35,
    "escalated_failures": 7 + 22,
    "rescued": 7,
    "rerun_successes": 20 + 1,
    "harmed": 1,
    "paired_wins": 12,                   # 7 rescued + 5 replay variance
    "paired_losses": 1,
    "replay_variance_wins": 5,
}
# --- exception-gate structural recall ceiling ---
exception = {"episodes": 590, "failures": 166, "escalations": 6}


def derive(pool: dict) -> dict:
    f = 1 - pool["first_run_success"]
    s = pool["first_run_success"]
    e_f = pool["escalated_failures"] / pool["failures"]
    e_s = pool["rerun_successes"] / (pool["episodes"] * s)
    r = pool["rescued"] / pool["escalated_failures"]
    h = pool["harmed"] / pool["rerun_successes"]
    gain = f * e_f * r - s * e_s * h
    s_star = e_f * r / (e_f * r + e_s * h)
    slope = f * r / (s * h)
    net_episodes = pool["rescued"] - pool["harmed"]
    net_with_replay = pool["paired_wins"] - pool["paired_losses"]
    return {
        "s": round(s, 4), "f": round(f, 4),
        "e_f": round(e_f, 4), "e_s": round(e_s, 4),
        "r": round(r, 4), "h": round(h, 4),
        "model_gain_pp": round(100 * gain, 2),
        "model_gain_episodes": round(gain * pool["episodes"], 1),
        "observed_net_episodes": net_episodes,
        "observed_net_with_replay": net_with_replay,
        "observed_net_pp": round(100 * net_episodes / pool["episodes"], 2),
        "observed_net_with_replay_pp": round(100 * net_with_replay / pool["episodes"], 2),
        "break_even_s": round(s_star, 4),
        "optimal_roc_slope": round(slope, 3),
    }


out = {
    "libero_590": derive(pool),
    "matched_72": derive(hard),
    "exception_gate_recall_ceiling": round(
        exception["escalations"] / exception["failures"], 4),
    "note": "rates from REA Table 14 (proxy attribution, replay variance excluded "
            "from r/h; paper calls the attribution a consistent explanation, not a "
            "tested mechanism)",
}
with open("criterion.json", "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
