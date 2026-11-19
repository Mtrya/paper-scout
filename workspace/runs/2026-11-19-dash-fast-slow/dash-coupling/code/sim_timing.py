#!/usr/bin/env python
"""Event simulation of DASH Eq.(11)-(12) temporal alignment.

One reduction, used everywhere
------------------------------
Every alignment scheme answers one question: at execution time t, at which index
of chunk k do we read?  Call it A_k(t).  Chunk k's entry j holds the reference
trajectory at time s_k + j*T (s_k = the chunk's own observation/anchor time).
The executed command is

    u(t) = sum_k w_k interp(C_k, A_k(t)) / sum_k w_k,   0 <= A_k(t) <= H-1

with C_k an exact copy of the reference trajectory.  Every error in u is then a
time-bookkeeping error and nothing else; prediction error is held at zero on
purpose so the arms differ by semantics, not by model quality.  Chunk arrays are
never resampled repeatedly: differences are semantic, not interpolation blur.

Because all chunks in an arm claim a definite absolute time, the ensemble's
effective time is also measurable without cross-correlation:

    t_eff(t) = sum_k w_k (a_k + A_k(t) * T) / sum_k w_k
    lead(t)  = t_eff(t) - t                        [ticks]

For a correct arm every chunk agrees and lead == 0.  A positive lead means the
ensemble is reading predictions for a future time (fast motion), negative means
stale (slow motion).

Three candidate semantics for Eq.(12)
-------------------------------------
S_k is chunk k's write position on the output timeline in ticks from a_0; the
reader sits at P(t) = (t - a_0)/T.

  (i)   position:  A_k(t) = P(t) - S_k          = (t - a_k)/T
  (ii)  shift:     A_k(t) = (t - a_k)/T + S_k   [reading "1.2" as the shift]
  (iii) integer:   A_k(t) = (t - a_k)/T + sum_{m>k}(round(rho_m) - rho_m)

(i) is exact and is what the numbers 0.6, 1.2 actually are (positions).  (ii) is
the surface reading: at the second arrival the table says 1.2, the chunk's age is
0.6, and the difference is applied.  (iii) is the class Eq.(12) replaces: an
integer running clock instead of a fractional one.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

T_EXEC = 0.05           # 20 Hz, DASH's action-path rate
HORIZON = 16            # chunk length in ticks
SEED = 20261119
DECAY = 0.10            # ACT-style exponential ensemble weight per tick of age


# --------------------------------------------------------------------------
# reference trajectories
# --------------------------------------------------------------------------
def min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def analytic_trajectory(t_end=7.0, dt=0.001):
    """Piecewise minimum-jerk pick-and-place in 2-D: two dwells, a slow place
    and a fast transfer, so a time warp shows up in position, speed and jerk."""
    keys = [
        (0.0, (0.00, 0.00)),
        (0.4, (0.00, 0.00)),
        (1.6, (0.30, 0.10)),
        (2.0, (0.30, 0.10)),
        (2.6, (0.30, 0.35)),
        (3.1, (0.30, 0.35)),
        (4.3, (0.05, 0.30)),
        (4.7, (0.05, 0.30)),
        (5.1, (0.05, 0.05)),
        (6.9, (0.05, 0.05)),
    ]
    ts = np.arange(0.0, t_end + dt, dt)
    out = np.zeros((len(ts), 2))
    seg = 1
    for i, t in enumerate(ts):
        while seg < len(keys) - 1 and t > keys[seg][0]:
            seg += 1
        t0, p0 = keys[seg - 1]
        t1, p1 = keys[seg]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        out[i] = np.array(p0) + min_jerk(u) * (np.array(p1) - np.array(p0))
    return ts, out


def aloha_trajectory(path, dt=0.001):
    """LeRobot aloha_sim_transfer_cube_human episode 0: 14-D joint actions at 50 Hz.

    Real teleoperated two-arm transfer; the left gripper closes through 0.5 at
    t~4.66 s and opens at t~5.94 s, so a time shift is visible as an early or
    late grasp and not only as a smooth-trajectory error.
    """
    rows = json.load(open(path))
    ep = [r for r in rows if r["episode_index"] == 0]
    A = np.array([r["action"] for r in ep], dtype=np.float64)
    t = np.array([r["timestamp"] for r in ep], dtype=np.float64)
    ts = np.arange(0.0, t[-1] + dt, dt)
    out = np.stack([np.interp(ts, t, A[:, d]) for d in range(A.shape[1])], axis=1)
    return ts, out


class Ref:
    def __init__(self, ts, xs):
        self.ts, self.xs = ts, xs
        self.rms_vel = float(np.sqrt(np.mean(np.sum(np.diff(xs, axis=0) ** 2, axis=1))))

    def __call__(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        return np.stack([np.interp(t, self.ts, self.xs[:, d]) for d in range(self.xs.shape[1])], axis=1)


# --------------------------------------------------------------------------
# latency schedules
# --------------------------------------------------------------------------
def rho_const(n, rho=0.6):
    return np.full(n, rho)


def rho_jitter(n, seed=SEED):
    rng = np.random.default_rng(seed)
    base = 1.0 + 0.30 * np.sin(2 * np.pi * np.arange(n) / 7.0)
    return np.clip(base + 0.15 * rng.standard_normal(n), 0.5, 1.6)


def rho_cross(n, seed=SEED):
    """Alternates across the integer and drifts slowly upward."""
    k = np.arange(n)
    rng = np.random.default_rng(seed + 1)
    return 1.0 + 0.10 * ((-1.0) ** k) + 0.0008 * k + 0.02 * rng.standard_normal(n)


SCHEDULES = {
    "const0.6": lambda n: rho_const(n, 0.6),
    "jitter": rho_jitter,
    "cross": rho_cross,
}


def chunks_from_rho(rho, a0, horizon, T, ref, latency=0.0):
    # rho is in ticks, a is in seconds
    a = a0 + T * np.concatenate([[0.0], np.cumsum(rho[1:])])
    return [{"k": k, "s": float(a[k] - latency), "a": float(a[k]),
             "C": ref(a[k] - latency + np.arange(horizon) * T).astype(np.float64)}
            for k in range(len(rho))]


def chunks_from_anchors(s, L, horizon, T, ref):
    return [{"k": k, "s": float(s[k]), "a": float(s[k] + L[k]),
             "C": ref(s[k] + np.arange(horizon) * T).astype(np.float64)}
            for k in range(len(s))]


def eq12_table(rho):
    """S_k, n_k, f_k exactly as Eq.(12)."""
    S, prev_floor = 0.0, 0
    Ss, ns, fs = [0.0], [0], [0.0]
    for k in range(1, len(rho)):
        S += float(rho[k])
        n = int(math.floor(S)) - prev_floor
        prev_floor = int(math.floor(S))
        Ss.append(S)
        ns.append(n)
        fs.append(S - math.floor(S))
    return np.array(Ss), np.array(ns), np.array(fs)


class Ctx:
    """Precomputed per-chunk tables shared by all arms."""

    def __init__(self, chunks, rho):
        self.a = np.array([c["a"] for c in chunks])
        self.s = np.array([c["s"] for c in chunks])
        self.k = np.array([c["k"] for c in chunks])
        self.S, self.n, self.f = eq12_table(rho)
        self.rho = np.asarray(rho, dtype=np.float64)
        self.cum = {
            "ceil": np.concatenate([[0.0], np.cumsum(np.ceil(self.rho[1:]) - self.rho[1:])]),
            "floor": np.concatenate([[0.0], np.cumsum(np.floor(self.rho[1:]) - self.rho[1:])]),
            "fixed": np.concatenate([[0.0], np.cumsum(float(round(float(self.rho.mean()))) - self.rho[1:])]),
        }

    def last_arrived(self, t):
        return int(np.searchsorted(self.a, t, side="right")) - 1


def arm_indices(arm, ctx, t):
    dt = T_EXEC
    Nt = ctx.last_arrived(t)
    if arm == "exact_anchor":
        return (t - ctx.s) / dt
    if arm == "exact_arrival":
        return (t - ctx.a) / dt
    if arm == "shiftpos":
        return (t - ctx.a) / dt + ctx.S
    if arm in ("ceil_clock", "floor_clock", "fixed_clock"):
        key = arm.split("_")[0]
        E = ctx.cum[key]
        return (t - ctx.a) / dt + (E[Nt] - E)
    if arm == "prefix_ceil":
        return (t - ctx.a) / dt + np.ceil((ctx.a - ctx.s) / dt)
    raise ValueError(arm)


def interp_chunk(C, A):
    lo = max(0, int(math.floor(A)))
    w = A - lo
    hi = min(lo + 1, len(C) - 1)
    return (1.0 - w) * C[lo] + w * C[hi]


def run_arm(chunks, rho, arm, ref, t0, t1, weights="exp"):
    ctx = Ctx(chunks, rho)
    ts = np.arange(t0, t1 + 0.5 * T_EXEC, T_EXEC)
    u = np.zeros((len(ts), ref.xs.shape[1]))
    lead = np.full(len(ts), np.nan)
    live_counts = np.zeros(len(ts), dtype=int)
    starved = 0
    first_starve = None
    for i, t in enumerate(ts):
        A = arm_indices(arm, ctx, t)
        idx = np.where((A >= -1e-9) & (A <= HORIZON - 1 + 1e-9))[0]
        if len(idx) == 0:
            u[i] = u[i - 1] if i > 0 else 0.0
            starved += 1
            if first_starve is None:
                first_starve = float(t - ts[0])
            continue
        if weights == "flat":
            w = np.ones(len(idx))
        else:
            w = np.exp(-DECAY * np.maximum(A[idx], 0.0))
        acc = np.zeros(u.shape[1])
        for kk, wk in zip(idx, w):
            acc += wk * interp_chunk(chunks[kk]["C"], A[kk])
        u[i] = acc / w.sum()
        live_counts[i] = len(idx)
        t_eff = float(np.sum(w * (ctx.a[idx] + A[idx] * T_EXEC)) / w.sum())
        lead[i] = (t_eff - t) / T_EXEC
    return ts, u, ref(ts), lead, live_counts, starved, first_starve


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def summarize(u, g, ts, lead, live, starved, first_starve=None):
    n_t = len(ts)
    warm = min(8, n_t // 4)
    sl = slice(warm, n_t - 2)
    du, dg = np.diff(u[sl], axis=0), np.diff(g[sl], axis=0)
    d2u, d2g = np.diff(u[sl], n=2, axis=0), np.diff(g[sl], n=2, axis=0)
    d3u, d3g = np.diff(u[sl], n=3, axis=0), np.diff(g[sl], n=3, axis=0)
    # vector-norm error, so the ratio to the per-tick velocity is a time error
    rmse_vec = float(np.sqrt(np.mean(np.sum((u[sl] - g[sl]) ** 2, axis=1))))
    rmse = float(np.sqrt(np.mean((u[sl] - g[sl]) ** 2)))
    rms_vel = float(np.sqrt(np.mean(np.sum(dg ** 2, axis=1))))
    lw = lead[warm:n_t - 2]
    lw = lw[np.isfinite(lw)]
    first = float(np.mean(lw[:min(25, len(lw))])) if len(lw) else float("nan")
    last = float(np.mean(lw[-min(25, len(lw)):])) if len(lw) else float("nan")
    return {
        "rmse": rmse,
        "rmse_norm": rmse / max(g.std(), 1e-12),
        "time_err_ticks": rmse_vec / max(rms_vel, 1e-12),
        "speed_ratio": float(np.mean(np.linalg.norm(du, axis=1)) / max(np.mean(np.linalg.norm(dg, axis=1)), 1e-12)),
        "accel_ratio": float(np.mean(np.linalg.norm(d2u, axis=1)) / max(np.mean(np.linalg.norm(d2g, axis=1)), 1e-12)),
        "jerk_ratio": float(np.mean(np.linalg.norm(d3u, axis=1)) / max(np.mean(np.linalg.norm(d3g, axis=1)), 1e-12)),
        "peak_speed_ratio": float(np.max(np.linalg.norm(du, axis=1)) / max(np.max(np.linalg.norm(dg, axis=1)), 1e-12)),
        "lead_first_ticks": round(first, 3),
        "lead_last_ticks": round(last, 3),
        "lead_drift_ticks": round(last - first, 3),
        "starved_frac": round(starved / n_t, 3),
        "mean_live": round(float(np.mean(live)), 2),
        "first_starve_s": None if first_starve is None else round(first_starve, 3),
    }


def grasp_shift(u, g, ts, channel=6, level=0.5, lo=4.0, hi=6.0):
    def first_cross(x):
        m = (ts >= lo) & (ts <= hi)
        tt, xx = ts[m], x[m, channel]
        sgn = np.sign(xx - level)
        for i in range(1, len(xx)):
            if sgn[i] != sgn[i - 1] and sgn[i] != 0:
                return float(tt[i])
        return None

    a, b = first_cross(u), first_cross(g)
    return None if a is None or b is None else round((a - b) * 1000.0, 1)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    global HORIZON
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--ref", default="both", choices=["aloha", "analytic", "both"])
    args = ap.parse_args()
    HORIZON = args.horizon
    os.makedirs(args.out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))

    refs = {}
    if args.ref in ("analytic", "both"):
        refs["analytic"] = Ref(*analytic_trajectory())
    if args.ref in ("aloha", "both"):
        refs["aloha"] = Ref(*aloha_trajectory(os.path.join(here, "data", "aloha_transfer_rows.json")))

    arms = ["exact_arrival", "shiftpos", "ceil_clock", "floor_clock", "fixed_clock"]
    rows = []

    # --- bookkeeping exhibit (no closed loop) ------------------------------
    book = {}
    for name, sched in (("const0.6", SCHEDULES["const0.6"]),
                        ("jitter", SCHEDULES["jitter"]),
                        ("cross", SCHEDULES["cross"])):
        rho = sched(10)
        S, nn, ff = eq12_table(rho)
        book[name] = {
            "rho": [round(float(x), 4) for x in rho],
            "S_k": [round(float(x), 4) for x in S],
            "n_k": [int(x) for x in nn],
            "f_k": [round(float(x), 4) for x in ff],
            "true_shift_at_own_arrival": [0.0] * len(rho),
            "true_shift_at_last_arrival": [round(float(S[-1] - x), 4) for x in S],
            "shiftpos_minus_true": [round(float(2 * x - S[-1]), 4) for x in S],
            "overshift_ms_if_Sk_is_shift": [round(float(x * T_EXEC * 1000), 1) for x in S],
            "drift_per_chunk_ceil": round(float(np.mean(np.ceil(rho) - rho)), 4),
            "drift_per_chunk_floor": round(float(np.mean(np.floor(rho) - rho)), 4),
        }
    # the paper's own example must fall out of the table
    assert np.allclose(book["const0.6"]["S_k"][:3], [0.0, 0.6, 1.2], atol=1e-12)
    assert book["const0.6"]["n_k"][:3] == [0, 0, 1]
    assert np.allclose(book["const0.6"]["f_k"][:3], [0.0, 0.6, 0.2], atol=1e-12)

    # --- identity: Eq.(12) placement == per-chunk arrival age ---------------
    identity = {}
    for name, sched in SCHEDULES.items():
        rho = sched(120)
        ch = chunks_from_rho(rho, 2 * T_EXEC, HORIZON, T_EXEC, refs[list(refs)[0]])
        ctx = Ctx(ch, rho)
        S, n, f = ctx.S, ctx.n, ctx.f
        tt = ch[0]["a"] + np.arange(120) * T_EXEC
        dmax = 0.0
        for t in tt:
            A1 = arm_indices("exact_arrival", ctx, t)
            A2 = (t - ch[0]["a"]) / T_EXEC - S
            dmax = max(dmax, float(np.max(np.abs(A1 - A2))))
        identity[name] = {"max_abs_diff_ticks": dmax,
                          "sum_n_plus_f_minus_S": float(np.max(np.abs(np.cumsum(n) + f - S)))}
    assert all(v["max_abs_diff_ticks"] < 1e-9 for v in identity.values())

    # --- same identity with rho measured on observation times ---------------
    # If T_k were (s_k - s_{k-1}) instead of (a_k - a_{k-1}), Eq.(12) would place
    # chunks on the observation clock.  Check it on a jittery-latency stream.
    rng = np.random.default_rng(SEED + 11)
    s_s = 2 * T_EXEC + np.arange(60) * T_EXEC
    L_s = T_EXEC * (1.0 + np.clip(0.25 * rng.standard_normal(60), -0.5, 0.5))
    ch_anchor = chunks_from_anchors(s_s, L_s, HORIZON, T_EXEC, refs[list(refs)[0]])
    rho_s = np.concatenate([[1.0], np.diff(s_s) / T_EXEC])
    S_obs = eq12_table(rho_s)[0]
    tt = ch_anchor[0]["s"] + np.arange(60) * T_EXEC
    dmax = max(float(np.max(np.abs((t - s_s[0]) / T_EXEC - S_obs - (t - s_s) / T_EXEC))) for t in tt)
    identity["observation_clock"] = {"max_abs_diff_ticks": dmax}
    assert dmax < 1e-9

    # --- closed loop -------------------------------------------------------
    for ref_name, ref in refs.items():
        for seq_name, sched in SCHEDULES.items():
            n = int(1.7 * (ref.ts[-1] / T_EXEC)) + 4
            rho = sched(n)
            chunks = chunks_from_rho(rho, a0=2 * T_EXEC, horizon=HORIZON, T=T_EXEC, ref=ref)
            t0 = chunks[0]["a"]
            t1 = min(ref.ts[-1] - 0.2, chunks[-1]["a"] + (HORIZON - 2) * T_EXEC)
            for weights in ("exp", "flat"):
                for arm in arms:
                    ts, u, g, lead, live, starved, fs = run_arm(chunks, rho, arm, ref, t0, t1, weights)
                    m = summarize(u, g, ts, lead, live, starved, fs) | {"arm": arm, "seq": seq_name,
                                                                   "ref": ref_name, "weights": weights,
                                                                   "mean_rho": round(float(rho.mean()), 4)}
                    if ref_name == "aloha":
                        m["grasp_shift_ms"] = grasp_shift(u, g, ts)
                    rows.append(m)
                    print(f"{ref_name:8s} {seq_name:8s} {weights:4s} {arm:14s} "
                          f"err={m['time_err_ticks']:5.2f}tk speed={m['speed_ratio']:5.2f} "
                          f"acc2={m['accel_ratio']:5.2f} jerk3={m['jerk_ratio']:6.2f} "
                          f"lead={m['lead_first_ticks']:+6.2f}->{m['lead_last_ticks']:+6.2f} "
                          f"live={m['mean_live']:4.1f} starv={m['starved_frac']:.2f}")

    # --- anchor study: constant observation cadence, jittery head latency ---
    anchor_rows = []
    for ref_name, ref in refs.items():
        n = int(1.6 * (ref.ts[-1] / T_EXEC))
        base_s = 2 * T_EXEC + np.arange(n) * T_EXEC
        rng = np.random.default_rng(SEED + 7)
        scenarios = {
            "gauss": T_EXEC * (1.0 + np.clip(0.25 * rng.standard_normal(n), -0.5, 0.5)),
            "spike": T_EXEC * (1.0 + np.where(np.arange(n) % 25 == 0, 0.9, 0.0)),
        }
        for sc_name, L in scenarios.items():
            chunks = chunks_from_anchors(base_s, L, HORIZON, T_EXEC, ref)
            rho = np.concatenate([[1.0], np.diff([c["a"] for c in chunks]) / T_EXEC])
            t0, t1 = chunks[0]["a"], min(ref.ts[-1] - 0.2, chunks[-1]["a"] + (HORIZON - 2) * T_EXEC)
            for arm in ("exact_anchor", "exact_arrival", "prefix_ceil"):
                ts, u, g, lead, live, starved, fs = run_arm(chunks, rho, arm, ref, t0, t1, "exp")
                m = summarize(u, g, ts, lead, live, starved, fs) | {
                    "arm": arm, "ref": ref_name, "scenario": sc_name,
                    "mean_L_ticks": round(float(np.mean(L / T_EXEC)), 4),
                    "sigma_L_ticks": round(float(np.std(L / T_EXEC)), 4),
                }
                if ref_name == "aloha":
                    m["grasp_shift_ms"] = grasp_shift(u, g, ts)
                anchor_rows.append(m)
                print(f"ANCHOR {ref_name:8s} {sc_name:5s} {arm:14s} err={m['time_err_ticks']:5.2f}tk "
                      f"speed={m['speed_ratio']:5.2f} jerk={m['jerk_ratio']:6.2f} "
                      f"grasp={m.get('grasp_shift_ms')}")

    # --- horizon sensitivity ----------------------------------------------
    sweep = []
    base_h = HORIZON
    for hh in (8, 16, 32):
        HORIZON = hh
        rho = rho_cross(int(1.3 * (refs["analytic"].ts[-1] / T_EXEC)) + 4)
        chn = chunks_from_rho(rho, 2 * T_EXEC, hh, T_EXEC, refs["analytic"])
        for arm in ("exact_arrival", "shiftpos", "ceil_clock"):
            ts, u, g, lead, live, starved, fs = run_arm(chn, rho, arm, refs["analytic"], chn[0]["a"],
                                                   min(refs["analytic"].ts[-1] - 0.2,
                                                       chn[-1]["a"] + (hh - 2) * T_EXEC), "exp")
            m = summarize(u, g, ts, lead, live, starved, fs)
            sweep.append({"H": hh, "arm": arm, "time_err_ticks": m["time_err_ticks"],
                          "speed_ratio": m["speed_ratio"], "starved_frac": m["starved_frac"]})
    HORIZON = base_h

    out = {"bookkeeping": book, "identity": identity, "chunk_rows": rows, "anchor_rows": anchor_rows,
           "horizon_sweep": sweep,
           "config": {"T_exec": T_EXEC, "H": args.horizon, "seed": SEED, "decay": DECAY,
                      "note": "arms are index rules A_k(t); see module docstring"}}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
