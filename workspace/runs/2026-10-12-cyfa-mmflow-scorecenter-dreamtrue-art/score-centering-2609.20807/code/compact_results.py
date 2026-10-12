"""Shrink the experiment JSONs to run-packet sizes (<2 MB each) by rounding to
3 decimals, dropping unused series and decimating long curves (every 5th step
for the 1500-step constant-reward lanes; the figures use the raw files).

Usage: uv run --with numpy python3 compact_results.py
"""

import json
import os

import numpy as np

FIGDIR = os.environ.get("SC_FIGDIR", "figs")
R = lambda x, n=3: [round(float(v), n) for v in x]  # noqa: E731

EXACT_KEYS = ("reward_trainer", "entropy", "tv_pq", "tv_p0", "logit_norm")


def _load_raw(name, producer):
    path = os.path.join(FIGDIR, name)
    if not os.path.exists(path):
        print(f"skip {name}: run {producer} first")
        return None
    with open(path) as f:
        return json.load(f)


def compact_exp3():
    d = _load_raw("exp3_signal.json", "run_exp3_signal.py")
    if d is None:
        return
    out = {"config": d["config"], "anatomy": d["anatomy"], "results": {}}
    for key, block in d["results"].items():
        rec = {"exact": {}, "sampled": {}}
        for m, h in block["exact"].items():
            rec["exact"][m] = {k: R(h[k]) for k in EXACT_KEYS}
            rec["exact"][m]["final"] = round(h["final"], 4)
        for m, h in block["sampled"].items():
            curves = np.array(h["curves"])
            rec["sampled"][m] = {
                "final_mean": round(h["final_mean"], 4),
                "final_std": round(h["final_std"], 4),
                "finals": R(h["finals"]),
                "reward_mean": R(curves.mean(0)),
                "reward_std": R(curves.std(0)),
            }
        out["results"][key] = rec
    write(out, "exp3_signal_compact.json")


def compact_exp2(stride=10):
    d = _load_raw("exp2_drift.json", "run_exp2_drift.py")
    if d is None:
        return
    out = {"config": d["config"], "noise_sweep": d["noise_sweep"], "lanes": {}}
    for lane, block in d["lanes"].items():
        rec = {}
        for m, h in block.items():
            keep = list(range(0, len(h["step"]), stride))
            rec[m] = {k: R(np.asarray(h[k])[keep]) for k in h
                      if k != "step"}
            rec[m]["step"] = [int(i + 1) for i in keep]
        out["lanes"][lane] = rec
    write(out, "exp2_drift_compact.json")


def write(obj, name):
    dst = os.path.join(FIGDIR, name)
    with open(dst, "w") as f:
        json.dump(obj, f)
    print(f"{name}: {os.path.getsize(dst) / 1e3:.0f} kB")


def compact_exp2b(stride=5):
    d = _load_raw("exp2b_severity.json", "run_exp2b_severity.py")
    if d is None:
        return
    out = {"config": d["config"], "lanes": {}}
    for lane, block in d["lanes"].items():
        out["lanes"][lane] = {}
        for m, h in block.items():
            keep = list(range(0, len(h["entropy"]), stride))
            out["lanes"][lane][m] = {k: R(np.asarray(v)[keep]) for k, v in h.items()}
            out["lanes"][lane][m]["step"] = [int(i + 1) for i in keep]
    write(out, "exp2b_severity_compact.json")


def main():
    compact_exp3()
    compact_exp2()
    compact_exp2b()


if __name__ == "__main__":
    main()
