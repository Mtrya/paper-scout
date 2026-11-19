"""Aggregate the E1 (estimation) and E2 (control ablation) episodes into the numbers the run packet reports.

  python analyze.py --e1 runs/dev/e1_full.jsonl --e2 runs/dev/e2_full.jsonl --out-dir runs/dev/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

MID = {"weight_sort": 0.375, "weight_sort_ood": 0.375, "com_sort": 4.0, "com_sort_ood": 2.5}


def _rows_e1(path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _lift(rec):
    for r in rec.get("rows", []):
        if r["primitive"] == "lift":
            return r
    return None


def _angular_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def episode_metrics_e1(rec):
    lift = _lift(rec)
    if lift is None:
        return None
    ts = rec.get("truth_episode") or {}
    task = rec["task"]
    out = dict(task=task, arm=rec["arm"], seed=rec["seed"], success=rec["success"],
               verdict=rec.get("verdict"), verdict_correct=rec.get("verdict_correct"),
               termination=lift["termination"], n_window=lift.get("payload_n"),
               mass_est=lift.get("payload_mass"), mass_sigma=lift.get("payload_sigma"),
               mass_flags="|".join(lift.get("payload_flags") or []),
               com_est_cm=lift.get("com_r_cm"), com_sigma_cm=lift.get("com_sigma_cm"),
               com_flags="|".join(lift.get("com_flags") or []))
    grasps = [r for r in rec.get("rows", []) if r["primitive"] == "grasp"]
    out["grasp_ok"] = bool((grasps[-1].get("derived") or {}).get("closed_on_something")) if grasps else None
    out["n_grasps"] = len(grasps)
    ex = lift.get("payload_extra") or {}
    out["hold_force_z_N"] = ex.get("hold_force_z_N")
    out["floor_kg"] = ex.get("floor_kg")
    ce = lift.get("com_extra") or {}
    out["com_x_cm"] = ce.get("x_cm")
    out["com_y_cm"] = ce.get("y_cm")
    out["com_dir_deg"] = ce.get("direction_deg")
    out["balanced_floor_cm"] = ce.get("balanced_floor_cm")
    out["torque_xy"] = ce.get("torque_xy_Nm")
    out["label"] = ts.get("label")
    out["mass_true"] = ts.get("mass")
    out["com_offset_true_m"] = ts.get("com_offset_m")
    tr = (lift.get("truth") or {}).get("task") or {}
    out["bar_tilt_deg"] = tr.get("bar_tilt_deg")
    out["lever_xy_mm"] = tr.get("com_lever_xy_mm")
    out["object_contact"] = (lift.get("truth") or {}).get("contact_fraction")

    if "weight" in task:
        m, s = out["mass_est"], out["mass_sigma"]
        if m is not None and s and out.get("mass_true") is not None:
            out["mass_err"] = m - out["mass_true"]
            out["mass_z"] = (m - out["mass_true"]) / s
            out["mass_decided"] = "heavy" if m > 0.375 else "light"
            out["mass_correct"] = out["mass_decided"] == out["label"]
            out["mass_margin_sigma"] = abs(m - 0.375) / s
    else:
        r, s = out["com_est_cm"], out["com_sigma_cm"]
        lever = out["lever_xy_mm"]
        if r is not None and s and lever is not None:
            true_r = math.hypot(*lever) / 10.0
            out["com_true_cm"] = true_r
            out["com_err"] = r - true_r
            out["com_z"] = (r - true_r) / s
            true_dir = math.degrees(math.atan2(lever[1], lever[0]))
            out["com_true_dir_deg"] = true_dir
            out["com_dir_err_deg"] = _angular_diff(out["com_dir_deg"], true_dir)
            # the task variable: 4.0 cm offset (nominal) or 2.5 cm for the OOD variant, 0 for balanced
            nominal = 4.0 if task == "com_sort" else 2.5
            out["com_nominal_cm"] = 0.0 if out["label"] == "balanced" else nominal
            out["com_decided"] = "offset" if r > 2 * s else "balanced"
            out["com_correct"] = out["com_decided"] == out["label"]
            out["com_margin_sigma"] = abs(r - 2 * s) / s
            # direction: which end of the bar carries the mass (sign along the bar's long axis, world frame)
            out["com_dir_relevant"] = out["label"] == "offset" and true_r > 2.0
    return out


def stats(rows, value_key, sigma_key, z_key):
    vals = [r[value_key] for r in rows if r.get(value_key) is not None]
    sigs = [r[sigma_key] for r in rows if r.get(sigma_key) is not None]
    z = [r[z_key] for r in rows if r.get(z_key) is not None]
    out = dict(n=len(rows), n_values=len(vals),
               mean=float(np.mean(vals)) if vals else None,
               sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
               mean_sigma=float(np.mean(sigs)) if sigs else None,
               rms_sigma_ratio=None, cover1=None, cover2=None, mean_abs_z=None)
    if z:
        out.update(cover1=float(np.mean([abs(x) <= 1 for x in z])),
                   cover2=float(np.mean([abs(x) <= 2 for x in z])),
                   mean_abs_z=float(np.mean([abs(x) for x in z])),
                   mean_z=float(np.mean(z)))
    if vals and sigs and out["mean_sigma"]:
        out["rms_sigma_ratio"] = float(np.sqrt(np.mean(np.square(vals))) / out["mean_sigma"])
    return out


def summarize_e1(path):
    recs = [episode_metrics_e1(r) for r in _rows_e1(path)]
    recs = [r for r in recs if r is not None]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["task"], r["arm"])].append(r)
    out = {}
    for (task, arm), rows in sorted(groups.items()):
        g = dict(n_episodes=len(rows),
                 success_rate=float(np.mean([bool(r["success"]) for r in rows])),
                 verdict_correct=float(np.mean([r["verdict_correct"] is True for r in rows])) if rows else None,
                 flags={})
        for f in ("mass_flags", "com_flags"):
            for r in rows:
                for flag in [x for x in (r.get(f) or "").split("|") if x]:
                    g["flags"][flag] = g["flags"].get(flag, 0) + 1
        if "weight" in task:
            g["mass"] = stats(rows, "mass_est", "mass_sigma", "mass_z")
            g["mass_err"] = stats(rows, "mass_err", "mass_sigma", "mass_z")
            g["decision_acc"] = float(np.mean([bool(r.get("mass_correct")) for r in rows if r.get("mass_correct") is not None]))
            g["mean_margin_sigma"] = float(np.mean([r["mass_margin_sigma"] for r in rows if r.get("mass_margin_sigma") is not None]))
            clean = [r for r in rows if r.get("grasp_ok") is not False]
            g["grasp_ok_rate"] = float(len(clean) / max(1, len(rows)))
            g["no_load_rate"] = float(np.mean(["no_load" in (r.get("mass_flags") or "") for r in rows]))
            light = [r for r in clean if r["label"] == "light" and r.get("mass_est") is not None]
            heavy = [r for r in clean if r["label"] == "heavy" and r.get("mass_est") is not None]
            if light and heavy:
                ml = float(np.mean([r["mass_est"] for r in light]))
                mh = float(np.mean([r["mass_est"] for r in heavy]))
                tl = float(np.mean([r["mass_true"] for r in light]))
                th = float(np.mean([r["mass_true"] for r in heavy]))
                slope = (mh - ml) / (th - tl)
                g["fit"] = dict(mean_light=ml, true_light=tl, mean_heavy=mh, true_heavy=th,
                                slope=float(slope), intercept=float(ml - slope * tl),
                                offset_kg=float(ml - slope * tl))
                # error rate implied by the *reported* sigma, per class, with the observed class means
                sig = float(np.mean([r["mass_sigma"] for r in rows if r.get("mass_sigma") is not None]))
                phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
                p_light = phi((ml - 0.375) / sig)          # light misread as heavy
                p_heavy = phi((0.375 - mh) / sig)          # heavy misread as light
                g["implied_err_rate"] = float(0.5 * (p_light + p_heavy))
                g["observed_err_rate"] = float(1 - g["decision_acc"])
                g["half_gap_kg"] = float(0.5 * (th - tl))
        else:
            g["com"] = stats(rows, "com_est_cm", "com_sigma_cm", "com_z")
            g["com_err"] = stats(rows, "com_err", "com_sigma_cm", "com_z")
            g["decision_acc"] = float(np.mean([bool(r.get("com_correct")) for r in rows if r.get("com_correct") is not None]))
            g["mean_margin_sigma"] = float(np.mean([r["com_margin_sigma"] for r in rows if r.get("com_margin_sigma") is not None]))
            bal = [r for r in rows if r["label"] == "balanced" and r.get("com_est_cm") is not None]
            off = [r for r in rows if r["label"] == "offset" and r.get("com_est_cm") is not None]
            g["false_offset_on_balanced"] = float(np.mean([r["com_est_cm"] > 2 * r["com_sigma_cm"] for r in bal])) if bal else None
            g["missed_offset"] = float(np.mean([r["com_est_cm"] <= 2 * r["com_sigma_cm"] for r in off])) if off else None
            g["expert_verdict_acc"] = float(np.mean([bool(r["verdict_correct"]) for r in rows if r.get("verdict_correct") is not None]))
            if bal:
                g["balanced_read"] = dict(mean_est_cm=float(np.mean([r["com_est_cm"] for r in bal])),
                                          mean_true_cm=float(np.mean([r["com_true_cm"] for r in bal if r.get("com_true_cm") is not None])),
                                          fraction_unflagged=float(np.mean(["within_noise" not in (r.get("com_flags") or "") for r in bal])),
                                          implied_torque_Nm=float(np.mean([r["com_est_cm"] for r in bal]) / 100 * 0.8 * 9.81))
            if off:
                nominal = 4.0 if task == "com_sort" else 2.5
                g["offset_lever"] = dict(
                    mean_true_cm=float(np.mean([r["com_true_cm"] for r in off if r.get("com_true_cm") is not None])),
                    nominal_cm=nominal,
                    fraction_below_75pct=float(np.mean([r["com_true_cm"] < 0.75 * nominal for r in off if r.get("com_true_cm") is not None])),
                    max_tilt_deg=float(np.max([abs(r["bar_tilt_deg"]) for r in off if r.get("bar_tilt_deg") is not None])))
            off = [r for r in rows if r.get("label") == "offset" and r.get("com_dir_err_deg") is not None]
            g["direction_err_deg"] = dict(
                n=len(off),
                median=float(np.median([r["com_dir_err_deg"] for r in off])) if off else None,
                mean=float(np.mean([r["com_dir_err_deg"] for r in off])) if off else None,
                within30=float(np.mean([r["com_dir_err_deg"] <= 30 for r in off])) if off else None,
                within45=float(np.mean([r["com_dir_err_deg"] <= 45 for r in off])) if off else None)
            g["true_vs_est"] = dict(
                mean_true_cm=float(np.mean([r["com_true_cm"] for r in rows if r.get("com_true_cm") is not None])),
                mean_tilt_deg=float(np.mean([abs(r["bar_tilt_deg"]) for r in rows if r.get("bar_tilt_deg") is not None])),
                max_tilt_deg=float(np.max([abs(r["bar_tilt_deg"]) for r in rows if r.get("bar_tilt_deg") is not None])))
            # correlation of the error with the bar tilt (a physical attenuation the sigma does not model)
            pairs = [(abs(r["bar_tilt_deg"]), r["com_err"]) for r in rows
                     if r.get("com_err") is not None and r.get("bar_tilt_deg") is not None]
            if len(pairs) > 3:
                t, e = np.array(pairs).T
                g["err_vs_tilt"] = dict(corr=float(np.corrcoef(t, e)[0, 1]), n=len(pairs),
                                        slope_cm_per_deg=float(np.polyfit(t, e, 1)[0]))
        out[f"{task}|{arm}"] = g
    return out, recs


def episode_metrics_e2(rec):
    if "fatal" in rec:
        return dict(task=rec["task"], arm=rec["arm"], seed=rec["seed"], fatal=rec["fatal"])
    ts = rec["truth_episode"]
    task = rec["task"]
    last = rec["steps"][-1]
    q = last["quantities"]
    grasp = next((s for s in rec["steps"] if s["name"] == "grasp"), {})
    body = "cube" if "weight" in task else "bar"
    p0 = rec["home"].get(f"{body}_com") or [0, 0, 0]
    p1 = rec["end"].get(f"{body}_com") or [0, 0, 0]
    z0, z1 = p0[2], p1[2]
    out = dict(task=task, arm=rec["arm"], seed=rec["seed"], label=ts.get("label"),
               mass_true=ts.get("mass"), com_offset_true_m=ts.get("com_offset_m"),
               mass_est=(q.get("payload_mass") or {}).get("value"),
               mass_sigma=(q.get("payload_mass") or {}).get("sigma"),
               mass_flags="|".join((q.get("payload_mass") or {}).get("flags") or []),
               com_est_cm=(q.get("com_offset") or {}).get("value"),
               com_sigma_cm=(q.get("com_offset") or {}).get("sigma"),
               com_flags="|".join((q.get("com_offset") or {}).get("flags") or []),
               com_x_cm=((q.get("com_offset") or {}).get("extra") or {}).get("x_cm"),
               com_y_cm=((q.get("com_offset") or {}).get("extra") or {}).get("y_cm"),
               bar_tilt_deg=((last.get("truth") or {}) or {}).get("bar_tilt_deg"),
               grasp_closed_on_something=(grasp.get("derived") or {}).get("closed_on_something"),
               grasp_opening=(grasp.get("derived") or {}).get("opening_m"),
               object_rise=float(z1 - z0) if z0 is not None and z1 is not None else None,
               object_shove_xy=float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])),
               descend_contact_fraction=[s.get("contact_fraction") for s in rec["steps"] if s["name"] in ("move_until_contact", "move_to")],
               descend_peak_force=[s.get("peak_force") for s in rec["steps"] if s["name"] in ("move_until_contact", "move_to")],
               termination=last.get("termination"))
    if "weight" in task:
        m, s = out["mass_est"], out["mass_sigma"]
        if m is not None and out["mass_true"] is not None:
            out["mass_err"] = m - out["mass_true"]
            out["mass_z"] = out["mass_err"] / s if s else None
            out["mass_decided"] = "heavy" if m > 0.375 else "light"
            out["mass_correct"] = out["mass_decided"] == out["label"]
    else:
        r, s = out["com_est_cm"], out["com_sigma_cm"]
        lever = (last.get("truth") or {}).get("com_lever_xy_mm") if isinstance(last.get("truth"), dict) else None
        lever = (last.get("truth") or {}).get("com_lever_xy_mm")
        if r is not None and s:
            out["com_decided"] = "offset" if r > 2 * s else "balanced"
            out["com_correct"] = out["com_decided"] == out["label"]
            if lever:
                true_r = math.hypot(*lever) / 10.0
                out["com_true_cm"] = true_r
                out["com_err"] = r - true_r
                true_dir = math.degrees(math.atan2(lever[1], lever[0]))
                out["com_dir_err_deg"] = _angular_diff(((q.get("com_offset") or {}).get("extra") or {}).get("direction_deg", 0.0), true_dir)
    return out


def summarize_e2(path):
    recs = [episode_metrics_e2(r) for r in _rows_e1(path)]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["task"], r["arm"])].append(r)
    out = {}
    for (task, arm), rows in sorted(groups.items()):
        g = dict(n=len(rows),
                 grasped=float(np.mean([bool(r.get("grasp_closed_on_something")) for r in rows])),
                 grasped_and_lifted=float(np.mean([bool(r.get("grasp_closed_on_something")) and (r.get("object_rise") or 0) > 0.05 for r in rows])),
                 mean_object_rise=float(np.mean([r.get("object_rise") or 0.0 for r in rows])),
                 mean_descend_contact=float(np.mean([np.mean([x for x in (r.get("descend_contact_fraction") or [0]) if x is not None] or [0]) for r in rows])),
                 mean_descend_peak_force=float(np.mean([max([x or 0 for x in (r.get("descend_peak_force") or [0])]) for r in rows])))
        if "weight" in task:
            g["mass_err"] = dict(mean=float(np.mean([r["mass_err"] for r in rows if r.get("mass_err") is not None])),
                                 sd=float(np.std([r["mass_err"] for r in rows if r.get("mass_err") is not None], ddof=1)),
                                 rms=float(np.sqrt(np.mean(np.square([r["mass_err"] for r in rows if r.get("mass_err") is not None])))),
                                 mean_sigma=float(np.mean([r["mass_sigma"] for r in rows if r.get("mass_sigma") is not None])),
                                 within1=float(np.mean([abs(r["mass_err"]) <= r["mass_sigma"] for r in rows if r.get("mass_err") is not None])),
                                 decision_acc=float(np.mean([bool(r.get("mass_correct")) for r in rows if r.get("mass_correct") is not None])))
        else:
            g["com_err"] = dict(mean=float(np.mean([r["com_err"] for r in rows if r.get("com_err") is not None])),
                                sd=float(np.std([r["com_err"] for r in rows if r.get("com_err") is not None], ddof=1)),
                                rms=float(np.sqrt(np.mean(np.square([r["com_err"] for r in rows if r.get("com_err") is not None])))),
                                mean_sigma=float(np.mean([r["com_sigma_cm"] for r in rows if r.get("com_sigma_cm") is not None])),
                                within1=float(np.mean([abs(r["com_err"]) <= r["com_sigma_cm"] for r in rows if r.get("com_err") is not None])),
                                decision_acc=float(np.mean([bool(r.get("com_correct")) for r in rows if r.get("com_correct") is not None])),
                                mean_tilt=float(np.mean([abs(r["bar_tilt_deg"]) for r in rows if r.get("bar_tilt_deg") is not None])),
                                dir_err_median=float(np.median([r["com_dir_err_deg"] for r in rows if r.get("com_dir_err_deg") is not None])))
        out[f"{task}|{arm}"] = g
    return out, recs


def write_csv(path, rows, keys):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="runs/dev/e1_full.jsonl")
    ap.add_argument("--e2", default="runs/dev/e2_full.jsonl")
    ap.add_argument("--out-dir", default="runs/dev/analysis")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s1, r1 = summarize_e1(args.e1)
    (out / "summary_e1.json").write_text(json.dumps(s1, indent=1))
    keys1 = ["task", "arm", "seed", "label", "grasp_ok", "n_grasps", "mass_true", "mass_est", "mass_sigma", "mass_err", "mass_z",
             "mass_decided", "mass_correct", "mass_margin_sigma", "mass_flags", "com_est_cm", "com_sigma_cm",
             "com_err", "com_z", "com_true_cm", "com_true_dir_deg", "com_dir_deg", "com_dir_err_deg",
             "com_decided", "com_correct", "com_est_nominal", "com_nominal_cm", "com_margin_sigma", "com_flags",
             "bar_tilt_deg", "lever_xy_mm", "hold_force_z_N", "floor_kg", "success", "verdict", "verdict_correct"]
    write_csv(out / "episodes_e1.csv", r1, keys1)
    if Path(args.e2).exists():
        s2, r2 = summarize_e2(args.e2)
        (out / "summary_e2.json").write_text(json.dumps(s2, indent=1))
        keys2 = ["task", "arm", "seed", "label", "mass_true", "mass_est", "mass_sigma", "mass_err", "mass_z",
                 "mass_decided", "mass_correct", "mass_flags", "com_est_cm", "com_sigma_cm", "com_err", "com_true_cm",
                 "com_dir_err_deg", "com_decided", "com_correct", "com_flags", "bar_tilt_deg",
                 "grasp_closed_on_something", "object_rise", "descend_peak_force", "termination"]
        write_csv(out / "episodes_e2.csv", r2, keys2)
    print(json.dumps({"e1": s1, "e2": s2 if Path(args.e2).exists() else None}, indent=1)[:200])
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
