"""Aggregate all analysis JSONs into a compact verdict summary."""
import glob
import json
import os

import numpy as np


def load(p):
    with open(p) as f:
        return json.load(f)


def ms(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "n/a"
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"


def main():
    W = load("analysis/weights.json") if os.path.exists("analysis/weights.json") else {}
    R = load("analysis/replace.json") if os.path.exists("analysis/replace.json") else {}
    P = load("analysis/probe.json") if os.path.exists("analysis/probe.json") else {}
    seeds = [load(p) for p in sorted(glob.glob("out/seed*/rl_metrics.json"))]
    out = {"n_seeds": len(seeds)}
    lines = []
    lines.append(f"seeds: {len(seeds)}")
    lines.append(f"BC success   : {ms([s.get('bc_success') for s in seeds])}")
    lines.append(f"RL success   : {ms([s.get('rl_success') for s in seeds])}")
    for s in seeds:
        lines.append(f"  seed {s['seed']}: bc {s['bc_success']:.3f} -> rl {s['rl_success']:.3f} "
                     f"(expert baseline in bc_metrics)")
    if W:
        lines.append("")
        lines.append("C1/C2 spectra (mean r95 / density / energy share):")
        for tag in ["bc_stage", "rl", "bc_cont", "bc_disc", "bc_cont_mid",
                    "bc_disc_mid", "bc_stage_mid"]:
            row = []
            for g in ["AdaRMS", "TimeMLP", "Rest"]:
                r95 = [d[tag]["groups"][g]["r95_mean"] for d in W.values() if tag in d]
                de = [d[tag]["groups"][g]["density_1e-05"] for d in W.values() if tag in d]
                es = [d[tag]["groups"][g]["energy_share"] for d in W.values() if tag in d]
                row.append(f"{g}: r95={ms(r95)} dens={ms(de)} eshare={ms(es)}")
            tse = [d[tag]["ts_energy_share"] for d in W.values() if tag in d]
            lines.append(f"  {tag}: " + " | ".join(row) + f" || TS energy share {ms(tse)}")
        lines.append("")
        lines.append("C3 |cos(base, delta)|:")
        for tag in ["ssg_rl_vs_bc", "ssg_disc_vs_bc", "ssg_cont_vs_bc"]:
            row = []
            for c in ["scale", "shift", "gate"]:
                v = [float(np.abs(np.array(d[tag]["abs_cos"][c])).mean())
                     for d in W.values() if tag in d]
                row.append(f"{c}={ms(v)}")
            lines.append(f"  {tag}: " + " ".join(row))
        # tau direction localisation at the RL timesteps vs elsewhere
        idx = np.arange(40) / 39 * 0.95
        near = [np.argmin(np.abs(idx - k / 10)) for k in range(10)]
        far = [i for i in range(40) if i not in near]
        lines.append("")
        lines.append("C2 tau-direction localisation (normalised |cos|, near vs between RL steps):")
        for tag in ["rl", "bc_disc", "bc_cont", "bc_disc_mid"]:
            nv, fv = [], []
            for d in W.values():
                key = f"tau_dirs_{tag}"
                if key not in d:
                    continue
                curves = [np.array(v["cos"]) for k, v in d[key].items()
                          if k.endswith("|v0")]
                if not curves:
                    continue
                m = np.mean(curves, 0)
                m = m / m.max()
                nv.append(float(m[near].mean()))
                fv.append(float(m[far].mean()))
            if nv:
                lines.append(f"  {tag}: at RL steps {ms(nv)}, elsewhere {ms(fv)}, "
                             f"ratio {np.mean(nv)/max(np.mean(fv),1e-9):.2f}")
    if R:
        lines.append("")
        lines.append("C4 module replacement:")
        for cond in ["bc", "rl", "ts_only", "nonts_only", "ts_top4", "ts_wo_top4",
                     "bc_plus_ts_top4_bc"]:
            vals = [d[cond]["success"] for d in R.values() if cond in d]
            if vals:
                lines.append(f"  {cond:18s} {ms(vals)}")
    if P:
        lines.append("")
        lines.append("C5 probe / steering:")
        for k in ["probe_all_auc", "probe_random_auc_mean", "probe_best_single_auc"]:
            lines.append(f"  {k:24s} {ms([d[k] for d in P.values() if k in d])}")
        for k in ["probe_bc_auc", "probe_rl_auc"]:
            lines.append(f"  {k:24s} {ms([d[k] for d in P.values() if k in d])}")
        for pol in ["rl", "bc"]:
            for mode in ["shift_adaptive", "shift_fixed", "random_fixed"]:
                row = []
                for a in [0.25, 0.5, 1.0, 2.0]:
                    key = f"{mode}_a{a}"
                    vals = []
                    for d in P.values():
                        sd = d.get(f"steer_{pol}")
                        if sd and key in sd:
                            vals.append(sd[key] - sd["alpha0"])
                    row.append(f"a={a}:{ms(vals)}")
                lines.append(f"  {pol} {mode:15s} " + "  ".join(row))
    out["text"] = "\n".join(lines)
    with open("analysis/summary.json", "w") as f:
        json.dump(out, f, indent=1)
    print(out["text"])


if __name__ == "__main__":
    main()
