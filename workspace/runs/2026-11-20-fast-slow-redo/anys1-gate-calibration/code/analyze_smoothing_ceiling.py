#!/usr/bin/env python3
"""Label-smoothing confidence ceiling, checked against AnyS1 tau-sweep data.

Derivation (ours, this redo): AnyS1 trains with label smoothing
L = CE(p, (1-eps) e_y + eps u_O), eps = 0.1. The probability-matching optimum
puts mass (1-eps) + eps/K on the true option, so the maximum per-decision
confidence achievable in steady state is

    ceiling(K) = 1 - eps + eps/K.

Consequences: a gate threshold must sit below the ceiling to ever fire;
tau >= 1 - eps + eps/2 = 0.95 (K >= 2) can find no samples; hard-label CE
(eps = 0) removes the ceiling, which is why its fitted threshold inflates to
0.998. We check this against the 600-sample tau sweep recorded from the
official AnyS1-4B weights (see README for provenance).
"""
import json

EPS = 0.1
data = json.load(open("tau_result.json"))
rows = data["rows"]

ceilings = [1 - EPS + EPS / r["n_options"] for r in rows]
overshoot = [r["p_max"] - c for r, c in zip(rows, ceilings) if r["p_max"] > c + 1e-9]
above_097 = [r for r in rows if r["p_max"] >= 0.97]
above_095 = [r for r in rows if r["p_max"] >= 0.95]
k_hist: dict[int, int] = {}
for r in rows:
    k_hist[r["n_options"]] = k_hist.get(r["n_options"], 0) + 1

out = {
    "eps": EPS,
    "n": len(rows),
    "formula": "ceiling(K) = 1 - eps + eps/K",
    "k_distribution": dict(sorted(k_hist.items())),
    "median_K": sorted(k_hist.items(), key=lambda kv: -kv[1])[0][0],
    "modal_ceiling": round(1 - EPS + EPS / sorted(k_hist.items(), key=lambda kv: -kv[1])[0][0], 4),
    "max_p_max": round(max(r["p_max"] for r in rows), 4),
    "rows_ge_095": len(above_095),
    "rows_ge_097": len(above_097),
    "ceiling_violations": len(overshoot),
    "max_overshoot": round(max(overshoot), 4) if overshoot else 0.0,
    "tau_cold": 0.918,
    "tau_cold_coverage": data["tau_cold_0918"]["coverage"],
    "tau_cold_agreement": data["tau_cold_0918"]["agreement"],
    "tau_cold_n_accepted": data["tau_cold_0918"]["n_accepted"],
    "top1_accuracy": data["top1_accuracy"],
    "curve": data["curve"],
    "note": "ceiling is the loss optimum, not a hard constraint: the 1-epoch LoRA "
            "model overshoots it on a minority of samples; the explanatory content "
            "is where the confidence mass concentrates",
}
with open("ceiling_analysis.json", "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
