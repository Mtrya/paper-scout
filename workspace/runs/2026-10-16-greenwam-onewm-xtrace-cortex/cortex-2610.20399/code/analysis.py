"""Paired item-level analysis of eval_main.json (mean over seeds, bootstrap CI)."""

import json
import random
from collections import defaultdict
from pathlib import Path

ART = Path(__file__).resolve().parent / "artifacts"
res = json.load(open(ART / "eval_main.json"))

ARMS = defaultdict(list)
for key in res:
    if key.startswith(("vanilla/", "ctx/")):
        continue
    run_id = key.split("/")[0]
    ARMS["-".join(run_id.split("-")[:2])].append(run_id)


def item_correct(run_ids, kind, mode):
    """per-item mean correctness across seeds (items in the same shuffle order)."""
    per_seed = []
    for ridd in run_ids:
        items = res[f"{ridd}/{kind}/{mode}"]["items"]
        per_seed.append([1.0 if it["ok"] else 0.0 for it in items])
    n = len(per_seed[0])
    return [sum(s[i] for s in per_seed) / len(per_seed) for i in range(n)]


def bootstrap_diff(a, b, n=20000, seed=0):
    rng = random.Random(seed)
    diffs = []
    m = len(a)
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        diffs.append(sum(a[i] - b[i] for i in idx) / m)
    diffs.sort()
    return {
        "mean": sum(x - y for x, y in zip(a, b)) / m,
        "lo": diffs[int(0.025 * n)],
        "hi": diffs[int(0.975 * n)],
        "p_gt0": sum(d > 0 for d in diffs) / n,
    }


print("arms:", {k: len(v) for k, v in ARMS.items()})
table = {}
for arm, rids in ARMS.items():
    for kind in ("restate", "use"):
        for mode in ("free", "letter"):
            accs = [res[f"{r}/{kind}/{mode}"]["acc"] for r in rids]
            mmlu = [res[f"{r}/mmlu"]["acc"] for r in rids]
            per_item = item_correct(rids, kind, mode)
            table[f"{arm}|{kind}|{mode}"] = {
                "acc_mean": sum(accs) / len(accs),
                "acc_seeds": accs,
                "item_mean": per_item,
                "mmlu_mean": sum(mmlu) / len(mmlu),
            }

print("\narm-level held-out accuracy (mean over seeds)")
for arm in ("R-kl", "P-kl", "M-kl", "R-ce", "P-ce"):
    if f"{arm}|use|free" not in table:
        continue
    row = []
    for kind in ("restate", "use"):
        for mode in ("free", "letter"):
            row.append(f"{kind[:4]}/{mode[0]} {100*table[f'{arm}|{kind}|{mode}']['acc_mean']:.1f}")
    print(f"  {arm:5s} " + "  ".join(row) + f"   MMLU {100*table[f'{arm}|use|free']['mmlu_mean']:.1f}")

print("\npaired bootstrap over held-out USE items (free readout, 20 items)")
boot = {}
for a, b in (("M-kl", "R-kl"), ("P-kl", "R-kl"), ("M-kl", "P-kl")):
    if f"{a}|use|free" not in table or f"{b}|use|free" not in table:
        continue
    d = bootstrap_diff(table[f"{a}|use|free"]["item_mean"], table[f"{b}|use|free"]["item_mean"])
    boot[f"{a.replace('-kl','')}-{b.replace('-kl','')}"] = d
    print(f"  {a} - {b}: +{100*d['mean']:.1f}pp  95% CI [{100*d['lo']:.1f}, {100*d['hi']:.1f}]  P(>0)={d['p_gt0']:.3f}")

print("\npaired bootstrap over held-out RESTATE items")
for a, b in (("R-kl", "P-kl"), ("M-kl", "R-kl")):
    d = bootstrap_diff(table[f"{a}|restate|free"]["item_mean"], table[f"{b}|restate|free"]["item_mean"])
    boot[f"restate:{a.replace('-kl','')}-{b.replace('-kl','')}"] = d
    print(f"  {a} - {b}: +{100*d['mean']:.1f}pp  95% CI [{100*d['lo']:.1f}, {100*d['hi']:.1f}]  P(>0)={d['p_gt0']:.3f}")
json.dump(boot, open(ART / "bootstrap.json", "w"), indent=1)

print("\nCR per arm (using reference gap)")
ref = {"restate/free": (res["vanilla/restate/free"]["acc"], res["ctx/restate/free"]["acc"]),
       "use/free": (res["vanilla/use/free"]["acc"], res["ctx/use/free"]["acc"]),
       "restate/letter": (res["vanilla/restate/letter"]["acc"], res["ctx/restate/letter"]["acc"]),
       "use/letter": (res["vanilla/use/letter"]["acc"], res["ctx/use/letter"]["acc"])}
for arm in ("R-kl", "P-kl", "M-kl", "R-ce", "P-ce"):
    if f"{arm}|use|free" not in table:
        continue
    out = []
    for k, (v0, vc) in ref.items():
        kind, mode = k.split("/")
        raw = table[f"{arm}|{kind}|{mode}"]["acc_mean"]
        out.append(f"{kind[:4]}/{mode[0]}: {100*(raw-v0)/(vc-v0):+.0f}%")
    print(f"  {arm:5s} " + "  ".join(out))

json.dump({k: {kk: vv for kk, vv in v.items() if kk != "item_mean"} for k, v in table.items()},
          open(ART / "analysis_table.json", "w"), indent=1)
