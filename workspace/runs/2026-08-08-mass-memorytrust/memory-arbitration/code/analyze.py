"""Analysis: parse raw model outputs, compute all probe statistics.

Reads data/tasks.jsonl, data/gold.json and results/raw/<model>.jsonl files;
writes tables to results/tables/ and prints a summary. Pure local, no GPU.
"""
from __future__ import annotations

import difflib
import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(HERE, "results", "raw")
TABLES = os.path.join(HERE, "results", "tables")

# ---------------- parsing ----------------

def extract_json(text: str):
    """Extract first JSON array or object from model output (fence-tolerant)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = t.find(open_ch)
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except Exception:
                        break
    return None


def parse_judgments(response: str) -> dict[str, dict] | None:
    """-> {memory_id: {is_stale, confidence, reasoning}} or None on hard failure."""
    # thinking models: only parse the post-</think> answer section
    answer = response.split("</think>")[-1] if "</think>" in response else response
    obj = extract_json(answer)
    if obj is None and answer is not response:
        obj = extract_json(response)  # fallback: whole response
    if obj is None:
        return None
    if isinstance(obj, dict):
        obj = [obj]
    out = {}
    for item in obj:
        if not isinstance(item, dict):
            continue
        mid = item.get("memory_id")
        stale = item.get("is_stale")
        if mid is None or stale is None:
            continue
        if isinstance(stale, str):
            stale = stale.strip().lower() in ("true", "yes", "stale", "1")
        out[str(mid)] = {
            "is_stale": bool(stale),
            "confidence": item.get("confidence"),
            "reasoning": str(item.get("reasoning", "")),
        }
    return out or None


# ---------------- metrics ----------------

def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def load_model_raw(path: str) -> dict[str, dict]:
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            recs[r["task_id"]] = r
    return recs


def main() -> None:
    os.makedirs(TABLES, exist_ok=True)
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(os.path.join(DATA, "tasks.jsonl"))}
    gold = json.load(open(os.path.join(DATA, "gold.json")))
    models = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(RAW, "*.jsonl")))
    if not models:
        print("no raw outputs found")
        return

    md = ["# Memory-arbitration results\n"]

    # ---------------- probe 1 ----------------
    md.append("## Probe 1: detection P/R/F1 (pooled entry-level; per-seed F1 mean+-std)\n")
    md.append("| model | mode | regime | P | R | F1 | per-seed F1 | parse-fail |")
    md.append("|---|---|---|---|---|---|---|---|")
    for m in models:
        recs = load_model_raw(os.path.join(RAW, f"{m}.jsonl"))
        for mode in ("text", "vision"):
            for regime in ("L1", "L2"):
                tp = fp = fn = 0
                per_seed = defaultdict(lambda: [0, 0, 0])
                parse_fail = 0
                n_tasks = 0
                for tid, g in gold.items():
                    t = tasks[tid]
                    if t["probe"] != "p1" or t["mode"] != mode or t["regime"] != regime:
                        continue
                    if tid not in recs:
                        continue
                    n_tasks += 1
                    pred = parse_judgments(recs[tid]["response"])
                    if pred is None:
                        parse_fail += 1
                        continue
                    for mid, is_stale in g.items():
                        p = pred.get(mid)
                        if p is None:
                            continue
                        s = (t["seed"],)
                        if p["is_stale"] and is_stale:
                            tp += 1; per_seed[s][0] += 1
                        elif p["is_stale"] and not is_stale:
                            fp += 1; per_seed[s][1] += 1
                        elif not p["is_stale"] and is_stale:
                            fn += 1; per_seed[s][2] += 1
                P, R, F1 = prf(tp, fp, fn)
                seed_f1s = [prf(*v)[2] for v in per_seed.values()]
                import statistics
                sf = f"{statistics.mean(seed_f1s):.3f}+-{statistics.stdev(seed_f1s):.3f}" if len(seed_f1s) > 1 else "n/a"
                md.append(f"| {m} | {mode} | {regime} | {P:.3f} | {R:.3f} | {F1:.3f} | {sf} | {parse_fail}/{n_tasks} |")
    md.append("")

    # ---------------- probe 2 ----------------
    md.append("## Probe 2: image ablation (vision mode; per-entry judgment flips vs correct image, reasoning similarity)\n")
    md.append("| model | regime | comparison | flip rate | reason sim (difflib) | reason sim (jaccard) | flag rate correct | flag rate ablated | n |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for m in models:
        recs = load_model_raw(os.path.join(RAW, f"{m}.jsonl"))
        for regime in ("L1", "L2"):
            for abl in ("blank", "mismatch"):
                flips = sims_d = sims_j = 0
                n = 0
                flag_c = flag_a = 0
                for tid_c, g in gold.items():
                    t = tasks[tid_c]
                    if t["probe"] != "p1" or t["mode"] != "vision" or t["regime"] != regime:
                        continue
                    tid_a = tid_c.replace("p1_vision", f"p2_{abl}")
                    if tid_c not in recs or tid_a not in recs:
                        continue
                    pc = parse_judgments(recs[tid_c]["response"])
                    pa = parse_judgments(recs[tid_a]["response"])
                    if pc is None or pa is None:
                        continue
                    for mid in g:
                        if mid not in pc or mid not in pa:
                            continue
                        n += 1
                        sc, sa = pc[mid]["is_stale"], pa[mid]["is_stale"]
                        flips += int(sc != sa)
                        flag_c += int(sc); flag_a += int(sa)
                        rc, ra = pc[mid]["reasoning"], pa[mid]["reasoning"]
                        sims_d += difflib.SequenceMatcher(None, rc, ra).ratio()
                        tc, ta = set(rc.lower().split()), set(ra.lower().split())
                        sims_j += len(tc & ta) / len(tc | ta) if tc | ta else 1.0
                if n:
                    md.append(f"| {m} | {regime} | correct vs {abl} | {flips/n:.3f} | {sims_d/n:.3f} | {sims_j/n:.3f} | {flag_c/n:.3f} | {flag_a/n:.3f} | {n} |")
    md.append("")

    # ---------------- probe 3 ----------------
    md.append("## Probe 3: arbitration weights (text mode; P(judge stale) by factor level)\n")
    md.append("| model | factor | level | P(stale\\|conflict) | P(stale\\|control) | n_conflict | n_control |")
    md.append("|---|---|---|---|---|---|---|")
    for m in models:
        recs = load_model_raw(os.path.join(RAW, f"{m}.jsonl"))
        cells = defaultdict(lambda: [0, 0, 0, 0])  # (factor,level): [stale|conf, n_conf, stale|ctrl, n_ctrl]
        for tid, t in tasks.items():
            if t["probe"] != "p3" or tid not in recs:
                continue
            pred = parse_judgments(recs[tid]["response"])
            if pred is None:
                continue
            p = pred.get(t["memory_ids"][0])
            if p is None:
                continue
            key = (t["factor"], t["level"])
            if t["condition"] == "conflict":
                cells[key][0] += int(p["is_stale"]); cells[key][1] += 1
            else:
                cells[key][2] += int(p["is_stale"]); cells[key][3] += 1
        order = {"F1": ["definitely", "probably", "neutral"], "F2": ["now", "ago", "none"],
                 "F3": ["frequent", "rare", "none"], "F4": ["accurate", "plain"]}
        for factor, levels in order.items():
            for lv in levels:
                sc, nc, sv, nv = cells.get((factor, lv), [0, 0, 0, 0])
                if nc + nv == 0:
                    continue
                md.append(f"| {m} | {factor} | {lv} | {sc/nc:.3f} | {sv/nv:.3f} | {nc} | {nv} |" if nc and nv else
                          f"| {m} | {factor} | {lv} | n/a | n/a | {nc} | {nv} |")
        md.append("")

    out = "\n".join(md)
    with open(os.path.join(TABLES, "RESULTS.md"), "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
