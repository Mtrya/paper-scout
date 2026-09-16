#!/usr/bin/env python3
"""感知探针判分:perception/runs/<case>/<channel>/result.json vs perception/cases.json。
Q1: |winding| >= 0.5 -> yes/no;Q2: 精确整数;Q3: A->E0, B->E1(图像通道不问 Q3)。
"""
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
GT = json.load(open(os.path.join(ROOT, "perception", "cases.json")))["cases"]

def truth(c):
    g = c["ground_truth"]
    return {
        "q1": "yes" if abs(g["winding_about_peg"]) >= g["wound_threshold"] else "no",
        "q2": g["self_crossings"],
        "q3": "E0" if g["end_dist_to_peg"]["E0"] <= g["end_dist_to_peg"]["E1"] else "E1",
    }

rows = []
for c in GT:
    case = c["case"]
    t = truth(c)
    for ch in ["numeric", "image"]:
        p = os.path.join(ROOT, "perception", "runs", case, ch, "result.json")
        if not os.path.exists(p):
            rows.append((case, ch, "MISSING", "", "", ""))
            continue
        r = json.load(open(p))
        a = r.get("answer") or {}
        q1 = str(a.get("q1", "")).strip().lower()
        q1_ok = q1 == t["q1"]
        try:
            q2_ok = int(a.get("q2")) == t["q2"]
        except Exception:
            q2_ok = False
        line = [case, ch, f"Q1 {q1 or '?'}/{t['q1']} {'OK' if q1_ok else 'X'}",
                f"Q2 {a.get('q2')}/{t['q2']} {'OK' if q2_ok else 'X'}"]
        if ch == "numeric":
            q3 = {"a": "E0", "b": "E1"}.get(str(a.get("q3", "")).strip().lower(), "?")
            q3_ok = q3 == t["q3"]
            line.append(f"Q3 {q3}/{t['q3']} {'OK' if q3_ok else 'X'}")
        rows.append(tuple(line))

for r in rows:
    print(" | ".join(r))
# 汇总
def score(ch, qidx):
    got = tot = 0
    for c in GT:
        p = os.path.join(ROOT, "perception", "runs", c["case"], ch, "result.json")
        if not os.path.exists(p):
            continue
        a = json.load(open(p)).get("answer") or {}
        t = truth(c)
        tot += 1
        if qidx == "q1":
            got += str(a.get("q1", "")).strip().lower() == t["q1"]
        elif qidx == "q2":
            try:
                got += int(a.get("q2")) == t["q2"]
            except Exception:
                pass
        elif qidx == "q3":
            got += {"a": "E0", "b": "E1"}.get(str(a.get("q3", "")).strip().lower()) == t["q3"]
    return got, tot

for ch in ["numeric", "image"]:
    for q in (["q1", "q2", "q3"] if ch == "numeric" else ["q1", "q2"]):
        g, t = score(ch, q)
        print(f"{ch} {q}: {g}/{t}")
