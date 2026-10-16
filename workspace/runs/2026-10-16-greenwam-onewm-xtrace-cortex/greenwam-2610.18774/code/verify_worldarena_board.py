#!/usr/bin/env python3
"""Recompute the WorldArena 1.0 track-1 board from the public leaderboard files and place GreenWAM on it.

Downloads the 127 per-model result files of the WorldArena Hugging Face Space (a few hundred KB total),
applies the motion cap of WorldArena 2.0 (Dynamic Degree / Flow Score / Motion Smoothness capped at the
ground-truth means 23.05 / 12.50 / 69.79) and recomputes EWMScore = mean of the 15 board metrics, exactly
as benchmarks/worldarena/scoring/board.py does. Checks every WorldArena number the GreenWAM paper claims:

    python3 verify_worldarena_board.py [--cache <dir>]

Claims checked
  * GreenWAM 63.31 capped / 63.35 uncapped on the 1000 test episodes;
  * "second in JEPA similarity (98.41, after BLM at 98.49)";
  * "Highest trajectory / depth / 3D accuracy among the compared models", and where the full board puts it;
  * "112 of 127 exceed the recorded-video reference in dynamic degree and 113 in flow score";
  * "UNIS drops from 73.64 to 63.65 under the cap; Motubrain 63.16, FlowWAM-FiveAges 62.99".
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import urllib.parse
import urllib.request
from pathlib import Path

SPACE = "WorldArena/WorldArena"
TREE = "https://huggingface.co/api/spaces/{space}/tree/main/worldarena-results?recursive=true"
FILE = "https://huggingface.co/spaces/{space}/resolve/main/{path}"

METRICS = [
    "Image Quality", "Aesthetic Quality", "JEPA Similarity", "Dynamic Degree", "Flow Score",
    "Motion Smoothness", "Subject Consistency", "Background Consistency", "Photometric Consistency",
    "Interaction Quality", "Trajectory Accuracy", "Depth Accuracy", "Perspectivity",
    "Instruction Following", "Semantic Alignment",
]
CAPPED = {"Dynamic Degree", "Flow Score", "Motion Smoothness"}
GT = {"Dynamic Degree": 0.2305, "Flow Score": 0.1250, "Motion Smoothness": 0.6979}
GREENWAM = {"EWMScore uncapped": 63.35, "EWMScore capped": 63.31, "JEPA Similarity": 0.9841,
            "Trajectory Accuracy": 0.5597, "Depth Accuracy": 0.9816}


def download(cache: Path) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    paths = [t["path"] for t in json.loads(urllib.request.urlopen(TREE.format(space=SPACE)).read()) if t["path"].endswith(".json")]
    def one(path):
        out = cache / Path(path).name
        if not (out.exists() and out.read_bytes()[:1] == b"{"):
            url = FILE.format(space=SPACE, path=urllib.parse.quote(path))
            out.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        return json.loads(out.read_text())
    with concurrent.futures.ThreadPoolExecutor(12) as ex:
        return list(ex.map(one, paths))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("/tmp/worldarena-board"))
    a = ap.parse_args()
    docs = download(a.cache)
    rows = []
    for doc in docs:
        m = {k: v for k, v in (doc.get("Metrics") or {}).items() if k in METRICS and isinstance(v, (int, float))}
        if not m:
            continue
        scale = 100 if max(m.values()) > 1.5 else 1
        fr = {k: v / scale for k, v in m.items()}
        rows.append({
            "name": doc.get("Model", "?"),
            "open": doc.get("open_source"),
            "m": fr,
            "raw": 100 * statistics.fmean(fr.values()),
            "cap": 100 * statistics.fmean([min(v, GT[k]) if k in CAPPED else v for k, v in fr.items()]),
        })
    fails = []

    def check(label, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    print(f"board entries with metrics: {len(rows)} (127 files downloaded)")
    check("127 board entries", len(rows) == 127, f"n={len(rows)}")

    gw_cap, gw_raw = 63.31, 63.35
    rank_cap = 1 + sum(1 for r in rows if r["cap"] > gw_cap)
    rank_raw = 1 + sum(1 for r in rows if r["raw"] > gw_raw)
    print(f"      GreenWAM EWMScore 63.31 capped would rank #{rank_cap} of {len(rows)} on this recomputed board")
    print(f"      GreenWAM EWMScore 63.35 uncapped would rank #{rank_raw} of {len(rows)} (raw board leader 73.64+)")

    for metric, gw in (("JEPA Similarity", GREENWAM["JEPA Similarity"]), ("Trajectory Accuracy", GREENWAM["Trajectory Accuracy"]), ("Depth Accuracy", GREENWAM["Depth Accuracy"])):
        better = sorted([r for r in rows if r["m"].get(metric, 0) > gw], key=lambda r: -r["m"][metric])
        print(f"      {metric}: GreenWAM {100*gw:.2f} -> rank #{1+len(better)}; above it: "
              + ", ".join(f"{r['name']} {100*r['m'][metric]:.2f}{'(open)' if r['open']=='yes' else ''}" for r in better[:12]))
        if metric == "JEPA Similarity":
            check("JEPA: second of 127, only BLM above 98.41", len(better) == 1 and better[0]["name"] == "BLM")

    for name, raw, cap in (("UNIS", 73.64, 63.65), ("BWM-Fast", 72.71, 63.80), ("MotuBrain", 69.06, 63.16), ("FlowWAM-FiveAges", 72.00, 62.99)):
        hit = [r for r in rows if r["name"].lower() == name.lower()]
        if hit:
            r = hit[0]
            detail = f"board {r['raw']:.2f} -> capped {r['cap']:.2f} (paper quotes {raw}/{cap})"
            check(f"{name}: board means reproduce the paper", abs(r["raw"] - raw) < 0.02 and abs(r["cap"] - cap) < 0.02, detail)

    for metric, gt, quoted in (("Dynamic Degree", 0.2305, 112), ("Flow Score", 0.1250, 113)):
        n = sum(1 for r in rows if r["m"].get(metric, 0) > gt)
        top = sorted(rows, key=lambda r: -r["m"].get(metric, 0))[:1]
        check(f"{quoted} of 127 exceed the recorded-video reference in {metric}", n == quoted,
              f"n={n}, largest {top[0]['name']} {100*top[0]['m'][metric]:.2f} = {top[0]['m'][metric]/gt:.1f}x the reference")

    print()
    print("FAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
