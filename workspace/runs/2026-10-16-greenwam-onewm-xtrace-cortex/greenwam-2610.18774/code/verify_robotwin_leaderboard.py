#!/usr/bin/env python3
"""Place GreenWAM's RoboTwin 2.0 result (86.98 clean / 48.10 randomized / 67.54 mean) on the official board.

The leaderboard page ships its entries as an inline JSON blob; this script reads the page, extracts the
entries, recomputes the Average (c2c+C2R) ranking and reports where 67.54 would sit, which entries the
paper's Table 3 omits, and whether the protocol (50 clean demos per task) is the board's.

    python3 verify_robotwin_leaderboard.py
"""

from __future__ import annotations

import re
import statistics
import urllib.request

PAGE = "https://robotwin-platform.github.io/leaderboard"
ROW = re.compile(r'\{"id":"([^"]+)","name":"([^"]+)","setting":"([^"]+)","easy_mean":([0-9.]+),"hard_mean":([0-9.]+)')
NEWS = re.compile(r'\{"date":"(20[0-9-]+)","text":"([^"]+)"')
GREENWAM = (86.98, 48.10, 67.54)
# the rows of the paper's Table 3 upper block, keyed by leaderboard name
PAPER_TABLE3 = {
    "ME-Dex-1.0", "UniWAM", "OLA-Sem", "GigaBrain-0.7", "WorldScape Policy 2.0", "4D-WAM", "π0.5", "OLA-Geo",
    "Spatial Forcing", "X-WAM", "X-VLA", "Abot-M0", "EventVLA", "Xiaomi Robotics-0", "FastWAM", "GalaxeaVLA",
    "AHA-WAM", "starVLA",
}


def main() -> int:
    html = urllib.request.urlopen(PAGE, timeout=60).read().decode()
    entries = ROW.findall(html)
    news = NEWS.findall(html)
    rows = [(name, setting, float(e), float(h), (float(e) + float(h)) / 2) for _, name, setting, e, h in entries]
    cotrain = sorted([r for r in rows if r[1] == "cotrain"], key=lambda r: -r[4])
    fails = []

    def check(label, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    print(f"{len(rows)} leaderboard entries extracted ({len(cotrain)} co-train, {len(rows)-len(cotrain)} single-task SFT)")
    print(f"board updated: {'; '.join(f'{d} {t}' for d, t in news[:3])}")
    gw = GREENWAM[2]
    rank = 1 + sum(1 for r in cotrain if r[4] > gw)
    check("GreenWAM clean/randomized/mean match the board formula",
          abs((GREENWAM[0] + GREENWAM[1]) / 2 - gw) < 0.02, f"mean {gw}")
    print(f"      GreenWAM 67.54 would rank #{rank} of {len(cotrain)} co-train entries, #{1 + sum(1 for r in rows if r[4] > gw)} of all {len(rows)}")
    for i, r in enumerate(cotrain[:8], 1):
        print(f"        #{i:<2} {r[0]:24s} clean {r[2]:6.2f}  rand {r[3]:6.2f}  avg {r[4]:6.2f}")
    above = [r for r in cotrain if r[4] > gw]
    missing = [r for r in above if r[0] not in PAPER_TABLE3]
    check("entries above GreenWAM that the paper's Table 3 does not list", True,
          "; ".join(f"{r[0]} avg {r[4]:.2f} (clean {r[2]:.2f}, rand {r[3]:.2f})" for r in missing) or "none")
    check("every table-3 name except OLA-Sem (renamed UniWAM) is on the board",
          {"UniWAM"} <= {r[0] for r in rows} and len(PAPER_TABLE3 - {r[0] for r in rows} - {"OLA-Sem"}) == 0,
          str(sorted(PAPER_TABLE3 - {r[0] for r in rows})))
    print()
    print("FAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
