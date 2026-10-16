#!/usr/bin/env python3
"""Print the instructions one spliced probe is made of, plus what the kernel had there.

Usage: show_probe.py <spliced.cubin> <result.json> <site instruction index>
"""

import json
import sys

import sasslib


def main():
    spliced, result_json, site = sys.argv[1], sys.argv[2], str(sys.argv[3])
    result = json.load(open(result_json))
    pcs = result["probe_pcs"][f"site{site}"]
    view = next(iter(sasslib.load_kernels(spliced).values()))
    isize = 16
    print(f"kernel {view.name}: site at instruction index {site}")
    report = result["site_reports"].get(site) or result["site_reports"].get(int(site))
    print(f"splice: {json.dumps(report, default=str)[:400]}")
    print(f"probe = {len(pcs)} instructions at pcs {[hex(p) for p in pcs]}")
    for pc in pcs:
        print(f"   {hex(pc)}  {view.texts[pc // isize]}")
    first = min(pcs) // isize
    print("\nneighbourhood:")
    for i in range(max(0, first - 3), min(view.n, max(pcs) // isize + 4)):
        marker = "probe" if i * isize in pcs else "kernel"
        print(f"   [{marker}] {i:4d} {view.texts[i]}")

    # the trace pointer prologue, spliced at the kernel entry
    init_pcs = result["probe_pcs"].get("init")
    if init_pcs:
        print(f"\nINIT prologue = {len(init_pcs)} instructions:")
        for pc in init_pcs:
            print(f"   {hex(pc)}  {view.texts[pc // isize]}")


if __name__ == "__main__":
    main()
