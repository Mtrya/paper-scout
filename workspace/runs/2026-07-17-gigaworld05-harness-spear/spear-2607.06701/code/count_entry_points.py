#!/usr/bin/env python3
"""Count SPEAR's hand-crafted RPC server entry points from source.

The paper (Table 1) claims 193 hand-crafted server entry points. Entry points are
registered in cpp/unreal_plugins/SpServices/Source/**.h via calls of the form:

    entry_point_binder->bindFuncToExecuteOnGameThread("<service>", "<name>", lambda...)
    entry_point_binder->bindFuncToExecuteOnWorkerThread("<service>", "<name>", lambda...)

This script counts unique (service, name) pairs, grouped by service.

Usage: python count_entry_points.py <path-to-spear-repo>
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

def main(repo: str) -> None:
    src = Path(repo) / "cpp" / "unreal_plugins" / "SpServices" / "Source"
    pattern = re.compile(
        r'bindFuncToExecuteOn(?:GameThread|WorkerThread)\(\s*"([a-z_]+)"\s*,\s*"([a-z_0-9]+)"'
    )
    services = defaultdict(set)
    files_scanned = 0
    for path in sorted(src.rglob("*.h")):
        text = path.read_text(encoding="utf-8", errors="replace")
        files_scanned += 1
        for m in pattern.finditer(text):
            services[m.group(1)].add(m.group(2))

    total = sum(len(v) for v in services.values())
    print(f"scanned {files_scanned} headers under {src}")
    for svc in sorted(services):
        print(f"  {svc:28s} {len(services[svc]):4d} entry points")
    print(f"  {'TOTAL':28s} {total:4d} unique (service, entry point) pairs")
    print("paper Table 1 claims: 193 hand-crafted functions")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
