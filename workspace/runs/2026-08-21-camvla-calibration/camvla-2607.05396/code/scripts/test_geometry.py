#!/usr/bin/env python3
"""A1：几何变换单元测试入口。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from camvla.experiment import run_geometry_unit_tests  # noqa: E402


class _Stdout:
    def write(self, s: str) -> None:
        sys.stdout.write(s)

    def flush(self) -> None:
        sys.stdout.flush()


def main() -> None:
    result = run_geometry_unit_tests(_Stdout())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit("GEOMETRY_FAIL")
    print("GEOMETRY_OK")


if __name__ == "__main__":
    main()
