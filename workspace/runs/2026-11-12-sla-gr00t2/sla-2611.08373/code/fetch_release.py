#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def fetch_small(url, output, expected_size, retries=20):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = response.read()
            if len(payload) != expected_size:
                raise RuntimeError(f"expected {expected_size} bytes, received {len(payload)}")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(output)
            return
        except Exception as error:
            print(f"small-file retry {attempt + 1}/{retries}: {output.name}: {error!r}", flush=True)
            time.sleep(min(20, attempt + 2))
    raise RuntimeError(f"failed to download {url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--connections", type=int, default=32)
    args = parser.parse_args()
    items = json.loads(args.manifest.read_text())
    for index, item in enumerate(items, 1):
        relative = Path(item["path"])
        expected_size = int(item["size"])
        output = args.output / relative
        if output.exists() and output.stat().st_size == expected_size:
            print(f"[{index}/{len(items)}] verified size: {relative}", flush=True)
            continue
        if output.exists():
            output.unlink()
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(items)}] {relative} ({expected_size / 1e6:.1f} MB)", flush=True)
        if expected_size <= 5_000_000:
            fetch_small(item["url"], output, expected_size)
        else:
            for attempt in range(20):
                try:
                    subprocess.run(
                        [sys.executable, str(Path(__file__).with_name("pget.py")), item["url"], str(output), str(args.connections)],
                        check=True,
                    )
                    if output.stat().st_size != expected_size:
                        raise RuntimeError(f"size mismatch: {output.stat().st_size} != {expected_size}")
                    break
                except Exception as error:
                    print(f"large-file retry {attempt + 1}/20: {relative}: {error!r}", flush=True)
                    time.sleep(10)
            else:
                raise RuntimeError(f"failed to download {relative}")
    print("ALL_FILES_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
