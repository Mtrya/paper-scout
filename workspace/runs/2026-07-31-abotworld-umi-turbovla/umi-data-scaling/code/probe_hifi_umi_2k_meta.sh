#!/usr/bin/env bash
# Probe: verify what signals the public HiFi-UMI-2K dataset actually records.
# Fetches LeRobot v3 meta files from chunk-0000 of simple-world-lab/HiFi-UMI-2K
# and prints the schema (fps, features, state/action layout).
#
# Usage: bash probe_hifi_umi_2k_meta.sh [out_dir]
# The fetched info.json / modality.json are preserved alongside this script
# in hifi-umi-2k-meta/ (fetched 2026-07-31).

set -euo pipefail
OUT="${1:-/tmp/hifi-umi-2k-meta}"
BASE="https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K/resolve/main/chunk-0000/part-0000"
mkdir -p "$OUT"
for f in meta/info.json meta/modality.json; do
  curl -sL "$BASE/$f" -o "$OUT/$(basename "$f")"
done
python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
info = json.load(open(f"{out}/info.json"))
print("fps:", info["fps"], "| episodes(chunk):", info["total_episodes"],
      "| tasks:", info["total_tasks"])
for name, spec in info["features"].items():
    print(f"  {name}: {spec.get('dtype')} {spec.get('shape')}")
print("state_layout:", info["state_layout"]["per_side"])
print("action type:", info["action_layout"]["type"])
PY
