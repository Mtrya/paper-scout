#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/inspire/qb-ilm/project/exploration-topic/ky26021/attention-1109}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
OUT="$ROOT/results/qkv_full.json"
if [ ! -s "$OUT" ]; then
  "$ROOT/.venv/bin/python" "$ROOT/code/attention-1109/attention-1109/qkv_probe.py" \
    --model-dir "$ROOT/models/Qwen3-0.6B" \
    --text-file "$ROOT/data/wikitext-2-raw-validation.txt" \
    --layers 0 13 27 --sequences 4 --length 512 --device cuda \
    --kappas 0 0.25 0.5 1 2 --rhos 0.0001 0.001 0.01 0.1 \
    --seed 20261109 --out "$OUT"
fi
touch "$ROOT/results/qkv.ready"
