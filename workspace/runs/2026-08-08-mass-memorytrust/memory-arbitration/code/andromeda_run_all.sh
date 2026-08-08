#!/usr/bin/env bash
# andromeda-side: run both models over the full manifest, sequentially.
# Idempotent resume via runner_hf.py skip-done. Usage:
#   setsid nohup bash andromeda_run_all.sh > logs/run_all.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."   # script lives in code/, project root is one up
mkdir -p logs results/raw

run_model () {  # $1 = weights dir, $2 = short name, $3 = max tokens
  local weights="$1" name="$2" maxtok="$3"
  echo "=== $(date -Is) start $name ==="
  for attempt in 1 2 3 4 5; do
    .venv/bin/python code/runner_hf.py \
      --model-path "$weights" --model-name "$name" \
      --data code/data --out "results/raw/${name}.jsonl" \
      --load int8 --batch-size 4 --max-tokens "$maxtok" \
      && break
    echo "=== $name attempt $attempt failed, sleeping 30s ==="
    sleep 30
  done
  echo "=== $(date -Is) done $name ==="
}

run_model models/Qwen3-VL-8B-Instruct qwen3vl8b 4096
run_model models/GLM-4.1V-9B-Thinking glm41v9b 4096
echo "ALL RUNS DONE"
