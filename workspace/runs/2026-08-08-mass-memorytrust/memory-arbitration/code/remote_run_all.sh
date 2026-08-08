#!/usr/bin/env bash
# Run both models sequentially over the full task manifest. Idempotent resume:
# remote_runner skips task_ids already present in the output jsonl.
# Usage: setsid nohup bash remote_run_all.sh > logs/run_all.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs results/raw

run_model () {  # $1 = weights dir, $2 = short name
  local weights="$1" name="$2"
  echo "=== $(date -Is) start $name ==="
  for attempt in 1 2 3 4; do
    .venv/bin/python code/remote_runner.py \
      --model-path "$weights" --model-name "$name" \
      --data code/data --out "results/raw/${name}.jsonl" \
      && break
    echo "=== $name attempt $attempt failed, sleeping 60s ==="
    sleep 60
  done
  echo "=== $(date -Is) done $name ==="
}

run_model models/Qwen3-VL-8B-Instruct qwen3vl8b
run_model models/GLM-4.1V-9B-Thinking glm41v9b
echo "ALL RUNS DONE"
