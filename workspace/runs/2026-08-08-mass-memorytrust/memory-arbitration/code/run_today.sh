#!/usr/bin/env bash
# Qwen3-VL-8B full run (nf4, vision tower bf16), then GLM p3 (text-only short tasks).
# Idempotent resume. Usage:
#   setsid nohup bash run_today.sh > logs/run_today.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/raw
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {  # $1 model-dir $2 name $3 probes $4 modes $5 maxtok $6 limit
  for attempt in 1 2 3 4 5; do
    .venv/bin/python -u code/runner_hf.py \
      --model-path "$1" --model-name "$2" --data code/data \
      --out "results/raw/$2.jsonl" --load nf4 --skip-modules visual \
      --max-mem-gib 8 --batch-size 2 --max-tokens "$5" \
      --probes "$3" ${4:+--modes "$4"} ${6:+--limit "$6"} && break
    echo "=== $2 attempt $attempt failed $(date -Is), retry in 60s ==="
    sleep 60
  done
}

echo "=== $(date -Is) Qwen p1+p2"
run models/Qwen3-VL-8B-Instruct qwen3vl8b p1,p2 "" 1200 ""
echo "=== $(date -Is) Qwen p3"
run models/Qwen3-VL-8B-Instruct qwen3vl8b p3 "" 600 ""
echo "=== $(date -Is) GLM p3"
run models/GLM-4.1V-9B-Thinking glm41v9b p3 "" 2000 ""
echo "=== $(date -Is) STAGE1 DONE"
