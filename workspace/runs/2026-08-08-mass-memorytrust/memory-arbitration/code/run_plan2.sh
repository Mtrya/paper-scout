#!/usr/bin/env bash
# Optimized pipeline v2 (2026-08-08 14:35; measured ~14.5s/task under contention).
# Coverage: Qwen full-detail on 15 seeds for p1-vision/p2, full for p3;
#           GLM full p3, p1-text 4 seeds tonight; remainders accumulate overnight.
# Idempotent resume throughout. Usage:
#   setsid nohup bash run_plan2.sh > logs/run_plan2.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/raw
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QWEN=models/Qwen3-VL-8B-Instruct
GLM=models/GLM-4.1V-9B-Thinking

run () {  # $1 weights $2 outname $3 probes $4 modes $5 maxtok $6 batch $7 limit
  local extra=""
  [ -n "$4" ] && extra="--modes $4"
  for attempt in 1 2 3 4 5; do
    .venv/bin/python -u code/runner_hf.py \
      --model-path "$1" --model-name "$2" --data code/data \
      --out "results/raw/$2.jsonl" --load nf4 --skip-modules visual \
      --max-mem-gib 8 --batch-size "$6" --max-tokens "$5" \
      --probes "$3" $extra ${7:+--limit "$7"} && break
    echo "=== $2/$3 attempt $attempt failed $(date -Is), retry 60s ==="
    sleep 60
  done
  echo "=== phase done: $2 $3 $4 limit=$7 $(date -Is) ==="
}

echo "=== A1 Qwen p1 vision 15 seeds $(date -Is)"
run $QWEN qwen3vl8b p1 vision 900 4 210
echo "=== A2 Qwen p2 15 seeds $(date -Is)"
run $QWEN qwen3vl8b p2 "" 900 4 420
echo "=== A3 Qwen p3 full $(date -Is)"
run $QWEN qwen3vl8b p3 "" 400 4 ""
echo "=== B1 GLM p3 full $(date -Is)"
run $GLM glm41v9b p3 "" 1500 2 ""
echo "=== B2 GLM p1 text 4 seeds $(date -Is)"
run $GLM glm41v9b p1 text 4096 2 56
echo "=== A4 Qwen p1 text topup-to-10-seeds $(date -Is)"
run $QWEN qwen3vl8b p1 text 900 2 28
echo "=== C1 GLM p1 text remainder $(date -Is)"
run $GLM glm41v9b p1 text 4096 2 ""
echo "=== C2 GLM p1 vision remainder $(date -Is)"
run $GLM glm41v9b p1 vision 4096 2 ""
echo "=== C3 GLM p2 remainder $(date -Is)"
run $GLM glm41v9b p2 "" 4096 2 ""
echo "=== C4 Qwen p1 vision remainder $(date -Is)"
run $QWEN qwen3vl8b p1 vision 900 4 ""
echo "=== C5 Qwen p2 remainder $(date -Is)"
run $QWEN qwen3vl8b p2 "" 900 4 ""
echo "=== C6 Qwen p1 text remainder $(date -Is)"
run $QWEN qwen3vl8b p1 text 900 2 ""
echo "=== ALL PHASES DONE $(date -Is)"
