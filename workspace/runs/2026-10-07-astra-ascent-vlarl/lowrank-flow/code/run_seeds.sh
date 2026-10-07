#!/bin/bash
# Full C1-C5 pipeline for one or more seeds. Usage: bash run_seeds.sh "0 1 2 3 4"
set -u
PY="${PY:-python3}"  # scout-exp venv python if rebuilt; override via $PY
cd "$(dirname "$0")"
SEEDS="${1:-0 1 2 3 4}"
JOBS="${2:-3}"
COMMON="--n-demos 2500 --demo-noise 0.15 --max-steps 60 --K 10 --pred-mode x --eval-episodes 200"

run_seed () {
  local s=$1
  local out="out/seed$s"
  mkdir -p "$out"
  echo "=== seed $s BC $(date +%T)"
  PYTHONPATH=. $PY train_bc.py --seed "$s" --out "$out" $COMMON --steps 6000 \
      --lr 2e-3 --threads 3 > "$out/bc.log" 2>&1
  echo "=== seed $s RL $(date +%T)"
  PYTHONPATH=. $PY train_rl.py --seed "$s" --out "$out" $COMMON \
      --iterations 40 --sigma 0.5 --lr 1e-4 --lr-vf 1e-3 --n-envs 64 \
      --decisions 30 --bc-match-steps 640 --branch-steps-long 2560 --threads 3 > "$out/rl.log" 2>&1
  echo "=== seed $s done $(date +%T)"
}

pids=()
for s in $SEEDS; do
  run_seed "$s" &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
done
wait
echo "ALL SEEDS DONE $(date +%T)"
