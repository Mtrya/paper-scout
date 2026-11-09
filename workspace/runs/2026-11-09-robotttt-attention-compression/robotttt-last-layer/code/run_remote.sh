#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/inspire/qb-ilm/project/exploration-topic/ky26021/attention-1109}"
CODE="$ROOT/code/robotttt-1109"
PY="$ROOT/.venv/bin/python"
mkdir -p "$ROOT/results/toy/ckpts"
cd "$CODE"
if [ ! -s "$ROOT/results/toy/pair_T512_2500.json" ]; then
  "$PY" train.py --arms ttt ttt_last --T 512 --iters 2500 --batch 32 --seg 64 \
    --lr 3e-4 --d 64 --layers 2 --seed 0 --threads 8 --device cpu \
    --eval-episodes 64 --eval-batch 32 --eval-seed 999 --eval-long 1024 \
    --diag-T 1024 --diag-batch 16 --log-every 20 \
    --save-ckpt-dir "$ROOT/results/toy/ckpts" \
    --out "$ROOT/results/toy/pair_T512_2500.json"
fi
if [ ! -s "$ROOT/results/toy/latency_d256.json" ]; then
  "$PY" latency.py --d 256 --T 2048 --reps 3 --threads 1 \
    --out "$ROOT/results/toy/latency_d256.json"
fi
touch "$ROOT/results/toy/toy.ready"
