#!/bin/bash
# GPU pipeline on andromeda: eval typed -> probe1/2 -> train dense -> eval dense
# -> probe3 (H=1024, H=4096). Idempotent per stage (skips existing outputs).
set -u
cd "$(dirname "$0")"
PY=/home/alpheratz/Projects/chess-transformer/.venv/bin/python
LOG=results/gpu_pipeline.log
mkdir -p results

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

if [ ! -f results/eval.json ]; then
  say "eval typed: start"
  $PY eval_rollout.py --typed ckpt_typed.pt --horizon 128 >> results/eval_typed.log 2>&1
  say "eval typed: done ($?)"
fi

if [ ! -f results/probe1_drift.json ]; then
  say "probe1: start"
  $PY probe1_drift.py --horizon 128 >> results/probe1.log 2>&1
  say "probe1: done ($?)"
fi

if [ ! -f results/probe2_determinism.json ]; then
  say "probe2: start"
  $PY probe2_determinism.py --horizon 128 >> results/probe2.log 2>&1
  say "probe2: done ($?)"
fi

if [ ! -f ckpt_dense.pt ]; then
  say "train dense (GPU, seed 0): start"
  $PY train_dense.py --steps 20000 --seed 0 --out ckpt_dense.pt \
    >> results/train_dense.log 2>&1
  say "train dense: done ($?)"
fi

if [ -f ckpt_dense.pt ] && [ ! -f results/eval_dense.json ]; then
  say "eval dense: start"
  $PY eval_rollout.py --dense ckpt_dense.pt --horizon 128 --tag _dense \
    >> results/eval_dense.log 2>&1
  say "eval dense: done ($?)"
fi

if [ ! -f results/probe3_attractor.json ]; then
  say "probe3 H=1024: start"
  $PY probe3_attractor.py --horizon 1024 --episodes 4 >> results/probe3.log 2>&1
  say "probe3 H=1024: done ($?)"
fi

if [ ! -f results/probe3_attractor_4096.json ]; then
  say "probe3 H=4096: start"
  $PY probe3_attractor.py --horizon 4096 --episodes 4 --tag _4096 \
    >> results/probe3_4096.log 2>&1
  say "probe3 H=4096: done ($?)"
fi

say "gpu pipeline complete"
