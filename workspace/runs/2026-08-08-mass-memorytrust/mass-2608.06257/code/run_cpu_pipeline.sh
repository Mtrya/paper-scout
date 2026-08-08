#!/bin/bash
# Full CPU pipeline for the MASS reproduction. Idempotent: each stage skips if
# its marker/output exists, so it can be re-launched after interruptions.
# Order chosen to produce the core reproduction table as early as possible.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=results/pipeline.log
mkdir -p results

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# 1. typed Logic Engine, 20k updates (checkpoints every 5k)
if [ ! -f ckpt_typed.pt ]; then
  say "train typed: start"
  OMP_NUM_THREADS=12 $PY train_typed.py --steps 20000 --out ckpt_typed.pt \
    >> results/train_typed.log 2>&1
  say "train typed: done ($?)"
else
  say "train typed: skip (ckpt exists)"
fi

# 2. typed eval on val (16 eps, H=128)
if [ -f ckpt_typed.pt ] && [ ! -f results/eval.json ]; then
  say "eval typed: start"
  OMP_NUM_THREADS=12 $PY eval_rollout.py --typed ckpt_typed.pt --horizon 128 \
    >> results/eval_typed.log 2>&1
  say "eval typed: done ($?)"
fi

# 3. probe 1 (drift localization) + probe 2 (determinism isolation)
if [ -f ckpt_typed.pt ] && [ ! -f results/probe1_drift.json ]; then
  say "probe1: start"
  OMP_NUM_THREADS=12 $PY probe1_drift.py --horizon 128 >> results/probe1.log 2>&1
  say "probe1: done ($?)"
fi
if [ -f ckpt_typed.pt ] && [ ! -f results/probe2_determinism.json ]; then
  say "probe2: start"
  OMP_NUM_THREADS=12 $PY probe2_determinism.py --horizon 128 >> results/probe2.log 2>&1
  say "probe2: done ($?)"
fi

# 4. dense baseline 20k + in parallel probe 3 (attractor, H=1024)
if [ ! -f ckpt_dense.pt ]; then
  say "train dense: start"
  OMP_NUM_THREADS=8 $PY train_dense.py --steps 20000 --out ckpt_dense.pt \
    >> results/train_dense.log 2>&1 &
  DENSE_PID=$!
  if [ -f ckpt_typed.pt ] && [ ! -f results/probe3_attractor.json ]; then
    say "probe3 (H=1024): start"
    OMP_NUM_THREADS=4 $PY probe3_attractor.py --horizon 1024 --episodes 4 \
      >> results/probe3.log 2>&1
    say "probe3: done ($?)"
  fi
  wait $DENSE_PID
  say "train dense: done ($?)"
else
  say "train dense: skip (ckpt exists)"
fi

# 5. dense eval
if [ -f ckpt_dense.pt ] && [ ! -f results/eval_dense.json ]; then
  say "eval dense: start"
  OMP_NUM_THREADS=12 $PY eval_rollout.py --dense ckpt_dense.pt --horizon 128 \
    --tag _dense >> results/eval_dense.log 2>&1
  say "eval dense: done ($?)"
fi

# 6. optional: attractor at paper horizon H=4096 if not already done at 1024 only
if [ -f ckpt_typed.pt ] && [ ! -f results/probe3_attractor_4096.json ]; then
  say "probe3 (H=4096): start"
  OMP_NUM_THREADS=12 $PY probe3_attractor.py --horizon 4096 --episodes 4 \
    --tag _4096 >> results/probe3_4096.log 2>&1
  say "probe3 (H=4096): done ($?)"
fi

say "pipeline complete"
