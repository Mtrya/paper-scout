#!/bin/bash
# TACD (arXiv 2610.06617) toy reproduction.
#   stage 1  BC teachers (one per seed, skipped if the checkpoint already exists)
#   stage 2  distillation arms (bare / cd / ta / cdta / cdta1)
# Usage: bash run_all.sh <bench> <seeds> <jobs> <arms> <steps> [tang-sign]
#   bench = clean (single-mode expert, previous round's data) | mm (per-episode circling sign)
set -u
PY="${PY:-.venv/bin/python}"
cd "$(dirname "$0")"
BENCH="${1:-clean}"
SEEDS="${2:-0 1 2}"
JOBS="${3:-4}"
ARMS="${4:-cd ta cdta}"
STEPS="${5:-3000}"
TS="${6:-fixed}"
THREADS="${THREADS:-4}"
BASE="out/$BENCH"
DEMO="--n-demos 2500 --demo-noise 0.15 --max-steps 60 --tang-sign $TS"

mkdir -p "$BASE"
throttle () { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done; }

echo "=== $BENCH stage 1: teachers $(date +%T)"
for s in $SEEDS; do
  out="$BASE/teacher$s"
  [ -f "$out/M_bc.pt" ] && { echo "  teacher$s exists, skip"; continue; }
  mkdir -p "$out"
  PYTHONPATH=. $PY train_bc.py --seed "$s" --out "$out" --K 10 --pred-mode x \
    --eval-episodes 200 --steps 6000 --lr 2e-3 --threads "$THREADS" $DEMO \
    > "$out/bc.log" 2>&1 &
  throttle
done
wait

echo "=== $BENCH stage 2: distillation $(date +%T)"
for s in $SEEDS; do
  for arm in $ARMS; do
    tag="$arm"; LAM=""
    if [ "$arm" = "cdta1" ]; then tag="cdta_lam1"; LAM="--lam 1.0"; fi
    out="$BASE/s${s}_$tag"; mkdir -p "$out"
    PYTHONPATH=. $PY distill.py --arm "$arm" --seed "$s" --out "$out" \
      --teacher "$BASE/teacher$s/M_bc.pt" $DEMO $LAM \
      --steps "$STEPS" --eval-every $((STEPS / 12)) --threads "$THREADS" &
    throttle
  done
  out="$BASE/s${s}_bare"; mkdir -p "$out"
  PYTHONPATH=. $PY distill.py --arm bare --seed "$s" --out "$out" \
    --teacher "$BASE/teacher$s/M_bc.pt" $DEMO --steps 0 \
    --eval-teacher-steps 400 --threads "$THREADS" &
  throttle
done
wait
echo "=== $BENCH ALL DONE $(date +%T)"
