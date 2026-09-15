set -u
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021
D=$W/opd-pawb
PY=$D/.venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com
cd $D

run_cond() {  # run_cond <cond> <steps>
  local cond=$1 steps=$2
  local OUT=results/a2_$cond
  if [ ! -s $OUT/log.json ]; then
    mkdir -p $OUT
    echo "=== A2 $cond ($steps steps) start $(date '+%H:%M:%S') ==="
    $PY a2_train.py --condition $cond \
      --student models/Qwen3-0.6B --teacher models/Qwen3-1.7B \
      --train-data data/dapo-math-17k.jsonl --eval-data data/math500-subset.jsonl \
      --n-train 300 --n-eval 20 --steps $steps --batch 12 --max-new 2048 \
      --eval-every 20 --out $OUT > $OUT/run.log 2>&1
    local rc=$?
    echo "=== A2 $cond done $(date '+%H:%M:%S') rc=$rc ==="
  else
    echo "=== A2 $cond already done ==="
  fi
}

run_cond opd 40
run_cond opd-oneshot 40
run_cond fixed-neg 40
run_cond opsa 40
run_cond fixed-pos 25

echo "=== A2 done $(date '+%H:%M:%S') ==="
