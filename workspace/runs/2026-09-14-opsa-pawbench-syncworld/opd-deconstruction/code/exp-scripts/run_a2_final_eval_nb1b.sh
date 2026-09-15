set -u
D=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/opd-pawb
cd $D
N=50

run_eval () {  # cond ckpt_step maxwait_min
  cond=$1; step=$2; maxw=$3
  ck="results/a2_$cond/ckpt_$step.pt"
  for i in $(seq 1 $(( maxw * 2 ))); do
    if [ -f "$ck" ] && ! pgrep -f "[a]2_train.py --condition $cond" > /dev/null; then
      break
    fi
    sleep 30
  done
  if [ ! -f "$ck" ]; then
    echo "=== skip $cond: $ck never appeared $(date '+%H:%M:%S') ==="
    return 1
  fi
  echo "=== final-eval $cond (ckpt_$step, n=$N) $(date '+%H:%M:%S') ==="
  .venv/bin/python a2_final_eval.py --cond "$cond" \
    --ckpt "$ck" --n "$N" --k 4 \
    --out "results/a2_${cond}_final.json" \
    > "results/a2_${cond}_final.log" 2>&1
  echo "=== done $cond rc=$? $(date '+%H:%M:%S') ==="
}

run_eval opd 40 5
run_eval opd-oneshot 40 5
run_eval fixed-neg 40 5
run_eval fixed-pos 25 5
run_eval opsa 40 200

echo "=== base eval (n=$N) $(date '+%H:%M:%S') ==="
.venv/bin/python a2_final_eval.py --cond base --model models/Qwen3-0.6B \
  --ckpt none --n "$N" --k 4 --out results/a2_base_final.json \
  > results/a2_base_final.log 2>&1
echo "=== done base rc=$? $(date '+%H:%M:%S') ==="
echo "=== all final evals done $(date '+%H:%M:%S') ==="
