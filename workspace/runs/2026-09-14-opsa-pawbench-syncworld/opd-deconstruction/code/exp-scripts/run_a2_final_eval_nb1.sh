set -u
D=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/opd-pawb
cd $D

# phase 0: wait (bounded ~1.5h) for the A3 opsa eval on this box to release the GPU
for i in $(seq 1 90); do
  pgrep -f "[a]3_eval.py" > /dev/null || break
  sleep 60
done
sleep 20
echo "=== nb1 free, start A2 final evals $(date '+%H:%M:%S') ==="

run_eval () {  # cond ckpt_step maxwait_min
  cond=$1; step=$2; maxw=$3
  ck="results/a2_$cond/ckpt_$step.pt"
  n=$(( maxw * 2 ))
  for i in $(seq 1 $n); do
    if [ -f "$ck" ] && ! pgrep -f "[a]2_train.py --condition $cond" > /dev/null; then
      break
    fi
    sleep 30
  done
  if [ ! -f "$ck" ]; then
    echo "=== skip $cond: $ck never appeared $(date '+%H:%M:%S') ==="
    return 1
  fi
  echo "=== final-eval $cond (ckpt_$step) $(date '+%H:%M:%S') ==="
  .venv/bin/python a2_final_eval.py --cond "$cond" \
    --ckpt "$ck" --n 100 --k 4 \
    --out "results/a2_${cond}_final.json" \
    > "results/a2_${cond}_final.log" 2>&1
  echo "=== done $cond rc=$? $(date '+%H:%M:%S') ==="
}

run_eval opd 40 5
run_eval opd-oneshot 40 5
run_eval fixed-neg 40 5
run_eval fixed-pos 25 60
run_eval opsa 40 200
echo "=== all final evals done $(date '+%H:%M:%S') ==="
