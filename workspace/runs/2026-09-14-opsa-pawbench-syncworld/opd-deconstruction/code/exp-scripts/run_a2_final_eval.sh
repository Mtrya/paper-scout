set -u
D=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/opd-pawb
cd $D

# wait (bounded, ~4h) for the opsa re-run to finish; [a] trick avoids
# matching this waiter's own command line
for i in $(seq 1 240); do
  if ! pgrep -f "[a]2_train.py --condition opsa" > /dev/null; then
    break
  fi
  sleep 60
done
sleep 30

for spec in "opd 40" "opd-oneshot 40" "fixed-neg 40" "fixed-pos 25" "opsa 40"; do
  set -- $spec
  cond=$1; step=$2
  echo "=== final-eval $cond (ckpt_$step) $(date '+%H:%M:%S') ==="
  .venv/bin/python a2_final_eval.py --cond "$cond" \
    --ckpt "results/a2_$cond/ckpt_$step.pt" \
    --n 100 --k 4 --out "results/a2_${cond}_final.json" \
    > "results/a2_${cond}_final.log" 2>&1
  echo "=== done $cond rc=$? $(date '+%H:%M:%S') ==="
done
echo "=== all final evals done $(date '+%H:%M:%S') ==="
