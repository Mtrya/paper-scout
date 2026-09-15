set -u
D=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/opd-pawb
cd $D

# wait (bounded ~5h) for the 5-condition final-eval chain on this box to end
for i in $(seq 1 600); do
  pgrep -f "[r]un_a2_final_eval_nb1.sh" > /dev/null || break
  sleep 30
done
sleep 20
echo "=== base eval start $(date '+%H:%M:%S') ==="
.venv/bin/python a2_final_eval.py --cond base --model models/Qwen3-0.6B \
  --ckpt none --n 100 --k 4 --out results/a2_base_final.json \
  > results/a2_base_final.log 2>&1
echo "=== base eval done rc=$? $(date '+%H:%M:%S') ==="
