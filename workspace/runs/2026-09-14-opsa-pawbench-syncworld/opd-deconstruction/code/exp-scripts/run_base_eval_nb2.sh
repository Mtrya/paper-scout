set -u
D=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/opd-pawb
cd $D

# wait (bounded ~4h) for the opsa re-run to release the GPU on this box
for i in $(seq 1 480); do
  pgrep -f "[a]2_train.py --condition opsa" > /dev/null || break
  sleep 30
done
sleep 20
echo "=== base eval start $(date '+%H:%M:%S') ==="
.venv/bin/python a2_final_eval.py --cond base --model models/Qwen3-0.6B \
  --ckpt none --n 100 --k 4 --out results/a2_base_final.json \
  > results/a2_base_final.log 2>&1
echo "=== base eval done rc=$? $(date '+%H:%M:%S') ==="
