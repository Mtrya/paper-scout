set -u
D=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/opd-pawb
cd $D
MAXQ=${1:-6}

# wait (bounded ~4h) until opsa training has released this box's GPU
for i in $(seq 1 480); do
  pgrep -f "[a]2_train.py --condition opsa" > /dev/null || break
  sleep 30
done
# and until the main A3 eval on the other box has all 30 questions
for i in $(seq 1 120); do
  n=$(.venv/bin/python -c "
import json
try:
    print(len(json.load(open('results/a3_opsa.json'))['results']))
except Exception:
    print(0)")
  [ "$n" -ge 30 ] && break
  sleep 30
done
sleep 20

IND=$(.venv/bin/python -c "
import json
d=json.load(open('results/a3_opsa.json'))
bad=[r for r in d['results'] if not r['ok']]
# most-truncated failed questions first: those are the ones where the token
# budget is the prime suspect
bad.sort(key=lambda r: -sum(1 for l in r['lens'] if l >= 16384))
u=[r['i'] for r in bad[:$MAXQ]]
print(','.join(str(i) for i in u))")
echo "=== extend eval on questions: $IND  $(date '+%H:%M:%S') ==="
.venv/bin/python a3_eval.py --model models/Qwen3-1.7B-OPSA \
  --data data/aime-2024.jsonl --n 30 --k 4 --max-new 32768 \
  --indices "$IND" --save-text --out results/a3_opsa_ext.json \
  > a3_opsa_ext.log 2>&1
echo "=== extend eval done rc=$? $(date '+%H:%M:%S') ==="
