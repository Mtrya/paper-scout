#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/inspire/qb-ilm/project/exploration-topic/ky26021/attention-1109}"
for _ in $(seq 1 240); do
  if [ -f "$ROOT/results/toy/toy.ready" ]; then
    "$ROOT/.venv/bin/python" "$ROOT/code/robotttt-1109/spectral_diag.py" \
      --ckpt "$ROOT/results/toy/ckpts/ttt_last_T512.pt" \
      --T 1024 --batch 16 --seed 999 --threads 4 \
      --out "$ROOT/results/toy/spectral_ttt_last_T1024.json"
    touch "$ROOT/results/toy/spectral.ready"
    exit 0
  fi
  sleep 60
done
echo "timed out waiting for toy.ready" >&2
exit 1
