#!/usr/bin/env bash
# Round-trip blind-spot probe: train + probe all Lorenz regimes.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

for r in 0.5 3.0 10.0 20.0 28.0; do
  if [ ! -f "runs/r${r}/model.pt" ]; then
    $PY train.py --r "$r" --out "runs/r${r}"
  fi
  if [ ! -f "results/r${r}/metrics.json" ]; then
    $PY probe.py --r "$r" --run "runs/r${r}" --out "results/r${r}"
  fi
done
echo "ALL DONE"
