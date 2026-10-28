#!/bin/bash
# Milestone evaluation driver for the SuperMimic generalization experiment.
# Waits for each checkpoint and evaluates policy (both splits), drift, and WM nse.
set -euo pipefail
cd "$(dirname "$0")/SuperMimic-Reproduced"
PY=.venv/bin/python
RUN=logs/train30_1024
mkdir -p "$RUN/eval_fixed" "$RUN/drift_fixed"
for it in 16000 24000 32000; do
  ck=$RUN/ckpt_$(printf %07d "$it").pt
  while [ ! -f "$ck" ]; do sleep 60; done
  echo "=== checkpoint $it ==="
  for split in heldout10 train30; do
    out=$RUN/eval_fixed/eval_${split}_$(printf %07d "$it").json
    [ -f "$out" ] || $PY scripts/eval_benchmark.py --ckpt "$ck" --clip-list data/splits/${split}.txt --num-envs 256 --out "$out"
  done
  out=$RUN/drift_fixed/drift_heldout10_$(printf %07d "$it").json
  [ -f "$out" ] || $PY scripts/diag_drift.py --ckpt "$ck" --clips $(cat data/splits/heldout10.txt | tr '\n' ' ') --out "$out"
  out=$RUN/wm_nse_$(printf %07d "$it").json
  [ -f "$out" ] || $PY scripts/wm_nse_probe.py --ckpt "$ck" --clips dance2_subject2 dance2_subject1 run2_subject1 sprint1_subject2 --num-envs 4 --out "$out"
  echo "=== done $it ==="
done
echo "ALL MILESTONES DONE"
