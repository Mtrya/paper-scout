#!/bin/bash
# run_all.sh: A1 -> A3 -> B (sequential, resumable). Usage: bash run_all.sh
set -u
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021
D=$W/opd-pawb
PY=$D/.venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com
cd $D

# ---------- A1: noise measurement (1.7B student, 4B + 14B teachers) ----------
if [ ! -s results/a1_noise.json ]; then
  echo "=== A1 ==="
  $PY a1_noise.py \
    --student models/Qwen3-1.7B \
    --teachers models/Qwen3-4B-Instruct models/Qwen3-14B-Instruct \
    --data data/dapo-math-17k.jsonl --n 250 --max-att 6 --group 6 \
    --out results/a1_noise.json 2>&1 | tail -20
fi

# ---------- A3: released OPSA checkpoint vs base on AIME24 (avg@4) ----------
if [ ! -s results/a3_base.json ]; then
  echo "=== A3 base ==="
  $PY a3_eval.py --model models/Qwen3-1.7B --data data/aime-2024.jsonl \
    --n 30 --k 4 --max-new 16384 --out results/a3_base.json 2>&1 | tail -8
fi
if [ ! -s results/a3_opsa.json ]; then
  echo "=== A3 opsa ==="
  $PY a3_eval.py --model models/Qwen3-1.7B-OPSA --data data/aime-2024.jsonl \
    --n 30 --k 4 --max-new 16384 --out results/a3_opsa.json 2>&1 | tail -8
fi

# ---------- B: toy stochastic world (train 3 variants + eval) ----------
# (moved to notebook 2 — see run_a2.sh)

echo "=== run_all done ==="
