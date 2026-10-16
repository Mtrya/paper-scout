#!/usr/bin/env bash
# Full prospective-query probe. Run from code/cortex-probe (env at code/cortex-env).
# ~35 min end to end on an RTX 3060 12GB.
set -euo pipefail
cd "$(dirname "$0")"
PY=../cortex-env/bin/python

echo "== 1/6 generate rehearsal questions (Qwen2.5-3B-Instruct) =="
$PY probe.py gen --n 32

echo "== 2/6 teacher answers with context (Qwen3-0.6B) =="
$PY probe.py build

echo "== 3/6 manipulation check: classify question kinds =="
$PY probe.py classify

echo "== 4/6 train FFN-only LoRA (5 arms at seed 0, + 3 KL arms at seeds 1-2) =="
for arm in R P M; do
  $PY probe.py train --arm "$arm" --loss kl --seed 0
done
for arm in R P; do
  $PY probe.py train --arm "$arm" --loss ce --seed 0
done
for seed in 1 2; do
  for arm in R P M; do
    $PY probe.py train --arm "$arm" --loss kl --seed "$seed"
  done
done

echo "== 5/6 evaluate (20+20 held-out items x 2 readouts, MMLU 150) =="
$PY probe.py eval --runs R:kl:0 P:kl:0 M:kl:0 R:ce:0 P:ce:0 \
                       R:kl:1 P:kl:1 M:kl:1 R:kl:2 P:kl:2 M:kl:2 --tag main
$PY probe.py diag

echo "== 6/6 summarise, bootstrap, plot =="
$PY probe.py report
$PY analysis.py
$PY plot.py
