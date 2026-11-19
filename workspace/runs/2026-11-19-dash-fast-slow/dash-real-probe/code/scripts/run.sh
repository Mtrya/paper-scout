#!/usr/bin/env bash
# Stage runner for the DASH real-weight probe.  Every stage is idempotent; logs go to logs/.
set -euo pipefail
W=${W:?set W to the project workroot}
D=$W/dash-probe
cd "$D"
VP=$D/venv/bin/python
export HF_HOME=$W/cache/hf
export PYTHONPATH=$D/src
M=models/Qwen3-VL-2B-Instruct
B=${BATCH:-16}
ARMS_CORE="sync_nofresh sync_fresh desync8_nofresh desync8_fresh"
ARMS_EXTRA="desync4_nofresh desync4_fresh leak_nofresh leak_fresh"

norm_for() { case "$1" in leak_*) echo packed/norm_leak.json;; *) echo packed/norm.json;; esac; }

stage=${1:-help}
case "$stage" in
  extract)
    $VP src/extract_features.py --data-root data --split train --out feats --model $M --batch $B
    $VP src/extract_features.py --data-root data --split val --out feats --model $M --batch $B
    $VP src/extract_text_pure.py $M feats/text_pure.npz
    ;;
  pack)
    $VP src/pack_cli.py --ep-dir feats --out packed --split train
    $VP src/pack_cli.py --ep-dir feats --out packed --split val
    ;;
  norms)
    $VP src/make_norms.py --packed packed --out packed/norm.json --scene
    ;;
  smoke-train)
    $VP src/train.py --packed packed --norm packed/norm.json --arm sync_nofresh --seed 0 \
        --epochs 1 --smoke --out out
    $VP src/evaluate.py --packed packed --norm packed/norm.json --ckpt out/ckpt/sync_nofresh_s0.pt \
        --arm sync_nofresh --seed 0 --out out --text-pure feats/text_pure.npz --limit 256 --stride 32
    ;;
  train)
    for arm in $ARMS_CORE; do
      for s in 0 1 2; do
        $VP src/train.py --packed packed --norm "$(norm_for $arm)" --arm $arm --seed $s \
            --epochs ${EPOCHS:-15} --out out
      done
    done
    ;;
  train-extra)
    for arm in $ARMS_EXTRA; do
      for s in 0 1 2; do
        $VP src/train.py --packed packed --norm "$(norm_for $arm)" --arm $arm --seed $s \
            --epochs ${EPOCHS:-15} --out out
      done
    done
    ;;
  eval)
    for arm in $ARMS_CORE $ARMS_EXTRA; do
      for s in 0 1 2; do
        $VP src/evaluate.py --packed packed --norm "$(norm_for $arm)" --ckpt out/ckpt/${arm}_s${s}.pt \
            --arm $arm --seed $s --out out --text-pure feats/text_pure.npz --stride 4
      done
    done
    ;;
  analyze)
    $VP src/analyze.py --out out --packed packed
    ;;
  *)
    echo "stages: extract pack norms smoke-train train train-extra eval analyze"
    ;;
esac
