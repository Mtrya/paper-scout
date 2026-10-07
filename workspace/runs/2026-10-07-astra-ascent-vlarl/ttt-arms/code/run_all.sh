#!/bin/bash
# Full 6-arm online-TTT comparison.  Idempotent: each arm resumes from its checkpoint.
set -u
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021
D=$W/embodied-research/ttt-arms
export HF_HOME=$W/cache/hf HF_HUB_OFFLINE=1 TTT_MODEL=$W/cache/models/Qwen3-0.6B
export TTT_LEVEL_SHIFT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $D/code && . $D/.venv/bin/activate
R=${1:-$D/results/run1}
mkdir -p $R
COMMON="--out-dir $R --n-tasks 240 --eval-every 20 --val-per-family 5 --buffer-tasks 8 \
        --steps-per-update 8 --lr 1e-4 --max-seconds 10800"
for arm in frozen loop-imitate rft rft-settlement ascent; do
  echo "########## $arm $(date +%H:%M:%S)"
  python run_arm.py --arm $arm $COMMON --fixed-gen-dir $R 2>&1 | tail -40
done
echo "########## loop-imitate-frozen-gen $(date +%H:%M:%S)"
python run_arm.py --arm loop-imitate-frozen-gen $COMMON --fixed-gen-dir $R 2>&1 | tail -40
echo "ALL ARMS DONE $(date +%H:%M:%S)"
