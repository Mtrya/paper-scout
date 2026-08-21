#!/bin/bash
# 双 worker 并行评测编排(与 run_all.sh 同一 JSON,幂等)
# 用法: bash run_parallel.sh <phase> <worker>
#   phase=probe: worker=A → raw+15, rescue+15/s+, rescue-15/s+ ; worker=B → raw-15, rescue+15/s-, rescue-15/s-
#   phase=sweep: worker=A → raw 全部 ; worker=B → rescue 全部(符号取 RESCUE_SIGN)
set -u
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/openvla-viewprobe
export OPENVLA_ROOT=$W/openvla
export HF_HOME=$W/cache/hf
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export LIBERO_ENV_GPU_ID=0
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
CKPT=$W/models/openvla-7b-finetuned-libero-spatial
PY=$W/.venv/bin/python
JSON=$W/results/viewprobe_results.json   # 默认;probe/sweep 双 worker 各用 *_A/_B.json 避免并发互踩
mkdir -p $W/results $W/logs

run_cfg() {
  OPENVLA_ROOT=$W/openvla $PY $W/code/run_viewprobe.py \
    --task-suite libero_spatial --tasks 0 1 \
    --checkpoint $CKPT --num-trials $4 \
    --mode $1 --theta-deg $2 --rescue-sign $3 \
    --out-json $JSON --tag "$5"
}

PHASE=$1; WORKER=$2
JSON=$W/results/viewprobe_results_${WORKER}.json
case $PHASE:$WORKER in
  probe:A)
    run_cfg rescue 15 1 5 "probe-rescue-p15-s+"
    run_cfg rescue -15 1 5 "probe-rescue-m15-s+"
    run_cfg raw 15 1 5 "probe-raw-p15"
    ;;
  probe:B)
    run_cfg rescue 15 -1 5 "probe-rescue-p15-s-"
    run_cfg rescue -15 -1 5 "probe-rescue-m15-s-"
    run_cfg raw -15 1 5 "probe-raw-m15"
    ;;
  sweep:A)
    run_cfg baseline 0 1 5 "pil-baseline"
    for th in 5 10 15 -5 -10 -15; do
      run_cfg raw $th 1 5 "raw-th${th}"
    done
    ;;
  sweep:B)
    for th in 5 10 15 -5 -10 -15; do
      run_cfg rescue $th ${RESCUE_SIGN:-1} 5 "rescue-th${th}"
    done
    ;;
esac
echo "WORKER_${PHASE}_${WORKER}_DONE"
