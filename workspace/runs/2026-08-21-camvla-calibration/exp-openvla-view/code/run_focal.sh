#!/bin/bash
# 实验 E:焦距扰动编排(双 worker,独立 JSON)
# A: fovy +2.5/+5/+10%;B: fovy -2.5/-5/-10% + crop5% 预处理对照
# 每条件每任务 10 集;θ=0 用已有 PIL 锚点(20/20)
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
mkdir -p $W/results $W/logs

run_cfg() {
  # $1=json $2=extra args...
  OPENVLA_ROOT=$W/openvla $PY $W/code/run_viewprobe.py \
    --task-suite libero_spatial --tasks 0 1 \
    --checkpoint $CKPT --num-trials 10 \
    --mode focal --theta-deg 0 --rescue-sign 1 \
    --out-json "$1" "${@:2}"
}

WORKER=${1:-A}
JSON=$W/results/focal_results_${WORKER}.json
case $WORKER in
  A)
    run_cfg $JSON --fovy-pct 2.5 --tag "fovy-p2.5"
    run_cfg $JSON --fovy-pct 5 --tag "fovy-p5"
    run_cfg $JSON --fovy-pct 10 --tag "fovy-p10"
    ;;
  B)
    run_cfg $JSON --fovy-pct -2.5 --tag "fovy-m2.5"
    run_cfg $JSON --fovy-pct -5 --tag "fovy-m5"
    run_cfg $JSON --fovy-pct -10 --tag "fovy-m10"
    run_cfg $JSON --fovy-pct 0 --crop-pct 5 --tag "crop5"
    ;;
esac
echo "FOCAL_${WORKER}_DONE"
