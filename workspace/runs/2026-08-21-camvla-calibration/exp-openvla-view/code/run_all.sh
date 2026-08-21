#!/bin/bash
# OpenVLA×LIBERO 视角扰动全量实验编排(幂等断点续跑)
# 阶段0: baseline θ=0(20 集);阶段1: ±15° rescue 符号探针(各 5 集);
# 阶段2: θ∈{±5,±10,±15} × {raw,rescue} × 20 集(θ=0 复用 baseline)
# rescue sweep 的符号由 RESCUE_SIGN 环境变量传入(探针后决定)。
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
cd $W/openvla

mkdir -p $W/results
JSON=$W/results/viewprobe_results.json
LOGS=$W/logs
mkdir -p $LOGS

run_cfg() {
  # $1=mode $2=theta $3=sign $4=trials $5=tag
  echo "===== $5 : mode=$1 theta=$2 sign=$3 trials=$4 ====="
  OPENVLA_ROOT=$W/openvla $PY $W/code/run_viewprobe.py \
    --task-suite libero_spatial --tasks 0 1 \
    --checkpoint $CKPT --num-trials $4 \
    --mode $1 --theta-deg $2 --rescue-sign $3 \
    --out-json $JSON --tag "$5"
}

STAGE=${1:-all}
case $STAGE in
  phase0)
    run_cfg baseline 0 1 20 "baseline"
    ;;
  probe)
    run_cfg raw 15 1 5 "probe-raw-p15"
    run_cfg raw -15 1 5 "probe-raw-m15"
    run_cfg rescue 15 1 5 "probe-rescue-p15-s+"
    run_cfg rescue 15 -1 5 "probe-rescue-p15-s-"
    run_cfg rescue -15 1 5 "probe-rescue-m15-s+"
    run_cfg rescue -15 -1 5 "probe-rescue-m15-s-"
    ;;
  sweep)
    for th in 5 10 15 -5 -10 -15; do
      run_cfg raw $th 1 20 "raw-th${th}"
      run_cfg rescue $th ${RESCUE_SIGN:-1} 20 "rescue-th${th}"
    done
    ;;
  all)
    run_cfg baseline 0 1 20 "baseline"
    run_cfg raw 15 1 5 "probe-raw-p15"
    run_cfg raw -15 1 5 "probe-raw-m15"
    run_cfg rescue 15 1 5 "probe-rescue-p15-s+"
    run_cfg rescue 15 -1 5 "probe-rescue-p15-s-"
    run_cfg rescue -15 1 5 "probe-rescue-m15-s+"
    run_cfg rescue -15 -1 5 "probe-rescue-m15-s-"
    for th in 5 10 15 -5 -10 -15; do
      run_cfg raw $th 1 20 "raw-th${th}"
      run_cfg rescue $th 1 20 "rescue-th${th}"
    done
    ;;
esac
echo "STAGE_${STAGE}_DONE"
