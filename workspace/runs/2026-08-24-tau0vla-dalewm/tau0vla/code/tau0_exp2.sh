#!/bin/bash
# 实验 A2:τ₀-VLA openloop 评测
#  1) 零样本基线:base ckpt + 借来的 run_spec.json,K=10
#  2) FT500:K = 1,2,4,10 消融
# 幂等:out-dir 存在即跳过。
set -x
D=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/tau0vla
cd $D/repo
V=$D/.venv/bin/python
BASE=$D/checkpoints/tau-0-vla-base
FT=$D/outputs/gong_ft/agibot_world_gong_ft/checkpoint-500
RS=$D/outputs/gong_ft/agibot_world_gong_ft/run_spec.json
MAX=100

# 零样本:base ckpt 缺 run_spec.json,借 post-train 的(同 config 同数据)
if [ ! -f $BASE/run_spec.json ]; then cp $RS $BASE/run_spec.json; fi

run_eval () { # ckpt K out
  if [ -d "$3" ]; then echo "skip $3"; return; fi
  TAU0_NUM_STEPS=$2 $V deploy/openloop.py --ckpt $1 \
    --config agibot_world_gong_ft \
    --out-dir $3 --max-inferences $MAX \
    --policy-module deploy.policy_stepoverride --policy-class Tau0VLAPolicyStepOverride
}

run_eval $BASE 10 $D/tau0_out/zeroshot_k10
for K in 1 2 4 10; do
  run_eval $FT $K $D/tau0_out/ft500_k$K
done
echo EXP2_DONE_MARKER
