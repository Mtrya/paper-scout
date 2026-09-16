#!/bin/bash
# SyncWorld 试点+可观性探针一条龙。幂等:已有输出的 tag 跳过。
# 用法: bash run_probe.sh [pilot|probe]
set -u
WR=/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/ky26021/embodied-research/syncworld
REPO=$WR/SyncWorld
ENV=${ENV:-/root/sw-env}
EVALSET=$WR/SyncWorld-Evaluation/evaluation_libero
OUT=$WR/outputs
export PATH=$WR/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$WR/cache/hf
export WAN_VAE_PATH=$WR/wan22-vae/Wan2.2_VAE.pth
cd $REPO
PY=$ENV/bin/python

# 应用可观性补丁(幂等)
if ! grep -q calib-drop-slots examples/eval_gripperhead_fdm_rollout.py; then
  patch -p1 < $WR/syncworld_drop_slots.patch || exit 1
fi

run_tag() { # tag extra_args...
  tag=$1; shift
  if [ -f "$OUT/$tag/metrics_full_episode.csv" ] || [ -d "$OUT/$tag" ]; then
    echo "[skip] $tag exists"; return 0
  fi
  echo "=== $tag $(date +%H:%M:%S) ==="
  $PY examples/eval_gripperhead_fdm_rollout.py \
    --checkpoint $WR/SyncWorld-ckpt \
    --eval-set $EVALSET \
    --tag $tag --out $OUT \
    --max-episodes ${MAXEP:-2} --episode-indices "${EPIDX:-}" \
    --num-steps ${NSTEPS:-20} --action-cfg-scale ${CFG:-1.0} \
    "$@" 2>&1 | tail -20
}

case "${1:-pilot}" in
  sanity)
    $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    ;;
  pilot)
    # 论文口径,1 episode,完整标定——计时基线
    MAXEP=1 NSTEPS=35 CFG=5.0 run_tag pilot_full
    ;;
  probe)
    # 三条件对照(快速采样预设),同批 episodes
    run_tag full
    run_tag null --calib-null
    run_tag droprot --calib-drop-slots 3,4,5
    ;;
  probe_trans)
    run_tag droptrans --calib-drop-slots 0,1,2
    ;;
esac
echo "DONE ${1:-pilot} $(date +%H:%M:%S)"
