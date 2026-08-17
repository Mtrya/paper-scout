#!/bin/bash
# PSG-JEPA (lambda_g=0.1) training as a batch job (no 24h notebook limit).
set -ex
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/psgjepa
cd $W
export PYTHONPATH=$W/repo
echo "=== preflight: python/torch ==="
./.venv/bin/python -c "import sys,torch,transformers,lightning,torchvision,stable_worldmodel,psgjepa; print(sys.version); print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
mkdir -p /tmp/stablewm/datasets/ogbench
cp data/stablewm/datasets/ogbench/cube_single_expert.h5 /tmp/stablewm/datasets/ogbench/
export STABLEWM_HOME=/tmp/stablewm
cd repo
export PYTHONPATH=.
../.venv/bin/python train.py data=ogb_cm subdir=$W/out_psg num_workers=8 trainer.devices=1
echo PSG_JOB_DONE
