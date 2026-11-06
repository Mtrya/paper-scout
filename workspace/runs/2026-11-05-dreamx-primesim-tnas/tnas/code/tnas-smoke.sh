#!/bin/bash
# TNAS smoke: 1 example, small alpha grid, validates mechanics end to end.
set -x
W=/inspire/qb-ilm/project/exploration-topic/ky26021/tnas-1105
cd $W
export HF_HOME=$W/cache/hf
export HF_HUB_OFFLINE=1
.venv/bin/python tnas_probe.py --work $W --n 1 --alphas 0.5 1 2 --out $W/tnas_smoke.json
echo SMOKE_EXIT=$?
