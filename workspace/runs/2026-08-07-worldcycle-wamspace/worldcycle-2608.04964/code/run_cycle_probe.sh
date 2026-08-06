#!/bin/bash
# ABot-World-0 cycle probe runner (remote)
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021
R=$W/embodied-research/ABot-World
export HF_HOME=$W/cache/hf
export CYCLE_OUT=$W/embodied-research/cycle-probe-out
mkdir -p $R/checkpoints
ln -sfn $W/embodied-research/ABot-World-0-5B-LF $R/checkpoints/ABot-World-0-5B-LF
cd $R
$W/embodied-research/.venv/bin/python cycle_probe.py "$@" > $CYCLE_OUT/probe.log 2>&1
echo PROBE_EXIT=$?
tail -5 $CYCLE_OUT/probe.log
