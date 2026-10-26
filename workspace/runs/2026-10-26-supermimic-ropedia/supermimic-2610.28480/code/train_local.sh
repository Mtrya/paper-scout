#!/bin/bash
set -euo pipefail

env -u PYTHONPATH .venv/bin/sm-train \
  --run-name train30_1024 \
  --data.clips $(cat data/splits/train30.txt | tr '\n' ' ') \
  --env.num-envs 1024 \
  --env.anchor-obs \
  --policy-trainer.batch-size 1024 \
  --wm-trainer.batch-size 1024 \
  --schedule.total-iterations 32000 \
  --schedule.checkpoint-interval 2000
