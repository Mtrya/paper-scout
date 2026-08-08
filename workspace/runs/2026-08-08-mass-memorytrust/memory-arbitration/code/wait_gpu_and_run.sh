#!/usr/bin/env bash
# Wait until the 4060 Ti has >=12GB free (other tenants done), then run the full
# two-model pipeline. Usage:
#   setsid nohup bash wait_gpu_and_run.sh > logs/wait.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
THRESH=13000
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  echo "$(date -Is) free=${FREE}MiB"
  if [ "$FREE" -ge "$THRESH" ]; then
    echo "$(date -Is) GPU free enough, launching pipeline"
    break
  fi
  sleep 120
done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash code/andromeda_run_all.sh
