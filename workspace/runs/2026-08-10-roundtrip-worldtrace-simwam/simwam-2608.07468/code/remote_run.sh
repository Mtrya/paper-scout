#!/usr/bin/env bash
# SimWAM 推理探针运行:prefill 截断敏感性 + 延迟分解。
# 用法:bash remote_run.sh <workroot>
set -euo pipefail

W="${1:?usage: bash remote_run.sh <workroot>}"
ROOT="$W/embodied-research/simwam"
cd "$ROOT"

export HF_HOME="$W/cache/hf"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false

mkdir -p results
.venv/bin/python probe_prefill.py \
  --ckpt checkpoints/SimWAM.pt \
  --scenes probe_assets \
  --out results/prefill_probe.json \
  --k-list 0,3,6,10,15,20,25,30 \
  --steps 20 --seed 42
echo RUN_DONE
