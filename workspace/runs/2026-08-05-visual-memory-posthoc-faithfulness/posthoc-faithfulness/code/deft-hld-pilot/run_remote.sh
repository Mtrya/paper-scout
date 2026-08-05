#!/usr/bin/env bash
set -euo pipefail

workroot="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/deft-probe"
mkdir -p "$workroot" "$workroot/cache/pip" "$workroot/outputs" "$workroot/logs"

python3 -m venv --system-site-packages "$workroot/.venv"
"$workroot/.venv/bin/pip" config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$workroot/.venv/bin/pip" install --upgrade \
  'transformers>=4.57.0' accelerate 'qwen-vl-utils[decord]' requests

export HF_HOME="$workroot/cache/hf"
export PIP_CACHE_DIR="$workroot/cache/pip"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$workroot/.venv/bin/python" "$workroot/code/run_commitment_probe.py" \
  --rows-json "$workroot/data/rows-0-100.json" \
  --model "$workroot/models/Qwen3-VL-8B-Instruct" \
  --work-dir "$workroot" \
  --output "$workroot/outputs/commitment_probe.jsonl"
