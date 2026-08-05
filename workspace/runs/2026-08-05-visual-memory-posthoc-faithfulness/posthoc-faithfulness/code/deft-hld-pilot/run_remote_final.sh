#!/usr/bin/env bash
set -euo pipefail

workroot="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/deft-probe"
export HF_HOME="$workroot/cache/hf"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$workroot/.venv/bin/python" "$workroot/code/run_commitment_probe.py" \
  --rows-json "$workroot/data/rows-0-100.json" \
  --model "$workroot/models/Qwen3-VL-8B-Instruct" \
  --work-dir "$workroot" \
  --output "$workroot/outputs/commitment_probe_final.jsonl"
