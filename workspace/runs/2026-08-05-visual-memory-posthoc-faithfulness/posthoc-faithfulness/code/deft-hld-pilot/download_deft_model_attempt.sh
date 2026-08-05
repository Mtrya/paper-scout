#!/usr/bin/env bash
set -euo pipefail

workroot="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/deft-probe"
model_dir="$workroot/models/DEFT-RLVR-model-HF"
base_url="https://hf-mirror.com/hzxllll/DEFT-RLVR-model-HF/resolve/main"

mkdir -p "$model_dir" "$workroot/logs"

files=(
  .gitattributes README.md chat_template.json config.json generation_config.json
  merges.txt model-00001-of-00004.safetensors model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors model-00004-of-00004.safetensors
  model.safetensors.index.json preprocessor_config.json tokenizer.json
  tokenizer_config.json video_preprocessor_config.json vocab.json
)

for file in "${files[@]}"; do
  curl -fL -C - --retry 20 --retry-delay 5 \
    -o "$model_dir/$file" "$base_url/$file"
done

# 2026-08-05：大分片经镜像重定向到当前节点不可达的 HF endpoint，
# model-00001 仅取到约 97MB；本脚本作为精确失败路径保留，未用于最终实验。
