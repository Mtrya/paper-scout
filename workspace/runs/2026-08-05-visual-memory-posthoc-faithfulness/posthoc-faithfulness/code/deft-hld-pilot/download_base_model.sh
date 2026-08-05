#!/usr/bin/env bash
set -euo pipefail

workroot="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/deft-probe"
model_dir="$workroot/models/Qwen3-VL-8B-Instruct"
base_url="https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct/resolve/master"

mkdir -p "$model_dir" "$workroot/logs"

files=(
  .gitattributes README.md chat_template.json config.json configuration.json
  generation_config.json merges.txt model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors model.safetensors.index.json
  preprocessor_config.json tokenizer.json tokenizer_config.json
  video_preprocessor_config.json vocab.json
)

for file in "${files[@]}"; do
  curl -fsSL -C - --retry 20 --retry-delay 5 \
    -o "$model_dir/$file" "$base_url/$file"
done

python - <<'PY'
from pathlib import Path

root = Path("/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/deft-probe/models/Qwen3-VL-8B-Instruct")
expected = {
    "model-00001-of-00004.safetensors": 4_902_275_944,
    "model-00002-of-00004.safetensors": 4_915_962_496,
    "model-00003-of-00004.safetensors": 4_999_831_048,
    "model-00004-of-00004.safetensors": 2_716_270_024,
}
for name, size in expected.items():
    actual = (root / name).stat().st_size
    if actual != size:
        raise SystemExit(f"size mismatch for {name}: {actual} != {size}")
print("BASE_MODEL_DOWNLOAD_OK")
PY
