#!/usr/bin/env bash
# SimWAM 推理探针远端环境:venv + 最小依赖 + 权重下载。
# 在实例上执行:bash remote_setup.sh <workroot>
set -euo pipefail

W="${1:?usage: bash remote_setup.sh <workroot>}"
ROOT="$W/embodied-research/simwam"
cd "$ROOT"

export HF_HOME="$W/cache/hf"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

python3 -m venv --system-site-packages .venv
PIP=.venv/bin/pip
$PIP config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
$PIP install --no-cache-dir \
  "numpy==1.26.4" "scipy==1.14.1" \
  omegaconf hydra-core einops safetensors \
  "transformers==4.49.0" sentencepiece protobuf pillow

mkdir -p checkpoints
if [ "$(stat -c %s checkpoints/SimWAM.pt 2>/dev/null || echo 0)" != "12041719353" ]; then
  for i in 1 2 3 4 5 6 7 8; do
    curl -sL -C - --retry 5 --retry-delay 10 -o checkpoints/SimWAM.pt \
      "https://hf-mirror.com/H-EmbodVis/SimWAM/resolve/main/weights/SimWAM.pt" || true
    [ "$(stat -c %s checkpoints/SimWAM.pt 2>/dev/null || echo 0)" = "12041719353" ] && break
    sleep 10
  done
fi
echo "SimWAM.pt size=$(stat -c %s checkpoints/SimWAM.pt)"
echo SETUP_DONE
