#!/bin/bash
# SyncWorld 探针环境链:基础工具 -> miniconda -> clone -> conda env -> uv sync -> 权重/评测集
# 幂等可重放:每步完成才写 .done 标记(set -e,任何失败不留标记)。
set -euo pipefail
WR=/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/ky26021/embodied-research/syncworld
mkdir -p "$WR"/{cache/uv,cache/hf,envs,logs,bin}
export UV_CACHE_DIR=$WR/cache/uv
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_HOME=$WR/cache/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
LOG=$WR/logs/setup.log
exec >>"$LOG" 2>&1
set -x

# 0. 基础工具(裸 NGC 镜像无 curl/git)
if ! command -v curl >/dev/null || ! command -v git >/dev/null; then
  apt-get update && apt-get install -y curl git ca-certificates
fi

# 1. miniconda
if [ ! -f "$WR/.done.miniconda" ]; then
  if [ ! -x "$WR/miniconda3/bin/conda" ]; then
    curl -LsSf https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
      || curl -LsSf https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$WR/miniconda3"
  fi
  touch "$WR/.done.miniconda"
fi
source "$WR/miniconda3/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# 2. clone repo
if [ ! -f "$WR/.done.clone" ]; then
  [ -d "$WR/SyncWorld/.git" ] || git clone https://github.com/UMass-Embodied-AGI/SyncWorld "$WR/SyncWorld"
  touch "$WR/.done.clone"
fi

# 3. conda env (python 解释器层)
cd "$WR/SyncWorld"
if [ ! -f "$WR/.done.condaenv" ]; then
  conda env create -p "$WR/envs/syncworld" -f environment.yml -y
  touch "$WR/.done.condaenv"
fi
conda activate "$WR/envs/syncworld"

# 4. uv 本体(独立安装,不进环境)
if [ ! -x "$WR/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WR/bin" sh
fi
export PATH="$WR/bin:$PATH"

# 5. uv sync(pinned lockfile,cu130 train 组)
if [ ! -f "$WR/.done.uvsync" ]; then
  UV_PROJECT_ENVIRONMENT="$WR/envs/syncworld" uv sync --frozen --extra train --group=cu130-train
  touch "$WR/.done.uvsync"
fi

# 6. 验证 torch + CUDA
python -c "import torch; print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available())"

# 7. 下载权重 + VAE + 评测集(hf CLI,幂等)
if [ ! -f "$WR/.done.ckpt" ]; then
  hf download yyuncong/SyncWorld --local-dir "$WR/SyncWorld-ckpt" && touch "$WR/.done.ckpt"
fi
if [ ! -f "$WR/.done.vae" ]; then
  hf download Wan-AI/Wan2.2-TI2V-5B --include Wan2.2_VAE.pth --local-dir "$WR/wan22-vae" && touch "$WR/.done.vae"
fi
if [ ! -f "$WR/.done.evalset" ]; then
  hf download yyuncong/SyncWorld-Evaluation --repo-type dataset --local-dir "$WR/SyncWorld-Evaluation" && touch "$WR/.done.evalset"
fi

echo "SETUP_ALL_DONE"
