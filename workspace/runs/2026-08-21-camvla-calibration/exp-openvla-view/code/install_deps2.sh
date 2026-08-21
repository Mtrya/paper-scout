#!/bin/bash
# openvla LIBERO 评测依赖安装 v2(mujoco 3.9.0 cp312 wheel;TF 2.16.1 py3.12;幂等,每步重试)
set -e
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/openvla-viewprobe
export HF_HOME=$W/cache/hf
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
cd $W
source .venv/bin/activate
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

retry() { for i in 1 2 3 4 5; do if "$@"; then return 0; fi; echo "[retry $i] $*"; sleep 8; done; return 1; }

echo "=== step1: mujoco 3.9.0 + robosuite 1.4.1 + env deps ==="
retry pip install mujoco==3.9.0 2>&1 | tail -1
retry pip install numba 2>&1 | tail -1
retry pip install robosuite==1.4.1 2>&1 | tail -1
retry pip install bddl easydict cloudpickle gym==0.25.2 imageio[ffmpeg] opencv-python-headless pillow pynput termcolor 2>&1 | tail -1

echo "=== step2: openvla model deps ==="
retry pip install transformers==4.40.1 timm==0.9.10 tokenizers==0.19.1 "sentencepiece>=0.1.99" peft==0.11.1 accelerate==0.33.0 einops huggingface_hub json-numpy jsonlines draccus==0.8.0 rich protobuf matplotlib wandb 2>&1 | tail -1

echo "=== step3: tensorflow-cpu 2.16.1 (eval 图像预处理用) ==="
retry pip install tensorflow-cpu==2.16.1 2>&1 | tail -1

echo "=== step4: editable installs ==="
retry pip install -e $W/openvla --no-deps 2>&1 | tail -1
retry pip install -e $W/LIBERO 2>&1 | tail -1

echo "=== verify ==="
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import numpy
print("numpy", numpy.__version__)
import tensorflow as tf
print("tensorflow", tf.__version__)
import transformers, timm, peft
print("transformers", transformers.__version__)
import mujoco, robosuite
print("mujoco", mujoco.__version__, "robosuite", robosuite.__version__)
import libero
print("libero OK")
from libero.libero.envs import OffScreenRenderEnv
print("OffScreenRenderEnv import OK")
EOF
echo INSTALL_V2_DONE
