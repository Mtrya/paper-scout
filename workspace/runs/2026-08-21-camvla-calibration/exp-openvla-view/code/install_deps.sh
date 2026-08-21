#!/bin/bash
# openvla LIBERO 评测依赖安装(幂等)
set -e
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/openvla-viewprobe
export HF_HOME=$W/cache/hf
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
cd $W
source .venv/bin/activate
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== whoami / egl ==="
whoami; ls /usr/lib/x86_64-linux-gnu/libEGL* /usr/lib/x86_64-linux-gnu/libGL* 2>/dev/null | head || echo "no EGL/GL libs listed"

echo "=== step1: mujoco + robosuite + libero env deps ==="
pip install mujoco==2.3.7 2>&1 | tail -2
pip install robosuite==1.4.1 2>&1 | tail -2
pip install bddl easydict cloudpickle gym==0.25.2 imageio[ffmpeg] 2>&1 | tail -2

echo "=== step2: openvla model deps (pin per openvla pyproject, skip torch/torchvision/tf) ==="
pip install transformers==4.40.1 timm==0.9.10 tokenizers==0.19.1 "sentencepiece>=0.1.99" peft==0.11.1 accelerate==0.33.0 einops huggingface_hub json-numpy jsonlines draccus==0.8.0 rich protobuf matplotlib wandb 2>&1 | tail -2

echo "=== step3: editable installs ==="
pip install -e $W/openvla --no-deps 2>&1 | tail -2
pip install -e $W/LIBERO 2>&1 | tail -2

echo "=== verify ==="
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import numpy
print("numpy", numpy.__version__)
import transformers, timm, tokenizers, peft, accelerate
print("transformers", transformers.__version__)
import mujoco
print("mujoco", mujoco.__version__)
import robosuite
print("robosuite", robosuite.__version__)
import libero
print("libero OK")
EOF
echo INSTALL_ALL_DONE
