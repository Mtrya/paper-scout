#!/bin/bash
# 修复 numpy 2.5.2 污染 + bddl 3.6.0 版本错配,并续装剩余依赖(幂等)
set -e
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/openvla-viewprobe
export HF_HOME=$W/cache/hf
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
cd $W
source .venv/bin/activate
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

retry() { for i in 1 2 3 4 5; do if "$@"; then return 0; fi; echo "[retry $i] $*"; sleep 8; done; return 1; }

echo "=== fix1: opencv 降到 numpy1 兼容 ==="
retry pip install "opencv-python-headless<4.11" 2>&1 | tail -1

echo "=== fix2: numpy 回 1.26.4 ==="
retry pip install "numpy==1.26.4" --force-reinstall 2>&1 | tail -1

echo "=== fix3: bddl 回 1.0.1 ==="
retry pip install "bddl==1.0.1" 2>&1 | tail -1

echo "=== 续装 step2: openvla model deps ==="
retry pip install transformers==4.40.1 timm==0.9.10 tokenizers==0.19.1 "sentencepiece>=0.1.99" peft==0.11.1 accelerate==0.33.0 einops huggingface_hub json-numpy jsonlines draccus==0.8.0 rich protobuf matplotlib wandb 2>&1 | tail -1

echo "=== 续装 step3: tensorflow-cpu ==="
retry pip install tensorflow-cpu==2.16.1 2>&1 | tail -1

echo "=== 续装 step4: editable ==="
retry pip install -e $W/openvla --no-deps 2>&1 | tail -1
retry pip install -e $W/LIBERO 2>&1 | tail -1

echo "=== verify ==="
python - <<'EOF'
import numpy, torch
x = torch.ones(3)
try:
    x.numpy()
    print("torch.numpy interop OK")
except Exception as e:
    print("torch.numpy interop FAIL:", e)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy", numpy.__version__)
import numba; print("numba", numba.__version__)
import tensorflow as tf; print("tensorflow", tf.__version__)
import transformers, timm, peft; print("transformers", transformers.__version__)
import mujoco, robosuite; print("mujoco", mujoco.__version__, "robosuite", robosuite.__version__)
import bddl; print("bddl", getattr(bddl, "__version__", "?"))
import libero
from libero.libero.envs import OffScreenRenderEnv
print("libero + OffScreenRenderEnv import OK")
EOF
echo FIX_ALL_DONE
