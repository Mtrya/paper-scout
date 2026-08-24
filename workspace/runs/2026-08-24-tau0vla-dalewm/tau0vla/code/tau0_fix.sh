#!/bin/bash
# tau0vla-probe 抢救:杀掉 setup_v2(它正为 lerobot 下载 torch2.13+CUDA 轮子),
# 卸掉 venv 内 pip 版 torch 栈让 NGC torch 透出,lerobot 改 --no-deps + 迭代补缺,
# 装 tau0_vla 包,下载 ckpt。幂等。
set -x
D=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/tau0vla
cd $D
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com

# 0. 杀掉还在跑的 setup_v2 及其 pip 子进程
pkill -f setup_v2.sh || true
sleep 2
pkill -f "pip install" || true
sleep 2

source .venv/bin/activate

# 1. 卸掉 venv 内的 pip 版 torch 栈(只影响 venv site-packages,不动系统)
pip uninstall -y torch torchvision torchcodec triton 2>&1 | tail -2 || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 2. lerobot --no-deps + 迭代补齐缺失模块
python -c "import lerobot" 2>/dev/null || pip install -q --no-input --no-deps lerobot==0.4.1 2>&1 | tail -2
python - <<'EOF'
import subprocess, sys
MAP = {"cv2": "opencv-python-headless", "PIL": "pillow", "yaml": "pyyaml",
       "imageio_ffmpeg": "imageio-ffmpeg", "datasets": "datasets",
       "rerun": "rerun-sdk", "av": "av"}
for _ in range(25):
    try:
        import lerobot  # noqa
        print("LEROBOT_IMPORT_OK")
        break
    except ModuleNotFoundError as e:
        mod = str(e).replace("No module named", "").strip().strip("'").split(".")[0]
        pkg = MAP.get(mod, mod)
        print("installing missing:", mod, "->", pkg, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input", pkg])
    except Exception as e:
        print("LEROBOT_OTHER_ERR", repr(e)[:300])
        break
EOF

# 3. numpy/torchvision ABI 兜底
if ! python -c "import torchvision" 2>/dev/null; then
  pip install -q --no-input "numpy==1.26.4" 2>&1 | tail -2
fi

# 4. tau0_vla 包
python -c "import tau0_vla" 2>/dev/null || pip install -q --no-input -e repo --no-deps 2>&1 | tail -2

# 5. 校验
python - <<'EOF'
import importlib
for m in ["torch", "torchvision", "transformers", "lerobot", "fla", "deepspeed", "accelerate", "peft", "tau0_vla"]:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "FAIL", repr(e)[:200])
import torch
print("cuda:", torch.cuda.is_available())
EOF

# 6. ckpt 下载(6GB)
if [ ! -f checkpoints/tau-0-vla-base/model.safetensors ]; then
  hf download sii-research/tau-0-vla --local-dir checkpoints/tau-0-vla-base || \
  huggingface-cli download sii-research/tau-0-vla --local-dir checkpoints/tau-0-vla-base || true
fi
ls -la checkpoints/tau-0-vla-base/ 2>/dev/null | head -5
echo FIX_DONE_MARKER
