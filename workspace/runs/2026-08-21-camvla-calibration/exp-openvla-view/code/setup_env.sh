#!/bin/bash
# openvla-viewprobe 环境准备(幂等,可断点续跑)
set -e
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/openvla-viewprobe
export HF_HOME=$W/cache/hf
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

cd $W
[ -d .venv ] || python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo "=== python/venv ==="
python -V
python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== clone repos ==="
if [ ! -d LIBERO ]; then git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git 2>&1 | tail -2; else echo "LIBERO exists"; fi
if [ ! -d openvla ]; then git clone --depth 1 https://github.com/openvla/openvla.git 2>&1 | tail -2; else echo "openvla exists"; fi

echo "=== show requirements ==="
sed -n '1,120p' LIBERO/setup.py 2>/dev/null | head -80
echo "---- openvla pyproject ----"
ls openvla/ | head -30
[ -f openvla/pyproject.toml ] && sed -n '1,150p' openvla/pyproject.toml
echo SETUP_DONE
