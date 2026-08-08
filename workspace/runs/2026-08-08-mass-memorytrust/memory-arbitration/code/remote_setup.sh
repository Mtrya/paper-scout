#!/usr/bin/env bash
# Remote setup on the Inspire instance. Run from <workroot>/memory-arbitration/.
# Idempotent: skips steps whose outputs already exist.
set -euo pipefail
cd "$(dirname "$0")"
W="$(pwd)"
echo "workdir: $W"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/cache/uv}"

# ---- 0. uv bootstrap ----
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# ---- 1. project venv + vllm (PyPI tuna mirror) ----
if [ ! -f .venv/bin/python ]; then
  uv venv .venv --python 3.12
fi
if ! .venv/bin/python -c "import vllm" 2>/dev/null; then
  uv pip install --python .venv/bin/python -i https://pypi.tuna.tsinghua.edu.cn/simple vllm pillow requests
fi
.venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"

# ---- 2. model weights via ModelScope + pget.py ----
dl_model () {  # $1 = modelscope repo, $2 = local dir
  local repo="$1" dir="$2"
  mkdir -p "$dir"
  # file list via ModelScope API
  .venv/bin/python - "$repo" <<'EOF' > /tmp/msfiles.txt
import json, sys, urllib.request
repo = sys.argv[1]
url = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"
d = json.load(urllib.request.urlopen(url, timeout=60))
for f in d["Data"]["Files"]:
    if f["Type"] == "blob":
        print(f["Path"])
EOF
  while read -r path; do
    local out="$dir/$path"
    if [ -f "$out.done" ]; then continue; fi
    mkdir -p "$(dirname "$out")"
    local url="https://modelscope.cn/models/$repo/resolve/master/$path"
    local size
    size=$(.venv/bin/python -c "import urllib.request,sys; r=urllib.request.urlopen(urllib.request.Request('$url', method='HEAD'), timeout=60); print(int(r.headers['Content-Length']))")
    if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then
      touch "$out.done"; echo "skip $path (complete)"; continue
    fi
    if [ "$size" -gt 200000000 ]; then
      .venv/bin/python code/pget.py "$url" "$out" 24
    else
      curl -sL -C - --retry 5 -o "$out" "$url"
    fi
    [ "$(stat -c%s "$out")" = "$size" ] && touch "$out.done" || { echo "SIZE MISMATCH $path"; exit 1; }
  done < /tmp/msfiles.txt
  echo "model ready: $dir"
}

dl_model "Qwen/Qwen3-VL-8B-Instruct" "$W/models/Qwen3-VL-8B-Instruct"
dl_model "ZhipuAI/GLM-4.1V-9B-Thinking" "$W/models/GLM-4.1V-9B-Thinking"
echo "SETUP DONE"
