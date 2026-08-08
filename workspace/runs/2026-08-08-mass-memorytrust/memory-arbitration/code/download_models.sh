#!/usr/bin/env bash
# Download both models from ModelScope to local models/ (idempotent).
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

dl_model () {  # $1 = modelscope repo, $2 = local dir
  local repo="$1" dir="$2"
  mkdir -p "$dir"
  $PY - "$repo" <<'EOF' > /tmp/msfiles_$$.txt
import json, sys, urllib.request
repo = sys.argv[1]
url = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"
d = json.load(urllib.request.urlopen(url, timeout=60))
for f in d["Data"]["Files"]:
    if f["Type"] == "blob":
        print(f["Path"], f["Size"])
EOF
  while read -r path size; do
    local out="$dir/$path"
    [ -f "$out.done" ] && continue
    mkdir -p "$(dirname "$out")"
    local url="https://modelscope.cn/models/$repo/resolve/master/$path"
    if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then
      touch "$out.done"; echo "skip $path"; continue
    fi
    if [ "$size" -gt 200000000 ]; then
      $PY pget.py "$url" "$out" 24 || { echo "PGET FAIL $path"; exit 1; }
    else
      curl -sL --retry 5 -o "$out" "$url" || { echo "CURL FAIL $path"; exit 1; }
    fi
    [ "$(stat -c%s "$out")" = "$size" ] && touch "$out.done" || { echo "SIZE MISMATCH $path"; exit 1; }
  done < /tmp/msfiles_$$.txt
  rm -f /tmp/msfiles_$$.txt
  echo "MODEL READY: $dir"
}

dl_model "Qwen/Qwen3-VL-8B-Instruct" "models/Qwen3-VL-8B-Instruct"
dl_model "ZhipuAI/GLM-4.1V-9B-Thinking" "models/GLM-4.1V-9B-Thinking"
echo "ALL MODELS DONE"
