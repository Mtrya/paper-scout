#!/bin/bash
# dl2: fast parallel downloads (ModelScope direct + pget2 for hf-mirror)
set -u
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021
D=$W/opd-pawb
P2=$W/embodied-research/psgjepa/pget2.py
mkdir -p $D/models

ms_dl() {  # ms_dl <repo> <destdir>
  local repo=$1 dst=$2
  mkdir -p $dst
  python3 - "$repo" "$dst" <<'PYEOF'
import json, os, sys, subprocess
repo, dst = sys.argv[1], sys.argv[2]
u = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"
d = json.load(__import__('urllib.request', fromlist=['urlopen']).urlopen(u))
files = d['Data']['Files']
for f in files:
    name, size = f['Name'], f['Size']
    out = os.path.join(dst, name)
    if os.path.exists(out) and abs(os.path.getsize(out) - size) < 10:
        continue
    url = f"https://modelscope.cn/models/{repo}/resolve/master/{name}"
    print("downloading", name, size, flush=True)
    subprocess.run(["curl", "-sL", "-C", "-", "--max-time", "3600", "-o", out, url], check=True)
    if abs(os.path.getsize(out) - size) > 10:
        raise SystemExit(f"size mismatch {name}")
print("done", repo, flush=True)
PYEOF
}

echo "== Qwen3-1.7B =="
ms_dl Qwen/Qwen3-1.7B $D/models/Qwen3-1.7B
echo "== Qwen3-4B-Instruct-2507 =="
ms_dl Qwen/Qwen3-4B-Instruct-2507 $D/models/Qwen3-4B-Instruct

# hf-mirror files via pget2 (multi-thread range downloader)
hf_dl() {  # hf_dl <repo> <filename> <dest>
  local repo=$1 fn=$2 dst=$3
  local url="https://hf-mirror.com/$repo/resolve/main/$fn"
  if [ ! -s "$dst" ]; then
    echo "== $repo/$fn =="
    python3 $P2 "$url" "$dst" 16
  fi
}
mkdir -p $D/models/Qwen3-14B-Instruct $D/models/Qwen3-1.7B-OPSA
for i in 01 02 03 04 05 06 07; do
  hf_dl Qwen/Qwen3-14B-Instruct-2507 model-000$i-of-00007.safetensors $D/models/Qwen3-14B-Instruct/model-000$i-of-00007.safetensors
done
for f in config.json generation_config.json model.safetensors.index.json tokenizer.json tokenizer_config.json merges.txt vocab.json configuration.json; do
  hf_dl Qwen/Qwen3-14B-Instruct-2507 $f $D/models/Qwen3-14B-Instruct/$f
done
for i in 01 02; do
  hf_dl Tuwhy/Qwen3-1.7B-OPSA model-000$i-of-00002.safetensors $D/models/Qwen3-1.7B-OPSA/model-000$i-of-00002.safetensors
done
for f in config.json generation_config.json model.safetensors.index.json tokenizer.json tokenizer_config.json merges.txt vocab.json configuration.json; do
  hf_dl Tuwhy/Qwen3-1.7B-OPSA $f $D/models/Qwen3-1.7B-OPSA/$f
done

echo "== MATH-500 =="
if [ ! -s $D/data/math500.parquet ]; then
  curl -sL --max-time 300 -o $D/data/math500.parquet \
    "https://hf-mirror.com/datasets/HuggingFaceH4/MATH-500/resolve/main/test-00000-of-00001.parquet"
fi
$D/.venv/bin/python - <<'EOF'
import json, os
import pandas as pd
D = "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/opd-pawb/data"
out = os.path.join(D, "math500-subset.jsonl")
if not os.path.exists(out):
    df = pd.read_parquet(os.path.join(D, "math500.parquet"))
    rows = df[["problem", "answer"]].head(100).to_dict("records")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({"prompt": [{"role": "user", "content": r["problem"]}],
                                "label": str(r["answer"])}) + "\n")
    print("wrote math500-subset.jsonl", len(rows))
EOF
echo "== dl2 done =="
du -sh $D/models/*
