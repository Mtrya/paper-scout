#!/bin/bash
# 下载 openvla-7b-finetuned-libero-spatial(hf-mirror,pget2 断点续传,幂等)
set -u
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/openvla-viewprobe
BASE=https://hf-mirror.com/openvla/openvla-7b-finetuned-libero-spatial/resolve/main
OUT=$W/models/openvla-7b-finetuned-libero-spatial
mkdir -p $OUT
cd $OUT
PY=$W/.venv/bin/python

BIG="model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors"
SMALL="model.safetensors.index.json config.json generation_config.json preprocessor_config.json dataset_statistics.json added_tokens.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json"

for f in $BIG; do
  if [ -f "$f.done" ] && [ -f "$f" ]; then
    echo "skip $f (done)"; continue
  fi
  for a in 1 2 3 4 5; do
    $PY $W/code/pget2.py $BASE/$f $OUT/$f 12 && break
    echo "retry $f (attempt $a)"
    sleep 5
  done
done

for f in $SMALL; do
  if [ -f "$f" ]; then echo "skip $f"; continue; fi
  for a in 1 2 3 4 5; do
    curl -sL -C - -o $f $BASE/$f && [ -s "$f" ] && break
    echo "retry small $f (attempt $a)"
    sleep 5
  done
done

echo "=== verify safetensors headers ==="
$PY - <<'EOF'
import json, glob, struct
idx = json.load(open('model.safetensors.index.json'))
print("index keys:", list(idx.keys())[:6])
for w, loc in idx['weight_map'].items():
    pass
shards = idx['weight_map']
files = {}
for w, loc in shards.items():
    files.setdefault(loc, 0)
    files[loc] += 1
print("shards:", {k: f"{v} weights" for k, v in files.items()})
from safetensors import safe_open
for f in sorted(glob.glob('model-*.safetensors')):
    try:
        with safe_open(f, framework='pt') as sf:
            n = len(sf.keys())
        print(f"{f}: OK ({n} tensors)")
    except Exception as e:
        print(f"{f}: FAIL {e}")
EOF
echo DOWNLOAD_ALL_DONE
