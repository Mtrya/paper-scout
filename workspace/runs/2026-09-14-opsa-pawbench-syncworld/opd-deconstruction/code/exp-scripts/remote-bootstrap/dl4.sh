#!/bin/bash
# dl4: snapshot_download for 14B-Instruct + Tuwhy (dl3 已部分下载 Tuwhy)
set -u
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021
D=$W/opd-pawb
export HF_ENDPOINT=https://hf-mirror.com
PY=$D/.venv/bin/python
$PY - <<'EOF'
import os
from huggingface_hub import snapshot_download
jobs = [
    ("Qwen/Qwen3-14B-Instruct-2507", "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/opd-pawb/models/Qwen3-14B-Instruct"),
    ("Tuwhy/Qwen3-1.7B-OPSA", "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/opd-pawb/models/Qwen3-1.7B-OPSA"),
]
for repo, dst in jobs:
    print("downloading", repo, flush=True)
    try:
        snapshot_download(repo, local_dir=dst, max_workers=8)
        print("done", repo, flush=True)
    except Exception as e:
        print("FAIL", repo, repr(e)[:300], flush=True)
EOF
echo "== dl4 done =="
du -sh $D/models/*
