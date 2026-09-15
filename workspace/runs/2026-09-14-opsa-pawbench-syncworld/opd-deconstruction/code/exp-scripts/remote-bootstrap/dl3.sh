#!/bin/bash
# dl3: hf-mirror via huggingface-cli for 14B-Instruct + Tuwhy OPSA + MATH-500
set -u
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021
D=$W/opd-pawb
export HF_ENDPOINT=https://hf-mirror.com
PY=$D/.venv/bin/python

echo "== Qwen3-14B-Instruct-2507 =="
$PY -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen3-14B-Instruct-2507 \
  --local-dir $D/models/Qwen3-14B-Instruct 2>&1 | tail -2 || true

echo "== Tuwhy/Qwen3-1.7B-OPSA =="
$PY -m huggingface_hub.commands.huggingface_cli download Tuwhy/Qwen3-1.7B-OPSA \
  --local-dir $D/models/Qwen3-1.7B-OPSA 2>&1 | tail -2 || true

echo "== MATH-500 =="
if [ ! -s $D/data/math500.parquet ]; then
  $PY -m huggingface_hub.commands.huggingface_cli download HuggingFaceH4/MATH-500 \
    test-00000-of-00001.parquet --local-dir $D/data/math500-dl 2>&1 | tail -2 || true
  cp $D/data/math500-dl/test-00000-of-00001.parquet $D/data/math500.parquet 2>/dev/null || true
fi
$PY - <<'EOF'
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
echo "== dl3 done =="
du -sh $D/models/*
