#!/bin/bash
# Remote setup: venv + packages + model/data downloads (idempotent).
set -u
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021
D=$W/opd-pawb
mkdir -p $D/{models,data,results}
export HF_ENDPOINT=https://hf-mirror.com

# ---- venv (inherit NGC torch 2.7 cu128) ----
if [ ! -d $D/.venv ]; then
  python3 -m venv --system-site-packages $D/.venv
fi
PIP="$D/.venv/bin/pip install -q"
$PIP -i https://pypi.tuna.tsinghua.edu.cn/simple "numpy==1.26.4" "transformers>=4.51,<5" accelerate safetensors modelscope math-verify 2>&1 | tail -3
echo "== venv check =="
$D/.venv/bin/python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)"

# ---- Qwen models via ModelScope (fast) ----
MSC="$D/.venv/bin/modelscope download --model"
for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B-Instruct Qwen/Qwen3-14B-Instruct; do
  tgt=$D/models/$(basename $m)
  if [ ! -f $tgt/config.json ]; then
    echo "== downloading $m =="
    $MSC $m --local_dir $tgt 2>&1 | tail -1
  else
    echo "== $m present =="
  fi
done

# ---- Tuwhy OPSA checkpoint via hf-mirror ----
TGT=$D/models/Qwen3-1.7B-OPSA
if [ ! -f $TGT/config.json ]; then
  echo "== downloading Tuwhy/Qwen3-1.7B-OPSA =="
  $D/.venv/bin/huggingface-cli download Tuwhy/Qwen3-1.7B-OPSA --local-dir $TGT 2>&1 | tail -1
else
  echo "== Tuwhy OPSA present =="
fi

# ---- datasets ----
if [ ! -f $D/data/math500.parquet ]; then
  echo "== downloading MATH-500 =="
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
    print("wrote", out, len(rows), "rows")
EOF
echo "== setup done =="
ls $D/models
