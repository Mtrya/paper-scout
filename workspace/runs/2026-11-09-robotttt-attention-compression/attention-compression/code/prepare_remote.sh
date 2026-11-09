#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/inspire/qb-ilm/project/exploration-topic/ky26021/attention-1109}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/inspire/qb-ilm/project/exploration-topic/ky26021/cache/hf
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$ROOT"/{models,data,results}
MODEL="$ROOT/models/Qwen3-0.6B"
if [ ! -f "$MODEL/model.safetensors" ]; then
  "$ROOT/.venv/bin/modelscope" download --model Qwen/Qwen3-0.6B --local_dir "$MODEL"
fi
"$ROOT/.venv/bin/python" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
from safetensors import safe_open

root = Path(sys.argv[1])
text_path = root / "data/wikitext-2-raw-validation.txt"
if not text_path.exists():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text_path.write_text("\n".join(ds["text"]), encoding="utf-8")
model_path = root / "models/Qwen3-0.6B/model.safetensors"
with safe_open(model_path, framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    shape = handle.get_slice(keys[0]).get_shape()
config = json.loads((model_path.parent / "config.json").read_text())
assert config["model_type"] == "qwen3"
assert config["num_hidden_layers"] == 28
print({"text_bytes": text_path.stat().st_size, "model_bytes": model_path.stat().st_size,
       "tensor_count": len(keys), "first_tensor_shape": shape,
       "heads": config["num_attention_heads"], "kv_heads": config["num_key_value_heads"]})
(root / "results/stage.ready").touch()
PY
