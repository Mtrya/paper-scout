#!/usr/bin/env bash
# andromeda-side setup: venv + offline wheelhouse install. Idempotent.
# Run from ~/Projects/memory-arbitration/
set -euo pipefail
cd "$(dirname "$0")/.."   # script lives in code/, project root is one up

if [ ! -f .venv/bin/python ]; then
  python3.14 -m venv .venv
fi
.venv/bin/python -m pip install --quiet --no-index --find-links wheelhouse/ \
  torch transformers accelerate bitsandbytes sentencepiece pillow numpy
.venv/bin/python - <<'EOF'
import torch, transformers, accelerate, bitsandbytes
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("bitsandbytes", bitsandbytes.__version__)
EOF
echo "SETUP DONE"
