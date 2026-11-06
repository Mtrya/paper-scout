#!/bin/bash
set -u
cd "$(dirname "$0")"
PY=../cosmos-venv/bin/python
echo "=== dit train (full) ==="
$PY extract_cosmos.py --manifest manifests/calvin_train.csv --out_dir feats_full/dit/train --arm dit --sigma 1.0 --cond_mask 1 --layers 6 13 20 27
echo "=== dit val (full) ==="
$PY extract_cosmos.py --manifest manifests/calvin_val.csv --out_dir feats_full/dit/val --arm dit --sigma 1.0 --cond_mask 1 --layers 6 13 20 27
echo "=== vae train (full) ==="
$PY extract_cosmos.py --manifest manifests/calvin_train.csv --out_dir feats_full/vae/train --arm vae
echo "=== vae val (full) ==="
$PY extract_cosmos.py --manifest manifests/calvin_val.csv --out_dir feats_full/vae/val --arm vae
echo EXTRACT_FULL_DONE
