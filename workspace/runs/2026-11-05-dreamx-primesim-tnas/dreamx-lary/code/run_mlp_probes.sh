#!/bin/bash
# LARY-official MLP arms (flat tokens -> residual MLP-4096, Huber 1.0, lr 1e-4 const+10% warmup).
# Run AFTER run_full_probes.sh finishes (GPU is single).
set -u
cd "$(dirname "$0")"
PY=../scout-exp/bin/python
COMMON="--dataset calvin --train_manifest manifests/calvin_train.csv --val_manifest manifests/calvin_val.csv --epochs 40 --batch_size 256 --out results_full"

$PY probe.py $COMMON --train_features feats_full/vae/train --val_features feats_full/vae/val \
  --tag vae_mlp --probe mlp --preload --seed 42

$PY probe.py $COMMON --train_features feats_full/dit/train --val_features feats_full/dit/val \
  --feature_key layer27 --tag dit_l27_pool4_mlp --probe mlp --pool4 --seed 42

$PY probe.py $COMMON --train_features feats_full/dit/train --val_features feats_full/dit/val \
  --feature_key layer27 --tag dit_l27_mean_mlp --probe mlp --meanpool --seed 42

echo MLP_PROBES_DONE
