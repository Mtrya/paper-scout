#!/bin/bash
# Full-scale probes on CALVIN: DiT(sigma=1.0,mask=1) layers + pool4 + VAE tokens.
set -u
cd "$(dirname "$0")"
PY=../scout-exp/bin/python
TR=manifests/calvin_train.csv
VA=manifests/calvin_val.csv
mkdir -p results_full

run() { # tag feature_dir key extra...  (no --preload: 35GB/arm exceeds RAM)
  local tag=$1 dir=$2 key=$3; shift 3
  for seed in "$@"; do
    $PY probe.py --dataset calvin --train_manifest $TR --val_manifest $VA \
      --train_features feats_full/$dir/train --val_features feats_full/$dir/val \
      --feature_key $key --tag $tag --epochs 40 --batch_size 256 \
      --seed $seed --out results_full
  done
}

# headline arms first
run dit_l27 dit layer27 42 0 1

# DreamX token-budget arm: 2x2 pooled tokens per latent frame
for seed in 42 0 1; do
  $PY probe.py --dataset calvin --train_manifest $TR --val_manifest $VA \
    --train_features feats_full/dit/train --val_features feats_full/dit/val \
    --feature_key layer27 --pool4 --tag dit_l27_pool4 --epochs 40 --batch_size 256 \
    --seed $seed --out results_full
done

# VAE token arm (small batch: 1800 tokens/sample; preload OK: 2.2GB)
for seed in 42 0 1; do
  $PY probe.py --dataset calvin --train_manifest $TR --val_manifest $VA \
    --train_features feats_full/vae/train --val_features feats_full/vae/val \
    --feature_key tokens --tag vae --epochs 40 --batch_size 16 \
    --preload --seed $seed --out results_full
done

# layer curve
run dit_l6  dit layer6  42
run dit_l13 dit layer13 42
run dit_l20 dit layer20 42
echo FULL_PROBES_DONE
