#!/bin/bash
# Download LARYBench regression subsets (CALVIN + VLABench) from HF mirror.
set -u
BASE="https://hf-mirror.com/datasets/meituan-longcat/LARYBench/resolve/main/regression"
OUT="$(dirname "$0")"
mkdir -p "$OUT/calvin/train_stride5" "$OUT/calvin/val_stride5" "$OUT/vlabench"

dl() { # url out
  if [ -s "$2" ]; then echo "skip $2"; return 0; fi
  curl -sL -C - --retry 5 --retry-delay 3 -o "$2" "$1" && echo "ok $2" || echo "FAIL $1"
}

dl "$BASE/calvin/val_stride5/val_stride5.progress" "$OUT/calvin/val_stride5/val_stride5.progress"
dl "$BASE/calvin/train_stride5/train_stride5.progress" "$OUT/calvin/train_stride5/train_stride5.progress"
for i in $(seq -w 0 11); do
  dl "$BASE/calvin/val_stride5/val_stride5_part000${i}.tar.gz" "$OUT/calvin/val_stride5/val_stride5_part000${i}.tar.gz"
done
for i in $(seq -w 0 9); do
  dl "$BASE/calvin/train_stride5/train_stride5_part000${i}.tar.gz" "$OUT/calvin/train_stride5/train_stride5_part000${i}.tar.gz"
done
dl "$BASE/vlabench/vlabench_part00000_train.tar.gz" "$OUT/vlabench/vlabench_part00000_train.tar.gz"
dl "$BASE/vlabench/vlabench_part00000_val.tar.gz" "$OUT/vlabench/vlabench_part00000_val.tar.gz"
echo DONE
