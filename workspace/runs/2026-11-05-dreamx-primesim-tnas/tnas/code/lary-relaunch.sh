#!/bin/bash
# Relaunch the LARYBench train-volume chain (pget skips finished files).
W=/inspire/qb-ilm/project/exploration-topic/ky26021/tnas-1105
cd $W
for i in $(seq -w 19 197); do
  until python3 pget.py "https://hf-mirror.com/datasets/meituan-longcat/LARYBench/resolve/main/regression/calvin/train_stride5/train_stride5_part00${i}.tar.gz" "lary_train/train_stride5_part00${i}.tar.gz" 24 >> pget_lary.log 2>&1; do
    echo "RETRY part00${i} $(date -u)" >> pget_lary.log; sleep 10
  done
done
echo LARY_CHAIN_DONE >> pget_lary.log
