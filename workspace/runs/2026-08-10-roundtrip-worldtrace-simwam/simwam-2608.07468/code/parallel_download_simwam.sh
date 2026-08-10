#!/usr/bin/env bash
# SimWAM.pt 并行分块下载(HF 单连接 ~1MB/s,16 并发拉满出口带宽)
# 用法:bash parallel_download_simwam.sh <workroot>
set -uo pipefail

W="${1:?usage: bash parallel_download_simwam.sh <workroot>}"
ROOT="$W/embodied-research/simwam"
URL="https://huggingface.co/H-EmbodVis/SimWAM/resolve/main/weights/SimWAM.pt"
SIZE=12041719353
NCHUNK=16
DEST="$ROOT/checkpoints/SimWAM.pt"
TMPD="$ROOT/checkpoints/.simwam_chunks"
mkdir -p "$TMPD"

if [ "$(stat -c %s "$DEST" 2>/dev/null || echo 0)" = "$SIZE" ]; then
  echo "already complete: $DEST"
  exit 0
fi

CHUNK=$(( (SIZE + NCHUNK - 1) / NCHUNK ))
pids=()
for i in $(seq 0 $((NCHUNK-1))); do
  s=$((i * CHUNK))
  e=$((s + CHUNK - 1))
  [ $e -ge $SIZE ] && e=$((SIZE - 1))
  (
    want=$((e - s + 1))
    for try in 1 2 3 4 5 6 7 8 9 10; do
      have=$(stat -c %s "$TMPD/chunk_$i" 2>/dev/null || echo 0)
      [ "$have" = "$want" ] && break
      off=$((s + have))
      curl -sL --retry 3 --retry-delay 5 -r "$off-$e" "$URL" >> "$TMPD/chunk_$i" || true
      sleep 3
    done
    have=$(stat -c %s "$TMPD/chunk_$i" 2>/dev/null || echo 0)
    echo "chunk_$i: $have/$want"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

ok=1
for i in $(seq 0 $((NCHUNK-1))); do
  s=$((i * CHUNK)); e=$((s + CHUNK - 1)); [ $e -ge $SIZE ] && e=$((SIZE - 1))
  want=$((e - s + 1))
  have=$(stat -c %s "$TMPD/chunk_$i" 2>/dev/null || echo 0)
  [ "$have" = "$want" ] || { echo "chunk_$i INCOMPLETE $have/$want"; ok=0; }
done
[ "$ok" = 1 ] || { echo "DOWNLOAD_INCOMPLETE"; exit 1; }

cat $(for i in $(seq 0 $((NCHUNK-1))); do echo "$TMPD/chunk_$i"; done) > "$DEST"
final=$(stat -c %s "$DEST")
echo "final size=$final"
[ "$final" = "$SIZE" ] && rm -rf "$TMPD" && echo DOWNLOAD_DONE || { echo SIZE_MISMATCH; exit 1; }
