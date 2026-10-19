#!/usr/bin/env bash
# Reproduce the Natura-1 video audit: download four public clips, probe them,
# measure frame-to-frame scene scores (hard-cut detector), extract the frames
# used in the contact sheets, then rebuild the sheets.
#
# Videos are kept in the ignored workspace lab bench (workspace/code/), not in
# the run packet. Override NATURA_VIDEO_DIR to use another staging directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THREAD="$(cd "$HERE/.." && pwd)"
WORKSPACE="$(cd "$THREAD/../../.." && pwd)"
VID_DIR="${NATURA_VIDEO_DIR:-$WORKSPACE/code/natura-videos}"
FRAMES="$VID_DIR"
mkdir -p "$VID_DIR" "$FRAMES"

BASE="https://media.naturewill.com/research-blog/media"
declare -A URLS=(
  [wontons]="$BASE/v002-05981bd1/20261012T210313Z-089ca3c35ad7/mp4/high.mp4"
  [rod_basket]="$BASE/v034-79043c4d/20261012T210313Z-089ca3c35ad7/mp4/high.mp4"
  [lemon]="$BASE/v026-094fb206/20261012T210313Z-089ca3c35ad7/mp4/high.mp4"
  [card_edge]="$BASE/v030-c28b682f/20261012T210313Z-089ca3c35ad7/mp4/high.mp4"
)

for name in wontons rod_basket lemon card_edge; do
  f="$VID_DIR/$name.mp4"
  if [[ ! -s "$f" ]]; then
    echo "downloading $name"
    curl -sL --max-time 300 -o "$f" "${URLS[$name]}"
  fi
done

for name in wontons rod_basket lemon card_edge; do
  f="$VID_DIR/$name.mp4"
  echo "== $name"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate,duration,nb_frames,codec_name \
    -of default=noprint_wrappers=1 "$f"
  tmp="$(mktemp)"
  ffmpeg -hide_banner -i "$f" \
    -vf "select='gte(scene,0)',metadata=print:file=$tmp" -an -f null - 2>/dev/null || true
  max="$(grep -o 'scene_score=[0-9.]*' "$tmp" | cut -d= -f2 | sort -nr | head -1 || true)"
  echo "max_scene_score=$max  (hard cuts are typically >0.3; <0.05 means no cut)"
  rm -f "$tmp"
done

# Frames used by make_figures.py's sheets.
extract() { # video, prefix, times...
  local video="$1" prefix="$2"; shift 2
  for t in "$@"; do
    ffmpeg -y -v error -ss "$t" -i "$VID_DIR/$video.mp4" -frames:v 1 "$FRAMES/${prefix}_${t}.png"
  done
}
extract wontons ww 5 20 35 45 55 65
extract rod_basket r 0 2 4 6 8
extract card_edge c 0 2 4 6 8
extract lemon l 0 8 16 24 32 40

echo "frames written to $FRAMES"
echo "rebuilding sheets with NATURA_VIDEO_DIR=$VID_DIR"
NATURA_VIDEO_DIR="$VID_DIR" python3 "$HERE/make_figures.py"
# The wonton contact sheet is photographic and exceeds the 2 MB per-file cap
# as a PNG; package it as JPEG while keeping the lossless PNG as an intermediate.
ffmpeg -y -v error -i "$THREAD/../assets/natura_wontons_sheet.png" -q:v 2 "$THREAD/../assets/natura_wontons_sheet.jpg"
rm -f "$THREAD/../assets/natura_wontons_sheet.png"
