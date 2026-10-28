#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN=logs/train30_1024
SM="$ROOT/code/SuperMimic-Reproduced"
cd "$SM"

while pgrep -f '[s]cripts/eval_benchmark.py' >/dev/null; do sleep 60; done
mkdir -p "$RUN/eval_fixed" "$ROOT/code/evals_final"
for it in 2000 4000 8000 12000; do
  ck=$RUN/ckpt_$(printf %07d "$it").pt
  echo "=== corrected whole-clip evaluation $it ==="
  for split in heldout10 train30; do
    out=$RUN/eval_fixed/eval_${split}_$(printf %07d "$it").json
    [ -f "$out" ] || env -u PYTHONPATH .venv/bin/python scripts/eval_benchmark.py \
      --ckpt "$ck" --clip-list data/splits/${split}.txt --num-envs 256 --out "$out"
  done
done

until [ -f "$SM/$RUN/wm_nse_0032000.json" ]; do
  if ! pgrep -f '[s]m-train --run-name train30_1024' >/dev/null && ! pgrep -f '[e]val_milestones.sh' >/dev/null; then
    echo "Training and milestone evaluation stopped before the final checkpoint probe" >&2
    exit 1
  fi
  sleep 120
done
env -u PYTHONPATH .venv/bin/python scripts/wm_nse_probe.py \
  --ckpt "$RUN/ckpt_0032000.pt" \
  --clips dance1_subject1 dance2_subject2 fallAndGetUp1_subject1 fallAndGetUp2_subject2 fight1_subject2 jumps1_subject1 run2_subject4 sprint1_subject4 walk3_subject1 dance1_subject3 dance2_subject1 fallAndGetUp1_subject5 fallAndGetUp2_subject3 fight1_subject3 jumps1_subject2 run2_subject1 sprint1_subject2 walk3_subject5 walk4_subject1 \
  --num-envs 19 --out "$RUN/wm_nse_paired_0032000.json"

cp "$RUN"/eval_fixed/eval_*.json "$ROOT/code/evals_final/"
for it in 16000 24000 32000; do
  for split in heldout10 train30; do
    cp "$RUN/eval_fixed/eval_${split}_$(printf %07d "$it").json" "$ROOT/code/evals_final/"
  done
done
cp data/splits/train30.txt data/splits/heldout10.txt "$ROOT/code/evals_final/"
env -u PYTHONPATH .venv/bin/python "$ROOT/code/analyze_sm.py" \
  --eval-dir "$ROOT/code/evals_final" --out "$ROOT/code/analysis_final"
env -u PYTHONPATH .venv/bin/python "$ROOT/code/analyze_wm_pairs.py" \
  --probe "$RUN/wm_nse_paired_0032000.json" --eval-dir "$ROOT/code/evals_final" \
  --out-json "$ROOT/code/analysis_final/wm_pairs.json" \
  --figure "$ROOT/code/analysis_final/wm_pairs.png"

cd "$ROOT/code/ropedia-w0"
.venv-torch/bin/python audit_run.py --cases cases_local.json \
  --out "$ROOT/code/ropedia-local/audit" --ckpt weights/w0-5b-midtrain.safetensors \
  --text_layer_offload --num_inference_steps 50 --skip_existing
.venv-torch/bin/python audit_metrics.py --dir "$ROOT/code/ropedia-local/audit" \
  --names rice_full,rice_noskel,rice_none,rice_synthcam,rice_skelshift,rice_skelshuffle,cut_full,cut_nocam \
  --out "$ROOT/code/ropedia-local/metrics.json" \
  --sheet "$ROOT/code/ropedia-local/contact.png" \
  --sheet_rows rice_full,rice_noskel,rice_none,rice_synthcam,rice_skelshift,rice_skelshuffle,cut_full,cut_nocam
for name in cut_full cut_nocam; do
  "$ROOT/code/ropedia-hand-venv/bin/python" audit_hand_landmarker.py \
    --video "$ROOT/code/ropedia-local/audit/${name}_pred.mp4" \
    --labels examples/xp10m_cut_cardboard/data.npz \
    --model models/hand_landmarker.task \
    --out "$ROOT/code/ropedia-local/${name}_hand.json" \
    --overlay "$ROOT/code/ropedia-local/${name}_hand.png"
done
echo "LOCAL EXPERIMENTS DONE"
