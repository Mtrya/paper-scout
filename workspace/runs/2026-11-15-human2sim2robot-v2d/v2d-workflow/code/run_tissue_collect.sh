#!/usr/bin/env bash
set -euo pipefail
W="${1:?usage: run_tissue_collect.sh ABSOLUTE_WORKROOT [CONFIG] [LABEL]}"
CONFIG="${2:-$W/runs/tissue_box/e2e_config.json}"
ROOT="$(dirname "$CONFIG")"
LABEL="${3:-tissue}"
export PATH="$W/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
export V2D_WORKROOT="$W"
export V2D_RECONSTRUCTION_ROOT="$W/video_to_data/reconstruction"
export V2D_NATIVE_PYTHON="$W/native/isaac-python"
export V2D_ISAAC_PYTHON="$W/native/isaac-python"
export PYTHON="$W/native/isaac-python"
export PYTHONPATH="$W/video_to_data/reconstruction/modules/v2d_task_library_loader/_rg_pkg:$W/video_to_data/robotic_grounding:$W/video_to_data/robotic_grounding/source/robotic_grounding${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$W/cache/hf"
export HF_HUB_DISABLE_XET=1
export PYOPENGL_PLATFORM=egl
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
DRIVER_DIR="$W/native/nvidia-driver-$DRIVER_VERSION"
[ -d "$DRIVER_DIR" ] || DRIVER_DIR="$W/native/nvidia-driver"
export VK_ICD_FILENAMES="$W/native/nvidia-icd.json"
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] && [ -e /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
else
  export LD_LIBRARY_PATH="$DRIVER_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$DRIVER_DIR/10_nvidia.json"
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$W/logs" "$W/manifests"
rm -f "$W/manifests/$LABEL-collect.done" "$W/manifests/$LABEL-collect.exit"
trap 'status=$?; printf "%s\n" "$status" > "$W/manifests/$LABEL-collect.exit"' EXIT
CHECKPOINT="$W/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/policies/e2e_example/tissue_box/vega_sharpa_policy.onnx"
cd "$W/video_to_data"
if [ "${SKIP_COLLECT:-0}" != "1" ]; then
  ./run_e2e.sh --config "$CONFIG" collect \
    --rl-checkpoint "$CHECKPOINT" \
    --collection-max-steps 600 \
    --target-successes 10 --pilot-attempts 32 --render-envs 1 \
    2>&1 | tee "$W/logs/$LABEL-collect.log"
fi
"$PYTHON" "$W/native/evaluate_source_tasks.py" \
  --run-root "$ROOT" \
  --source "selected=$ROOT/selected" \
  --source "pilot=$ROOT/source/pilot" \
  --output "$ROOT/source_task_metrics.json" \
  > "$W/logs/$LABEL-source-task-metrics.log"
EXPORT_ARGS=()
if [ "${FORCE_EXPORT:-0}" = "1" ]; then
  EXPORT_ARGS=(--force)
fi
"$PYTHON" "$W/native/export_lerobot.py" --config "$CONFIG" "${EXPORT_ARGS[@]}" \
  2>&1 | tee "$W/logs/$LABEL-lerobot.log"
touch "$W/manifests/$LABEL-collect.done"
