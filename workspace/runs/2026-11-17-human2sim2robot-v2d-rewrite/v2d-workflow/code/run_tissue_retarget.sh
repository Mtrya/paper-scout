#!/usr/bin/env bash
set -euo pipefail
W="${1:?usage: run_tissue_retarget.sh ABSOLUTE_WORKROOT}"
export PATH="$W/bin:/usr/local/bin:/usr/bin:/bin"
export V2D_WORKROOT="$W"
export V2D_RECONSTRUCTION_ROOT="$W/video_to_data/reconstruction"
export V2D_NATIVE_PYTHON="$W/native/isaac-python"
export V2D_ISAAC_PYTHON="$W/native/isaac-python"
export PYTHON="$W/native/isaac-python"
export PYTHONPATH="$W/video_to_data/reconstruction/modules/v2d_task_library_loader/_rg_pkg:$W/video_to_data/robotic_grounding/source/robotic_grounding${PYTHONPATH:+:$PYTHONPATH}"
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
rm -f "$W/manifests/tissue-retarget.done" "$W/manifests/tissue-retarget.exit"
trap 'status=$?; printf "%s\n" "$status" > "$W/manifests/tissue-retarget.exit"' EXIT
cd "$W/video_to_data"
./run_e2e.sh retarget --embodiment vega 2>&1 | tee "$W/logs/tissue-retarget.log"
./run_e2e.sh simulate --embodiment vega 2>&1 | tee "$W/logs/tissue-simulate.log"
touch "$W/manifests/tissue-retarget.done"
