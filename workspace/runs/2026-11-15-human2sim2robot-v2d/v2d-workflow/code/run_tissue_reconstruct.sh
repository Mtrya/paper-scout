#!/usr/bin/env bash
set -euo pipefail
W="${1:?usage: run_tissue_reconstruct.sh ABSOLUTE_WORKROOT}"
export PATH="$W/native-venv/bin:$W/bin:/usr/local/bin:/usr/bin:/bin"
export V2D_WORKROOT="$W"
export V2D_NATIVE=1
export V2D_RECONSTRUCTION_ROOT="$W/video_to_data/reconstruction"
export PYTHON="$W/native-venv/bin/python"
export HF_HOME="$W/cache/hf"
export TORCH_HOME="$W/cache/torch"
export HF_HUB_DISABLE_XET=1
export PYOPENGL_PLATFORM=egl
export CUDA_HOME=/usr/local/cuda
export CONDA_PREFIX=/opt/conda
export LIDRA_SKIP_INIT=1
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
rm -f "$W/manifests/tissue-reconstruct.done" "$W/manifests/tissue-reconstruct.exit"
trap 'status=$?; printf "%s\n" "$status" > "$W/manifests/tissue-reconstruct.exit"' EXIT
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
cd "$W/video_to_data"
./run_e2e.sh reconstruct \
  --video "$PWD/reconstruction/assets/tissue_box.mp4" \
  --hand-tracking hamer \
  --run-gsplat-refinement 2>&1 | tee "$W/logs/tissue-reconstruct.log"
touch "$W/manifests/tissue-reconstruct.done"
