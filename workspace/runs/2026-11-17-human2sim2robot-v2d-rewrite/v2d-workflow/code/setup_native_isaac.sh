#!/usr/bin/env bash
set -euo pipefail
W="${1:?usage: setup_native_isaac.sh ABSOLUTE_WORKROOT}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
export PIP_CONFIG_FILE=/dev/null
export PIP_EXTRA_INDEX_URL=
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple
export PIP_NO_CACHE_DIR=1
export PIP_CONSTRAINT=/dev/null
export HF_HOME="$W/cache/hf"
export HF_HUB_DISABLE_XET=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$W"/{bin,native,logs,manifests,cache/hf}
rm -f "$W/manifests/native-isaac-setup.exit" "$W/manifests/native-isaac-setup.done"
trap 'status=$?; printf "%s\n" "$status" > "$W/manifests/native-isaac-setup.exit"' EXIT
if ! command -v ffprobe >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends ffmpeg
fi
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
DRIVER_DIR="$W/native/nvidia-driver-$DRIVER_VERSION"
if [ -e /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]; then
    if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
        cp /usr/share/vulkan/icd.d/nvidia_icd.json "$W/native/nvidia-icd.json"
    else
        printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.2.155"}}' > "$W/native/nvidia-icd.json"
    fi
else
    DRIVER_ARCHIVE="$W/cache/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
    if [ ! -f "$DRIVER_DIR/libGLX_nvidia.so.${DRIVER_VERSION}" ]; then
        curl -fL --retry 3 -C - -o "$DRIVER_ARCHIVE" \
            "https://download.nvidia.com/XFree86/Linux-x86_64/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
        sh "$DRIVER_ARCHIVE" --extract-only --target "$DRIVER_DIR"
    fi
    for file in "$DRIVER_DIR"/lib*.so.*; do
        [ -f "$file" ] || continue
        soname=$(objdump -p "$file" 2>/dev/null | awk '/SONAME/{print $2}')
        if [ -n "$soname" ] && [ ! -e "$DRIVER_DIR/$soname" ]; then
            ln -s "$(basename "$file")" "$DRIVER_DIR/$soname"
        fi
    done
    cp "$DRIVER_DIR/nvidia_icd.json" "$W/native/nvidia-icd.json"
fi
cat > "$W/native/isaac-python" <<'EOF'
#!/usr/bin/env bash
ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
for USD_ROOT in "$ISAACLAB_PATH"/_isaac_sim/extscache/omni.usd.libs-*; do
    if [ -d "$USD_ROOT/pxr" ] && [ -d "$USD_ROOT/bin" ]; then
        export PYTHONPATH="$USD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        export LD_LIBRARY_PATH="$USD_ROOT/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
exec "$ISAACLAB_PATH/isaaclab.sh" -p "$@"
EOF
chmod +x "$W/native/isaac-python"
ln -sfn "$W/native/isaac-python" "$W/bin/python3"
P=("${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install --no-cache-dir)
"${P[@]}" --upgrade pip
"${P[@]}" \
    tensordict==0.8.3 datasets==2.19.0 wandb==0.22.2 importlib_metadata \
    plotly viser==1.0.15 dearpygui scipy scikit-learn tzdata py-soma-x==0.2.1
"${P[@]}" \
    mujoco 'judo-rai>=0.0.5' pin==3.7.0 pin-pink==4.2.0 coal==3.0.1 \
    eigenpy==3.10.3 cmeel==0.57.3 cmeel-boost==1.87.0.1 \
    cmeel-urdfdom==4.0.1 cmeel-console-bridge==1.0.2.3 \
    cmeel-tinyxml2==10.0.0 cmeel-tinyxml==2.6.2.3 cmeel-assimp==5.4.3.1 \
    cmeel-octomap==1.10.0 cmeel-qhull==8.0.2.1 cmeel-zlib==1.3.1 \
    deprecation
"${P[@]}" --upgrade osqp 'qpsolvers[clarabel,daqp,proxqp,scs,osqp]==4.9.0'
"${P[@]}" \
    onnxruntime-gpu==1.24.4 imageio imageio-ffmpeg pyrender pytorch-ignite \
    pytest boto3 pyzmq msgpack-numpy numpy==1.26.4 pyarrow==19.0.1 python-dateutil uv
R="$W/video_to_data/reconstruction"
G="$W/video_to_data/robotic_grounding"
cp "$W/native/native_rg.py" "$G/workflow/native_run.py"
chmod +x "$G/workflow/native_run.py"
"${P[@]}" --no-build-isolation -e "$G/source/robotic_grounding"
"${P[@]}" --no-build-isolation -e "$R/modules/v2d_common" -e "$R/modules/v2d_docker" \
    -e "$R/modules/v2d_task_library_loader/docker" \
    -e "$R/modules/v2d_task_library_loader/lib" \
    git+https://github.com/mattloper/chumpy.git \
    git+https://github.com/lixiny/manotorch.git
"${ISAACLAB_PATH}/isaaclab.sh" -p -m pip uninstall -y usd-core
mkdir -p "$R/modules/v2d_task_library_loader/_rg_pkg"
ln -sfn "$G/source/robotic_grounding/robotic_grounding" \
    "$R/modules/v2d_task_library_loader/_rg_pkg/robotic_grounding"
export PYTHONPATH="$R/modules/v2d_task_library_loader/_rg_pkg:$G/source/robotic_grounding${PYTHONPATH:+:$PYTHONPATH}"
"${ISAACLAB_PATH}/isaaclab.sh" -p -c 'import numpy, torch, pinocchio, onnxruntime, msgpack_numpy, zmq, robotic_grounding; from v2d.task_library_loader.docker import run_loader; print(numpy.__version__, torch.__version__, torch.version.cuda, torch.cuda.is_available(), pinocchio.__version__); assert numpy.__version__.startswith("1.26.") and torch.cuda.is_available()'
touch "$W/manifests/native-isaac-setup.done"
