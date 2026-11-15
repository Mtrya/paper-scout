#!/usr/bin/env bash
set -euo pipefail
W="${1:?usage: setup_native_recon.sh ABSOLUTE_WORKROOT}"
V="$W/native-venv"
R="$W/video_to_data/reconstruction"
T="$W/third_party"
export PATH="$V/bin:$W/bin:/usr/local/bin:/usr/bin:/bin"
export V2D_WORKROOT="$W"
export V2D_RECONSTRUCTION_ROOT="$R"
export PYTHON="$V/bin/python"
export HF_HOME="$W/cache/hf"
export TORCH_HOME="$W/cache/torch"
export HF_HUB_DISABLE_XET=1
export PIP_CONFIG_FILE=/dev/null
export PIP_EXTRA_INDEX_URL=
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple
export PIP_NO_CACHE_DIR=1
export PIP_CONSTRAINT="$W/native/constraints.txt"
export TORCH_CUDA_ARCH_LIST="8.9"
export FORCE_CUDA=1
export MAX_JOBS=6
export CUDA_HOME=/usr/local/cuda
export PYOPENGL_PLATFORM=egl
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$W"/{bin,native,logs,manifests,third_party,cache/hf,cache/torch}
rm -f "$W/manifests/native-recon-setup.exit"
trap 'status=$?; printf "%s\n" "$status" > "$W/manifests/native-recon-setup.exit"' EXIT

if [ "${WEIGHTS_ONLY:-0}" != "1" ]; then
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build libeigen3-dev pybind11-dev \
    libboost-dev libboost-system-dev libboost-program-options-dev \
    libgl1 libegl1 libgles2 libglib2.0-0 libxrender1 libsm6 libxext6 \
    libglfw3 libglm-dev ffmpeg git wget curl python3-venv
python3 -m venv --system-site-packages "$V"
printf 'numpy<2\ntorch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\n' > "$PIP_CONSTRAINT"
"$V/bin/python" -m pip install --upgrade 'pip<26' setuptools wheel ninja --index-url https://pypi.org/simple
if ! "$V/bin/python" -c 'import torch; assert torch.__version__.startswith("2.5.1") and torch.cuda.is_available()' 2>/dev/null; then
    PIP_CONSTRAINT=/dev/null "$V/bin/python" -m pip install --no-deps --ignore-installed \
        --index-url https://mirrors.ustc.edu.cn/pypi/simple \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
fi
"$V/bin/python" -m pip install triton==3.1.0
"$V/bin/python" -c 'import torch; x=torch.randn(256,256,device="cuda"); print(torch.__version__, torch.version.cuda, float(x.std())); assert torch.__version__.startswith("2.5.1") and torch.cuda.is_available()'
ln -sf "$W/native/native_process.py" "$W/bin/docker"
chmod +x "$W/native/native_process.py"

cd "$R"
if [ ! -f "$W/manifests/native-host-packages.done" ]; then
    bash scripts/install_ego_reconstruction_packages.sh
    touch "$W/manifests/native-host-packages.done"
fi
"$V/bin/python" -m pip install \
    'numpy<2' scipy matplotlib Pillow h5py plyfile opencv-python \
    opencv-contrib-python==4.11.0.86 \
    imageio imageio-ffmpeg av decord open3d trimesh pyrender smplx \
    yacs hydra-core omegaconf pytorch-lightning lightning timm einops \
    webdataset braceexpand dill joblib loguru scikit-image scikit-learn \
    pyvista pymeshfix igraph xatlas astor easydict networkx rtree roma \
    warp-lang==1.0.2 meshcat webdataset pypng seaborn openpyxl fast_simplification \
    gdown ruamel.yaml transformations kornia ultralytics safetensors \
    pyvista pyglet==2.1.15 cumm-cu124 spconv-cu124
"$V/bin/python" -c 'import tensorrt as trt; print("TensorRT runtime:", trt.__version__)'
"$V/bin/python" -m pip install \
    --index-url https://repo.huaweicloud.com/repository/pypi/simple \
    'transformers==4.27.4' 'huggingface-hub==0.25.2'
ln -sf "$V/bin/huggingface-cli" "$W/bin/hf"
if ! "$V/bin/python" -c 'import chumpy, manotorch, wilor_mini' 2>/dev/null; then
    "$V/bin/python" -m pip install --no-build-isolation --no-deps \
        git+https://github.com/mattloper/chumpy.git \
        git+https://github.com/lixiny/manotorch.git \
        git+https://github.com/warmshao/WiLoR-mini.git
fi
"$V/bin/python" -m pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
"$V/bin/python" -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
"$V/bin/python" -m pip install --no-build-isolation --no-binary gsplat gsplat==1.4.0
if ! "$V/bin/python" -c 'from pytorch3d import ops' 2>/dev/null; then
    "$V/bin/python" -m pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
fi

clone() {
    local url="$1" directory="$2"
    if [ ! -d "$directory/.git" ]; then
        git clone --recursive "$url" "$directory"
    fi
}
clone https://github.com/geopavlakos/hamer.git "$T/hamer"
git -C "$T/hamer" checkout 3a01849f4148352e9260b69bf28b65d1671a4905
git -C "$T/hamer" submodule update --init --recursive
"$V/bin/python" -m pip install --no-deps -e "$T/hamer"
clone https://github.com/princeton-vl/DROID-SLAM.git "$T/DROID-SLAM"
if ! "$V/bin/python" -c 'import lietorch' 2>/dev/null; then
    "$V/bin/python" -m pip install --no-build-isolation -e "$T/DROID-SLAM/thirdparty/lietorch"
fi
if ! "$V/bin/python" -c 'import droid_backends' 2>/dev/null; then
    "$V/bin/python" -m pip install --no-build-isolation -e "$T/DROID-SLAM"
fi
clone https://github.com/IDEA-Research/GroundingDINO.git "$T/GroundingDINO"
"$V/bin/python" -m pip install --no-build-isolation "$T/GroundingDINO"
clone https://github.com/facebookresearch/sam-3d-objects.git "$T/sam-3d-objects"
"$V/bin/python" -m pip install --no-deps "$T/sam-3d-objects"
clone https://github.com/NVlabs/nvdiffrast.git "$T/nvdiffrast"
"$V/bin/python" -m pip install --no-build-isolation "$T/nvdiffrast"

clone https://github.com/microsoft/MoGe.git "$T/MoGe"
git -C "$T/MoGe" checkout 925b8ed835a7a9cdb7578ba15c658a0afc969030
clone https://github.com/microsoft/MoGe.git "$T/MoGe-sam3d"
git -C "$T/MoGe-sam3d" checkout a8c37341bc0325ca99b9d57981cc3bb2bd3e255b
clone https://github.com/EasternJournalist/utils3d.git "$T/utils3d-sam3d"
git -C "$T/utils3d-sam3d" checkout 3913c65d81e05e47b9f367250cf8c0f7462a0900
clone https://github.com/facebookresearch/sam2.git "$T/sam2"
git -C "$T/sam2" checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
clone https://github.com/javrtg/AnyCalib.git "$T/AnyCalib"
clone https://github.com/cvg/GeoCalib.git "$T/GeoCalib"
for package in MoGe sam2 AnyCalib GeoCalib; do
    "$V/bin/python" -m pip install --no-build-isolation --no-deps "$T/$package"
done
for module in mesh moge sam2 grounding_dino sam3d foundation_pose hamer wilor anycalib geocalib droid_slam gsplat_refinement; do
    "$V/bin/python" -m pip install --no-build-isolation --no-deps -e "$R/modules/v2d_${module}/lib"
done
"$V/bin/python" -m pip install --no-deps 'transformers==4.27.4' 'huggingface-hub==0.25.2'
FP="$R/modules/v2d_foundation_pose/lib/FoundationPose"
cmake -S "$FP/mycpp" -B "$FP/mycpp/build" -DPYTHON_EXECUTABLE="$V/bin/python"
cmake --build "$FP/mycpp/build" -j6
"$V/bin/python" -m pip install --no-build-isolation -e "$FP/bundlesdf/mycuda"
"$V/bin/python" -c 'from transformers import AutoTokenizer, BertModel; import os; path=os.path.join(os.environ["V2D_WORKROOT"], "models/bert-base-uncased"); AutoTokenizer.from_pretrained(path); BertModel.from_pretrained(path)'
fi
printf '%s\n' "$T/DROID-SLAM" "$T/DROID-SLAM/droid_slam" > "$V/lib/python3.10/site-packages/v2d-droid-slam.pth"
cd "$R"
KORNIA_LIGHTGLUE="$V/lib/python3.10/site-packages/kornia/feature/lightglue.py"
if [ -f "$KORNIA_LIGHTGLUE" ]; then
    sed -i 's/^except ModuleNotFoundError:$/except ImportError:/' "$KORNIA_LIGHTGLUE"
fi
bash scripts/download_ego_reconstruction_weights.sh --mode hamer_prompt --accept-nvidia-model-eula
if [ ! -e "$R/data/weights/wilor/pretrained_models/models" ]; then
    ln -s "$W/mano/models" "$R/data/weights/wilor/pretrained_models/models"
fi
mkdir -p "$R/data/weights/hamer/_DATA/data"
ln -sfn "$R/data/weights/wilor/pretrained_models/mano_mean_params.npz" \
    "$R/data/weights/hamer/_DATA/data/mano_mean_params.npz"
ln -sfn "$W/mano/models" "$R/data/weights/hamer/_DATA/data/mano"
ln -sfn "$W/mano/models" "$R/data/weights/hamer/_DATA/data/models"
"$V/bin/python" -c 'import torch, gsplat, hamer, droid_slam; from droid import Droid; from v2d.sam3d.lib import render_textured_video; from v2d.pipelines import run_ego_reconstruction; print("native reconstruction imports OK", torch.cuda.get_device_name(0))'
cd "$W/video_to_data"
"$V/bin/python" "$W/native/native_setup_config.py" --config "$W/runs/tissue_box/e2e_config.json" --mano-dir "$W/mano" --isaac-groot-dir "$W/Isaac-GR00T"
touch "$W/manifests/native-recon-setup.done"
