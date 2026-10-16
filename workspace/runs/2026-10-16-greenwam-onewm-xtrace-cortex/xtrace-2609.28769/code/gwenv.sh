#!/usr/bin/env bash
# Environment for every measurement in this packet.
#
# gwatch's shipped native libraries are built against the CUDA 12 runtime and CUPTI, which
# the environment must supply on LD_LIBRARY_PATH:
#   * libcudart.so.12 and libcupti.so.12 come from the nvidia-*-cu12 wheels that torch's
#     cu128 wheel pulls in;
#   * libcheckpoint.so is the CUPTI checkpoint shim that triton ships.
# Source this file, then run the scripts in this directory with $GWATCH_PYTHON.
#
#   source gwenv.sh
#   "$GWATCH_PYTHON" probe_triton.py --tier sass --sites sche
#
# GWATCH_VENV overrides the interpreter location.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="${GWATCH_VENV:-$here/../../../../code/xtrace-env}"
site="$venv/lib/python3.12/site-packages"

export GWATCH_PYTHON="$venv/bin/python"
export LD_LIBRARY_PATH="$site/nvidia/cuda_runtime/lib:$site/nvidia/cuda_cupti/lib:$site/triton/backends/nvidia/lib/cupti${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
