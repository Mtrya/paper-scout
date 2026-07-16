#!/usr/bin/env bash
# =====================================================================
# Train the dual-stream (RGB + optical-flow) Wan2.2 world model with the
# IDM action expert on RoboTwin.
#
# Single node (all visible GPUs):
#   DATASET_BASE_PATH=/path/to/robotwin_data bash train.sh
#
# Multi node (run on every node, same NUM_MACHINES / MASTER_ADDR):
#   NUM_MACHINES=2 MASTER_ADDR=<ip> bash train.sh --machine_rank 0
#   NUM_MACHINES=2 MASTER_ADDR=<ip> bash train.sh --machine_rank 1
#
# Base Wan2.2-TI2V-5B weights are expected under training/models/Wan-AI
# (see README "Model download").
# =====================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
# export SWANLAB_API_KEY=...   # optional: enable SwanLab logging

mkdir -p "${SCRIPT_DIR}/models"

# ---- Distributed config ----
NUM_MACHINES="${NUM_MACHINES:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MACHINE_RANK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --machine_rank) MACHINE_RANK="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

# ---- Data ----
DATASET_BASE_PATH="${DATASET_BASE_PATH:?set DATASET_BASE_PATH to the RoboTwin data root}"
VARIANTS="${VARIANTS:-aloha-agilex_clean_50 aloha-agilex_randomized_500}"
CAMERAS="head_camera left_camera right_camera"

# ---- Base models (relative to training/models via --model_id_with_origin_paths) ----
MODEL_PATHS_DIT="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors"
MODEL_PATHS_T5="Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth"
MODEL_PATHS_VAE="Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"

# ---- Temporal config: 32 executed actions, 4x visual downsample ----
# (9-1)*4+1 = 33 action steps (1 anchor + 32); VAE latent T = (9-1)/4+1 = 3.
NUM_FRAMES="${NUM_FRAMES:-33}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-9}"
VISUAL_STRIDE="${VISUAL_STRIDE:-4}"
SIZE_W=320
SIZE_H=256

# ---- Flow ----
FLOW_METHOD=raft
FLOW_DEVICE=cuda
FLOW_MODE="${FLOW_MODE:-robot_only}"      # robot_only | full_scene
FLOW_MAX_MAGNITUDE="${FLOW_MAX_MAGNITUDE:-25.0}"
FLOW_MOTION_BOOST=2.0

# ---- Optimization ----
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS=1
TRAINABLE_MODELS=dit
ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-1.0}"
FLOW_LOSS_WEIGHT=0.1
ACTION_DIM=14
NUM_ACTION_LAYERS=30              # 1:1 with the Wan2.2-5B video DiT (30 layers)
ACTION_SNR_SHIFT="${ACTION_SNR_SHIFT:-5.0}"   # MUST match the inference server
TEXT_CONTEXT_DIM=4096
LOSS_TIMESTEP_WEIGHTING=on
REF_AUG_STRENGTH=0.1
FP32_MODULATION=true

# ---- IDM conditioning ----
COND_NOISE_PROB="${COND_NOISE_PROB:-0.5}"
COND_DETACH="${COND_DETACH:-false}"           # false = action loss also trains video backbone
COND_LAYER_STRIDE=1
ACTION_PRED_TARGET=velocity                   # velocity | x0
ACTION_POS_MODE=rope                          # rope | learned
PROPRIO_MODE=text                             # text | state_token

# ---- LR schedule ----
LR_SCHEDULER_TYPE=cosine
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}"
LR_MAX_STEPS="${LR_MAX_STEPS:-0}"             # 0 => derive from num_epochs

# ---- Checkpointing / resume ----
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_EVERY_N_EPOCHS=100
FULL_STATE_KEEP=2
RESUME_STATE_DIR="${RESUME_STATE_DIR:-}"
# Optional weights-only warm-start of the video DiT + flow_stream (the action
# expert always starts fresh). Leave empty to train the video DiT from the
# Wan2.2-5B pretrain.
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ---- Data mode: online (default) or precomputed latent cache ----
# LOAD_FROM_CACHE=false: RAFT optical flow + VAE computed on the fly.
# LOAD_FROM_CACHE=true : read {CACHE_ROOT}/... produced by precompute.sh.
LOAD_FROM_CACHE="${LOAD_FROM_CACHE:-false}"
CACHE_ROOT="${CACHE_ROOT:-}"

if [ "${LOAD_FROM_CACHE}" = "true" ]; then
  MODEL_PATHS="${MODEL_PATHS_DIT},${MODEL_PATHS_VAE}"   # T5 embeddings are cached
else
  MODEL_PATHS="${MODEL_PATHS_DIT},${MODEL_PATHS_T5},${MODEL_PATHS_VAE}"
fi

OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/models/train/flowwam_idm}"
ACTION_NORM_PATH="${OUTPUT_PATH}/action_norm_stats.npz"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s ',' 0 $((NUM_GPUS - 1)))}"
ACCELERATE_ARGS="--num_processes=$((NUM_GPUS * NUM_MACHINES)) --num_machines=${NUM_MACHINES}"
if [ "${NUM_MACHINES}" -gt 1 ]; then
  ACCELERATE_ARGS="${ACCELERATE_ARGS} --machine_rank=${MACHINE_RANK} --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT}"
fi

echo "========== FlowWAM IDM Flow+Action Training =========="
echo "  machines/gpus:   ${NUM_MACHINES} x ${NUM_GPUS}"
echo "  frames:          action=${NUM_FRAMES} video=${NUM_VIDEO_FRAMES} stride=${VISUAL_STRIDE}"
echo "  batch/lr:        ${BATCH_SIZE} / ${LEARNING_RATE}"
echo "  data mode:       $([ "${LOAD_FROM_CACHE}" = "true" ] && echo "cache: ${CACHE_ROOT}" || echo "online (RAFT flow)")"
echo "  output:          ${OUTPUT_PATH}"
echo "======================================================"

"${PYTHON}" -m accelerate.commands.launch \
  ${ACCELERATE_ARGS} \
  "${SCRIPT_DIR}/flow_action_train.py" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --num_frames ${NUM_FRAMES} \
  --num_video_frames ${NUM_VIDEO_FRAMES} \
  --visual_stride ${VISUAL_STRIDE} \
  --model_id_with_origin_paths "${MODEL_PATHS}" \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --size ${SIZE_W} ${SIZE_H} \
  --variants ${VARIANTS} \
  --cameras ${CAMERAS} \
  --trainable_models ${TRAINABLE_MODELS} \
  --flow_method ${FLOW_METHOD} \
  --flow_device ${FLOW_DEVICE} \
  --flow_mode ${FLOW_MODE} \
  --flow_max_magnitude ${FLOW_MAX_MAGNITUDE} \
  --dataset_num_workers 0 \
  --flow_loss_weight ${FLOW_LOSS_WEIGHT} \
  --action_loss_weight ${ACTION_LOSS_WEIGHT} \
  --action_dim ${ACTION_DIM} \
  --num_action_layers ${NUM_ACTION_LAYERS} \
  --action_norm_path "${ACTION_NORM_PATH}" \
  --action_snr_shift ${ACTION_SNR_SHIFT} \
  --cond_noise_prob ${COND_NOISE_PROB} \
  $([ "${COND_DETACH}" = "true" ] && echo "--cond_detach") \
  --cond_layer_stride ${COND_LAYER_STRIDE} \
  --action_pred_target ${ACTION_PRED_TARGET} \
  --action_pos_mode ${ACTION_POS_MODE} \
  --proprio_mode ${PROPRIO_MODE} \
  --loss_timestep_weighting ${LOSS_TIMESTEP_WEIGHTING} \
  --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
  --lr_warmup_steps ${LR_WARMUP_STEPS} \
  --lr_max_steps ${LR_MAX_STEPS} \
  --full_state_keep ${FULL_STATE_KEEP} \
  ${RESUME_STATE_DIR:+--resume_state_dir "${RESUME_STATE_DIR}"} \
  --flow_motion_boost ${FLOW_MOTION_BOOST} \
  --text_context_dim ${TEXT_CONTEXT_DIM} \
  --batch_size ${BATCH_SIZE} \
  --extra_inputs "input_image" \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --ref_aug_strength ${REF_AUG_STRENGTH} \
  --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
  ${SAVE_STEPS:+--save_steps ${SAVE_STEPS}} \
  ${FP32_MODULATION:+--fp32_modulation} \
  --use_gradient_checkpointing \
  ${RESUME_CHECKPOINT:+--resume_checkpoint "${RESUME_CHECKPOINT}"} \
  $([ "${LOAD_FROM_CACHE}" = "true" ] && echo "--load_from_cache --cache_root ${CACHE_ROOT} --cache_episode_batching --cache_chunks_per_episode 4 --cache_episode_lru_size 4")
