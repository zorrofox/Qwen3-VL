#!/bin/bash
# TPU training launch script for JAX Qwen3-VL fine-tuning.
#
# Usage:
#   bash jax_qwenvl/scripts/train_tpu.sh
#
# Environment variables (override defaults):
#   MODEL_PATH    - HuggingFace model path (default: Qwen/Qwen3-VL-2B-Instruct)
#   DATASETS      - Comma-separated dataset names (default: cambrian_737k)
#   OUTPUT_DIR    - Output directory (default: ./output)
#   BATCH_SIZE    - Per-device train batch size (default: 4)
#   GRAD_ACCUM    - Gradient accumulation steps (default: 4)
#   LR            - Learning rate (default: 2e-7)
#   NUM_EPOCHS    - Number of training epochs (default: 1)
#   REPORT_TO     - Logging backend: none, wandb, tensorboard (default: none)
#   RUN_NAME      - Run name for logging (default: qwen3vl-jax)
#   LOGGING_DIR   - Tensorboard log directory, supports gs:// paths (default: output_dir)

set -euo pipefail

# ── XLA/LIBTPU 优化 flag（Ironwood v7x 推荐配置）────────────────────────────
# 参考：https://docs.cloud.google.com/tpu/docs/ironwood-performance
#       https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} \
  --xla_tpu_scoped_vmem_limit_kib=98304 \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_use_enhanced_launch_barrier=true"

# Model configuration
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
DATASETS="${DATASETS:-cambrian_737k}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-2e-7}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# Module tuning flags
TUNE_VISION="${TUNE_VISION:-False}"
TUNE_MLP="${TUNE_MLP:-True}"
TUNE_LLM="${TUNE_LLM:-True}"

# Image/video resolution
MAX_PIXELS="${MAX_PIXELS:-50176}"
MIN_PIXELS="${MIN_PIXELS:-784}"

# Data settings
DATA_FLATTEN="${DATA_FLATTEN:-False}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"

# Logging
REPORT_TO="${REPORT_TO:-none}"
RUN_NAME="${RUN_NAME:-qwen3vl-jax}"
LOGGING_DIR="${LOGGING_DIR:-}"

# GCS upload (model + checkpoint sync)
GCS_OUTPUT_DIR="${GCS_OUTPUT_DIR:-}"

# Step limit (set > 0 to stop early, -1 = full epoch)
MAX_STEPS="${MAX_STEPS:--1}"

# Checkpointing
SAVE_STEPS="${SAVE_STEPS:-1000}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-3}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# FSDP (set to True for multi-device FSDP)
FSDP="${FSDP:-False}"
# FSDP_DEVICES: explicit FSDP axis size for hybrid DP+FSDP (0=use FSDP bool logic)
FSDP_DEVICES="${FSDP_DEVICES:-0}"

# LoRA
LORA_ENABLE="${LORA_ENABLE:-False}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"

# FP8 量化训练（Qwix）：节省 HBM，允许更大 batch size
# 预期：激活缓冲区减少 ~50%，HBM 从 93% 降到 ~55%，可支持 batch=4
ENABLE_FP8="${ENABLE_FP8:-False}"

echo "=== JAX Qwen3-VL Training ==="
echo "Model:      ${MODEL_PATH}"
echo "Datasets:   ${DATASETS}"
echo "Output:     ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE} (accum: ${GRAD_ACCUM})"
echo "LR:         ${LR}"
echo "Devices:    $(timeout 5 python3 -c 'import jax; print(len(jax.devices()))' 2>/dev/null || echo 'unknown')"
echo "=============================="

EXTRA_ARGS=()
if [ -n "${LOGGING_DIR}" ]; then
    EXTRA_ARGS+=(--logging_dir "${LOGGING_DIR}")
fi
if [ -n "${GCS_OUTPUT_DIR}" ]; then
    EXTRA_ARGS+=(--gcs_output_dir "${GCS_OUTPUT_DIR}")
fi
if [ "${MAX_STEPS}" != "-1" ]; then
    EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

python3 -m jax_qwenvl.train.train \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_use "${DATASETS}" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --tune_mm_vision "${TUNE_VISION}" \
    --tune_mm_mlp "${TUNE_MLP}" \
    --tune_mm_llm "${TUNE_LLM}" \
    --max_pixels "${MAX_PIXELS}" \
    --min_pixels "${MIN_PIXELS}" \
    --data_flatten "${DATA_FLATTEN}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --report_to "${REPORT_TO}" \
    --run_name "${RUN_NAME}" \
    --save_steps "${SAVE_STEPS}" \
    --max_checkpoints "${MAX_CHECKPOINTS}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --fsdp "${FSDP}" \
    --fsdp_devices "${FSDP_DEVICES}" \
    --lora_enable "${LORA_ENABLE}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --enable_fp8 "${ENABLE_FP8}" \
    --bf16 True \
    --logging_steps 1 \
    --seed 42 \
    "${EXTRA_ARGS[@]}"
