#!/bin/bash

# NVILA-Lite-8B Fine-Tuning and Evaluation Script
# Description: Fine-tunes the NVILA-Lite-8B model with DeepSpeed and evaluates it using vila-infer.

# Exit immediately if a command exits with a non-zero status
set -e

# Activate Conda environment
echo "Activating vila conda environment..."
conda activate vila

# Define workspace and input arguments
WORKSPACE_DIR="${PWD}"
STAGE_PATH=${1:-"Efficient-Large-Model/NVILA-Lite-8B"}
DATA_MIXTURE=${2:-"YOUR_DATASET_NAME"}  # Replace this if needed
OUTPUT_DIR=${3:-"runs/train/nvila-lite-8b-sft"}
DEEPSPEED_CONFIG="scripts/zero3.json"

# Fine-tuning hyperparameters
DEFAULT_RUN_NAME="nvila-lite-8b-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=64
DEFAULT_GRADIENT_ACCUMULATION_STEPS=2
VISION_TOWER="Efficient-Large-Model/paligemma-siglip-so400m-patch14-448"
GPUS_PER_NODE=8
MASTER_PORT=25001

# Print setup summary
echo "---------------------------------------"
echo "Fine-tuning NVILA-Lite-8B"
echo "Model Path: $STAGE_PATH"
echo "Dataset: $DATA_MIXTURE"
echo "Output Directory: $OUTPUT_DIR"
echo "GPUs Per Node: $GPUS_PER_NODE"
echo "---------------------------------------"

# Export PyTorch memory allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
torchrun \
  --nnodes=1 \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=$MASTER_PORT \
  llava/train/train_mem.py \
  --deepspeed $DEEPSPEED_CONFIG \
  --model_name_or_path $STAGE_PATH \
  --data_mixture $DATA_MIXTURE \
  --vision_tower $VISION_TOWER \
  --mm_vision_select_feature cls_patch \
  --mm_projector mlp_downsample_3x3_fix \
  --tune_vision_tower True \
  --tune_mm_projector True \
  --tune_language_model True \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio dynamic \
  --bf16 True \
  --output_dir $OUTPUT_DIR/model \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps $DEFAULT_GRADIENT_ACCUMULATION_STEPS \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 16 \
  --vflan_no_system_prompt True \
  --report_to none

echo "✅ Fine-tuning completed!"

# =====================
# Run Inference
# =====================

# Set evaluation arguments (fill these before running)
MODEL_PATH="${OUTPUT_DIR}/model"
TEST_CSV="PATH_TO_TEST_DATA.csv"        # <== Set this
QUERY="QUERY_TEXT"                      # <== Set this
CUDA_DEVICE="7"

echo "Starting evaluation..."
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

vila-infer \
  --model-path $MODEL_PATH \
  --conv-mode auto \
  --text "$QUERY" \
  --csv-file $TEST_CSV

echo "✅ Evaluation completed!"