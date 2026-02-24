#!/bin/bash

# LLaVA 13B Fine-Tuning with LoRA and Evaluation Script
# Description: This script performs fine-tuning using LoRA for the LLaVA 13B model with DeepSpeed, 
#              merges LoRA weights, and evaluates the fine-tuned model.

# Exit immediately if a command exits with a non-zero status
set -e

# Define the workspace directory
WORKSPACE_DIR="${PWD}"

# Define common variables for DeepSpeed and LLaVA model with LoRA
DEEPSPEED_CONFIG="./scripts/zero3.json"
MODEL_PATH="./checkpoints/llava-v1.5-13B/"
VISION_TOWER="openai/clip-vit-large-patch14-336"
LORA_OUTPUT_DIR="OUTPUT DIRECTORY"
IMAGE_FOLDER="IMAGE PATH"
DATA_PATH="JSON FILE DIRECTORY"

# Variables for LoRA merging script
MERGE_LORA_MODEL_PATH="LORA_OUTPUT_DIR"
MERGE_MODEL_BASE="MODEL_PATH"
SAVE_MODEL_PATH="MERGED_MODEL_PATH"

# Variables for Evaluation Step
TEST_IMAGE_FOLDER="TEST DATA DIRECTORY"
TEST_JSON="JSON TEST DIRECTORY"
QUERY="QUERY"


# Configuration 1: Fine-Tuning LLaVA 13B with LoRA
echo "Starting LLaVA 13B Fine-Tuning with LoRA..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/llava/train/train_mem.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_PATH \
    --version v1 \
    --image_folder $IMAGE_FOLDER \
    --data_path $DATA_PATH \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $LORA_OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --mm_projector_lr 2e-5 \
    --lora_alpha 256 \
    --lora_r 128 \
    --lora_enable True

echo "LLaVA 13B Fine-Tuning with LoRA Completed!"

# Configuration 2: Merging LoRA Weights
echo "Merging LoRA Weights..."
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model-path> <model-base> <save-model-path>"
    exit 1
fi

# Assign arguments for merging LoRA weights
MODEL_PATH=$1
MODEL_BASE=$2
SAVE_MODEL_PATH=$3

# Run LoRA weight merging script
python3 ./scripts/merge_lora_weights.py \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_BASE" \
    --save-model-path "$SAVE_MODEL_PATH"

echo "LoRA Weights Merged Successfully!"


# Configuration 3: Running Fine-Tuned LLaVA-13B Model for Evaluation
echo "Starting Evaluation of Fine-Tuned LLaVA-13B Model..."
PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/llava/eval/Run_LLaVA_Model.py \
    --model-path $SAVE_MODEL_PATH \
    --image-folder $TEST_IMAGE_FOLDER \
    --image-json $TEST_JSON \
    --query $QUERY

echo "Evaluation Completed!"

# End of script