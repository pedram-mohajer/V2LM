#!/bin/bash

# LLaVA 7B Fine-Tuning and Evaluation Script
# Description: This script runs two fine-tuning processes sequentially for the LLaVA 7B model using DeepSpeed, followed by the evaluation of the fine-tuned model.

# Exit immediately if a command exits with a non-zero status
set -e

# Define the workspace directory
WORKSPACE_DIR="${PWD}"

# Define common variables for DeepSpeed and LLaVA model
DEEPSPEED_CONFIG="./scripts/zero2.json"
MODEL_PATH="./checkpoints/llava-v1.5-7b/"
VISION_TOWER="openai/clip-vit-large-patch14-336"
OUTPUT_DIR="OUTPUT DIRECTORY"
IMAGE_FOLDER="IMAGE PATH"
DATA_PATH="JSON FILE DIRECTORY"

# Variables for Evaluation Step
TEST_IMAGE_FOLDER="TEST DATA DIRECTORY"
TEST_JSON="JSON TEST DIRECTORY"
QUERY="QUERY"

# Configuration 1: First Run without tuning MLP adapter
echo "Starting LLaVA 7B Fine-Tuning (Run 1)..."
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
    --output_dir $OUTPUT_DIR \
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
    --report_to wandb

echo "LLaVA 7B Fine-Tuning (Run 1) Completed!"

# Configuration 2: Second Run with MLP adapter tuning
echo "Starting LLaVA 7B Fine-Tuning (Run 2 with MLP Adapter Tuning)..."
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
    --output_dir $OUTPUT_DIR \
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
    --tune_mm_mlp_adapter True

echo "LLaVA 7B Fine-Tuning (Run 2) Completed!"

# Configuration 3: Running Fine-Tuned LLaVA Model for Evaluation
echo "Starting Evaluation of Fine-Tuned LLaVA Model..."
PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/llava/eval/Run_LLaVA_Model.py \
    --model-path $OUTPUT_DIR \
    --image-folder $TEST_IMAGE_FOLDER \
    --image-json $TEST_JSON \
    --query $QUERY

echo "Evaluation Completed!"

# End of script