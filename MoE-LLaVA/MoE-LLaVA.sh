#!/bin/bash

# LLaVA Pre-Training, Fine-Tuning, MoE Fine-Tuning, and Evaluation Script
# Description: # MoE-LLaVA Fine-Tuning and Inference Script

# Exit immediately if a command exits with a non-zero status
set -e

# Define the workspace directory
WORKSPACE_DIR="${PWD}"

# Define common variables for model training and fine-tuning
DEEPSPEED_CONFIG="./scripts/zero2.json"
MODEL_NAME="microsoft/phi-2"
IMAGE_TOWER="openai/clip-vit-large-patch14-336"
JSON_DIR="JSON DIRECTORY"
IMAGE_DIR="IMAGE FOLDER"
CACHE_DIR="./cache_dir"

# Define output directories for each stage
OUTPUT_PRE_TRAINING="OUTPUT_PRE_TRANING"
OUTPUT_FINE_TUNING="OUTPUT_FINE_TUNING"
OUTPUT_MOE_FINE_TUNING="OUTPUT"

# Define test directories for evaluation
TEST_IMAGE_DIR="TEST IMAGE DIRECTORY"
TEST_JSON_DIR="TEST JSON DIRECTORY"

# 1. Pre-Training Step
echo "Starting Pre-Training..."
CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/moellava/train/train_mem.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_NAME \
    --version plain \
    --data_path $JSON_DIR \
    --image_folder $IMAGE_DIR \
    --image_tower $IMAGE_TOWER \
    --image_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_PRE_TRAINING \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir $CACHE_DIR

echo "Pre-Training Completed!"

# 2. Fine-Tuning Step
echo "Starting Fine-Tuning..."
CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/MoE-LLaVA/moellava/train/train_mem.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_NAME \
    --version phi \
    --data_path $JSON_DIR \
    --image_folder $IMAGE_DIR \
    --image_tower $IMAGE_TOWER \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter $OUTPUT_PRE_TRAINING/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_FINE_TUNING \
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
    --report_to tensorboard \
    --cache_dir $CACHE_DIR

echo "Fine-Tuning Completed!"

# 3. MoE Fine-Tuning Step
echo "Starting MoE Fine-Tuning..."
CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/moellava/train/train_mem.py \
    --moe_enable True \
    --num_experts 4 \
    --top_k_experts 2 \
    --capacity_factor 1.5 \
    --moe_mode sparse \
    --use_residual False \
    --router_aux_loss_coef 0.01 \
    --train_modules fc1 fc2 wg \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $OUTPUT_FINE_TUNING \
    --version phi \
    --data_path $JSON_DIR \
    --image_folder $IMAGE_DIR \
    --image_tower $IMAGE_TOWER \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_MOE_FINE_TUNING \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 24000 \
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
    --report_to tensorboard \
    --cache_dir $CACHE_DIR

echo "MoE Fine-Tuning Completed!"

# 4. Model Evaluation with JSON Input
echo "Starting Model Evaluation..."
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/predict_json.py \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/predict.py \
    --directory_path $TEST_IMAGE_DIR \
    #--json_file $TEST_JSON_DIR \
    --model_path $OUTPUT_MOE_FINE_TUNING \
    --device cuda \
    --conv_mode phi

echo "Model Evaluation Completed!"
