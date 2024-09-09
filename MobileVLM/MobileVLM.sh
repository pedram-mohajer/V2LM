#!/bin/bash

# MobileVLM Fine-Tuning and Inference Script

# Exit immediately if a command exits with a non-zero status
set -e

# Define the workspace directory
WORKSPACE_DIR="${PWD}"

# Define common variables for DeepSpeed and MobileVLM model
DEEPSPEED_CONFIG="./scripts/deepspeed/zero2.json"
MODEL_PATH="./checkpoints/MobileVLM-1.7B/"
VISION_TOWER="openai/clip-vit-large-patch14-336"
IMAGE_FOLDER="IMAGE DIRECTORY"
DATA_PATH="JSON FILE DIRECTORY"
OUTPUT_FINE_TUNED="OUTPUT_FINE_TUNED"
CACHE_DIR="./cache_dir"

# Variables for inference step
TEST_JSON_DIR="TEST JSON DIRECTORY"
QUERY="QUERY"
OUTPUT_FILE="OUTPUT FILE"

# 1. Fine-Tuning Step
echo "Starting MobileVLM Fine-Tuning..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/mobilevlm/train/train_mem.py \
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
    --output_dir $OUTPUT_FINE_TUNED \
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

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/mobilevlm/train/train_mem.py \
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
    --output_dir $OUTPUT_FINE_TUNED \
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

echo "MobileVLM Fine-Tuning Completed!"

# 2. Inference Step
echo "Starting Inference on Fine-Tuned MobileVLM Model..."
PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/run_inference.py \
    --model-path $OUTPUT_FINE_TUNED \
    --conv-mode v1 \
    --temperature 0.2 \
    --top_p 0.9 \
    --num_beams 1 \
    --max_new_tokens 512 \
    --load_8bit False \
    --load_4bit False \
    --input-json $TEST_JSON_DIR \
    --prompt $QUERY \
    --output-file $OUTPUT_FILE

echo "Inference Completed!"

# End of script
