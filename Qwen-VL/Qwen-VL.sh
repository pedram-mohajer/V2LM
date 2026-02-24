#!/bin/bash

# Qwen-VL Fine-Tuning and Evaluation Script
# Description: Fine-tunes the Qwen-VL model and evaluates it on a test set.

set -e  # Exit on error

# ========================
# Configurable Variables
# ========================

WORKSPACE_DIR="${PWD}"

# General info
MODEL_NAME="Qwen/Qwen-VL-Chat"
DATASET_NAME="MY_DATASET"
QUERY="YOUR_QUESTION_HERE"  # Example: "What is this traffic sign?"

# Model paths
BASE_MODEL_PATH="${WORKSPACE_DIR}/checkpoints/${MODEL_NAME}"
OUTPUT_DIR="${WORKSPACE_DIR}/checkpoints/${MODEL_NAME}_finetuned_${DATASET_NAME}"

# Data paths
TRAIN_JSON="${WORKSPACE_DIR}/data/${DATASET_NAME}/train.json"
TRAIN_IMAGE_FOLDER="${WORKSPACE_DIR}/data/${DATASET_NAME}/train_images"

TEST_JSON="${WORKSPACE_DIR}/data/${DATASET_NAME}/test.json"
TEST_IMAGE_FOLDER="${WORKSPACE_DIR}/data/${DATASET_NAME}/test_images"

# Evaluation output
OUTPUT_CSV="${DATASET_NAME}_results.csv"
SINGLE_IMAGE=0

# DeepSpeed config
DEEPSPEED_CONFIG="${WORKSPACE_DIR}/finetune/ds_config_zero3.json"

# Distributed training config
export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
"

# ========================
# Fine-Tuning Qwen-VL
# ========================
echo "Starting Qwen-VL Fine-Tuning..."
torchrun $DISTRIBUTED_ARGS $WORKSPACE_DIR/finetune.py \
    --model_name_or_path $BASE_MODEL_PATH \
    --data_path $TRAIN_JSON \
    --bf16 True \
    --fix_vit True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed $DEEPSPEED_CONFIG

echo "Qwen-VL Fine-Tuning Completed!"

# ========================
# Evaluation of Qwen-VL
# ========================
echo "Starting Evaluation of Fine-Tuned Qwen-VL Model..."
export CUDA_VISIBLE_DEVICES=0
python3 $WORKSPACE_DIR/Run_QwenVL_Model.py \
    --single_image $SINGLE_IMAGE \
    --model_path $OUTPUT_DIR \
    --image_dir $TEST_IMAGE_FOLDER \
    --output_csv $OUTPUT_CSV \
    --query "$QUERY"

echo "Evaluation Completed!"

# End of script
