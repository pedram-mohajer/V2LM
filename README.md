## 🚀 Toward Inherently Robust VLMs Against Visual Perception Attacks
This repository contains shell scripts and Python configurations designed for fine-tuning and performing inference for the following vision-language models:
- LLaVA-13B-LoRA
- LLaVA-7B
- MoE-LLaVA
- MobileVLM
- Qwen-VL
- NVILA


This work was Accepted to the [2026 IEEE Intelligent Vehicles Symposium (IV 2026)](https://arxiv.org/abs/2506.11472).


---

## 🛠️ Installation and Setup
# Vision Language Model Fine-Tuning and Inference

1. Clone the repository:
   ```bash
   git clone MODEL-REPO
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up DeepSpeed by following their official installation guide: [DeepSpeed Documentation](https://www.deepspeed.ai/).

---

## 📝 Scripts Overview

### 1️⃣ LLaVA 13B Fine-Tuning (`LLaVA_13B_FINETUNE.sh`)
This script fine-tunes the LLaVA 13B model using DeepSpeed.

#### Arguments Overview:
- `--model_name_or_path`: Path to the pre-trained LLaVA 13B model.
- `--image_folder`: Directory containing the training images.
- `--data_path`: Directory containing the dataset in JSON format.
- `--output_dir`: Directory where the fine-tuned model checkpoints will be saved.

#### Command Example:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/llava-v1.5-13B/ \
    --image_folder IMAGE_DIRECTORY \
    --data_path JSON_FILE_DIRECTORY \
    --output_dir OUTPUT_FINE_TUNED
```

---

### 2️⃣ LLaVA 7B Fine-Tuning (`LLaVA_7B_FINETUNE.sh`)
This script fine-tunes the LLaVA 7B model using DeepSpeed.

#### Arguments Overview:
- `--model_name_or_path`: Path to the pre-trained LLaVA 7B model.
- `--image_folder`: Directory containing the training images.
- `--data_path`: Directory containing the dataset in JSON format.
- `--output_dir`: Directory where the fine-tuned model checkpoints will be saved.

#### Command Example:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/llava-v1.5-7B/ \
    --image_folder IMAGE_DIRECTORY \
    --data_path JSON_FILE_DIRECTORY \
    --output_dir OUTPUT_FINE_TUNED
```

---

### 3️⃣ MoE-LLaVA Fine-Tuning (`MoE_LLaVA_FINETUNE.sh`)
This script fine-tunes the MoE-LLaVA model using DeepSpeed with the Mixture of Experts (MoE) method.

#### Arguments Overview:
- `--model_name_or_path`: Path to the fine-tuned LLaVA model.
- `--image_folder`: Directory containing the training images.
- `--data_path`: Directory containing the dataset in JSON format.
- `--output_dir`: Directory where the fine-tuned model checkpoints will be saved.

#### Command Example:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/moellava/train/train_mem.py \
    --moe_enable True \
    --model_name_or_path ./checkpoints/MoE-v1.5-7B/ \
    --image_folder IMAGE_DIRECTORY \
    --data_path JSON_FILE_DIRECTORY \
    --output_dir OUTPUT_FINE_TUNED
```

---

### 4️⃣ MobileVLM Fine-Tuning (`MobileVLM_FINETUNE.sh`)
This script fine-tunes the MobileVLM model using DeepSpeed.

#### Arguments Overview:
- `--model_name_or_path`: Path to the pre-trained MobileVLM model.
- `--image_folder`: Directory containing the training images.
- `--data_path`: Directory containing the dataset in JSON format.
- `--output_dir`: Directory where the fine-tuned model checkpoints will be saved.

---

### 5️⃣ Qwen-VL Fine-Tuning (`Qwen-VL.sh`)
This script fine-tunes the Qwen-VL model and evaluates it on a test set.

#### Arguments Overview:
- `--model_name_or_path`: Path to the pre-trained Qwen-VL model.
- `--data_path`: Directory containing the dataset in JSON format.
- `--output_dir`: Directory where the fine-tuned model checkpoints will be saved.
- `--num_train_epochs`: Number of training epochs.
- `--learning_rate`: Learning rate for optimization.
- `--save_steps`: Frequency of checkpoint saving.
- `--evaluation_strategy`: Strategy for model evaluation.

#### Command Example:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 $WORKSPACE_DIR/finetune.py \
    --model_name_or_path ./checkpoints/Qwen/Qwen-VL-Chat \
    --data_path ./data/MY_DATASET/train.json \
    --output_dir ./checkpoints/Qwen-VL-finetuned \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --save_steps 1000 \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --deepspeed ./finetune/ds_config_zero3.json
```


### 6️⃣ NVILA-Lite-8B Fine-Tuning and Evaluation (`NVILA.sh`)  
This script performs **end-to-end fine-tuning and evaluation** of the NVILA-Lite-8B model using DeepSpeed and `vila-infer`. It supports both training and inference in one unified workflow.

#### Arguments Overview:  
- `STAGE_PATH`: Path to the pre-trained NVILA-Lite-8B model (default: `Efficient-Large-Model/NVILA-Lite-8B`).  
- `DATA_MIXTURE`: Name of the training dataset or mixture.  
- `OUTPUT_DIR`: Directory where the fine-tuned model and logs will be saved.  

#### Command Example:  
```bash
---

## 📚 File Structure
├── LLaVA-13B-LoRA
│   ├── LICENSE
│   ├── llava
│   ├── LLaVA-13-LoRA.sh
│   └── scripts
├── LLaVA-7B
│   ├── LICENSE
│   ├── llava
│   ├── LLaVA-7B.sh
│   └── scripts
├── MobileVLM
│   ├── LICENSE
│   ├── mobilevlm
│   ├── MobileVLM.sh
│   └── scripts
├── MoE-LLaVA
│   ├── LICENSE
│   ├── moellava
│   ├── MoE-LLaVA.sh
│   └── scripts
├── Qwen-VL
│   ├── LICENSE
│   ├── Qwen-VL.sh
│   └── finetune
├── NVILA
│   ├── LICENSE
│   ├── NVILA.sh
│   ├── scripts
│   └── llava
├── Sample
│   ├── DRP-Attack
│   ├── RAUCA
│   └── Shadow-Attack


---
```

## 📊 Training and Evaluation Metrics
Training scripts log progress via TensorBoard and W&B for visualization and debugging purposes. Modify logging steps and evaluation strategies as needed.

---

## 🔧 Customization
- Batch size: 32 (training), 4 (evaluation)
- Checkpoints saved every 50,000 steps
- DeepSpeed configurations adjustable in `zero2.json` or `zero3.json`

---

By following this guide, you can efficiently fine-tune and infer using the LLaVA, MoE-LLaVA, MobileVLM, and Qwen-VL models.

**Note**: Ensure you have access to GPUs with adequate memory for fine-tuning large models.


**Note**: Ensure that you have access to GPUs with adequate memory for fine-tuning large models.<br>
**Note**: The models are fine-tuned on an A100 40GB GPU, except for Qwen-VL (2×A100 80GB GPUs) and NVILA (4×A100 40GB GPUs).

 
