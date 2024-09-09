# Vision-Language Model Fine-Tuning and Inference

## 🚀 Overview
This repository contains shell scripts and Python configurations designed for fine-tuning and performing inference for the following vision-language models:
- LLaVA 13B
- LLaVA 7B
- MoE-LLaVA
- MobileVLM

---

## 🛠️ Installation and Setup

1. Clone the repository:
   git clone MODEL-REPO

2. Install required dependencies:
   pip install -r requirements.txt

3. Set up DeepSpeed by following their official installation guide: [DeepSpeed Documentation](https://www.deepspeed.ai/).

---

## 📜 Scripts Overview

### 1️⃣ LLaVA 13B Fine-Tuning (`LLaVA_13B_FINETUNE.sh`)
This script fine-tunes the LLaVA 13B model using DeepSpeed. It is designed to handle various configurations, including data path, image folders, batch sizes, and more.

#### Arguments Overview:
- --model_name_or_path: Path to the pre-trained LLaVA 13B model.
- --image_folder: Directory containing the training images.
- --data_path: Directory containing the dataset in JSON format.
- --output_dir: Directory where the fine-tuned model checkpoints will be saved.

#### Command Example:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/llava-v1.5-13B/ \
    --image_folder IMAGE_DIRECTORY \
    --data_path JSON_FILE_DIRECTORY \
    --output_dir OUTPUT_FINE_TUNED

---

### 2️⃣ LLaVA 7B Fine-Tuning (`LLaVA_7B_FINETUNE.sh`)
This script fine-tunes the LLaVA 7B model using DeepSpeed.

#### Arguments Overview:
- --model_name_or_path: Path to the pre-trained LLaVA 7B model.
- --image_folder: Directory containing the training images.
- --data_path: Directory containing the dataset in JSON format.
- --output_dir: Directory where the fine-tuned model checkpoints will be saved.

#### Command Example:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/llava-v1.5-7B/ \
    --image_folder IMAGE_DIRECTORY \
    --data_path JSON_FILE_DIRECTORY \
    --output_dir OUTPUT_FINE_TUNED

---

### 3️⃣ MoE-LLaVA Fine-Tuning (`MoE_LLaVA_FINETUNE.sh`)
This script fine-tunes the MoE-LLaVA model using DeepSpeed with the Mixture of Experts (MoE) method.

#### Arguments Overview:
- --model_name_or_path: Path to the fine-tuned LLaVA model.
- --image_folder: Directory containing the training images.
- --data_path: Directory containing the dataset in JSON format.
- --output_dir: Directory where the fine-tuned model checkpoints will be saved.

#### Command Example:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/moellava/train/train_mem.py \
    --moe_enable True \
    --num_experts 4 \
    --top_k_experts 2 \
    --model_name_or_path ./checkpoints/llava-v1.5-7B/ \
    --image_folder IMAGE_DIRECTORY \
    --data_path JSON_FILE_DIRECTORY \
    --output_dir OUTPUT_FINE_TUNED

---

### 4️⃣ MobileVLM Fine-Tuning (`MobileVLM_FINETUNE.sh`)
This script fine-tunes the MobileVLM model using DeepSpeed.

#### Arguments Overview:
- --model_name_or_path: Path to the pre-trained MobileVLM model.
- --image_folder: Directory containing the training images.
- --data_path: Directory containing the dataset in JSON format.
- --output_dir: Directory where the fine-tuned model checkpoints will be saved.

#### Command Example:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/mobilevlm/train/train_mem.py \
    --deepspeed ./scripts/deepspeed/zero2.json \
    --model_name_or_path ./checkpoints/MobileVLM-1.7B/ \
    --image_folder IMAGE_DIRECTORY \
    --data_path JSON_FILE_DIRECTORY \
    --output_dir OUTPUT_FINE_TUNED

---

### 5️⃣ Inference for MobileVLM (`Run_MobileVLM_Inference.sh`)
After fine-tuning the MobileVLM model, this script performs inference on a test dataset and outputs the results.

#### Arguments Overview:
- --model-path: Path to the fine-tuned MobileVLM model.
- --input-json: Directory containing the test dataset in JSON format.
- --output-file: File to save the inference results.

#### Command Example:
PYTHONPATH=$WORKSPACE_DIR python3 $WORKSPACE_DIR/run_inference.py \
    --model-path OUTPUT_FINE_TUNED \
    --input-json TEST_JSON_DIRECTORY \
    --output-file OUTPUT_FILE

---

## 📂 File Structure

vision-language-models/
│
├── llava/
│   ├── train/
│   │   └── train_mem.py       # Training script for LLaVA models
├── moellava/
│   ├── train/
│   │   └── train_mem.py       # Training script for MoE-LLaVA
│
├── mobilevlm/
│   ├── train/
│   │   └── train_mem.py       # Training script for MobileVLM
├── run_inference.py           # Script for running inference on fine-tuned models
├── LLaVA_13B_FINETUNE.sh      # Fine-tuning shell script for LLaVA 13B
├── LLaVA_7B_FINETUNE.sh       # Fine-tuning shell script for LLaVA 7B
├── MoE_LLaVA_FINETUNE.sh      # Fine-tuning shell script for MoE-LLaVA
├── MobileVLM_FINETUNE.sh      # Fine-tuning shell script for MobileVLM
├── Run_MobileVLM_Inference.sh # Inference shell script for MobileVLM
├── README.md                  # This README file

---

## 📊 Training and Evaluation Metrics
The training scripts log progress via TensorBoard and W&B for visualization and debugging purposes. You can adjust the logging steps and evaluation strategies by modifying the corresponding arguments in the shell scripts.

---

## 🔧 Customization

### Batch Size and Training Steps
- Batch size for training is set to 32.
- Evaluation batch size is set to 4.
- Checkpoints are saved every 50,000 steps.

These values can be adjusted in the fine-tuning scripts as per the available GPU memory.

### Using DeepSpeed
DeepSpeed significantly reduces memory consumption during training. Ensure it is installed correctly, and you can adjust the configurations in `zero2.json` or `zero3.json` to fit your hardware.

---

## 📝 Best Practices

- **Use DeepSpeed**: Make sure DeepSpeed is installed to take full advantage of memory optimization.
- **Experiment with Learning Rates**: Adjust the learning_rate parameter in the scripts based on your dataset size.
- **Use TensorBoard**: Track training metrics using TensorBoard for real-time monitoring and debugging.

---

## 👥 Contributions
We welcome contributions to improve these scripts or add new features. Feel free to submit a pull request or open an issue for discussion.

---

## 📄 License
This repository is licensed under the MIT License. See the LICENSE file for more details.

---

By following this guide, you can efficiently fine-tune and infer using the LLaVA, MoE-LLaVA, and MobileVLM models, ensuring optimal performance and accuracy.

---

**Note**: Ensure that you have access to GPUs with adequate memory for training large models like LLaVA 13B, LLaVA 7B, MoE-LLaVA, and MobileVLM. 
**Note**: The models are fine-tuned on an A100 40GB GPU.
 
