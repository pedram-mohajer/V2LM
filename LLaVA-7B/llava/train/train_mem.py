import os

# Set CUDA_VISIBLE_DEVICES before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import wandb

# Set the device to GPU 1
torch.cuda.set_device(0)

wandb.login()

# Print the number of GPUs available
print(f"Number of GPUs available: {torch.cuda.device_count()}")

# List the names of the available GPUs
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

torch.cuda.empty_cache()

# Now import your training script
from llava.train.train import train

# Continue with your training script

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'



'''
import torch

# Print the number of GPUs available
print(f"Number of GPUs available: {torch.cuda.device_count()}")

# List the names of the available GPUs
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Ensure correct device usage
target_gpu = 0  # Index of the GPU you want to use
if target_gpu < torch.cuda.device_count():
    torch.cuda.set_device(target_gpu)
    print(f"Using GPU {target_gpu}: {torch.cuda.get_device_name(target_gpu)}")
else:
    print(f"GPU {target_gpu} is not available. Using default GPU 0.")
    torch.cuda.set_device(0)
    print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
'''
# Print the current device being used
# print(f"Current device: {torch.cuda.current_device()}")
# print(f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")



if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")