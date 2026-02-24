import argparse
import os
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image

# Argument parser
parser = argparse.ArgumentParser(description="Run Qwen-VL model for image recognition")
parser.add_argument("--single_image", type=int, default=0, help="Set to 1 to run a single image")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
parser.add_argument("--image_dir", type=str, help="Directory containing images")
parser.add_argument("--output_csv", type=str, help="Path to output CSV file")
parser.add_argument("--query", type=str, help="query")

args = parser.parse_args()

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True).eval()

if args.single_image == 0:
    # Batch processing mode
    image_extensions = (".png", ".ppm", ".jpg")
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(image_extensions)]

    with open(args.output_csv, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Filename", "Output"])  # CSV header

        for img_file in image_files:
            image_path = os.path.join(args.image_dir, img_file)

            try:
                if not os.path.exists(image_path):
                    print(f"Skipping {img_file}: File not found.")
                    continue

                # Prepare query
                query = tokenizer.from_list_format([
                    {'image': image_path},  # Pass as string (path)
                    {'text':args.query},
                ])

                # Run inference
                response, history = model.chat(tokenizer, query=query, history=None)

                # Save result in CSV
                csv_writer.writerow([img_file, response])

                print(f"Processed: {img_file} -> {response}")

            except Exception as e:
                print(f"Error processing {img_file}: {e}")

    print(f"Processing complete. Results saved in {args.output_csv}.")

else:
    # Single image processing mode
    image_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"  # Change as needed
    image = Image.open(image_path).convert("RGB")

    query = tokenizer.from_list_format([
        {'image': image_path},  
        {'text': args.query},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    print("Generated Output:", response)