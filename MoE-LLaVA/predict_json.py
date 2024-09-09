import os
import csv
import json
import torch
import argparse
from PIL import Image
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# Set CUDA_VISIBLE_DEVICES to only GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Print available GPUs before setting device
print("Available GPUs before setting device:")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set the device to GPU 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

# Print available GPUs after setting device
print("\nAvailable GPUs after setting device:")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def get_image_paths(directory=None, json_file=None):
    image_paths = []
    
    if directory:
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')  # Add more extensions if needed
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    absolute_path = os.path.join(root, file)
                    image_paths.append(absolute_path)

    elif json_file:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for item in data:
                if 'image' in item:  # Ensure the 'image' key exists
                    image_paths.append(item['image'])

    return image_paths

def main(args):
    disable_torch_init()
    inp = "As a car driver, which direction should you turn the steering wheel?"
    
    device = args.device
    load_4bit = args.load_4bit
    load_8bit = args.load_8bit
    model_path = args.model_path
    conv_mode = args.conv_mode

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    image_processor = processor['image']
    conv_template = conv_templates[conv_mode].copy()
    roles = conv_template.roles

    directory_path = args.directory_path
    json_file = args.json_file
    image_paths = get_image_paths(directory=directory_path, json_file=json_file)

    # Prepare to write results to CSV
    csv_file = 'moe-llava-ALC-tandem.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Path', 'Prediction'])

        cnt = 0
        for image_path in image_paths:
            # Reinitialize conversation template for each image
            conv = conv_template.copy()

            image_tensor = image_processor.preprocess(Image.open(image_path).convert('RGB'), return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)

            inp_with_token = DEFAULT_IMAGE_TOKEN + '\n' + inp
            
            conv.append_message(conv.roles[0], inp_with_token)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

            # Write the image path and prediction to the CSV file
            writer.writerow([image_path, outputs])

            print(cnt, '---->', outputs)
            cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--directory_path', type=str, help='Path to the directory containing images')
    parser.add_argument('--json_file', type=str, help='Path to the JSON file containing image paths')
    parser.add_argument('--model_path',     type=str, required=True,  help='Path to the model')
    parser.add_argument('--device',         type=str, default='cuda', help='Device to use for computation (default: cuda)')
    parser.add_argument('--load_4bit',      action='store_true',      help='Load model in 4-bit precision')
    parser.add_argument('--load_8bit',      action='store_true',      help='Load model in 8-bit precision')
    parser.add_argument('--conv_mode',      type=str,                 choices=['phi', 'qwen', 'stablelm'], default='phi', help='Conversation mode (default: phi)')
    args = parser.parse_args()

    if not args.directory_path and not args.json_file:
        raise ValueError("Either --directory_path or --json_file must be provided.")
    
    main(args)
