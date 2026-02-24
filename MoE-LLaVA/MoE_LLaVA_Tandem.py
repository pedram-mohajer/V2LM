import os
import csv
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
print("Available GPUs before setting device:")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set the device to GPU 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
print("\nAvailable GPUs after setting device:")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def get_image_paths(directory):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                absolute_path = os.path.join(root, file)
                image_paths.append(absolute_path)
    return image_paths

def main(args):
    disable_torch_init()
    # The fixed query for all images
    inp = "Can you see a car here? only choose either Yes or No"
    
    device = args.device
    load_4bit = args.load_4bit
    load_8bit = args.load_8bit
    model_path = args.model_path
    conv_mode = args.conv_mode

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, None, model_name, load_8bit, load_4bit, device=device
    )
    image_processor = processor['image']
    conv_template = conv_templates[conv_mode].copy()

    directory_path = args.directory_path
    image_paths = get_image_paths(directory_path)

    # Ensure that we have at least 3 images
    if len(image_paths) < 3:
        print("Not enough images in the directory. Please provide at least 3 images.")
        return
    # Only take the first 3 images
    image_paths = image_paths[:3]

    # Build conversation prompts (one for each image)
    prompts = []
    for _ in range(3):
        conv = conv_template.copy()  # reinitialize conversation template
        inp_with_token = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp_with_token)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)

    # Tokenize prompts as a batch. Each prompt is tokenized then unsqueezed so that we can concatenate.
    input_ids = torch.cat([
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        for prompt in prompts
    ], dim=0).to(model.device)

    # Preprocess images as a batch (assumes image_processor supports a list of images)
    images = [Image.open(path).convert('RGB') for path in image_paths]
    image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']\
        .to(model.device, dtype=torch.float16)

    # Prepare stopping criteria (using the separator string from the conversation template)
    stop_str = conv_template.sep if conv_template.sep_style != SeparatorStyle.TWO else conv_template.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # Generate outputs for the batch of 3 images
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    # Decode each output by removing the prompt tokens
    final_outputs = []
    for i in range(len(prompts)):
        prompt_len = input_ids[i].shape[0]
        decoded = tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True).strip()
        final_outputs.append(decoded)

    # Write the image path and corresponding prediction to CSV
    csv_file = 'moe-llava-OD-tandem-AE.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Path', 'Prediction'])
        for i, path in enumerate(image_paths):
            writer.writerow([path, final_outputs[i]])
            print(f"Image {i+1} ({path}) ----> {final_outputs[i]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process three images with the model.')
    parser.add_argument('--directory_path', type=str, required=True, help='Path to the directory containing images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation (default: cuda)')
    parser.add_argument('--load_4bit', action='store_true', help='Load model in 4-bit precision')
    parser.add_argument('--load_8bit', action='store_true', help='Load model in 8-bit precision')
    parser.add_argument('--conv_mode', type=str, choices=['phi', 'qwen', 'stablelm'], default='phi', help='Conversation mode (default: phi)')
    args = parser.parse_args()
    main(args)
