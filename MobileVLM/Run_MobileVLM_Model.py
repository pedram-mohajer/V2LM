import torch
import argparse
from PIL import Image
from pathlib import Path
import sys
import json
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress tracking

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def load_image(image_file):
    return Image.open(image_file).convert("RGB")


def load_images_from_json(json_path):
    images = []
    image_paths = []
    with open(json_path, 'r') as f:
        data = json.load(f)
        for entry in data:
            image_path = entry['image']
            image = load_image(image_path)
            images.append(image)
            image_paths.append(image_path)
    return images, image_paths


def inference_once(model, tokenizer, image_processor, args, image, prompt_text):
    images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

    stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2

    # Input
    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    
    # Inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # Result-Decode
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]

    return outputs


def eval_model(args):
    # Model
    disable_torch_init()

    # Load model and tokenizer
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.load_8bit, args.load_4bit)

    # Load images from JSON
    images, image_paths = load_images_from_json(args.input_json)

    # Process each image
    data = []
    for image, image_path in tqdm(zip(images, image_paths), total=len(images), desc="Processing images"):
        conv_template = conv_templates[args.conv_mode].copy()
        conv_template.append_message(conv_template.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + args.prompt)
        conv_template.append_message(conv_template.roles[1], None)
        prompt_text = conv_template.get_prompt()

        classid = inference_once(model, tokenizer, image_processor, args, image, prompt_text)
        data.append([image_path, classid])

    # Save results to CSV
    df = pd.DataFrame(data, columns=["Image Path", "ClassId"])
    df.to_csv(args.output_file, index=False)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="mtgv/MobileVLM-1.7B")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--load_4bit", type=bool, default=False)
    parser.add_argument("--input-json", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the model")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    eval_model(args)
