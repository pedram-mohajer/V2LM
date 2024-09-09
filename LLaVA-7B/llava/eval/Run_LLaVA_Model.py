import argparse
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
# Print the number of GPUs available
print(f"Number of GPUs available: {torch.cuda.device_count()}")
# List the names of the available GPUs
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

import pandas as pd

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import json



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

def load_images_from_folder(folder_path):
    images = []
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image_path = os.path.join(folder_path, filename)
            image = load_image(image_path)
            images.append(image)
            image_paths.append(image_path)
    return images, image_paths


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Load images from folder or JSON
    if args.image_folder:
        images, image_paths = load_images_from_folder(args.image_folder)
    elif args.image_json:
        images, image_paths = load_images_from_json(args.image_json)
    else:
        raise ValueError("Either --image-folder or --image-json must be provided")


    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    data = []

    for image, image_path in tqdm(zip(images, image_paths), total=len(images)):
        qs = args.query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"------------------------>         Output for {image_path}: {outputs}")
        data.append([image_path, outputs])

    df = pd.DataFrame(data, columns=["Image Path", "ClassId"])
    df.to_csv("Prediction.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",     type=str,   default="facebook/opt-350m")
    parser.add_argument("--model-base",     type=str,   default=None)
    parser.add_argument("--query",          type=str,   required=True)
    parser.add_argument("--image-folder",   type=str,   default=None)
    parser.add_argument("--image-json",     type=str,   default=None)
    parser.add_argument("--conv-mode",      type=str,   default=None)
    parser.add_argument("--sep",            type=str,   default=",")
    parser.add_argument("--temperature",    type=float, default=0.2)
    parser.add_argument("--top_p",          type=float, default=None)
    parser.add_argument("--num_beams",      type=int,   default=1)
    parser.add_argument("--max_new_tokens", type=int,   default=512)
    args = parser.parse_args()

    eval_model(args)