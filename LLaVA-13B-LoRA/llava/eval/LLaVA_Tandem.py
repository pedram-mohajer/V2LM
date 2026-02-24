import argparse
import torch
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
from io import BytesIO
import re

import time


def image_parser(image_files, sep):
    return image_files.split(sep)


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files, sep):
    return [load_image(img) for img in image_parser(image_files, sep)]


def eval_model(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    queries = args.query.split(args.sep)
    image_files = image_parser(args.image_file, args.sep)
    images = load_images(args.image_file, args.sep)

    responses = []

    for index, (query, image) in enumerate(zip(queries, images)):
        modified_query = query
        if IMAGE_PLACEHOLDER in query:
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            modified_query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
        else:
            if model.config.mm_use_im_start_end:
                modified_query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query
            else:
                modified_query = DEFAULT_IMAGE_TOKEN + "\n" + query

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], modified_query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        times = []
        for i in range(50):
            start_time = time.perf_counter()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size for _ in range(len(image_tensor))],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)


        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        responses.append((index + 1, output_text))


    for idx, response in responses:
        print(f"Response {idx}: {response}\n")

    #elapsed_time_ms = (end_time - start_time) * 1000
    #print(f"Elapsed time: {elapsed_time_ms:.2f} ms")
    average_time_ms = sum(times) / len(times)
    print(f"Average processing time over 100 iterations: {average_time_ms:.2f} ms")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--sep", type=str, default="|")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)

