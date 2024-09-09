#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


def load_pretrained_model(
    lora_path,
    pretrained_path,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if pretrained_path is None:
        raise RuntimeError("Pretrained model missing. Provide path.")

    if lora_path is not None and (load_4bit or load_8bit):
        raise RuntimeError(
            "PEFT library does not support merging mixed precision. Need to quantize, save, then reload as 16 bit (not 4 or 8 bit)"
        )

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    # Load LLaVA model
    cfg = AutoConfig.from_pretrained(pretrained_path)
    if lora_path is not None:
        cfg = AutoConfig.from_pretrained(lora_path)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
    print("Loading LLaVA from base model...")
    model = LlavaLlamaForCausalLM.from_pretrained(
        pretrained_path, low_cpu_mem_usage=True, config=cfg, **kwargs
    )

    if lora_path is not None:
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(
                    token_num, tokem_dim, device=model.device, dtype=model.dtype
                )
            )
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(
                    token_num, tokem_dim, device=model.device, dtype=model.dtype
                )
            )

        print("Loading additional LLaVA weights...")
        if os.path.exists(os.path.join(lora_path, "non_lora_trainables.bin")):
            non_lora_trainables = torch.load(
                os.path.join(lora_path, "non_lora_trainables.bin"), map_location="cpu"
            )
        else:
            raise RuntimeError("missing non lora trainables")
        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v
            for k, v in non_lora_trainables.items()
        }
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in non_lora_trainables.items()
            }
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModelForCausalLM

        print("Loading LoRA weights...")
        model = PeftModelForCausalLM.from_pretrained(model, lora_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()

    print("Model is loaded...")

    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
