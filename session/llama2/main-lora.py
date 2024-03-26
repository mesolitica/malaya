# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da
# modified for custom dataset and instructions

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import random
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

# This example fine-tunes Llama v2 model on Guanace dataset
# using QLoRA. At the end of the script we perform merging the weights
# Use it by correctly passing --model_name argument when running the
# script.
#
# Versions used:
# accelerate ==0.21.0
# peft == 0.4.0
# bitsandbytes == 0.40.2
# transformers == 4.31.0
# trl == 0.4.7

# For models that have `config.pretraining_tp > 1` install:
# pip install git+https://github.com/huggingface/transformers.git

system_prompts = [
    'You are an AI assistant',
    'Anda adalah pembantu AI',
    'Anda adalah pembantu AI yang berguna',
    'You are a helpful assistant',
    'Anda seorang pembantu yang berguna',
]


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    use_flash_attention2: bool = field(
        default=False, metadata={
            "help": "use flash attention2"}, )


def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    print(script_args, training_args)

    compute_dtype = getattr(torch, script_args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=script_args.use_4bit,
        bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=script_args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and script_args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = {"": 0}

    if script_args.use_flash_attention2:
        from llama_patch import replace_attn_with_flash_attn
        replace_attn_with_flash_attn()

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=bnb_config,
        use_auth_token=True,
        use_cache=False,
        device_map="auto"
    )

    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    if script_args.use_flash_attention2:
        from llama_patch import upcast_layer_for_flash_attention
        model = upcast_layer_for_flash_attention(model, compute_dtype)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    tokenizer.padding_side = "right"
    model.config.use_cache = False

    def generate_and_tokenize_prompt(row):
        system_prompt = random.choice(system_prompts)
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']

        if '<bot>:' in row['input'] and row['output'] is None:
            inputs, outputs = [], []
            splitted = row['input'].split('<bot>:')
            for i in range(len(splitted) - 1):
                if i == 0:
                    human = splitted[i].replace('<manusia>:', '')
                else:
                    human = splitted[i].split('<manusia>:')[1]
                bot = splitted[i + 1].split('<manusia>:')[0]
                inputs.append(human.strip())
                outputs.append(bot.strip())
        else:
            inputs = [row['input']]
            outputs = [row['output']]
        for input, output in zip(inputs, outputs):
            texts.append(f'{input} [/INST] {output} </s><s>[INST] ')
        return {'text': ''.join(texts)}

    dataset = load_dataset("json", data_files=script_args.dataset_name, split="train")
    dataset = dataset.map(generate_and_tokenize_prompt)
    dataset = dataset.remove_columns(['prompt_input', 'input', 'output'])

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=script_args.packing,
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if last_checkpoint:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
