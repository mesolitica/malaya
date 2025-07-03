#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling
# task. Pointers for this are left as comments.

import torch

torch._dynamo.config.optimize_ddp=False

# monkey patch pytorch validation
def _is_valid_woq_optimization_pattern():
    def fn(match):
        assert all(k in match.kwargs for k in ("x", "weight", "scales"))
        try:
            x = match.kwargs["x"].meta["val"]
            weight = match.kwargs["weight"].meta["val"]
            print(x.dtype, weight.dtype, x.device)
            scales = match.kwargs["scales"].meta["val"]
            
            return (
                # For now, we only support woq mm kernels
                # with x.type=bfloat16 and w.type=int8
                x.dtype == torch.bfloat16
                and weight.dtype == torch.int8
                and scales.dtype == torch.bfloat16
                # _weight_int8pack_mm kernel only supports cpu now
                # TODO: add cuda kernel support instead of calling mul+sum
                and x.device.type == "cpu"
                and x.device == weight.device
                and x.device == scales.device
            )
        except Exception as e:
            print(e, match)
            return False

    return fn

from torch._inductor.fx_passes import quantization
quantization._is_valid_woq_optimization_pattern = _is_valid_woq_optimization_pattern

from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
from datasets import load_dataset

import transformers
import random
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AddedToken,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import Qwen2ForCausalLM
import streaming
import json
import numpy as np
import pandas as pd
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from cut_cross_entropy import linear_cross_entropy

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    rank: Optional[int] = field(
        default=32,
        metadata={
            "help": (
                "rank"
            )
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " +
            ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None, metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")}, )
    config_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"}, )
    use_fast_tokenizer: bool = field(
        default=True, metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}, )
    model_revision: str = field(
        default="main", metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."}, )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine.")}, )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."),
            "choices": [
                "auto",
                "bfloat16",
                "float16",
                "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
                self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

def pad_attention_mask_3d(attention_mask, max_size = 4096, value = 0.0):
    maxlen = attention_mask.shape[-1]
    return F.pad(
        attention_mask,
        (0, max_size - maxlen, 0, max_size - maxlen),
        value = value,
    )

def block_diagonal_concat_inverted(*masks, dtype=torch.bfloat16):
    total_size = sum(mask.size(0) for mask in masks)
    combined_mask = torch.zeros(total_size, total_size, dtype=dtype)

    current_pos = 0

    for mask in masks:
        size = mask.size(0)
        combined_mask[current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size

    min_value = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
    inverted_mask = torch.where(combined_mask == 1, torch.tensor(0, dtype=dtype), min_value)
    return inverted_mask.unsqueeze(0)

class Model(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, input_ids, attention_mask = None, position_ids = None, labels = None, **kwargs):
        super_out = self.model.forward(
            input_ids = input_ids, 
            position_ids = position_ids, 
            attention_mask = attention_mask, 
            output_hidden_states = True,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            auto_shift_loss = linear_cross_entropy(
                embeddings.to(torch.bfloat16), 
                self.lm_head.weight.to(torch.bfloat16), 
                labels, 
                shift=True,
            )
            return {'loss': auto_shift_loss}
        return super_out

class LinearLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = linear.in_features
        out_features = linear.out_features
        
        device = self.linear.weight.device
        dtype = self.linear.weight.dtype
        
        self.lora_A = nn.Linear(
            in_features, r, bias=False, 
            device = device,
            dtype = dtype,
        )
        self.lora_B = nn.Linear(
            r, out_features, bias=False, 
            device = device,
            dtype = dtype,
        )

        for param in self.lora_A.parameters():
            param.requires_grad = True
        for param in self.lora_B.parameters():
            param.requires_grad = True

        init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        init.zeros_(self.lora_B.weight)

    def forward(self, x):
        out = self.linear(x)
        lora_update = self.lora_B(self.lora_A(x)) * self.scaling
        return out + lora_update

def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34.",
            FutureWarning)
        if model_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level
        # at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    min_dtype = torch.finfo(torch_dtype).min
    sequence_length = data_args.block_size

    class UInt32(Encoding):
        def encode(self, obj) -> bytes:
            return obj.tobytes()

        def decode(self, data: bytes):
            return np.frombuffer(data, np.uint32)

    _encodings['uint32'] = UInt32

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, local):
            self.dataset = LocalDataset(local=local)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            data.pop('audio', None)
            data.pop('text', None)
            data.pop('token_type_ids', None)

            for k in data.keys():
                data[k] = data[k].astype(np.int64)

            data['input_ids'] = np.pad(data['input_ids'], 
                (0, sequence_length - data['input_ids'].shape[0]), 
                'constant', constant_values=(0, tokenizer.pad_token_id))
            data['position_ids'] = np.pad(data['position_ids'], 
                (0, sequence_length - data['position_ids'].shape[0]), 
                'constant', constant_values=(0, 32000))
            data['labels'] = data["input_ids"].copy()
            data['labels'][data['labels'] == tokenizer.pad_token_id] = -100

            masking = data.pop('attention_mask')
            masks = []
            for m in masking:
                masks.append(torch.tril(torch.ones(m, m)))
            attention_mask = block_diagonal_concat_inverted(*masks)
            data['attention_mask'] = pad_attention_mask_3d(
                attention_mask, sequence_length, min_dtype)
        
            return data

        def __len__(self):
            return len(self.dataset)

    dataset = DatasetFixed(data_args.train_file)
    print('dataset', len(dataset), dataset[0]['attention_mask'].shape)

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation = 'sdpa',
        torch_dtype = torch_dtype,
    )
    for name, param in model.named_parameters():
        param.requires_grad = False

    if model_args.rank > 0:
        selected = [
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj",
            "gate_proj", 
            "up_proj", 
            "down_proj",
        ]
        r = model_args.rank
        alpha = r * 2
        for name, module in model.named_modules():
            if 'encoder' in name:
                continue
            for child_name, child in module.named_children():
                if len(child_name) and any([a in child_name for a in selected]) and isinstance(child, nn.Linear):
                    lora = LinearLoRA(child, r=r, alpha=alpha)
                    setattr(module, child_name, lora)

    model.model.embed_tokens.weight.requires_grad = True
    model.lm_head.weight.requires_grad = True
    print(model)
    
    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
