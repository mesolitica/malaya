#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task.
# Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Any, Union
import json
import torch
import torch.nn.functional as F
import math
import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from tokenizers import AddedToken
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from streaming import LocalDataset

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.23.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    M2M100Tokenizer]

def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets

def compute_bias(
    query_length, 
    key_length,
    relative_attention_bias,
    bidirectional = True, 
    num_buckets = 32, 
    max_distance = 128, 
    device=None,
):
    """Compute binned relative position bias"""
    if device is None:
        device = relative_attention_bias.weight.device
    context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = _relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=bidirectional,
        num_buckets=num_buckets,
        max_distance=max_distance,
    )
    values = relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    return values

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

def pad_attention_mask(attention_mask, maxlen = 2048):
    maxlen_right = maxlen
    maxlen_bottom = maxlen
    attention_mask = [
        F.pad(
            attention_mask[i],
            (0, maxlen_right - attention_mask[i].shape[1], 0, maxlen_bottom - attention_mask[i].shape[0])) for i in range(
            len(attention_mask))]
    return torch.stack(attention_mask)

def pad_attention_mask_4d(attention_mask, maxlen = 2048):
    maxlen_right = maxlen
    maxlen_bottom = maxlen
    attention_mask = [
        F.pad(
            attention_mask[i],
            (0, maxlen_right - attention_mask[i].shape[-2], 0, maxlen_bottom - attention_mask[i].shape[-1])) for i in range(
            len(attention_mask))]
    return torch.stack(attention_mask)

def block_diagonal_concat(*masks, dtype=torch.bfloat16):
    total_size = sum(mask.size(0) for mask in masks)
    combined_mask = torch.zeros(total_size, total_size, dtype=dtype)

    current_pos = 0

    for mask in masks:
        size = mask.size(0)
        combined_mask[current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size

    return combined_mask

def block_diagonal_concat_4d(*masks, dtype=torch.bfloat16):
    total_size = sum(mask.size(1) for mask in masks)
    combined_mask = torch.zeros(masks[0].shape[0], 
                                total_size, total_size, dtype=dtype)

    current_pos = 0

    for mask in masks:
        size = mask.size(1)
        combined_mask[:, current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size

    return combined_mask

def block_diagonal_concat_cross(*masks, dtype=torch.bfloat16):
    total_rows = sum(mask.size(0) for mask in masks)
    total_cols = sum(mask.size(1) for mask in masks)
    
    combined_mask = torch.zeros((total_rows, total_cols), dtype=dtype)
    
    current_row, current_col = 0, 0

    for mask in masks:
        rows, cols = mask.size()
        combined_mask[current_row:current_row + rows, current_col:current_col + cols] = mask
        current_row += rows
        current_col += cols
        
    return combined_mask

def collator(batch, pad_token_id = 1, label_pad = -100, maxlen = 2048):
    max_length = maxlen
    results = {}
    results['input_ids'] = [
        b['input_ids'] + [pad_token_id] * (max_length - len(b['input_ids']))
        for b in batch
    ]
    results['input_ids'] = torch.tensor(results['input_ids'], dtype = torch.int64)
    
    max_length = maxlen
    results['labels'] = [
        b['labels'] + [label_pad] * (max_length - len(b['labels']))
        for b in batch
    ]
    results['labels'] = torch.tensor(results['labels'], dtype = torch.int64)
    
    results['position_bias'] = pad_attention_mask_4d([b['position_bias'] for b in batch])
    results['decoder_position_bias'] = pad_attention_mask_4d([b['decoder_position_bias'] for b in batch])
    
    attention_mask = [b['attention_mask'] for b in batch]
    results['attention_mask'] = pad_attention_mask(attention_mask)
    encoder_attention_mask = [b['encoder_attention_mask'] for b in batch]
    results['encoder_attention_mask'] = pad_attention_mask(encoder_attention_mask)
    decoder_attention_mask = [b['decoder_attention_mask'] for b in batch]
    results['decoder_attention_mask'] = pad_attention_mask(decoder_attention_mask)
    
    dtype = results['attention_mask'].dtype
    encoder_extended_attention_mask = results['attention_mask'][:, None, :, :]
    encoder_extended_attention_mask = encoder_extended_attention_mask
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(dtype).min
    results['position_bias'] = results['position_bias'] + encoder_extended_attention_mask
    
    dtype = results['decoder_attention_mask'].dtype
    encoder_extended_attention_mask = results['decoder_attention_mask'][:, None, :, :]
    encoder_extended_attention_mask = encoder_extended_attention_mask
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(dtype).min
    results['decoder_position_bias'] = results['decoder_position_bias'] + encoder_extended_attention_mask
    
    return results

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}, )
    model_revision: str = field(
        default="main", metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."}, )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models).")},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={
            "help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."})
    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacrebleu) on a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacrebleu) on a jsonlines file."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128, metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded.")}, )
    val_max_target_length: Optional[int] = field(
        default=None, metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``.")}, )
    pad_to_max_length: bool = field(
        default=False, metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU.")}, )
    max_train_samples: Optional[int] = field(
        default=None, metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set.")}, )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set.")}, )
    max_predict_samples: Optional[int] = field(
        default=None, metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set.")}, )
    num_beams: Optional[int] = field(
        default=None, metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``.")}, )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."})
    forced_bos_token: Optional[str] = field(
        default=None, metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)")}, )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(training_args)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_translation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        attn_implementation = 'sdpa',
    )
    print(model)
    model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight._requires_grad = False
    model.decoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight._requires_grad = False
    model.config.use_cache = False
    encoder_emb = torch.nn.Embedding(
        model.config.relative_attention_num_buckets, 
        model.config.num_heads
    )

    with torch.no_grad():
        encoder_emb.weight.copy_(model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight.detach())

    encoder_emb.weight.requires_grad = False

    decoder_emb = torch.nn.Embedding(
        model.config.relative_attention_num_buckets, 
        model.config.num_heads
    )

    with torch.no_grad():
        decoder_emb.weight.copy_(model.decoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight.detach())

    decoder_emb.weight.requires_grad = False

    def multipack(input_ids, labels, lengths):
        results = {
            'input_ids': input_ids,
            'labels': labels,
        }
        attention_mask = []
        encoder_attention_mask = []
        decoder_attention_mask = []
        encoder_biases = []
        decoder_biases = []
        
        for length in lengths:
            left_len = length[0]
            right_len = length[1]
            
            attention_mask.append(torch.ones(left_len, left_len))
            encoder_attention_mask.append(torch.ones(right_len, left_len))
            decoder_attention_mask.append(torch.tril(torch.ones(right_len, right_len)))
            
            encoder_bias = compute_bias(
                left_len, left_len,
                encoder_emb,
                bidirectional=True,
                num_buckets=model.config.relative_attention_num_buckets,
                max_distance=model.config.relative_attention_max_distance,
            )
            encoder_biases.append(encoder_bias[0])
            
            decoder_bias = compute_bias(
                right_len, right_len,
                decoder_emb,
                bidirectional=False,
                num_buckets=model.config.relative_attention_num_buckets,
                max_distance=model.config.relative_attention_max_distance,
            )
            decoder_biases.append(decoder_bias[0])
            
        results['attention_mask'] = block_diagonal_concat(*attention_mask)
        results['encoder_attention_mask'] = block_diagonal_concat_cross(*encoder_attention_mask)
        results['decoder_attention_mask'] = block_diagonal_concat(*decoder_attention_mask)
        
        results['position_bias'] = block_diagonal_concat_4d(*encoder_biases)
        results['decoder_position_bias'] = block_diagonal_concat_4d(*decoder_biases)
        
        return results

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, remote):

            self.dataset = LocalDataset(local=remote)

        def __getitem__(self, idx):
            row = self.dataset[idx]
            d = json.loads(row['data'])
            return multipack(**d)

        def __len__(self):
            return len(self.dataset)

    train_dataset = DatasetFixed(data_args.train_file)

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
