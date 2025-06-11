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
import transformers
import random
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2ForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    set_seed,
    WhisperPreTrainedModel,
    WhisperConfig,
    AddedToken,
    AutoProcessor,
)
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.modeling_outputs import BaseModelOutput
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from datasets import Audio
import streaming
import json
import numpy as np
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from peft import LoraConfig, get_peft_model
from cut_cross_entropy import linear_cross_entropy

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    rank: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "rank"
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
        default="bfloat16",
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

    dataset_name: Optional[str] = field(
        default=None, metadata={
            "help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."})
    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None, metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}, )
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
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
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

def pad_attention_mask_4d(attention_mask, max_size = 4096, value = 0.0):
    maxlen_right = max_size
    maxlen_bottom = max_size
    attention_mask = [
        F.pad(
            attention_mask[i],
            (0, maxlen_right - attention_mask[i].shape[-2], 0, maxlen_bottom - attention_mask[i].shape[-1]), value = value) for i in range(
            len(attention_mask))]
    return torch.stack(attention_mask)

class WhisperEncoder(WhisperPreTrainedModel):
    
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        
        self.range_max_source_positions = torch.arange(self.max_source_positions)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions(self.range_max_source_positions)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class Model(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = WhisperEncoder(config.audio_encoder_config)
        self.projection = nn.Linear(self.encoder.config.d_model, self.config.hidden_size, bias=False)
    
    def forward(
        self, 
        input_ids, 
        attention_mask,
        position_ids,
        input_features = None, 
        feature_attention_mask = None, 
        labels = None, 
        **kwargs,
    ):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if input_features is not None:
            batch_size, _, max_mel_seq_len = input_features.shape
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            audio_feat_lengths = self.encoder._get_feat_extract_output_lengths(feature_attention_mask.sum(-1))
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.encoder.conv1.weight.dtype, device=self.encoder.conv1.weight.device
            )
            audio_attention_mask[audio_attention_mask_] = float("-inf")
            audio_outputs = self.encoder(input_features, attention_mask=audio_attention_mask)
            selected_audio_feature = audio_outputs.last_hidden_state
            audio_features = self.projection(selected_audio_feature)
            num_audio_tokens = audio_feat_lengths
            num_audios, max_audio_tokens, embed_dim = audio_features.shape
            audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
                num_audio_tokens.device
            ) < num_audio_tokens.unsqueeze(1)
            masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)
            inputs_embeds[input_ids == self.config.audio_token_index] = masked_audio_features.contiguous()
        
        super_out = self.model.forward(
            inputs_embeds = inputs_embeds, 
            attention_mask = attention_mask,
            position_ids = position_ids,
            output_hidden_states = True,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            auto_shift_loss = linear_cross_entropy(
                embeddings, 
                self.lm_head.weight, 
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

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    chat_template = "{% set audio_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if 'audio' in content or 'audio_url' in content or message['type'] == 'audio' %}{% set audio_count.value = audio_count.value + 1 %}Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    tokenizer.chat_template = chat_template
    audio_token = "<|AUDIO|>"
    audio_bos_token = "<|audio_bos|>"
    audio_eos_token = "<|audio_eos|>"
    new_tokens = [AddedToken(audio_token), AddedToken(audio_bos_token), AddedToken(audio_eos_token)]
    tokenizer.add_tokens(new_tokens)
    audio_token_id = tokenizer.vocab[audio_token]
    pad_token_id = tokenizer.pad_token_id
        
    processor = AutoProcessor.from_pretrained('openai/whisper-large-v3')
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    audio_encoder_config = AutoConfig.from_pretrained('huseinzol05/whisper-large-v3-encoder')
    config.audio_encoder_config = audio_encoder_config
    config.audio_token_index = audio_token_id

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
            self.audio = Audio(sampling_rate=16000)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            try:
                data.pop('text', None)
                audio_files = data.pop('audio', '')
                data['labels'] = data["input_ids"].copy()
                masking = data.pop('attention_mask')

                data.pop('token_type_ids', None)
                for k in data.keys():
                    data[k] = torch.tensor(data[k].astype(np.int64))

                masks = []
                for m in masking:
                    masks.append(torch.tril(torch.ones(m, m)))
                attention_mask = block_diagonal_concat_inverted(*masks)
                data['attention_mask'] = attention_mask

                data['labels'][data['labels'] == audio_token_id] = -100
                data['labels'][data['labels'] == pad_token_id] = -100

                if len(audio_files):
                    files = json.loads(audio_files)
                    
                    audios = []
                    new_files = []
                    for f in files:
                        audio = self.audio.decode_example(
                        self.audio.encode_example(f))['array']
                        audios.append(audio)
                        new_files.append(f)

                    inputs_audio = processor.feature_extractor(
                        audios, return_attention_mask=True, 
                        sampling_rate=16000,
                        padding="max_length", return_tensors = 'pt')

                    input_lengths = (inputs_audio['attention_mask'].sum(-1) - 1) // 2 + 1
                    output_lengths = input_lengths
                    output_lengths = sum(output_lengths).tolist()
                    audio_tokens = data['input_ids'][data['input_ids'] == audio_token_id].shape[0]
                    if audio_tokens != output_lengths:
                        print(idx, audio_tokens, output_lengths, 'length speech tokens not match', new_files)
                        return

                    data['input_features'] = inputs_audio['input_features']
                    data['feature_attention_mask'] = inputs_audio['attention_mask']
                    
                return data

            except Exception as e:
                print('Exception', e)
                return None

        def __len__(self):
            return len(self.dataset)

    def collator(batch):
        batch = [b for b in batch if b is not None] 
        input_ids, attention_mask, position_ids, labels = [], [], [], []
        input_features, feature_attention_mask = [], []

        for b in batch:
            if 'input_features' in b:
                input_features.append(b['input_features'])
                feature_attention_mask.append(b['feature_attention_mask'])
            input_ids.append(b['input_ids'][None])
            attention_mask.append(b['attention_mask'])
            position_ids.append(b['position_ids'][None])
            labels.append(b['labels'][None])

        input_ids = {
            'input_ids': torch.concat(input_ids, 0),
            'attention_mask': pad_attention_mask_4d(attention_mask, sequence_length, min_dtype),
            'position_ids': torch.concat(position_ids, 0),
            'labels': torch.concat(labels, 0),
        }
        if len(input_features):
            input_ids['input_features'] = torch.concat(input_features, 0)
            input_ids['feature_attention_mask'] = torch.concat(feature_attention_mask, 0)

        return input_ids

    dataset = DatasetFixed(data_args.train_file)
    print('dataset', len(dataset), dataset[0])
    print(collator([dataset[0], dataset[1]]))

    model = Model.from_pretrained(
        model_args.model_name_or_path, 
        config = config,
        torch_dtype = torch.bfloat16,
    )
    model.encoder = model.encoder.from_pretrained(
        'huseinzol05/whisper-large-v3-encoder',
        torch_dtype = torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

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

    model.projection.weight.requires_grad = True
    model.model.embed_tokens.weight.requires_grad = True
    model.lm_head.weight.requires_grad = True
    
    print(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

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
