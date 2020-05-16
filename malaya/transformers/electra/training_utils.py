# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Utilities for training the models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import re
import time
import tensorflow.compat.v1 as tf

from . import modeling


def get_bert_config(config):
    """Get model hyperparameters based on a pretraining/finetuning config"""
    if config.model_size == 'large':
        args = {'hidden_size': 1024, 'num_hidden_layers': 24}
    elif config.model_size == 'base':
        args = {'hidden_size': 768, 'num_hidden_layers': 12}
    elif config.model_size == 'small':
        args = {'hidden_size': 256, 'num_hidden_layers': 12}
    else:
        raise ValueError('Unknown model size', config.model_size)
    args['vocab_size'] = config.vocab_size
    args.update(**config.model_hparam_overrides)
    # by default the ff size and num attn heads are determined by the hidden size
    args['num_attention_heads'] = max(1, args['hidden_size'] // 64)
    args['intermediate_size'] = 4 * args['hidden_size']
    args.update(**config.model_hparam_overrides)
    return modeling.BertConfig.from_dict(args)
