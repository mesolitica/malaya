# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

import tensorflow as tf
from electra import modeling
from malaya.text.bpe import (
    bert_tokenization,
    padding_sequence,
    merge_wordpiece_tokens,
)
from malaya.transformers.sampling import top_k_logits, top_p_logits
from collections import defaultdict
import numpy as np
import os
from herpetologist import check_type
from typing import List

bert_num_layers = {'electra': 12, 'small-electra': 6}


def _extract_attention_weights(num_layers, tf_graph):
    attns = [
        {
            f'layer_{i}': tf_graph.get_tensor_by_name(
                f'bert/encoder/layer_{i}/attention/self/Softmax:0'
            )
        }
        for i in range(num_layers)
    ]

    return attns


def _extract_attention_weights_import(num_layers, tf_graph):
    attns = [
        {
            f'layer_{i}': tf_graph.get_tensor_by_name(
                f'import/bert/encoder/layer_{i}/attention/self/Softmax:0'
            )
        }
        for i in range(num_layers)
    ]

    return attns
