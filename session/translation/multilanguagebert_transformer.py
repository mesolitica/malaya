from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
from bert import modeling

BERT_CONFIG = 'multi_cased_L-12_H-768_A-12/bert_config.json'
bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)

from collections import defaultdict

BASE_PARAMS = defaultdict(
    lambda: None,  # Set default value to None.
    # Input params
    default_batch_size = 2048,  # Maximum number of tokens per batch of examples.
    default_batch_size_tpu = 32768,
    max_length = 256,  # Maximum number of tokens per example.
    # Model params
    initializer_gain = 1.0,  # Used in trainable variable initialization.
    vocab_size = 32000,  # Number of tokens defined in the vocabulary file.
    hidden_size = 768,  # Model dimension in the hidden layers.
    num_hidden_layers = 6,  # Number of layers in the encoder and decoder stacks.
    num_heads = 8,  # Number of heads to use in multi-headed attention.
    filter_size = 2048,  # Inner layer dimension in the feedforward network.
    # Dropout values (only used when training)
    layer_postprocess_dropout = 0.1,
    attention_dropout = 0.1,
    relu_dropout = 0.1,
    # Training params
    label_smoothing = 0.1,
    learning_rate = 1.0,
    learning_rate_decay_rate = 1.0,
    learning_rate_warmup_steps = 16000,
    # Optimizer params
    optimizer_adam_beta1 = 0.9,
    optimizer_adam_beta2 = 0.997,
    optimizer_adam_epsilon = 1e-09,
    # Default prediction params
    extra_decode_length = 50,
    beam_size = 4,
    alpha = 0.6,  # used to calculate length normalization in beam search
    # TPU specific parameters
    use_tpu = False,
    static_batch = False,
    allow_ffn_pad = True,
)

from transformer import model_utils
from transformer import utils
from transformer.transformer import DecoderStack
from transformer import beam_search
from tensor2tensor.utils import bleu_hook


class Model:
    def __init__(
        self,
        input_ids,
        input_mask,
        token_type_ids,
        Y,
        learning_rate = 2e-5,
        is_training = True,
    ):
        self.X = input_ids
        self.segment_ids = token_type_ids
        self.input_masks = input_mask
        self.Y = Y
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype = tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype = tf.int32)
        batch_size = tf.shape(self.X)[0]

        model = modeling.BertModel(
            config = bert_config,
            is_training = is_training,
            input_ids = self.X,
            input_mask = self.input_masks,
            token_type_ids = self.segment_ids,
            use_one_hot_embeddings = False,
        )

        self.decoder_stack = DecoderStack(BASE_PARAMS, is_training)
        attention_bias = model_utils.get_padding_bias(self.X)

        output_layer = model.get_sequence_output()
        pooled_output = model.get_pooled_output()
        embedding = model.get_embedding_table()

        with tf.name_scope('decode'):
            mask = tf.to_float(tf.not_equal(self.Y, 0))
            decoder_inputs = tf.gather(embedding, self.Y)
            decoder_inputs *= tf.expand_dims(mask, -1)
            with tf.name_scope('shift_targets'):
                decoder_inputs = tf.pad(
                    decoder_inputs, [[0, 0], [1, 0], [0, 0]]
                )[:, :-1, :]
            with tf.name_scope('add_pos_encoding'):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += model_utils.get_position_encoding(
                    length, BASE_PARAMS['hidden_size']
                )
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, 1 - BASE_PARAMS['layer_postprocess_dropout']
                )
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                length
            )
            outputs = self.decoder_stack(
                decoder_inputs,
                output_layer,
                decoder_self_attention_bias,
                attention_bias,
            )

        with tf.variable_scope('cls/predictions'):
            with tf.variable_scope('transform'):
                input_tensor = tf.layers.dense(
                    outputs,
                    units = bert_config.hidden_size,
                    activation = modeling.get_activation(
                        bert_config.hidden_act
                    ),
                    kernel_initializer = modeling.create_initializer(
                        bert_config.initializer_range
                    ),
                )
            input_tensor = modeling.layer_norm(input_tensor)

            output_bias = tf.get_variable(
                'output_bias',
                shape = [bert_config.vocab_size],
                initializer = tf.zeros_initializer(),
            )
            self.training_logits = tf.matmul(
                input_tensor, embedding, transpose_b = True
            )

    masks = tf.sequence_mask(
        self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype = tf.float32
    )
    self.cost = tf.contrib.seq2seq.sequence_loss(
        logits = self.training_logits, targets = self.Y, weights = masks
    )
    self.bleu, _ = bleu_hook.bleu_score(self.training_logits, self.Y)
