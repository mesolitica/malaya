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

BERT_CONFIG = '../multi_cased_L-12_H-768_A-12/bert_config.json'
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
    vocab_size = 119547,  # Number of tokens defined in the vocabulary file.
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

EOS = 1


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
            if is_training:
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

        print(self.training_logits)

    def get_sequence_output(self):
        return self.training_logits

        # masks = tf.sequence_mask(
        #     self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype = tf.float32
        # )
        # self.cost = tf.contrib.seq2seq.sequence_loss(
        #     logits = self.training_logits, targets = self.Y, weights = masks
        # )
        # # self.bleu, _ = bleu_hook.bleu_score(self.training_logits, self.Y)
        # y_t = tf.argmax(self.training_logits, axis = 2)
        # y_t = tf.cast(y_t, tf.int32)
        # self.prediction = tf.boolean_mask(y_t, masks)
        # mask_label = tf.boolean_mask(self.Y, masks)
        # correct_pred = tf.equal(self.prediction, mask_label)
        # correct_index = tf.cast(correct_pred, tf.float32)
        # self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # def _get_symbols_to_logits_fn(max_decode_length):
        #     timing_signal = model_utils.get_position_encoding(
        #         max_decode_length + 1, BASE_PARAMS['hidden_size']
        #     )
        #     decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
        #         max_decode_length
        #     )

        #     def symbols_to_logits_fn(ids, i, cache):
        #         decoder_input = ids[:, -1:]
        #         mask = tf.to_float(tf.not_equal(decoder_input, 0))
        #         decoder_input = tf.gather(embedding, decoder_input)
        #         decoder_input *= tf.expand_dims(mask, -1)
        #         decoder_input += timing_signal[i : i + 1]
        #         self_attention_bias = decoder_self_attention_bias[
        #             :, :, i : i + 1, : i + 1
        #         ]
        #         decoder_outputs = self.decoder_stack(
        #             decoder_input,
        #             cache.get('encoder_outputs'),
        #             self_attention_bias,
        #             cache.get('encoder_decoder_attention_bias'),
        #             cache,
        #         )

        #         with tf.variable_scope('cls/predictions', reuse = True):
        #             with tf.variable_scope('transform'):
        #                 input_tensor = tf.layers.dense(
        #                     decoder_outputs,
        #                     units = bert_config.hidden_size,
        #                     activation = modeling.get_activation(
        #                         bert_config.hidden_act
        #                     ),
        #                     kernel_initializer = modeling.create_initializer(
        #                         bert_config.initializer_range
        #                     ),
        #                 )
        #             input_tensor = modeling.layer_norm(input_tensor)

        #             output_bias = tf.get_variable(
        #                 'output_bias',
        #                 shape = [bert_config.vocab_size],
        #                 initializer = tf.zeros_initializer(),
        #             )
        #             logits = tf.matmul(
        #                 input_tensor, embedding, transpose_b = True
        #             )
        #         logits = tf.squeeze(logits, axis = [1])
        #         return logits, cache

        #     return symbols_to_logits_fn

        # batch_size = tf.shape(output_layer)[0]
        # input_length = tf.shape(output_layer)[1]
        # max_decode_length = input_length + BASE_PARAMS['extra_decode_length']
        # symbols_to_logits_fn = _get_symbols_to_logits_fn(max_decode_length)
        # initial_ids = tf.zeros([batch_size], dtype = tf.int32)
        # cache = {
        #     'layer_%d'
        #     % layer: {
        #         'k': tf.zeros([batch_size, 0, BASE_PARAMS['hidden_size']]),
        #         'v': tf.zeros([batch_size, 0, BASE_PARAMS['hidden_size']]),
        #     }
        #     for layer in range(BASE_PARAMS['num_hidden_layers'])
        # }
        # cache['encoder_outputs'] = output_layer
        # cache['encoder_decoder_attention_bias'] = attention_bias

        # decoded_ids, scores = beam_search.sequence_beam_search(
        #     symbols_to_logits_fn = symbols_to_logits_fn,
        #     initial_ids = initial_ids,
        #     initial_cache = cache,
        #     vocab_size = bert_config.vocab_size,
        #     beam_size = 1,
        #     alpha = BASE_PARAMS['alpha'],
        #     max_decode_length = max_decode_length,
        #     eos_id = EOS,
        # )

        # top_decoded_ids = decoded_ids[:, 0, 1:]
        # self.fast_result = top_decoded_ids

        # decoded_ids, scores = beam_search.sequence_beam_search(
        #     symbols_to_logits_fn = symbols_to_logits_fn,
        #     initial_ids = initial_ids,
        #     initial_cache = cache,
        #     vocab_size = bert_config.vocab_size,
        #     beam_size = 5,
        #     alpha = BASE_PARAMS['alpha'],
        #     max_decode_length = max_decode_length,
        #     eos_id = EOS,
        # )

        # top_decoded_ids = decoded_ids[:, 0, 1:]
        # self.beam_result = top_decoded_ids
