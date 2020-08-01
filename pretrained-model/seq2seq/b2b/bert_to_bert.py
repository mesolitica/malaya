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
import bert_decoder as modeling_decoder

GO = 1
EOS = 1


class Model:
    def __init__(
        self,
        bert_config,
        input_ids,
        input_mask,
        token_type_ids,
        Y,
        learning_rate = 2e-5,
        training = True,
    ):
        self.X = input_ids
        self.segment_ids = token_type_ids
        self.input_masks = input_mask
        self.Y = Y
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype = tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype = tf.int32)
        batch_size = tf.shape(self.X)[0]

        def forward(x, segment, masks, y, reuse = False, config = bert_config):
            with tf.variable_scope('bert', reuse = reuse):
                model = modeling.BertModel(
                    config = config,
                    is_training = training,
                    input_ids = x,
                    input_mask = masks,
                    token_type_ids = segment,
                    use_one_hot_embeddings = False,
                )
                memory = model.get_sequence_output()
            with tf.variable_scope('bert', reuse = True):
                Y_seq_len = tf.count_nonzero(y, 1, dtype = tf.int32)
                y_masks = tf.sequence_mask(
                    Y_seq_len, tf.reduce_max(Y_seq_len), dtype = tf.float32
                )

                model = modeling_decoder.BertModel(
                    config = config,
                    is_training = training,
                    input_ids = y,
                    input_mask = y_masks,
                    memory = memory,
                    memory_mask = masks,
                    use_one_hot_embeddings = False,
                )
                output_layer = model.get_sequence_output()
                embedding = model.get_embedding_table()

            with tf.variable_scope('cls/predictions', reuse = reuse):
                with tf.variable_scope('transform'):
                    input_tensor = tf.layers.dense(
                        output_layer,
                        units = config.hidden_size,
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
                logits = tf.matmul(input_tensor, embedding, transpose_b = True)
                return logits

        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)

        self.training_logits = forward(
            self.X, self.segment_ids, self.input_masks, decoder_input
        )
