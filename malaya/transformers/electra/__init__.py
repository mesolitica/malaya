# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

import tensorflow.compat.v1 as tf
from malaya.function import get_device, generate_session
from malaya.transformers.electra import modeling, tokenization, training_utils
from malaya.transformers.sampling import top_k_logits, top_p_logits
from malaya.text.bpe import (
    bert_tokenization,
    padding_sequence,
    merge_wordpiece_tokens,
)
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
                f'generator/encoder/layer_{i}/attention/self/Softmax:0'
            )
        }
        for i in range(num_layers)
    ]

    return attns


def _extract_attention_weights_import(num_layers, tf_graph):
    attns = [
        {
            f'layer_{i}': tf_graph.get_tensor_by_name(
                f'import/generator/encoder/layer_{i}/attention/self/Softmax:0'
            )
        }
        for i in range(num_layers)
    ]

    return attns


class Model:
    def __init__(self, bert_config, tokenizer, **kwargs):
        device = get_device(**kwargs)
        _graph = tf.Graph()
        with _graph.as_default():
            with tf.device(device):
                self.X = tf.placeholder(tf.int32, [None, None])
                self.segment_ids = tf.placeholder(tf.int32, [None, None])
                self.top_p = tf.placeholder(tf.float32, None)
                self.top_k = tf.placeholder(tf.int32, None)
                self.k = tf.placeholder(tf.int32, None)
                self.temperature = tf.placeholder(tf.float32, None)
                self.indices = tf.placeholder(tf.int32, [None, None])
                self.MASK = tf.placeholder(tf.int32, [None, None])
                self._tokenizer = tokenizer

                self.model = modeling.BertModel(
                    bert_config=bert_config,
                    is_training=False,
                    input_ids=self.X,
                    input_mask=self.MASK,
                    token_type_ids=self.segment_ids,
                    use_one_hot_embeddings=False,
                    scope='generator',
                    embedding_size=bert_config.embedding_size,
                )
                self.logits = self.model.get_pooled_output()
                output_layer = self.model.get_sequence_output()

                with tf.variable_scope('generator_predictions'):
                    hidden = tf.layers.dense(
                        output_layer,
                        units=modeling.get_shape_list(
                            self.model.get_embedding_table()
                        )[-1],
                        activation=modeling.get_activation(
                            bert_config.hidden_act
                        ),
                        kernel_initializer=modeling.create_initializer(
                            bert_config.initializer_range
                        ),
                    )
                    hidden = modeling.layer_norm(hidden)
                    output_bias = tf.get_variable(
                        'output_bias',
                        shape=[bert_config.vocab_size],
                        initializer=tf.zeros_initializer(),
                    )
                    logits = tf.matmul(
                        hidden,
                        self.model.get_embedding_table(),
                        transpose_b=True,
                    )
                    self._logits = tf.nn.bias_add(logits, output_bias)
                    self._log_softmax = tf.nn.log_softmax(self._logits)

                logits = tf.gather_nd(self._logits, self.indices)
                logits = logits / self.temperature

                def necleus():
                    return top_p_logits(logits, self.top_p)

                def select_k():
                    return top_k_logits(logits, self.top_k)

                logits = tf.cond(self.top_p > 0, necleus, select_k)
                self.samples = tf.multinomial(
                    logits, num_samples=self.k, output_dtype=tf.int32
                )

                self._sess = generate_session(_graph, **kwargs)
                self._sess.run(tf.global_variables_initializer())

                var_lists = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'
                )
                var_electra = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='electra'
                )

                self._saver = tf.train.Saver(var_list=var_lists + var_electra)
                attns = _extract_attention_weights(
                    bert_config.num_hidden_layers, tf.get_default_graph()
                )
                self.attns = attns

    def _log_vectorize(self, s_tokens, s_masks):
        """
        Log vectorize ids, suitable for spelling correction or any minimizing log probability.

        Parameters
        ----------
        s_tokens : list of tokenized word after sentencepiece.
        s_masks : list of mask tokens.

        Returns
        -------
        result: np.array
        """
        s_tokens = np.array(s_tokens)
        segments = np.zeros(s_tokens.shape)

        return self._sess.run(
            self._log_softmax,
            feed_dict={
                self.X: s_tokens,
                self.MASK: s_masks,
                self.segment_ids: segments,
            },
        )

    @check_type
    def vectorize(self, strings: List[str]):
        """
        Vectorize string inputs.

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: np.array
        """

        batch_x, batch_masks, batch_segments, _ = bert_tokenization(
            self._tokenizer, strings
        )
        return self._sess.run(
            self.logits,
            feed_dict={
                self.X: batch_x,
                self.MASK: batch_masks,
                self.segment_ids: batch_segments,
            },
        )

    def _attention(self, strings):
        batch_x, batch_masks, batch_segments, s_tokens = bert_tokenization(
            self._tokenizer, strings
        )
        maxlen = max([len(s) for s in s_tokens])
        s_tokens = padding_sequence(s_tokens, maxlen, pad_int='[SEP]')
        attentions = self._sess.run(
            self.attns,
            feed_dict={
                self.X: batch_x,
                self.MASK: batch_masks,
                self.segment_ids: batch_segments,
            },
        )
        return attentions, s_tokens, batch_masks

    @check_type
    def attention(self, strings: List[str], method: str = 'last', **kwargs):
        """
        Get attention string inputs.

        Parameters
        ----------
        strings : List[str]
        method : str, optional (default='last')
            Attention layer supported. Allowed values:

            * ``'last'`` - attention from last layer.
            * ``'first'`` - attention from first layer.
            * ``'mean'`` - average attentions from all layers.

        Returns
        -------
        result : List[List[Tuple[str, float]]]
        """

        method = method.lower()
        if method not in ['last', 'first', 'mean']:
            raise Exception(
                "method not supported, only support 'last', 'first' and 'mean'"
            )
        attentions, s_tokens, _ = self._attention(strings)

        if method == 'first':
            cls_attn = list(attentions[0].values())[0][:, :, 0, :]

        if method == 'last':
            cls_attn = list(attentions[-1].values())[0][:, :, 0, :]

        if method == 'mean':
            combined_attentions = []
            for a in attentions:
                combined_attentions.append(list(a.values())[0])
            cls_attn = np.mean(combined_attentions, axis=0).mean(axis=2)

        cls_attn = np.mean(cls_attn, axis=1)
        total_weights = np.sum(cls_attn, axis=-1, keepdims=True)
        attn = cls_attn / total_weights
        output = []
        for i in range(attn.shape[0]):
            output.append(
                merge_wordpiece_tokens(list(zip(s_tokens[i], attn[i])))
            )
        return output

    @check_type
    def visualize_attention(self, string: str):
        """
        Visualize attention.

        Parameters
        ----------
        string : str
        """

        from malaya.function.html import _attention

        strings = [string]
        attentions, s_tokens, _ = self._attention(strings)
        attn_dict = defaultdict(list)
        for layer, attn_data in enumerate(attentions):
            attn = list(attn_data.values())[0][0]
            attn_dict['all'].append(attn.tolist())

        results = {
            'all': {
                'attn': attn_dict['all'],
                'left_text': s_tokens[0],
                'right_text': s_tokens[0],
            }
        }
        _attention(results)


@check_type
def load(model: str = 'electra', **kwargs):
    """
    Load electra model.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'electra'`` - base electra-bahasa released by Malaya.
        * ``'small-electra'`` - small electra-bahasa released by Malaya.

    Returns
    -------
    result : malaya.transformers.electra.Model class
    """

    from malaya.path import PATH_ELECTRA, S3_PATH_ELECTRA
    from malaya.function import check_file

    model = model.lower()
    check_file(PATH_ELECTRA[model]['model'], S3_PATH_ELECTRA[model], **kwargs)

    if not os.path.exists(PATH_ELECTRA[model]['directory'] + 'model.ckpt'):
        import tarfile

        with tarfile.open(PATH_ELECTRA[model]['model']['model']) as tar:
            tar.extractall(path=PATH_ELECTRA[model]['path'])

    vocab = PATH_ELECTRA[model]['directory'] + 'bahasa.wordpiece'
    bert_checkpoint = PATH_ELECTRA[model]['directory'] + 'model.ckpt'
    bert_config = PATH_ELECTRA[model]['directory'] + 'config.json'

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab, do_lower_case=False, **kwargs
    )

    bert_config = modeling.BertConfig.from_json_file(bert_config)
    model = Model(bert_config, tokenizer, **kwargs)
    model._saver.restore(model._sess, bert_checkpoint)
    return model
