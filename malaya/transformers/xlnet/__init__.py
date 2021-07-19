# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

import tensorflow.compat.v1 as tf
from malaya.function import get_device, generate_session
from malaya.transformers.xlnet import xlnet as xlnet_lib
from malaya.text.bpe import (
    xlnet_tokenization,
    padding_sequence,
    merge_sentencepiece_tokens,
    SentencePieceTokenizer,
)
from malaya.path import PATH_XLNET, S3_PATH_XLNET
from malaya.function import check_file
from collections import defaultdict
import collections
import re
import os
import numpy as np
from herpetologist import check_type
from typing import List


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match('^(.*):\\d+$', name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ':0'] = 1

    return (assignment_map, initialized_variable_names)


def _extract_attention_weights_import(tf_graph):
    attentions = [
        n.name
        for n in tf_graph.as_graph_def().node
        if 'rel_attn/Softmax' in n.name
    ]

    return [tf_graph.get_tensor_by_name('%s:0' % (a)) for a in attentions]


class Model:
    def __init__(
        self, xlnet_config, tokenizer, checkpoint, pool_mode='last', **kwargs
    ):

        kwargs_config = dict(
            is_training=True,
            use_tpu=False,
            use_bfloat16=False,
            dropout=0.0,
            dropatt=0.0,
            init='normal',
            init_range=0.1,
            init_std=0.05,
            clamp_len=-1,
        )

        xlnet_parameters = xlnet_lib.RunConfig(**kwargs_config)

        self._tokenizer = tokenizer
        device = get_device(**kwargs)
        _graph = tf.Graph()
        with _graph.as_default():
            with tf.device(device):
                self.X = tf.placeholder(tf.int32, [None, None])
                self.segment_ids = tf.placeholder(tf.int32, [None, None])
                self.input_masks = tf.placeholder(tf.float32, [None, None])

                xlnet_model = xlnet_lib.XLNetModel(
                    xlnet_config=xlnet_config,
                    run_config=xlnet_parameters,
                    input_ids=tf.transpose(self.X, [1, 0]),
                    seg_ids=tf.transpose(self.segment_ids, [1, 0]),
                    input_mask=tf.transpose(self.input_masks, [1, 0]),
                )

                self.logits = xlnet_model.get_pooled_out(pool_mode, True)
                self._sess = generate_session(_graph, **kwargs)
                self._sess.run(tf.global_variables_initializer())
                tvars = tf.trainable_variables()
                assignment_map, _ = get_assignment_map_from_checkpoint(
                    tvars, checkpoint
                )
                self._saver = tf.train.Saver(var_list=assignment_map)
                attentions = [
                    n.name
                    for n in tf.get_default_graph().as_graph_def().node
                    if 'rel_attn/Softmax' in n.name
                ]
                g = tf.get_default_graph()
                self.attention_nodes = [
                    g.get_tensor_by_name('%s:0' % (a)) for a in attentions
                ]

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

        input_ids, input_masks, segment_ids, _ = xlnet_tokenization(
            self._tokenizer, strings
        )
        return self._sess.run(
            self.logits,
            feed_dict={
                self.X: input_ids,
                self.segment_ids: segment_ids,
                self.input_masks: input_masks,
            },
        )

    def _attention(self, strings):
        input_ids, input_masks, segment_ids, s_tokens = xlnet_tokenization(
            self._tokenizer, strings
        )
        maxlen = max([len(s) for s in s_tokens])
        s_tokens = padding_sequence(s_tokens, maxlen, pad_int='<cls>')
        attentions = self._sess.run(
            self.attention_nodes,
            feed_dict={
                self.X: input_ids,
                self.segment_ids: segment_ids,
                self.input_masks: input_masks,
            },
        )
        return attentions, s_tokens, input_masks

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
                "method not supported, only support ['last', 'first', 'mean']"
            )
        attentions, s_tokens, _ = self._attention(strings)

        if method == 'first':
            cls_attn = np.transpose(attentions[0][:, 0], (1, 0, 2))

        if method == 'last':
            cls_attn = np.transpose(attentions[-1][:, 0], (1, 0, 2))

        if method == 'mean':
            cls_attn = np.transpose(
                np.mean(attentions, axis=0).mean(axis=1), (1, 0, 2)
            )

        cls_attn = np.mean(cls_attn, axis=1)
        total_weights = np.sum(cls_attn, axis=-1, keepdims=True)
        attn = cls_attn / total_weights
        output = []
        for i in range(attn.shape[0]):
            output.append(
                merge_sentencepiece_tokens(
                    list(zip(s_tokens[i], attn[i])), model='xlnet'
                )
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
            attn = attn_data[:, :, 0]
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
def load(model: str = 'xlnet', pool_mode: str = 'last', **kwargs):
    """
    Load xlnet model.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'xlnet'`` - XLNET architecture from google.
    pool_mode : str, optional (default='last')
        Model logits architecture supported. Allowed values:

        * ``'last'`` - last of the sequence.
        * ``'first'`` - first of the sequence.
        * ``'mean'`` - mean of the sequence.
        * ``'attn'`` - attention of the sequence.

    Returns
    -------
    result : malaya.transformers.xlnet.Model class
    """

    model = model.lower()
    pool_mode = pool_mode.lower()

    if pool_mode not in ['last', 'first', 'mean', 'attn']:
        raise Exception(
            "pool_mode not supported, only support ['last', 'first', 'mean', 'attn']"
        )

    check_file(PATH_XLNET[model]['model'], S3_PATH_XLNET[model], **kwargs)

    if not os.path.exists(PATH_XLNET[model]['directory'] + 'model.ckpt'):
        import tarfile

        with tarfile.open(PATH_XLNET[model]['model']['model']) as tar:
            tar.extractall(path=PATH_XLNET[model]['path'])

    vocab_model = PATH_XLNET[model]['directory'] + 'sp10m.cased.v9.model'
    vocab = PATH_XLNET[model]['directory'] + 'sp10m.cased.v9.vocab'
    tokenizer = SentencePieceTokenizer(vocab_file=vocab, spm_model_file=vocab_model)
    xlnet_config = xlnet_lib.XLNetConfig(
        json_path=PATH_XLNET[model]['directory'] + 'config.json'
    )
    xlnet_checkpoint = PATH_XLNET[model]['directory'] + 'model.ckpt'
    model = Model(
        xlnet_config,
        tokenizer,
        xlnet_checkpoint,
        pool_mode=pool_mode,
        **kwargs
    )
    model._saver.restore(model._sess, xlnet_checkpoint)
    return model
