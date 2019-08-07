import tensorflow as tf
from ._xlnet import xlnet as xlnet_lib
from ._utils._paths import PATH_XLNET, S3_PATH_XLNET
from .texts._text_functions import (
    xlnet_tokenization,
    padding_sequence,
    merge_sentencepiece_tokens,
)
from ._utils._utils import check_file, check_available
import collections
import re
import os
import numpy as np


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


class _Model:
    def __init__(self, xlnet_config, tokenizer, checkpoint, pool_mode = 'last'):

        kwargs = dict(
            is_training = True,
            use_tpu = False,
            use_bfloat16 = False,
            dropout = 0.0,
            dropatt = 0.0,
            init = 'normal',
            init_range = 0.1,
            init_std = 0.05,
            clamp_len = -1,
        )

        xlnet_parameters = xlnet_lib.RunConfig(**kwargs)

        self._tokenizer = tokenizer
        _graph = tf.Graph()
        with _graph.as_default():
            self.X = tf.placeholder(tf.int32, [None, None])
            self.segment_ids = tf.placeholder(tf.int32, [None, None])
            self.input_masks = tf.placeholder(tf.float32, [None, None])

            xlnet_model = xlnet_lib.XLNetModel(
                xlnet_config = xlnet_config,
                run_config = xlnet_parameters,
                input_ids = tf.transpose(self.X, [1, 0]),
                seg_ids = tf.transpose(self.segment_ids, [1, 0]),
                input_mask = tf.transpose(self.input_masks, [1, 0]),
            )

            self.logits = xlnet_model.get_pooled_out(pool_mode, True)
            self._sess = tf.InteractiveSession()
            self._sess.run(tf.global_variables_initializer())
            tvars = tf.trainable_variables()
            assignment_map, _ = get_assignment_map_from_checkpoint(
                tvars, checkpoint
            )
            self._saver = tf.train.Saver(var_list = assignment_map)
            attentions = [
                n.name
                for n in tf.get_default_graph().as_graph_def().node
                if 'rel_attn/Softmax' in n.name
            ]
            g = tf.get_default_graph()
            self.attention_nodes = [
                g.get_tensor_by_name('%s:0' % (a)) for a in attentions
            ]

    def vectorize(self, strings):
        """
        Vectorize string inputs using bert attention.

        Parameters
        ----------
        strings : str / list of str

        Returns
        -------
        array: vectorized strings
        """

        if isinstance(strings, list):
            if not isinstance(strings[0], str):
                raise ValueError('input must be a list of strings or a string')
        else:
            if not isinstance(strings, str):
                raise ValueError('input must be a list of strings or a string')
        if isinstance(strings, str):
            strings = [strings]

        input_ids, input_masks, segment_ids, _ = xlnet_tokenization(
            self._tokenizer, strings
        )
        return self._sess.run(
            self.logits,
            feed_dict = {
                self.X: input_ids,
                self.segment_ids: segment_ids,
                self.input_masks: input_masks,
            },
        )

    def attention(self, strings, method = 'last', **kwargs):
        """
        Get attention string inputs from xlnet attention.

        Parameters
        ----------
        strings : str / list of str
        method : str, optional (default='last')
            Attention layer supported. Allowed values:

            * ``'last'`` - attention from last layer.
            * ``'first'`` - attention from first layer.
            * ``'mean'`` - average attentions from all layers.

        Returns
        -------
        array: attention
        """

        if isinstance(strings, list):
            if not isinstance(strings[0], str):
                raise ValueError('input must be a list of strings or a string')
        else:
            if not isinstance(strings, str):
                raise ValueError('input must be a list of strings or a string')
        if isinstance(strings, str):
            strings = [strings]

        method = method.lower()
        if method not in ['last', 'first', 'mean']:
            raise Exception(
                "method not supported, only support ['last', 'first', 'mean']"
            )

        input_ids, input_masks, segment_ids, s_tokens = xlnet_tokenization(
            self._tokenizer, strings
        )
        maxlen = max([len(s) for s in s_tokens])
        s_tokens = padding_sequence(s_tokens, maxlen, pad_int = '<cls>')
        attentions = self._sess.run(
            self.attention_nodes,
            feed_dict = {
                self.X: input_ids,
                self.segment_ids: segment_ids,
                self.input_masks: input_masks,
            },
        )

        if method == 'first':
            cls_attn = np.transpose(attentions[0][:, 0], (1, 0, 2))

        if method == 'last':
            cls_attn = np.transpose(attentions[-1][:, 0], (1, 0, 2))

        if method == 'mean':
            cls_attn = np.transpose(
                np.mean(attentions, axis = 0).mean(axis = 1), (1, 0, 2)
            )

        cls_attn = np.mean(cls_attn, axis = 1)
        total_weights = np.sum(cls_attn, axis = -1, keepdims = True)
        attn = cls_attn / total_weights
        output = []
        for i in range(attn.shape[0]):
            output.append(
                merge_sentencepiece_tokens(list(zip(s_tokens[i], attn[i])))
            )
        return output


def available_xlnet_model():
    """
    List available xlnet models.
    """
    return ['base', 'small']


def xlnet(model = 'base', pool_mode = 'last', validate = True):
    """
    Load xlnet model.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'base'`` - base xlnet-bahasa released by Malaya.
        * ``'small'`` - small xlnet-bahasa released by Malaya.
    pool_mode : str, optional (default='last')
        Model logits architecture supported. Allowed values:

        * ``'last'`` - last of the sequence.
        * ``'first'`` - first of the sequence.
        * ``'mean'`` - mean of the sequence.
        * ``'attn'`` - attention of the sequence.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    XLNET_MODEL: malaya.xlnet._Model class
    """

    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(pool_mode, str):
        raise ValueError('pool_mode must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

    model = model.lower()
    pool_mode = pool_mode.lower()
    if model not in available_xlnet_model():
        raise Exception(
            'model not supported, please check supported models from malaya.xlnet.available_xlnet_model()'
        )
    if pool_mode not in ['last', 'first', 'mean', 'attn']:
        raise Exception(
            "pool_mode not supported, only support ['last', 'first', 'mean', 'attn']"
        )

    if validate:
        check_file(PATH_XLNET[model]['model'], S3_PATH_XLNET[model])
    else:
        if not check_available(PATH_XLNET[model]['model']):
            raise Exception(
                'bert-model/%s is not available, please `validate = True`'
                % (model)
            )

    if not os.path.exists(PATH_XLNET[model]['directory']):
        import tarfile

        with tarfile.open(PATH_XLNET[model]['model']['model']) as tar:
            tar.extractall(path = PATH_XLNET[model]['path'])

    import sentencepiece as spm

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(PATH_XLNET[model]['directory'] + 'sp10m.cased.v5.model')
    xlnet_config = xlnet_lib.XLNetConfig(
        json_path = PATH_XLNET[model]['directory'] + 'config.json'
    )
    xlnet_checkpoint = PATH_XLNET[model]['directory'] + 'model.ckpt'
    model = _Model(
        xlnet_config, sp_model, xlnet_checkpoint, pool_mode = pool_mode
    )
    model._saver.restore(model._sess, xlnet_checkpoint)
    return model
