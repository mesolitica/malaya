import tensorflow as tf
from bert import modeling
from ..texts._text_functions import (
    bert_tokenization,
    padding_sequence,
    merge_sentencepiece_tokens,
    merge_wordpiece_tokens,
)
from ._sampling import top_k_logits, top_p_logits
from collections import defaultdict
import numpy as np
import os
from herpetologist import check_type

bert_num_layers = {'multilanguage': 12, 'base': 12, 'small': 6}


def _extract_attention_weights(num_layers, tf_graph):
    attns = [
        {
            'layer_%s'
            % i: tf_graph.get_tensor_by_name(
                'bert/encoder/layer_%s/attention/self/Softmax:0' % i
            )
        }
        for i in range(num_layers)
    ]

    return attns


def _extract_attention_weights_import(num_layers, tf_graph):
    attns = [
        {
            'layer_%s'
            % i: tf_graph.get_tensor_by_name(
                'import/bert/encoder/layer_%s/attention/self/Softmax:0' % i
            )
        }
        for i in range(num_layers)
    ]

    return attns


class _Model:
    def __init__(self, bert_config, tokenizer, cls, sep):
        _graph = tf.Graph()
        with _graph.as_default():
            self.X = tf.placeholder(tf.int32, [None, None])
            self.top_p = tf.placeholder(tf.float32, None)
            self.top_k = tf.placeholder(tf.int32, None)
            self.k = tf.placeholder(tf.int32, None)
            self.temperature = tf.placeholder(tf.float32, None)
            self.indices = tf.placeholder(tf.int32, [None, None])
            self._tokenizer = tokenizer
            self._cls = cls
            self._sep = sep

            self.model = modeling.BertModel(
                config = bert_config,
                is_training = False,
                input_ids = self.X,
                use_one_hot_embeddings = False,
            )
            self.logits = self.model.get_pooled_output()
            output_layer = self.model.get_sequence_output()
            embedding = self.model.get_embedding_table()

            with tf.variable_scope('cls/predictions'):
                with tf.variable_scope('transform'):
                    input_tensor = tf.layers.dense(
                        output_layer,
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
                logits = tf.matmul(input_tensor, embedding, transpose_b = True)
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
                logits, num_samples = self.k, output_dtype = tf.int32
            )

            self._sess = tf.InteractiveSession()
            self._sess.run(tf.global_variables_initializer())
            var_lists = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bert'
            )
            cls = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'cls'
            )
            self._saver = tf.train.Saver(var_list = var_lists + cls)
            attns = _extract_attention_weights(
                bert_config.num_hidden_layers, tf.get_default_graph()
            )
            self.attns = attns

    def _log_vectorize(self, s_tokens):

        """
        Log vectorize ids, suitable for spelling correction or any minimizing log probability.

        Parameters
        ----------
        s_tokens : list of tokenized word after sentencepiece.

        Returns
        -------
        array: vectorized strings
        """

        return self._sess.run(self._log_softmax, feed_dict = {self.X: s_tokens})

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

        batch_x, _, _, _ = bert_tokenization(
            self._tokenizer, strings, cls = self._cls, sep = self._sep
        )
        return self._sess.run(self.logits, feed_dict = {self.X: batch_x})

    def _attention(self, strings):
        batch_x, _, _, s_tokens = bert_tokenization(
            self._tokenizer, strings, cls = self._cls, sep = self._sep
        )
        maxlen = max([len(s) for s in s_tokens])
        s_tokens = padding_sequence(s_tokens, maxlen, pad_int = self._sep)
        attentions = self._sess.run(self.attns, feed_dict = {self.X: batch_x})
        return attentions, s_tokens

    def attention(self, strings, method = 'last', **kwargs):
        """
        Get attention string inputs from bert attention.

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
                "method not supported, only support 'last', 'first' and 'mean'"
            )
        attentions, s_tokens = self._attention(strings)

        if method == 'first':
            cls_attn = list(attentions[0].values())[0][:, :, 0, :]

        if method == 'last':
            cls_attn = list(attentions[-1].values())[0][:, :, 0, :]

        if method == 'mean':
            combined_attentions = []
            for a in attentions:
                combined_attentions.append(list(a.values())[0])
            cls_attn = np.mean(combined_attentions, axis = 0).mean(axis = 2)

        cls_attn = np.mean(cls_attn, axis = 1)
        total_weights = np.sum(cls_attn, axis = -1, keepdims = True)
        attn = cls_attn / total_weights
        output = []
        for i in range(attn.shape[0]):
            if '[' in self._cls:
                output.append(
                    merge_wordpiece_tokens(list(zip(s_tokens[i], attn[i])))
                )
            else:
                output.append(
                    merge_sentencepiece_tokens(list(zip(s_tokens[i], attn[i])))
                )
        return output

    @check_type
    def visualize_attention(self, string: str):
        from .._utils._html import _attention

        strings = [string]
        attentions, s_tokens = self._attention(strings)
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


def available_bert_model():
    """
    List available bert models.
    """
    return ['multilanguage', 'base', 'small']


@check_type
def bert(model: str = 'base', validate: bool = True):
    """
    Load bert model.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'base'`` - base bert-bahasa released by Malaya.
        * ``'small'`` - small bert-bahasa released by Malaya.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    BERT_MODEL: malaya._transformer._bert._Model class
    """

    from .._utils._paths import PATH_BERT, S3_PATH_BERT
    from .._utils._utils import check_file, check_available

    model = model.lower()
    if validate:
        check_file(PATH_BERT[model]['model'], S3_PATH_BERT[model])
    else:
        if not check_available(PATH_BERT[model]['model']):
            raise Exception(
                'bert-model/%s is not available, please `validate = True`'
                % (model)
            )
    if not os.path.exists(PATH_BERT[model]['directory'] + 'model.ckpt'):
        import tarfile

        with tarfile.open(PATH_BERT[model]['model']['model']) as tar:
            tar.extractall(path = PATH_BERT[model]['path'])

    import sentencepiece as spm
    from ..texts._text_functions import SentencePieceTokenizer

    bert_checkpoint = PATH_BERT[model]['directory'] + 'model.ckpt'
    vocab_model = PATH_BERT[model]['directory'] + 'sp10m.cased.v8.model'
    vocab = PATH_BERT[model]['directory'] + 'sp10m.cased.v8.vocab'
    bert_config = PATH_BERT[model]['directory'] + 'config.json'

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(vocab_model)

    with open(vocab) as fopen:
        v = fopen.read().split('\n')[:-1]
    v = [i.split('\t') for i in v]
    v = {i[0]: i[1] for i in v}
    tokenizer = SentencePieceTokenizer(v, sp_model)
    cls = '<cls>'
    sep = '<sep>'

    bert_config = modeling.BertConfig.from_json_file(bert_config)
    model = _Model(bert_config, tokenizer, cls = cls, sep = sep)
    model._saver.restore(model._sess, bert_checkpoint)
    return model
