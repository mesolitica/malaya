import tensorflow as tf
import numpy as np
from ..texts._text_functions import (
    str_idx,
    entities_textcleaning,
    char_str_idx,
    generate_char_seq,
    language_detection_textcleaning,
    tag_chunk,
)
from .._utils._parse_dependency import DependencyGraph
from .._utils._utils import add_neutral as neutral
from .._utils._html import (
    _render_binary,
    _render_toxic,
    _render_emotion,
    _render_relevancy,
)


def _convert_sparse_matrix_to_sparse_tensor(X, got_limit = True, limit = 5):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    if got_limit:
        coo.data[coo.data > limit] = limit
    return (
        tf.SparseTensorValue(indices, coo.col, coo.shape),
        tf.SparseTensorValue(indices, coo.data, coo.shape),
    )


class _LANG_MODEL:
    def __init__(self):
        self.X = tf.sparse_placeholder(tf.int32)
        self.W = tf.sparse_placeholder(tf.int32)
        self.Y = tf.placeholder(tf.int32, [None])
        embeddings = tf.Variable(tf.truncated_normal([660726, 40]))
        embed = tf.nn.embedding_lookup_sparse(
            embeddings, self.X, self.W, combiner = 'mean'
        )
        self.logits = tf.layers.dense(embed, 4)


class _SPARSE_SOFTMAX_MODEL:
    def __init__(self, output_size, embedded_size, vocab_size):
        self.X = tf.sparse_placeholder(tf.int32)
        self.W = tf.sparse_placeholder(tf.int32)
        self.Y = tf.placeholder(tf.int32, [None])
        embeddings = tf.Variable(
            tf.truncated_normal([vocab_size, embedded_size])
        )
        embed = tf.nn.embedding_lookup_sparse(
            embeddings, self.X, self.W, combiner = 'mean'
        )
        self.logits = tf.layers.dense(embed, output_size)


class DEEP_LANG:
    def __init__(self, path, vectorizer, label):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._model = _LANG_MODEL()
            self._sess = tf.InteractiveSession()
            self._sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(self._sess, path + '/model.ckpt')
        self._vectorizer = vectorizer
        self._label = label
        self._softmax = tf.nn.softmax(self._model.logits)

    def predict(self, string, get_proba = False):
        """
        classify a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        dictionary: results
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')

        string = language_detection_textcleaning(string)
        transformed = self._vectorizer.transform([string])
        batch_x = _convert_sparse_matrix_to_sparse_tensor(transformed)
        probs = self._sess.run(
            self._softmax,
            feed_dict = {self._model.X: batch_x[0], self._model.W: batch_x[1]},
        )[0]
        if get_proba:
            return {self._label[no]: i for no, i in enumerate(probs)}
        else:
            return self._label[np.argmax(probs)]

    def predict_batch(self, strings, get_proba = False):
        """
        classify list of strings

        Parameters
        ----------
        strings : list

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(strings[0], str):
            raise ValueError('input must be list of strings')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')

        strings = [language_detection_textcleaning(i) for i in strings]
        transformed = self._vectorizer.transform(strings)
        batch_x = _convert_sparse_matrix_to_sparse_tensor(transformed)
        probs = self._sess.run(
            self._softmax,
            feed_dict = {self._model.X: batch_x[0], self._model.W: batch_x[1]},
        )
        dicts = []
        if get_proba:
            for i in range(probs.shape[0]):
                dicts.append(
                    {self._label[no]: k for no, k in enumerate(probs[i])}
                )
        else:
            probs = np.argmax(probs, 1)
            for prob in probs:
                dicts.append(self._label[prob])
        return dicts
