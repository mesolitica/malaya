import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
from ..texts._text_functions import (
    str_idx,
    add_ngram,
    fasttext_str_idx,
    entities_textcleaning,
    char_str_idx,
    generate_char_seq,
    language_detection_textcleaning,
)
from ..stem import _classification_textcleaning_stemmer_attention


def _convert_sparse_matrix_to_sparse_tensor(X, limit = 5):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
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


class TAGGING:
    def __init__(
        self,
        X,
        X_char,
        logits,
        settings,
        sess,
        model,
        is_lower = True,
        story = None,
    ):
        self._X = X
        self._X_char = X_char
        self._logits = logits
        self._settings = settings
        self._sess = sess
        self._model = model
        self._is_lower = is_lower
        self._story = story

        self._settings['idx2word'] = {
            int(k): v for k, v in self._settings['idx2word'].items()
        }
        self._settings['idx2tag'] = {
            int(k): v for k, v in self._settings['idx2tag'].items()
        }

    def predict(self, string):
        """
        Tagging a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: tagged string
        """
        assert isinstance(string, str), 'input must be a string'
        string = string.lower() if self._is_lower else string
        string = entities_textcleaning(string)
        batch_x = char_str_idx([string], self._settings['word2idx'], 2)
        batch_x_char = generate_char_seq(
            batch_x, self._settings['idx2word'], self._settings['char2idx']
        )
        if self._model == 'entity-network':
            batch_x_expand = np.expand_dims(batch_x, axis = 1)
            predicted = self._sess.run(
                self._logits,
                feed_dict = {
                    self._X: batch_x,
                    self._X_char: batch_x_char,
                    self._story: batch_x_expand,
                },
            )[0]
        else:
            predicted = self._sess.run(
                self._logits,
                feed_dict = {self._X: batch_x, self._X_char: batch_x_char},
            )[0]
        return [
            (string[i], self._settings['idx2tag'][predicted[i]])
            for i in range(len(predicted))
        ]


class SOFTMAX:
    def __init__(
        self,
        X,
        logits,
        sess,
        mode,
        dictionary,
        ngram = None,
        alphas = None,
        input_mask = None,
        segment_ids = None,
        is_training = None,
        dropout_keep_prob = None,
        story = None,
        maxlen = 80,
        label = ['negative', 'positive'],
    ):
        self._X = X
        self._logits = logits
        self._sess = sess
        self._mode = mode
        self._dictionary = dictionary
        self._ngram = ngram
        self._alphas = alphas
        self._input_mask = input_mask
        self._segment_ids = segment_ids
        self._is_training = is_training
        self._dropout_keep_prob = dropout_keep_prob
        self._story = story
        self._maxlen = maxlen
        self._label = label

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
        assert isinstance(string, str), 'input must be a string'
        string = _classification_textcleaning_stemmer_attention(string)
        splitted = string[1].split()
        if self._mode == 'fast-text':
            batch_x = fasttext_str_idx([string[0]], self._dictionary)
            batch_x = add_ngram(batch_x, self._ngram)
        else:
            if self._mode in ['entity-network', 'bert']:
                batch_x = str_idx(
                    [string[0]], self._dictionary, self._maxlen, UNK = 3
                )
            else:
                batch_x = str_idx(
                    [string[0]], self._dictionary, len(splitted), UNK = 3
                )
        if self._mode in ['luong', 'bahdanau', 'hierarchical']:
            probs, alphas = self._sess.run(
                [tf.nn.softmax(self._logits), self._alphas],
                feed_dict = {self._X: batch_x},
            )
            if self._mode == 'hierarchical':
                alphas = alphas[0]
            words = []
            for i in range(alphas.shape[0]):
                words.append([splitted[i], alphas[i]])
        if self._mode in ['bidirectional', 'fast-text']:
            probs = self._sess.run(
                tf.nn.softmax(self._logits), feed_dict = {self._X: batch_x}
            )
        if self._mode == 'bert':
            np_mask = np.ones((1, self._maxlen), dtype = np.int32)
            np_segment = np.ones((1, self._maxlen), dtype = np.int32)
            probs = self._sess.run(
                tf.nn.softmax(self._logits),
                feed_dict = {
                    self._X: batch_x,
                    self._input_mask: np_mask,
                    self._segment_ids: np_segment,
                    self._is_training: False,
                },
            )
        if self._mode == 'entity-network':
            batch_x_expand = np.expand_dims(batch_x, axis = 1)
            probs = self._sess.run(
                tf.nn.softmax(self._logits),
                feed_dict = {
                    self._X: batch_x,
                    self._story: batch_x_expand,
                    self._dropout_keep_prob: 1.0,
                },
            )

        if get_proba:
            dict_result = {}
            for no, label in enumerate(self._label):
                dict_result[label] = probs[0, no]
            if self._mode in ['luong', 'bahdanau', 'hierarchical']:
                dict_result['attention'] = words
            return dict_result
        else:
            return self._label[np.argmax(probs[0])]

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
        assert isinstance(strings, list) and isinstance(
            strings[0], str
        ), 'input must be list of strings'
        strings = [
            _classification_textcleaning_stemmer_attention(i)[0]
            for i in strings
        ]
        maxlen = max([len(i.split()) for i in strings])
        if self._mode == 'fast-text':
            batch_x = fasttext_str_idx(strings, self._dictionary)
            batch_x = add_ngram(batch_x, self._ngram)
            batch_x = tf.keras.preprocessing.sequence.pad_sequences(
                batch_x, maxlen
            )
        else:
            if self._mode in ['entity-network', 'bert']:
                batch_x = str_idx(
                    strings, self._dictionary, self._maxlen, UNK = 3
                )
            else:
                batch_x = str_idx(strings, self._dictionary, maxlen, UNK = 3)
        if self._mode not in ['bert', 'entity-network']:
            probs = self._sess.run(
                tf.nn.softmax(self._logits), feed_dict = {self._X: batch_x}
            )
        if self._mode == 'bert':
            np_mask = np.ones((len(batch_x), self._maxlen), dtype = np.int32)
            np_segment = np.ones((len(batch_x), self._maxlen), dtype = np.int32)
            probs = self._sess.run(
                tf.nn.softmax(self._logits),
                feed_dict = {
                    self._X: batch_x,
                    self._input_mask: np_mask,
                    self._segment_ids: np_segment,
                    self._is_training: False,
                },
            )
        if self._mode == 'entity-network':
            batch_x_expand = np.expand_dims(batch_x, axis = 1)
            probs = self._sess.run(
                tf.nn.softmax(self._logits),
                feed_dict = {
                    self._X: batch_x,
                    self._story: batch_x_expand,
                    self._dropout_keep_prob: 1.0,
                },
            )

        results = []
        if get_proba:
            for prob in probs:
                dict_result = {}
                for no, label in enumerate(self._label):
                    dict_result[label] = prob[no]
                results.append(dict_result)
        else:
            probs = np.argmax(probs, 1)
            for prob in probs:
                results.append(self._label[prob])
        return results


class SIGMOID:
    def __init__(
        self,
        X,
        logits,
        sess,
        mode,
        dictionary,
        ngram = None,
        alphas = None,
        input_mask = None,
        segment_ids = None,
        is_training = None,
        dropout_keep_prob = None,
        story = None,
        maxlen = 80,
    ):
        self._X = X
        self._logits = logits
        self._sess = sess
        self._mode = mode
        self._dictionary = dictionary
        self._ngram = ngram
        self._alphas = alphas
        self._input_mask = input_mask
        self._segment_ids = segment_ids
        self._is_training = is_training
        self._dropout_keep_prob = dropout_keep_prob
        self._story = story
        self._maxlen = maxlen
        self._label = [
            'toxic',
            'severe_toxic',
            'obscene',
            'threat',
            'insult',
            'identity_hate',
        ]

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
        assert isinstance(string, str), 'input must be a string'
        string = _classification_textcleaning_stemmer_attention(string)
        splitted = string[1].split()
        if self._mode == 'fast-text':
            batch_x = fasttext_str_idx([string[0]], self._dictionary)
            batch_x = add_ngram(batch_x, self._ngram)
        else:
            if self._mode in ['entity-network', 'bert']:
                batch_x = str_idx(
                    [string[0]], self._dictionary, self._maxlen, UNK = 3
                )
            else:
                batch_x = str_idx(
                    [string[0]], self._dictionary, len(splitted), UNK = 3
                )
        if self._mode in ['luong', 'bahdanau', 'hierarchical']:
            probs, alphas = self._sess.run(
                [tf.nn.sigmoid(self._logits), self._alphas],
                feed_dict = {self._X: batch_x},
            )
            if self._mode == 'hierarchical':
                alphas = alphas[0]
            words = []
            for i in range(alphas.shape[0]):
                words.append([splitted[i], alphas[i]])
        if self._mode in ['fast-text']:
            probs = self._sess.run(
                tf.nn.softmax(self._logits), feed_dict = {self._X: batch_x}
            )
        if self._mode == 'entity-network':
            batch_x_expand = np.expand_dims(batch_x, axis = 1)
            probs = self._sess.run(
                tf.nn.softmax(self._logits),
                feed_dict = {
                    self._X: batch_x,
                    self._story: batch_x_expand,
                    self._dropout_keep_prob: 1.0,
                },
            )
        if get_proba:
            dict_result = {}
            for no, label in enumerate(self._label):
                dict_result[label] = probs[0, no]
            if self._mode in ['luong', 'bahdanau', 'hierarchical']:
                dict_result['attention'] = words
            return dict_result
        else:
            result = []
            probs = np.around(probs[0])
            for no, label in enumerate(self._label):
                if probs[no]:
                    result.append(label)
            return result

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
        assert isinstance(strings, list) and isinstance(
            strings[0], str
        ), 'input must be list of strings'
        strings = [
            _classification_textcleaning_stemmer_attention(i)[0]
            for i in strings
        ]
        maxlen = max([len(i.split()) for i in strings])
        if self._mode == 'fast-text':
            batch_x = fasttext_str_idx(strings, self._dictionary)
            batch_x = add_ngram(batch_x, self._ngram)
            batch_x = tf.keras.preprocessing.sequence.pad_sequences(
                batch_x, maxlen
            )
        else:
            if self._mode in ['entity-network']:
                batch_x = str_idx(
                    strings, self._dictionary, self._maxlen, UNK = 3
                )
            else:
                batch_x = str_idx(strings, self._dictionary, maxlen, UNK = 3)
        if self._mode not in ['entity-network']:
            probs = self._sess.run(
                tf.nn.sigmoid(self._logits), feed_dict = {self._X: batch_x}
            )
        if self._mode == 'entity-network':
            batch_x_expand = np.expand_dims(batch_x, axis = 1)
            probs = self._sess.run(
                tf.nn.sigmoid(self._logits),
                feed_dict = {
                    self._X: batch_x,
                    self._story: batch_x_expand,
                    self._dropout_keep_prob: 1.0,
                },
            )
        results = []
        if get_proba:
            for prob in probs:
                dict_result = {}
                for no, label in enumerate(self._label):
                    dict_result[label] = prob[no]
                results.append(dict_result)
        else:
            probs = np.around(probs)
            for prob in probs:
                list_result = []
                for no, label in enumerate(self._label):
                    if prob[no]:
                        list_result.append(label)
                results.append(list_result)

        return results


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
        assert isinstance(string, str), 'input must be a string'
        string = language_detection_textcleaning(string)
        transformed = self._vectorizer.transform([string])
        batch_x = _convert_sparse_matrix_to_sparse_tensor(transformed)
        probs = self._sess.run(
            tf.nn.softmax(self._model.logits),
            feed_dict = {self._model.X: batch_x[0], self._model.W: batch_x[1]},
        )[0]
        if get_proba:
            return {self._label[no]: i for no, i in enumerate(probs)}
        else:
            return self._label[np.argmax(probs)]

    def predict_batch(self, strings):
        """
        classify list of strings

        Parameters
        ----------
        strings : list

        Returns
        -------
        list_dictionaries: list of results
        """
        assert isinstance(strings, list) and isinstance(
            strings[0], str
        ), 'input must be list of strings'
        strings = [language_detection_textcleaning(i) for i in strings]
        transformed = self._vectorizer.transform(strings)
        batch_x = _convert_sparse_matrix_to_sparse_tensor(transformed)
        probs = self._sess.run(
            tf.nn.softmax(self._model.logits),
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


class SPARSE_SOFTMAX:
    def __init__(
        self, path, vectorizer, label, output_size, embedded_size, vocab_size
    ):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._model = _SPARSE_SOFTMAX_MODEL(
                output_size, embedded_size, vocab_size
            )
            self._sess = tf.InteractiveSession()
            self._sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(self._sess, path + '/model.ckpt')
        self._vectorizer = vectorizer
        self._label = label

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
        assert isinstance(string, str), 'input must be a string'
        string = _classification_textcleaning_stemmer_attention(string)[0]
        transformed = self._vectorizer.transform([string])
        batch_x = _convert_sparse_matrix_to_sparse_tensor(transformed)
        probs = self._sess.run(
            tf.nn.softmax(self._model.logits),
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
        assert isinstance(strings, list) and isinstance(
            strings[0], str
        ), 'input must be list of strings'
        strings = [
            _classification_textcleaning_stemmer_attention(i)[0]
            for i in strings
        ]
        transformed = self._vectorizer.transform(strings)
        batch_x = _convert_sparse_matrix_to_sparse_tensor(transformed)
        probs = self._sess.run(
            tf.nn.softmax(self._model.logits),
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
