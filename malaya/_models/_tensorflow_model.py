import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
from ..texts._text_functions import (
    str_idx,
    entities_textcleaning,
    char_str_idx,
    generate_char_seq,
    language_detection_textcleaning,
)
from .._utils._parse_dependency import DependencyGraph
from ..stem import _classification_textcleaning_stemmer


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


class DEPENDENCY:
    def __init__(
        self,
        X,
        X_char,
        logits,
        logits_depends,
        settings,
        sess,
        model,
        transitions,
        transitions_depends,
        features,
    ):
        self._X = X
        self._X_char = X_char
        self._logits = logits
        self._logits_depends = logits_depends
        self._settings = settings
        self._sess = sess
        self._model = model
        self._settings['idx2tag'] = {
            int(k): v for k, v in self._settings['idx2tag'].items()
        }
        self.transitions, self.transitions_depends, self.features = self._sess.run(
            [transitions, transitions_depends, features]
        )

    def print_transitions_tag(self, top_k = 10):
        """
        Print important top-k transitions for tagging dependency

        Parameters
        ----------
        top_k : int
        """
        if not isinstance(top_k, int):
            raise ValueError('input must be an integer')
        print('Top-%d likely transitions:' % (top_k))
        indices = np.argsort(self.transitions.flatten())[::-1]
        top_trans = [
            np.unravel_index(i, self.transitions.shape) for i in indices[:top_k]
        ]
        for i in range(top_k):
            print(
                '%s -> %s: %f'
                % (
                    self._settings['idx2tag'][top_trans[i][0]],
                    self._settings['idx2tag'][top_trans[i][1]],
                    self.transitions[top_trans[i]],
                )
            )

        bottom_trans = [
            np.unravel_index(i, self.transitions.shape)
            for i in indices[::-1][:top_k]
        ]
        print('\nTop-%d unlikely transitions:' % (top_k))
        for i in range(top_k):
            print(
                '%s -> %s: %f'
                % (
                    self._settings['idx2tag'][bottom_trans[i][0]],
                    self._settings['idx2tag'][bottom_trans[i][1]],
                    self.transitions[bottom_trans[i]],
                )
            )

    def print_transitions_index(self, top_k = 10):
        """
        Print important top-k transitions for indexing dependency

        Parameters
        ----------
        top_k : int
        """
        if not isinstance(top_k, int):
            raise ValueError('input must be an integer')
        print('Top-%d likely transitions:' % (top_k))
        indices = np.argsort(self.transitions_depends.flatten())[::-1]
        top_trans = [
            np.unravel_index(i, self.transitions_depends.shape)
            for i in indices[:top_k]
        ]
        for i in range(top_k):
            print(
                '%d -> %d: %f'
                % (
                    top_trans[i][0],
                    top_trans[i][1],
                    self.transitions_depends[top_trans[i]],
                )
            )

        bottom_trans = [
            np.unravel_index(i, self.transitions_depends.shape)
            for i in indices[::-1][:top_k]
        ]
        print('\nTop-%d unlikely transitions:' % (top_k))
        for i in range(top_k):
            print(
                '%d -> %d: %f'
                % (
                    bottom_trans[i][0],
                    bottom_trans[i][1],
                    self.transitions_depends[bottom_trans[i]],
                )
            )

    def print_features(self, top_k = 10):
        """
        Print important top-k features

        Parameters
        ----------
        top_k : int
        """
        if not isinstance(top_k, int):
            raise ValueError('input must be an integer')
        _features = self.features.sum(axis = 1)
        indices = np.argsort(_features)[::-1]
        rev_indices = indices[::-1]
        print('Top-%d positive:' % (top_k))
        for i in range(top_k):
            print(
                '%s: %f'
                % (
                    self._settings['idx2word'][str(indices[i])],
                    _features[indices[i]],
                )
            )

        print('\nTop-%d negative:' % (top_k))
        for i in range(top_k):
            print(
                '%s: %f'
                % (
                    self._settings['idx2word'][str(rev_indices[i])],
                    _features[rev_indices[i]],
                )
            )

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
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        original_string, string = entities_textcleaning(string)
        if len(string) > 120:
            raise Exception(
                'Dependency parsing only able to accept string less than 120 words'
            )
        batch_x = char_str_idx([string], self._settings['word2idx'], 2)
        batch_x_char = generate_char_seq(
            [string], self._settings['char2idx'], 2
        )
        tagging, depend = self._sess.run(
            [self._logits, self._logits_depends],
            feed_dict = {self._X: batch_x, self._X_char: batch_x_char},
        )
        tagging = [self._settings['idx2tag'][i] for i in tagging[0]]
        depend = depend[0] - 1
        for i in range(len(depend)):
            if depend[i] == 0 and tagging[i] != 'root':
                tagging[i] = 'UNK'
            elif depend[i] != 0 and tagging[i] == 'root':
                tagging[i] = 'UNK'
            elif depend[i] > len(tagging):
                depend[i] = len(tagging)
        tagging = [(original_string[i], tagging[i]) for i in range(len(depend))]
        indexing = [(original_string[i], depend[i]) for i in range(len(depend))]
        result = []
        for i in range(len(tagging)):
            result.append(
                '%d\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_'
                % (i + 1, tagging[i][0], int(indexing[i][1]), tagging[i][1])
            )
        d = DependencyGraph('\n'.join(result), top_relation_label = 'root')
        return d, tagging, indexing


class TAGGING:
    def __init__(
        self,
        X,
        X_char,
        logits,
        settings,
        sess,
        model,
        transitions,
        features,
        is_lower = True,
        story = None,
        tags_state_fw = None,
        tags_state_bw = None,
    ):
        self._X = X
        self._X_char = X_char
        self._logits = logits
        self._settings = settings
        self._sess = sess
        self._model = model
        self._is_lower = is_lower
        self._story = story
        self._settings['idx2tag'] = {
            int(k): v for k, v in self._settings['idx2tag'].items()
        }
        self.transitions, self.features = self._sess.run(
            [transitions, features]
        )
        self._tags_state_fw = tags_state_fw
        self._tags_state_bw = tags_state_bw

    def get_alignment(self, string):
        if 'bahdanau' not in self._model and 'luong' not in self._model:
            print(
                'alignment visualization only supports `bahdanau` or `luong` model'
            )
        else:
            original_string, string = entities_textcleaning(
                string, lowering = self._is_lower
            )
            batch_x = char_str_idx([string], self._settings['word2idx'], 2)
            batch_x_char = generate_char_seq(
                [string], self._settings['char2idx'], 2
            )
            predicted, state_fw, state_bw = self._sess.run(
                [self._logits, self._tags_state_fw, self._tags_state_bw],
                feed_dict = {self._X: batch_x, self._X_char: batch_x_char},
            )
            tag = [
                '%s-%s' % (original_string[no], self._settings['idx2tag'][t])
                for no, t in enumerate(predicted[0])
            ]
            r = np.argmax((state_bw[::-1] + state_fw)[:, 0], axis = 1)
            result = []
            for i in range(len(tag)):
                result.append(
                    '%d\t%s\t_\t_\t_\t_\t%d\t_\t_\t_'
                    % (i + 1, tag[i], int(r[i]))
                )
            d = DependencyGraph('\n'.join(result))
            return d, predicted, state_fw, state_bw

    def print_transitions(self, top_k = 10):
        """
        Print important top-k transitions

        Parameters
        ----------
        top_k : int
        """
        if not isinstance(top_k, int):
            raise ValueError('input must be an integer')
        print('Top-%d likely transitions:' % (top_k))
        indices = np.argsort(self.transitions.flatten())[::-1]
        top_trans = [
            np.unravel_index(i, self.transitions.shape) for i in indices[:top_k]
        ]
        for i in range(top_k):
            print(
                '%s -> %s: %f'
                % (
                    self._settings['idx2tag'][top_trans[i][0]],
                    self._settings['idx2tag'][top_trans[i][1]],
                    self.transitions[top_trans[i]],
                )
            )

        bottom_trans = [
            np.unravel_index(i, self.transitions.shape)
            for i in indices[::-1][:top_k]
        ]
        print('\nTop-%d unlikely transitions:' % (top_k))
        for i in range(top_k):
            print(
                '%s -> %s: %f'
                % (
                    self._settings['idx2tag'][bottom_trans[i][0]],
                    self._settings['idx2tag'][bottom_trans[i][1]],
                    self.transitions[bottom_trans[i]],
                )
            )

    def print_features(self, top_k = 10):
        """
        Print important top-k features

        Parameters
        ----------
        top_k : int
        """
        if not isinstance(top_k, int):
            raise ValueError('input must be an integer')
        _features = self.features.sum(axis = 1)
        indices = np.argsort(_features)[::-1]
        rev_indices = indices[::-1]
        print('Top-%d positive:' % (top_k))
        for i in range(top_k):
            print(
                '%s: %f'
                % (
                    self._settings['idx2word'][str(indices[i])],
                    _features[indices[i]],
                )
            )

        print('\nTop-%d negative:' % (top_k))
        for i in range(top_k):
            print(
                '%s: %f'
                % (
                    self._settings['idx2word'][str(rev_indices[i])],
                    _features[rev_indices[i]],
                )
            )

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
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        original_string, string = entities_textcleaning(
            string, lowering = self._is_lower
        )
        batch_x = char_str_idx([string], self._settings['word2idx'], 2)
        batch_x_char = generate_char_seq(
            [string], self._settings['char2idx'], 2
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
            (original_string[i], self._settings['idx2tag'][predicted[i]])
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
        self._alphas = alphas
        self._input_mask = input_mask
        self._segment_ids = segment_ids
        self._is_training = is_training
        self._dropout_keep_prob = dropout_keep_prob
        self._story = story
        self._maxlen = maxlen
        self._label = label

    def get_dictionary(self):
        return self._dictionary

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
        string = _classification_textcleaning_stemmer(string, attention = True)
        splitted = string[1].split()
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
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(strings[0], str):
            raise ValueError('input must be list of strings')
        strings = [_classification_textcleaning_stemmer(i) for i in strings]
        maxlen = max([len(i.split()) for i in strings])
        if self._mode in ['entity-network', 'bert']:
            batch_x = str_idx(strings, self._dictionary, self._maxlen, UNK = 3)
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

    def get_dictionary(self):
        return self._dictionary

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
        string = _classification_textcleaning_stemmer(string, attention = True)
        splitted = string[1].split()
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
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(strings[0], str):
            raise ValueError('input must be list of strings')
        strings = [
            _classification_textcleaning_stemmer(i, attention = True)[0]
            for i in strings
        ]
        maxlen = max([len(i.split()) for i in strings])
        if self._mode in ['entity-network']:
            batch_x = str_idx(strings, self._dictionary, self._maxlen, UNK = 3)
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
        if not isinstance(string, str):
            raise ValueError('input must be a string')
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
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        string = _classification_textcleaning_stemmer(string, attention = True)[
            0
        ]
        transformed = self._vectorizer.transform([string])
        batch_x = _convert_sparse_matrix_to_sparse_tensor(
            transformed, got_limit = False
        )
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
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(strings[0], str):
            raise ValueError('input must be list of strings')
        strings = [
            _classification_textcleaning_stemmer(i, attention = True)[0]
            for i in strings
        ]
        transformed = self._vectorizer.transform(strings)
        batch_x = _convert_sparse_matrix_to_sparse_tensor(
            transformed, got_limit = False
        )
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
