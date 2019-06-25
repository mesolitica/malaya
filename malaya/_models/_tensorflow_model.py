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
    tag_chunk,
    bert_tokenization,
    bert_tokenization_siamese,
)
from .._utils._parse_dependency import DependencyGraph
from ..preprocessing import preprocessing_classification_index
from ..stem import _classification_textcleaning_stemmer
from .._utils._utils import add_neutral as neutral
from .._utils._html import _render_binary, _render_toxic, _render_emotion


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
        Print important top-k transitions for tagging dependency.

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
        Print important top-k transitions for indexing dependency.

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
        Print important top-k features.

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
        Tag a string.

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
        Print important top-k transitions.

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
        Print important top-k features.

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

    def analyze(self, string):
        """
        Analyze a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: analyzed string
        """
        predicted = self.predict(string)
        return tag_chunk(predicted)

    def predict(self, string):
        """
        Tag a string.

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


class BERT:
    def __init__(
        self,
        X,
        segment_ids,
        input_masks,
        logits,
        sess,
        tokenizer,
        maxlen,
        label = ['negative', 'positive'],
    ):
        self._X = X
        self._segment_ids = segment_ids
        self._input_masks = input_masks
        self._logits = logits
        self._sess = sess
        self._tokenizer = tokenizer
        self._maxlen = maxlen
        self._label = label


class BINARY_BERT(BERT):
    def __init__(
        self,
        X,
        segment_ids,
        input_masks,
        logits,
        sess,
        tokenizer,
        maxlen,
        label = ['negative', 'positive'],
    ):
        BERT.__init__(
            self,
            X,
            segment_ids,
            input_masks,
            logits,
            sess,
            tokenizer,
            maxlen,
            label,
        )

    def predict(self, string, get_proba = False, add_neutral = True):
        """
        classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        dictionary: results
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        if not isinstance(add_neutral, bool):
            raise ValueError('add_neutral must be a boolean')

        if add_neutral:
            label = self._label + ['neutral']
        else:
            label = self._label

        input_ids, input_masks, segment_ids = bert_tokenization(
            self._tokenizer, [string], self._maxlen
        )

        result = self._sess.run(
            tf.nn.softmax(self._logits),
            feed_dict = {
                self._X: input_ids,
                self._segment_ids: segment_ids,
                self._input_masks: input_masks,
            },
        )
        if add_neutral:
            result = neutral(result)
        result = result[0]
        if get_proba:
            return {label[i]: result[i] for i in range(len(result))}
        else:
            return label[np.argmax(result)]

    def predict_batch(self, strings, get_proba = False, add_neutral = True):
        """
        classify list of strings.

        Parameters
        ----------
        strings : list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        if not isinstance(add_neutral, bool):
            raise ValueError('add_neutral must be a boolean')

        if add_neutral:
            label = self._label + ['neutral']
        else:
            label = self._label

        input_ids, input_masks, segment_ids = bert_tokenization(
            self._tokenizer, strings, self._maxlen
        )
        results = self._sess.run(
            tf.nn.softmax(self._logits),
            feed_dict = {
                self._X: input_ids,
                self._segment_ids: segment_ids,
                self._input_masks: input_masks,
            },
        )
        if add_neutral:
            results = neutral(results)

        if get_proba:
            outputs = []
            for result in results:
                outputs.append(
                    {label[i]: result[i] for i in range(len(result))}
                )
            return outputs
        else:
            return [label[result] for result in np.argmax(results, axis = 1)]


class MULTICLASS_BERT(BERT):
    def __init__(
        self,
        X,
        segment_ids,
        input_masks,
        logits,
        sess,
        tokenizer,
        maxlen,
        label = ['negative', 'positive'],
    ):
        BERT.__init__(
            self,
            X,
            segment_ids,
            input_masks,
            logits,
            sess,
            tokenizer,
            maxlen,
            label,
        )

    def predict(self, string, get_proba = False):
        """
        classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        dictionary: results
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')

        input_ids, input_masks, segment_ids = bert_tokenization(
            self._tokenizer, [string], self._maxlen
        )

        result = self._sess.run(
            tf.nn.softmax(self._logits),
            feed_dict = {
                self._X: input_ids,
                self._segment_ids: segment_ids,
                self._input_masks: input_masks,
            },
        )
        result = result[0]
        if get_proba:
            return {self._label[i]: result[i] for i in range(len(result))}
        else:
            return self._label[np.argmax(result)]

    def predict_batch(self, strings, get_proba = False):
        """
        classify list of strings.

        Parameters
        ----------
        strings : list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')

        input_ids, input_masks, segment_ids = bert_tokenization(
            self._tokenizer, strings, self._maxlen
        )
        results = self._sess.run(
            tf.nn.softmax(self._logits),
            feed_dict = {
                self._X: input_ids,
                self._segment_ids: segment_ids,
                self._input_masks: input_masks,
            },
        )

        if get_proba:
            outputs = []
            for result in results:
                outputs.append(
                    {self._label[i]: result[i] for i in range(len(result))}
                )
            return outputs
        else:
            return [
                self._label[result] for result in np.argmax(results, axis = 1)
            ]


class SIGMOID_BERT(BERT):
    def __init__(
        self,
        X,
        segment_ids,
        input_masks,
        logits,
        sess,
        tokenizer,
        maxlen,
        label = ['negative', 'positive'],
    ):
        BERT.__init__(
            self,
            X,
            segment_ids,
            input_masks,
            logits,
            sess,
            tokenizer,
            maxlen,
            label,
        )

    def predict(self, string, get_proba = False):
        """
        classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        dictionary: results
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')

        input_ids, input_masks, segment_ids = bert_tokenization(
            self._tokenizer, [string], self._maxlen
        )

        result = self._sess.run(
            tf.nn.sigmoid(self._logits),
            feed_dict = {
                self._X: input_ids,
                self._segment_ids: segment_ids,
                self._input_masks: input_masks,
            },
        )
        result = result[0]
        if get_proba:
            return {self._label[i]: result[i] for i in range(len(result))}
        else:
            probs = np.around(result)
            return [label for no, label in enumerate(self._label) if probs[no]]

    def predict_batch(self, strings, get_proba = False):
        """
        classify list of strings.

        Parameters
        ----------
        strings : list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')

        input_ids, input_masks, segment_ids = bert_tokenization(
            self._tokenizer, strings, self._maxlen
        )
        probs = self._sess.run(
            tf.nn.sigmoid(self._logits),
            feed_dict = {
                self._X: input_ids,
                self._segment_ids: segment_ids,
                self._input_masks: input_masks,
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


class SOFTMAX:
    def __init__(
        self,
        X,
        logits,
        logits_seq,
        alphas,
        sess,
        dictionary,
        class_name,
        label = ['negative', 'positive'],
    ):
        self._X = X
        self._logits = logits
        self._logits_seq = logits_seq
        self._alphas = alphas
        self._sess = sess
        self._dictionary = dictionary
        self._label = label
        self._class_name = class_name


class BINARY_SOFTMAX(SOFTMAX):
    def __init__(
        self,
        X,
        logits,
        logits_seq,
        alphas,
        sess,
        dictionary,
        class_name,
        label = ['negative', 'positive'],
    ):
        SOFTMAX.__init__(
            self,
            X,
            logits,
            logits_seq,
            alphas,
            sess,
            dictionary,
            class_name,
            label,
        )

    def predict(self, string, get_proba = False, add_neutral = True):
        """
        classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        dictionary: results
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        if not isinstance(add_neutral, bool):
            raise ValueError('add_neutral must be a boolean')

        if add_neutral:
            label = self._label + ['neutral']
        else:
            label = self._label

        tokenized_indices, splitted = preprocessing_classification_index(string)
        batch_x = str_idx(
            [' '.join(splitted)], self._dictionary, len(splitted), UNK = 3
        )
        result, alphas = self._sess.run(
            [tf.nn.softmax(self._logits), self._alphas],
            feed_dict = {self._X: batch_x},
        )
        if add_neutral:
            result = neutral(result)
        result = result[0]

        if get_proba:
            dict_result = {label[i]: result[i] for i in range(len(result))}
            dict_result['attention'] = {
                k: alphas[v] if v > -1 else 0.0
                for k, v in tokenized_indices.items()
            }
            return dict_result
        else:
            return label[np.argmax(result)]

    def predict_words(self, string, visualization = True):
        """
        classify words.

        Parameters
        ----------
        string : str
        visualization: bool, optional (default=True)
            If True, it will open the visualization dashboard.

        Returns
        -------
        dictionary: results
        """

        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(visualization, bool):
            raise ValueError('visualization must be a boolean')

        label = self._label + ['neutral']

        tokenized_indices, splitted = preprocessing_classification_index(string)
        batch_x = str_idx(
            [' '.join(splitted)], self._dictionary, len(splitted), UNK = 3
        )
        result, alphas, words = self._sess.run(
            [
                tf.nn.softmax(self._logits),
                self._alphas,
                tf.nn.softmax(self._logits_seq),
            ],
            feed_dict = {self._X: batch_x},
        )
        result = neutral(result)
        result = result[0]
        words = neutral(words[0])
        distribution_words = words[:, np.argmax(words.sum(axis = 0))]
        y_histogram, x_histogram = np.histogram(
            distribution_words, bins = np.arange(0, 1, 0.05)
        )
        y_histogram = y_histogram / y_histogram.sum()
        x_attention = np.arange(len(alphas))
        neutral_word = [0.0 for _ in range(len(self._label))]
        neutral_word.append(1.0)
        left, right = np.unique(
            np.argmax(words, axis = 1), return_counts = True
        )
        left = left.tolist()
        y_barplot = []
        for i in range(len(label)):
            if i not in left:
                y_barplot.append(i)
            else:
                y_barplot.append(right[left.index(i)])

        dict_result = {label[i]: result[i] for i in range(len(result))}
        dict_result['alphas'] = {
            k: alphas[v] if v > -1 else 0.0
            for k, v in tokenized_indices.items()
        }
        dict_result['word'] = {
            k: words[v] if v > -1 else neutral_word
            for k, v in tokenized_indices.items()
        }
        dict_result['histogram'] = {'x': x_histogram, 'y': y_histogram}
        dict_result['attention'] = {'x': x_attention, 'y': alphas}
        dict_result['barplot'] = {'x': label, 'y': y_barplot}
        dict_result['class_name'] = self._class_name
        if visualization:
            _render_binary(dict_result)
        else:
            return dict_result

    def predict_batch(self, strings, get_proba = False, add_neutral = True):
        """
        classify list of strings.

        Parameters
        ----------
        strings : list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        if not isinstance(add_neutral, bool):
            raise ValueError('add_neutral must be a boolean')

        if add_neutral:
            label = self._label + ['neutral']
        else:
            label = self._label

        strings = [
            ' '.join(preprocessing_classification_index(i)[1]) for i in strings
        ]
        maxlen = max([len(i.split()) for i in strings])
        batch_x = str_idx(strings, self._dictionary, maxlen, UNK = 3)
        results = self._sess.run(
            tf.nn.softmax(self._logits), feed_dict = {self._X: batch_x}
        )
        if add_neutral:
            results = neutral(results)

        if get_proba:
            outputs = []
            for result in results:
                outputs.append(
                    {label[i]: result[i] for i in range(len(result))}
                )
            return outputs
        else:
            return [label[result] for result in np.argmax(results, axis = 1)]


class MULTICLASS_SOFTMAX(SOFTMAX):
    def __init__(
        self,
        X,
        logits,
        logits_seq,
        alphas,
        sess,
        dictionary,
        class_name,
        label = ['negative', 'positive'],
    ):
        SOFTMAX.__init__(
            self,
            X,
            logits,
            logits_seq,
            alphas,
            sess,
            dictionary,
            class_name,
            label,
        )

    def predict(self, string, get_proba = False):
        """
        classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        dictionary: results
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')

        tokenized_indices, splitted = preprocessing_classification_index(string)
        batch_x = str_idx(
            [' '.join(splitted)], self._dictionary, len(splitted), UNK = 3
        )
        result, alphas = self._sess.run(
            [tf.nn.softmax(self._logits), self._alphas],
            feed_dict = {self._X: batch_x},
        )
        result = result[0]

        if get_proba:
            dict_result = {
                self._label[i]: result[i] for i in range(len(result))
            }
            dict_result['attention'] = {
                k: alphas[v] if v > -1 else 0.0
                for k, v in tokenized_indices.items()
            }
            return dict_result
        else:
            return self._label[np.argmax(result)]

    def predict_words(self, string, visualization = True):
        """
        classify words.

        Parameters
        ----------
        string : str
        visualization: bool, optional (default=True)
            If True, it will open the visualization dashboard.

        Returns
        -------
        dictionary: results
        """

        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(visualization, bool):
            raise ValueError('visualization must be a boolean')

        tokenized_indices, splitted = preprocessing_classification_index(string)
        batch_x = str_idx(
            [' '.join(splitted)], self._dictionary, len(splitted), UNK = 3
        )
        result, alphas, words = self._sess.run(
            [
                tf.nn.softmax(self._logits),
                self._alphas,
                tf.nn.softmax(self._logits_seq),
            ],
            feed_dict = {self._X: batch_x},
        )
        result = result[0]
        words = words[0]
        distribution_words = words[:, np.argmax(words.sum(axis = 0))]
        y_histogram, x_histogram = np.histogram(
            distribution_words, bins = np.arange(0, 1, 0.05)
        )
        y_histogram = y_histogram / y_histogram.sum()
        x_attention = np.arange(len(alphas))
        neutral_word = [0.0 for _ in range(len(self._label))]
        left, right = np.unique(
            np.argmax(words, axis = 1), return_counts = True
        )
        left = left.tolist()
        y_barplot = []
        for i in range(len(self._label)):
            if i not in left:
                y_barplot.append(i)
            else:
                y_barplot.append(right[left.index(i)])

        dict_result = {self._label[i]: result[i] for i in range(len(result))}
        dict_result['alphas'] = {
            k: alphas[v] if v > -1 else 0.0
            for k, v in tokenized_indices.items()
        }
        dict_result['word'] = {
            k: words[v] if v > -1 else neutral_word
            for k, v in tokenized_indices.items()
        }
        dict_result['histogram'] = {'x': x_histogram, 'y': y_histogram}
        dict_result['attention'] = {'x': x_attention, 'y': alphas}
        dict_result['barplot'] = {'x': self._label, 'y': y_barplot}
        dict_result['class_name'] = self._class_name
        if visualization:
            _render_emotion(dict_result)
        else:
            return dict_result

    def predict_batch(self, strings, get_proba = False):
        """
        classify list of strings.

        Parameters
        ----------
        strings : list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')

        strings = [
            ' '.join(preprocessing_classification_index(i)[1]) for i in strings
        ]
        maxlen = max([len(i.split()) for i in strings])
        batch_x = str_idx(strings, self._dictionary, maxlen, UNK = 3)
        results = self._sess.run(
            tf.nn.softmax(self._logits), feed_dict = {self._X: batch_x}
        )

        if get_proba:
            outputs = []
            for result in results:
                outputs.append(
                    {self._label[i]: result[i] for i in range(len(result))}
                )
            return outputs
        else:
            return [
                self._label[result] for result in np.argmax(results, axis = 1)
            ]


class SIGMOID:
    def __init__(self, X, logits, logits_seq, alphas, sess, dictionary):
        self._X = X
        self._logits = logits
        self._logits_seq = logits_seq
        self._alphas = alphas
        self._sess = sess
        self._dictionary = dictionary
        self._label = [
            'toxic',
            'severe_toxic',
            'obscene',
            'threat',
            'insult',
            'identity_hate',
        ]
        self._class_name = 'toxicity'

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

        tokenized_indices, splitted = preprocessing_classification_index(string)
        batch_x = str_idx(
            [' '.join(splitted)], self._dictionary, len(splitted), UNK = 3
        )
        result, alphas = self._sess.run(
            [tf.nn.sigmoid(self._logits), self._alphas],
            feed_dict = {self._X: batch_x},
        )
        result = result[0]

        if get_proba:
            dict_result = {
                label: result[no] for no, label in enumerate(self._label)
            }
            dict_result['attention'] = {
                k: alphas[v] if v > -1 else 0.0
                for k, v in tokenized_indices.items()
            }
            return dict_result
        else:
            probs = np.around(result)
            return [label for no, label in enumerate(self._label) if probs[no]]

    def predict_words(self, string, visualization = True):
        """
        classify words.

        Parameters
        ----------
        string : str
        visualization: bool, optional (default=True)
            If True, it will open the visualization dashboard.

        Returns
        -------
        dictionary: results
        """

        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(visualization, bool):
            raise ValueError('visualization must be a boolean')

        tokenized_indices, splitted = preprocessing_classification_index(string)
        batch_x = str_idx(
            [' '.join(splitted)], self._dictionary, len(splitted), UNK = 3
        )
        result, alphas, words = self._sess.run(
            [
                tf.nn.sigmoid(self._logits),
                self._alphas,
                tf.nn.sigmoid(self._logits_seq),
            ],
            feed_dict = {self._X: batch_x},
        )
        result = result[0]
        words = words[0]
        distribution_words = words[:, np.argmax(words.sum(axis = 0))]
        y_histogram, x_histogram = np.histogram(
            distribution_words, bins = np.arange(0, 1, 0.05)
        )
        y_histogram = y_histogram / y_histogram.sum()
        x_attention = np.arange(len(alphas))
        neutral_word = [0.0 for _ in range(len(self._label))]
        around_words = np.around(words)
        y_barplot = np.sum(around_words, axis = 0).tolist()

        dict_result = {self._label[i]: result[i] for i in range(len(result))}
        dict_result['alphas'] = {
            k: alphas[v] if v > -1 else 0.0
            for k, v in tokenized_indices.items()
        }
        dict_result['word'] = {
            k: words[v] if v > -1 else neutral_word
            for k, v in tokenized_indices.items()
        }
        dict_result['histogram'] = {'x': x_histogram, 'y': y_histogram}
        dict_result['attention'] = {'x': x_attention, 'y': alphas}
        dict_result['barplot'] = {'x': self._label, 'y': y_barplot}
        dict_result['class_name'] = self._class_name
        if visualization:
            _render_toxic(dict_result)
        else:
            return dict_result

    def predict_batch(self, strings, get_proba = False):
        """
        classify list of strings.

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
        strings = [
            ' '.join(preprocessing_classification_index(i)[1]) for i in strings
        ]
        maxlen = max([len(i.split()) for i in strings])
        batch_x = str_idx(strings, self._dictionary, maxlen, UNK = 3)
        probs = self._sess.run(
            tf.nn.sigmoid(self._logits), feed_dict = {self._X: batch_x}
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
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')

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
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')

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
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        string = ' '.join(preprocessing_classification_index(string)[1])
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
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        strings = [
            ' '.join(preprocessing_classification_index(i)[1]) for i in strings
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


class SPARSE_SIGMOID:
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
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        string = ' '.join(preprocessing_classification_index(string)[1])
        transformed = self._vectorizer.transform([string])
        batch_x = _convert_sparse_matrix_to_sparse_tensor(
            transformed, got_limit = False
        )
        result = self._sess.run(
            tf.nn.sigmoid(self._model.logits),
            feed_dict = {self._model.X: batch_x[0], self._model.W: batch_x[1]},
        )

        result = result[0]

        if get_proba:
            return {label: result[no] for no, label in enumerate(self._label)}
        else:
            probs = np.around(result)
            return [label for no, label in enumerate(self._label) if probs[no]]

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
        strings = [
            ' '.join(preprocessing_classification_index(i)[1]) for i in strings
        ]
        transformed = self._vectorizer.transform(strings)
        batch_x = _convert_sparse_matrix_to_sparse_tensor(
            transformed, got_limit = False
        )
        probs = self._sess.run(
            tf.nn.sigmoid(self._model.logits),
            feed_dict = {self._model.X: batch_x[0], self._model.W: batch_x[1]},
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


class SIAMESE:
    def __init__(self, X_left, X_right, logits, sess, dictionary):
        self._X_left = X_left
        self._X_right = X_right
        self._logits = logits
        self._sess = sess
        self._dictionary = dictionary

    def predict(self, string_left, string_right):
        """
        calculate similarity for two different texts.

        Parameters
        ----------
        string_left : str
        string_right : str

        Returns
        -------
        float: float
        """
        if not isinstance(string_left, str):
            raise ValueError('string_left must be a string')
        if not isinstance(string_right, str):
            raise ValueError('string_right must be a string')
        _, splitted = preprocessing_classification_index(string_left)
        batch_x_left = str_idx(
            [' '.join(splitted)], self._dictionary, len(splitted), UNK = 3
        )
        _, splitted = preprocessing_classification_index(string_right)
        batch_x_right = str_idx(
            [' '.join(splitted)], self._dictionary, len(splitted), UNK = 3
        )
        return self._sess.run(
            1 - self._logits,
            feed_dict = {
                self._X_left: batch_x_left,
                self._X_right: batch_x_right,
            },
        )[0]

    def predict_batch(self, strings_left, strings_right):
        """
        calculate similarity for two different batch of texts.

        Parameters
        ----------
        string_left : str
        string_right : str

        Returns
        -------
        list: list of float
        """
        if not isinstance(strings_left, list):
            raise ValueError('strings_left must be a list')
        if not isinstance(strings_left[0], str):
            raise ValueError('strings_left must be list of strings')
        if not isinstance(strings_right, list):
            raise ValueError('strings_right must be a list')
        if not isinstance(strings_right[0], str):
            raise ValueError('strings_right must be list of strings')

        strings = [
            ' '.join(preprocessing_classification_index(i)[1])
            for i in strings_left
        ]
        maxlen = max([len(i.split()) for i in strings])
        batch_x_left = str_idx(strings, self._dictionary, maxlen, UNK = 3)

        strings = [
            ' '.join(preprocessing_classification_index(i)[1])
            for i in strings_right
        ]
        maxlen = max([len(i.split()) for i in strings])
        batch_x_right = str_idx(strings, self._dictionary, maxlen, UNK = 3)

        return self._sess.run(
            1 - self._logits,
            feed_dict = {
                self._X_left: batch_x_left,
                self._X_right: batch_x_right,
            },
        )


class SIAMESE_BERT(BERT):
    def __init__(
        self,
        X,
        segment_ids,
        input_masks,
        logits,
        sess,
        tokenizer,
        maxlen,
        label = ['not similar', 'similar'],
    ):
        BERT.__init__(
            self,
            X,
            segment_ids,
            input_masks,
            logits,
            sess,
            tokenizer,
            maxlen,
            label,
        )

    def _base(self, strings_left, strings_right):
        input_ids, input_masks, segment_ids = bert_tokenization_siamese(
            self._tokenizer, strings_left, strings_right, self._maxlen
        )

        return self._sess.run(
            tf.nn.softmax(self._logits),
            feed_dict = {
                self._X: input_ids,
                self._segment_ids: segment_ids,
                self._input_masks: input_masks,
            },
        )

    def predict(self, string_left, string_right):
        """
        calculate similarity for two different texts.

        Parameters
        ----------
        string_left : str
        string_right : str

        Returns
        -------
        float: float
        """
        if not isinstance(string_left, str):
            raise ValueError('string_left must be a string')
        if not isinstance(string_right, str):
            raise ValueError('string_right must be a string')

        return self._base([string_left], [string_right])[0, 1]

    def predict_batch(self, strings_left, strings_right):
        """
        calculate similarity for two different batch of texts.

        Parameters
        ----------
        string_left : str
        string_right : str

        Returns
        -------
        list: list of float
        """
        if not isinstance(strings_left, list):
            raise ValueError('strings_left must be a list')
        if not isinstance(strings_left[0], str):
            raise ValueError('strings_left must be list of strings')
        if not isinstance(strings_right, list):
            raise ValueError('strings_right must be a list')
        if not isinstance(strings_right[0], str):
            raise ValueError('strings_right must be list of strings')

        return self._base(strings_left, strings_right)[:, 1]
