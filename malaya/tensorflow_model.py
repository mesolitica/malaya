import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
from .text_functions import (
    str_idx,
    add_ngram,
    fasttext_str_idx,
    entities_textcleaning,
    char_str_idx,
    generate_char_seq,
)
from .stemmer import classification_textcleaning_stemmer_attention


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


class SENTIMENT:
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

    def predict(self, string):
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
        string = classification_textcleaning_stemmer_attention(string)
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
            return {
                'negative': probs[0, 0],
                'positive': probs[0, 1],
                'attention': words,
            }
        if self._mode in ['bidirectional', 'fast-text']:
            probs = self._sess.run(
                tf.nn.softmax(self._logits), feed_dict = {self._X: batch_x}
            )
            return {'negative': probs[0, 0], 'positive': probs[0, 1]}
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
            return {'negative': probs[0, 0], 'positive': probs[0, 1]}
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
            return {'negative': probs[0, 0], 'positive': probs[0, 1]}

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
        strings = [
            classification_textcleaning_stemmer_attention(i)[0] for i in strings
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

        dicts = []
        for i in range(probs.shape[0]):
            dicts.append({'negative': probs[i, 0], 'positive': probs[i, 1]})
        return dicts


class DEEP_TOXIC:
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

    def predict(self, string):
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
        string = classification_textcleaning_stemmer_attention(string)
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
            dict_result = {}
            for no in range(len(self._label)):
                dict_result[self._label[no]] = probs[0, no]
            dict_result['attention'] = words
            return dict_result
        if self._mode in ['fast-text']:
            probs = self._sess.run(
                tf.nn.softmax(self._logits), feed_dict = {self._X: batch_x}
            )
            dict_result = {}
            for no in range(len(self._label)):
                dict_result[self._label[no]] = probs[0, no]
            return dict_result
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
            dict_result = {}
            for no in range(len(self._label)):
                dict_result[self._label[no]] = probs[0, no]
            return dict_result

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
        strings = [
            classification_textcleaning_stemmer_attention(i)[0] for i in strings
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

        dict_result = {}
        for no in range(len(self._label)):
            dict_result[self._label[no]] = []
            for k in range(probs.shape[0]):
                dict_result[self._label[no]].append(probs[k, no])
        return dict_result
