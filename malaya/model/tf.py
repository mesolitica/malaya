import tensorflow as tf
import numpy as np
import re
from unidecode import unidecode
from malaya.text.function import (
    language_detection_textcleaning,
    split_into_sentences,
    transformer_textcleaning,
    translation_textcleaning,
    pad_sentence_batch,
)
from herpetologist import check_type
from typing import List


def _convert_sparse_matrix_to_sparse_tensor(X, got_limit = False, limit = 5):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    if got_limit:
        coo.data[coo.data > limit] = limit
    return (
        tf.SparseTensorValue(indices, coo.col, coo.shape),
        tf.SparseTensorValue(indices, coo.data, coo.shape),
    )


class _LANG_MODEL:
    def __init__(self, dimension = 32, output = 6):
        self.X = tf.sparse_placeholder(tf.int32)
        self.W = tf.sparse_placeholder(tf.int32)
        self.Y = tf.placeholder(tf.int32, [None])
        embeddings = tf.Variable(tf.truncated_normal([400000, dimension]))
        embed = tf.nn.embedding_lookup_sparse(
            embeddings, self.X, self.W, combiner = 'mean'
        )
        self.logits = tf.layers.dense(embed, output)


class DEEP_LANG:
    def __init__(self, path, vectorizer, label, bpe, type):
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
        self._bpe = bpe
        self._type = type

    def _classify(self, strings):
        strings = [language_detection_textcleaning(i) for i in strings]
        subs = [
            ' '.join(s)
            for s in self._bpe.encode(strings, output_type = self._type)
        ]
        transformed = self._vectorizer.transform(subs)
        batch_x = _convert_sparse_matrix_to_sparse_tensor(transformed)
        probs = self._sess.run(
            self._softmax,
            feed_dict = {self._model.X: batch_x[0], self._model.W: batch_x[1]},
        )
        return probs

    @check_type
    def predict(self, strings: List[str]):
        """
        classify list of strings.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[str]
        """

        probs = self._classify(strings)
        dicts = []
        probs = np.argmax(probs, 1)
        for prob in probs:
            dicts.append(self._label[prob])
        return dicts

    @check_type
    def predict_proba(self, strings: List[str]):
        """
        classify list of strings and return probability.

        Parameters
        ----------
        strings : List[str]


        Returns
        -------
        result: List[dict[str, float]]
        """

        probs = self._classify(strings)
        dicts = []
        for i in range(probs.shape[0]):
            dicts.append({self._label[no]: k for no, k in enumerate(probs[i])})
        return dicts


class PARAPHRASE:
    def __init__(self, X, greedy, beam, sess, tokenizer):

        self._X = X
        self._greedy = greedy
        self._beam = beam
        self._sess = sess
        self._tokenizer = tokenizer

    def _paraphrase(self, strings, beam_search = True):
        encoded = [
            self._tokenizer.encode(translation_textcleaning(string)) + [1]
            for string in strings
        ]
        if beam_search:
            output = self._beam
        else:
            output = self._greedy
        batch_x = pad_sentence_batch(encoded, 0)[0]
        p = self._sess.run(output, feed_dict = {self._X: batch_x}).tolist()
        result = []
        for row in p:
            result.append(
                self._tokenizer.decode([i for i in row if i not in [0, 1]])
            )
        return result

    @check_type
    def paraphrase(
        self, string: str, beam_search: bool = True, split_fullstop: bool = True
    ):
        """
        Paraphrase a string.

        Parameters
        ----------
        string : str
        beam_search : bool, (optional=True)
            If True, use beam search decoder, else use greedy decoder.
        split_fullstop: bool, (default=True)
            if True, will generate paraphrase for each strings splitted by fullstop.

        Returns
        -------
        result: str
        """

        if split_fullstop:

            splitted_fullstop = split_into_sentences(
                transformer_textcleaning(string)
            )

            results, batch, mapping = [], [], {}
            for no, splitted in enumerate(splitted_fullstop):
                if len(splitted.split()) < 4:
                    results.append(splitted)
                else:
                    mapping[len(batch)] = no
                    results.append('REPLACE-ME')
                    batch.append(splitted)

            if len(batch):
                output = self._paraphrase(batch, beam_search = beam_search)
                for no in range(len(output)):
                    results[mapping[no]] = output[no]

            return ' '.join(results)

        else:
            return self._paraphrase([string], beam_search = beam_search)[0]


class TRANSLATION:
    def __init__(self, X, greedy, beam, sess, tokenizer):

        self._X = X
        self._greedy = greedy
        self._beam = beam
        self._sess = sess
        self._tokenizer = tokenizer

    def _translate(self, strings, beam_search = True):
        encoded = [
            self._tokenizer.encode(translation_textcleaning(string)) + [1]
            for string in strings
        ]
        if beam_search:
            output = self._beam
        else:
            output = self._greedy
        batch_x = pad_sentence_batch(encoded, 0)[0]
        p = self._sess.run(output, feed_dict = {self._X: batch_x}).tolist()
        result = []
        for row in p:
            result.append(
                self._tokenizer.decode([i for i in row if i not in [0, 1]])
            )
        return result

    @check_type
    def translate(self, strings: List[str], beam_search: bool = True):
        """
        translate list of strings.

        Parameters
        ----------
        strings : List[str]
        beam_search : bool, (optional=True)
            If True, use beam search decoder, else use greedy decoder.

        Returns
        -------
        result: List[str]
        """
        return self._translate(strings, beam_search = beam_search)
