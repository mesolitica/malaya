import tensorflow as tf
import numpy as np
import re
from malaya.text.function import translation_textcleaning
from malaya.text.rouge import (
    filter_rouge,
    postprocessing_summarization,
    find_lapor_and_remove,
)
from malaya.text.bpe import bert_tokenization
from malaya.model.abstract import Classification, Seq2Seq
from herpetologist import check_type
from typing import List

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences


def cleaning(string):
    return re.sub(r'[ ]+', ' ', string).strip()


class Base:
    def __init__(
        self,
        X,
        logits,
        vectorizer,
        sess,
        tokenizer,
        label = ['negative', 'positive'],
    ):
        self._X = X
        self._logits = logits
        self._vectorizer = vectorizer
        self._sess = sess
        self._tokenizer = tokenizer
        self._label = label
        self._maxlen = 1024


class BigBird(Base):
    def __init__(
        self,
        X,
        logits,
        logits_seq,
        vectorizer,
        sess,
        tokenizer,
        class_name,
        label = ['negative', 'positive'],
    ):

        Base.__init__(
            self,
            X = X,
            logits = logits,
            vectorizer = vectorizer,
            sess = sess,
            tokenizer = tokenizer,
            label = label,
        )
        self._logits_seq = logits_seq
        self._class_name = class_name
        self._softmax = tf.nn.softmax(self._logits)
        self._softmax_seq = tf.nn.softmax(self._logits_seq)

    def _classify(self, strings):
        input_ids, _, _, _ = bert_tokenization(self._tokenizer, strings)
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(
            input_ids, padding = 'post', maxlen = self._maxlen
        )
        return self._sess.run(self._softmax, feed_dict = {self._X: input_ids})

    def _predict(self, strings, add_neutral = False):
        results = self._classify(strings)

        if add_neutral:
            result = neutral(results)
            label = self._label + ['neutral']
        else:
            label = self._label

        return [label[result] for result in np.argmax(results, axis = 1)]

    def _vectorize(self, strings, method = 'first'):
        method = method.lower()
        if method not in ['first', 'last', 'mean', 'word']:
            raise ValueError(
                "method not supported, only support 'first', 'last', 'mean' and 'word'"
            )
        input_ids, _, _, s_tokens = bert_tokenization(self._tokenizer, strings)
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(
            input_ids, padding = 'post', maxlen = self._maxlen
        )
        v = self._sess.run(self._vectorizer, feed_dict = {self._X: input_ids})
        if method == 'first':
            v = v[:, 0]
        elif method == 'last':
            v = v[:, -1]
        elif method == 'mean':
            v = np.mean(v, axis = 1)
        else:
            v = [
                merge_sentencepiece_tokens(
                    list(zip(s_tokens[i], v[i][: len(s_tokens[i])])),
                    weighted = False,
                    vectorize = True,
                )
                for i in range(len(v))
            ]
        return v

    def _predict_proba(self, strings, add_neutral = False):
        results = self._classify(strings)

        if add_neutral:
            results = neutral(results)
            label = self._label + ['neutral']
        else:
            label = self._label

        outputs = []
        for result in results:
            outputs.append({label[i]: result[i] for i in range(len(result))})
        return outputs


class MulticlassBigBird(BigBird, Classification):
    def __init__(
        self,
        X,
        logits,
        logits_seq,
        vectorizer,
        sess,
        tokenizer,
        class_name,
        label = ['negative', 'positive'],
    ):
        BigBird.__init__(
            self,
            X = X,
            logits = logits,
            logits_seq = logits_seq,
            vectorizer = vectorizer,
            sess = sess,
            tokenizer = tokenizer,
            class_name = class_name,
            label = label,
        )

    @check_type
    def vectorize(self, strings: List[str], method: str = 'first'):
        """
        vectorize list of strings.

        Parameters
        ----------
        strings: List[str]
        method : str, optional (default='first')
            Vectorization layer supported. Allowed values:

            * ``'last'`` - vector from last sequence.
            * ``'first'`` - vector from first sequence.
            * ``'mean'`` - average vectors from all sequences.
            * ``'word'`` - average vectors based on tokens.

        Returns
        -------
        result: np.array
        """

        return self._vectorize(strings = strings, method = method)

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

        return self._predict(strings = strings)

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

        return self._predict_proba(strings = strings)


class Translation(Seq2Seq):
    def __init__(self, X, greedy, sess, encoder, maxlen):

        self._X = X
        self._greedy = greedy
        self._sess = sess
        self._encoder = encoder
        self._maxlen = maxlen

    def _translate(self, strings):
        encoded = [
            self._encoder.encode(translation_textcleaning(string)) + [1]
            for string in strings
        ]
        batch_x = pad_sequences(
            encoded, padding = 'post', maxlen = self._maxlen
        )
        p = self._sess.run(self._greedy, feed_dict = {self._X: batch_x})
        result = []
        for r in p:
            result.append(
                self._encoder.decode([i for i in r.tolist() if i > 0])
            )
        return result

    def greedy_decoder(self, strings: List[str]):
        """
        translate list of strings.

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[str]
        """
        return self._translate(strings)


class Summarization(Seq2Seq):
    def __init__(self, X, greedy, temperature, top_p, sess, tokenizer, maxlen):

        self._X = X
        self._greedy = greedy
        self._temperature = temperature
        self._top_p = top_p
        self._sess = sess
        self._tokenizer = tokenizer
        self._maxlen = maxlen

    def _summarize(
        self,
        strings,
        mode,
        top_p = 0.7,
        temperature = 1.0,
        postprocess = True,
        **kwargs,
    ):
        mode = mode.lower()
        if mode not in ['ringkasan', 'tajuk']:
            raise ValueError('mode only supports [`ringkasan`, `tajuk`]')

        strings_ = [f'{mode}: {cleaning(string)}' for string in strings]
        batch_x = [self._tokenizer.encode(string) + [1] for string in strings_]
        batch_x = pad_sequences(
            batch_x, padding = 'post', maxlen = self._maxlen
        )

        p = self._sess.run(
            self._greedy,
            feed_dict = {
                self._X: batch_x,
                self._temperature: temperature,
                self._top_p: top_p,
            },
        ).tolist()
        results = []
        for no, r in enumerate(p):
            summary = self._tokenizer.decode(r)
            if postprocess and mode != 'tajuk':
                summary = filter_rouge(strings[no], summary, **kwargs)
                summary = postprocessing_summarization(summary)
                summary = find_lapor_and_remove(strings[no], summary)

            results.append(summary)

        return results

    def greedy_decoder(
        self,
        strings: List[str],
        mode: str = 'ringkasan',
        postprocess: bool = True,
        **kwargs,
    ):
        """
        Summarize strings using greedy decoder.

        Parameters
        ----------
        strings: List[str]
        mode: str
            mode for summarization. Allowed values:

            * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
            * ``'tajuk'`` - title summarization for long sentence, eg, news title.
        postprocess: bool, optional (default=True)
            If True, will filter sentence generated using ROUGE score and removed international news publisher.

        Returns
        -------
        result: List[str]
        """
        return self._summarize(
            strings = strings,
            mode = mode,
            top_p = 0.0,
            temperature = 0.0,
            postprocess = postprocess,
            **kwargs,
        )

    def nucleus_decoder(
        self,
        strings: List[str],
        mode: str = 'ringkasan',
        top_p: float = 0.7,
        temperature: float = 1.0,
        postprocess: bool = True,
        **kwargs,
    ):
        """
        Summarize strings using nucleus decoder.

        Parameters
        ----------
        strings: List[str]
        mode: str
            mode for summarization. Allowed values:

            * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
            * ``'tajuk'`` - title summarization for long sentence, eg, news title.
        top_p: float, (default=0.7)
            cumulative distribution and cut off as soon as the CDF exceeds `top_p`.
        temperature: float, (default=1.0)
            logits * -log(random.uniform) * temperature.
        postprocess: bool, optional (default=True)
            If True, will filter sentence generated using ROUGE score and removed international news publisher.

        Returns
        -------
        result: List[str]
        """
        return self._summarize(
            strings = strings,
            mode = mode,
            top_p = top_p,
            temperature = temperature,
            postprocess = postprocess,
            **kwargs,
        )
