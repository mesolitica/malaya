import tensorflow as tf
from herpetologist import check_type
from malaya.text.function import (
    postprocessing_summarization,
    transformer_textcleaning,
    split_into_sentences,
    upperfirst,
)
from typing import List
import re


def cleaning(string):
    return re.sub(r'[ ]+', ' ', string).strip()


def remove_repeat_fullstop(string):
    return ' '.join([k.strip() for k in string.split('.') if len(k.strip())])


class T5:
    def __init__(self, X, decode, sess, pred):
        self._X = X
        self._decode = decode
        self._sess = sess
        self._pred = pred

    def _predict(self, string):
        if self._pred:
            r = self._pred([string])[0].decode('utf-8')
        else:
            r = self._sess.run(self._decode, feed_dict = {self._X: [string]})[
                0
            ].decode('utf-8')
        return r


class SUMMARIZATION(T5):
    def __init__(self, X, decode, sess, pred):
        T5.__init__(self, X = X, decode = decode, sess = sess, pred = pred)

    def _summarize(self, string, mode):
        string = f'{mode}: {cleaning(string)}'
        return postprocessing_summarization(upperfirst(self._predict(string)))

    @check_type
    def summarize(self, string: str, mode: str = 'ringkasan'):
        """
        Summarize a string. Decoder is beam decoder with beam width size 1, alpha 0.5 .

        Parameters
        ----------
        string: str
        mode: str
            mode for summarization. Allowed values:

            * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
            * ``'tajuk'`` - title summarization for long sentence, eg, news title.

        Returns
        -------
        result: str
        """
        mode = mode.lower()
        if mode not in ['ringkasan', 'tajuk']:
            raise ValueError('mode only supports [`ringkasan`, `tajuk`]')

        results = self._summarize(string, mode)

        return results


class GENERATOR(T5):
    def __init__(self, X, decode, sess, pred):
        T5.__init__(self, X = X, decode = decode, sess = sess, pred = pred)

    @check_type
    def generate(self, strings: List[str]):
        """
        generate a long text given a isi penting. Decoder is beam decoder with beam width size 1, alpha 0.5 .

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: str
        """

        points = [
            f'{no + 1}. {remove_repeat_fullstop(string)}.'
            for no, string in enumerate(strings)
        ]
        points = ' '.join(points)
        points = f'karangan: {points}'
        return upperfirst(self._predict(cleaning(points)))


class PARAPHRASE(T5):
    def __init__(self, X, decode, sess, pred):
        T5.__init__(self, X = X, decode = decode, sess = sess, pred = pred)

    def _paraphrase(self, string):

        string = f'parafrasa: {cleaning(string)}'
        return upperfirst(self._predict(string))

    @check_type
    def paraphrase(self, string: str, split_fullstop: bool = True):
        """
        paraphrase a string. Decoder is beam decoder with beam width size 1, alpha 0.5 .

        Parameters
        ----------
        string: str
        split_fullstop: bool, (default=True)
            if True, will generate paraphrase for each strings splitted by fullstop.

        Returns
        -------
        result: str
        """

        if split_fullstop:

            splitted_fullstop = split_into_sentences(string)

            results = []
            for splitted in splitted_fullstop:
                if len(splitted.split()) < 4:
                    results.append(splitted)
                else:
                    results.append(self._paraphrase(splitted))
            return ' '.join(results)

        else:
            return self._paraphrase(string)
