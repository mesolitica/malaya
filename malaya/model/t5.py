import tensorflow as tf
from malaya.text.function import (
    transformer_textcleaning,
    split_into_sentences,
    upperfirst,
)
from malaya.text.rouge import (
    filter_rouge,
    postprocessing_summarization,
    find_lapor_and_remove,
)
from malaya.model.abstract import Seq2Seq
from herpetologist import check_type
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


class Summarization(T5, Seq2Seq):
    def __init__(self, X, decode, sess, pred):
        T5.__init__(self, X = X, decode = decode, sess = sess, pred = pred)

    def _summarize(self, string, mode, postprocess, **kwargs):
        summary = upperfirst(self._predict(f'{mode}: {cleaning(string)}'))
        if postprocess and mode != 'tajuk':
            summary = filter_rouge(string, summary, **kwargs)
            summary = postprocessing_summarization(summary)
            summary = find_lapor_and_remove(string, summary)
        return summary

    @check_type
    def greedy_decoder(
        self,
        strings: List[str],
        mode: str = 'ringkasan',
        postprocess: bool = True,
        **kwargs,
    ):
        """
        Summarize strings. Decoder is greedy decoder with beam width size 1, alpha 0.5 .

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
        mode = mode.lower()
        if mode not in ['ringkasan', 'tajuk']:
            raise ValueError('mode only supports [`ringkasan`, `tajuk`]')

        results = []
        for string in strings:
            results.append(self._summarize(string, mode, postprocess, **kwargs))

        return results


class Generator(T5, Seq2Seq):
    def __init__(self, X, decode, sess, pred):
        T5.__init__(self, X = X, decode = decode, sess = sess, pred = pred)

    @check_type
    def greedy_decoder(self, strings: List[str]):
        """
        generate a long text given a isi penting. 
        Decoder is greedy decoder with beam width size 1, alpha 0.5 .

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


class Paraphrase(T5, Seq2Seq):
    def __init__(self, X, decode, sess, pred):
        T5.__init__(self, X = X, decode = decode, sess = sess, pred = pred)

    def _paraphrase(self, string):

        string = f'parafrasa: {cleaning(string)}'
        return upperfirst(self._predict(string))

    @check_type
    def greedy_decoder(self, strings: List[str], split_fullstop: bool = True):
        """
        paraphrase strings. Decoder is greedy decoder with beam width size 1, alpha 0.5 .

        Parameters
        ----------
        strings: List[str]
        split_fullstop: bool, (default=True)
            if True, will generate paraphrase for each strings splitted by fullstop.

        Returns
        -------
        result: List[str]
        """
        results = []

        for string in strings:

            if split_fullstop:

                splitted_fullstop = split_into_sentences(string)

                output = []
                for splitted in splitted_fullstop:
                    if len(splitted.split()) < 4:
                        output.append(splitted)
                    else:
                        output.append(self._paraphrase(splitted))
                r = ' '.join(output)

            else:
                r = self._paraphrase(string)
            results.append(r)

        return results
