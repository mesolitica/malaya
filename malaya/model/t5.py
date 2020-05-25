import tensorflow as tf
from herpetologist import check_type
from malaya.text.function import transformer_textcleaning, split_into_sentences
from typing import List
import re


def cleaning(string):
    return re.sub(r'[ ]+', ' ', string).strip()


def remove_repeat_fullstop(string):
    return ' '.join([k.strip() for k in string.split('.') if len(k.strip())])


class SUMMARIZATION:
    def __init__(self, model):
        self._model = model

    def _summarize(self, string, mode):
        string = f'{mode}: {string}'

        return self._model([string])[0].decode('utf-8')

    @check_type
    def summarize(self, string: str, mode: str = 'ringkasan'):
        """
        Summarize a string.

        Parameters
        ----------
        string: str
        mode: str
            mode for summarization. Allowed values:

            * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
            * ``'tajuk'`` - title summarization for long sentence, eg, news title.
            * ``'perenggan'`` - summarization for each perenggan. This will automatically split sentences by EOS.

        Returns
        -------
        result: str
        """
        mode = mode.lower()
        if mode not in ['ringkasan', 'tajuk', 'perenggan']:
            raise ValueError(
                'mode only supports `ringkasan`, `tajuk`, and `perenggan`'
            )

        if mode == 'perenggan':
            splitted_fullstop = split_into_sentences(string)

            results = []
            for splitted in splitted_fullstop:
                if len(splitted.split()) < 8:
                    results.append(splitted)
                else:
                    results.append(self._summarize(splitted, mode).capitalize())
            results = '. '.join(results)

        else:
            results = self._summarize(cleaning(string), mode)

        return results


class GENERATOR:
    def __init__(self, model):
        self._model = model

    @check_type
    def generate(self, strings: List[str]):
        """
        generate a long text given a isi penting.

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
        return self._model([cleaning(points)])[0].decode('utf-8')
