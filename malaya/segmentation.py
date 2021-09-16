import json
import re
from functools import lru_cache
from math import log10
from malaya.text.regex import _expressions
from malaya.model.tf import Segmentation
from malaya.path import PATH_PREPROCESSING, S3_PATH_PREPROCESSING
from malaya.supervised import transformer as load_transformer
from malaya.function import check_file
from malaya.supervised import t5 as t5_load
from malaya.model.t5 import Segmentation as T5_Segmentation
from herpetologist import check_type
from typing import List

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.1,
        'WER': 0.208520,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 63.8,
        'WER': 0.1776236,
        'Suggested length': 256,
    },
    'super-tiny-t5': {
        'Size (MB)': 81.8,
        'Quantized Size (MB)': 27.1,
        'WER': 0.03298,
        'Suggested length': 256,
    },
    'super-super-tiny-t5': {
        'Size (MB)': 39.6,
        'Quantized Size (MB)': 12,
        'WER': 0.037882,
        'Suggested length': 256,
    }
}

REGEX_TOKEN = re.compile(r'\b[a-z]{2,}\b')
NGRAM_SEP = '_'


def _read_stats(gram=1):
    try:
        with open(PATH_PREPROCESSING[gram]['model']) as fopen:
            return json.load(fopen)
    except Exception as e:
        raise Exception(
            f"{e}, file corrupted due to some reasons, please run `malaya.clear_cache('preprocessing')` and try again"
        )


class _Pdist(dict):
    @staticmethod
    def default_unk_func(key, total):
        return 1.0 / total

    def __init__(self, data=None, total=None, unk_func=None, **kwargs):
        super().__init__(**kwargs)

        data = data or {}
        for key, count in data.items():
            self[key] = self.get(key, 0) + int(count)

        self.total = float(total or sum(self.values()))
        self.unk_prob = unk_func or self.default_unk_func

    def __call__(self, key):
        if key in self:
            return self[key] / self.total
        else:
            return self.unk_prob(key, self.total)


class Segmenter:
    def __init__(self, max_split_length=20):
        self.unigrams = _read_stats(1)
        self.bigrams = _read_stats(2)
        self.N = sum(self.unigrams.values())
        self.L = max_split_length

        self.Pw = _Pdist(self.unigrams, self.N, self.unk_probability)
        self.P2w = _Pdist(self.bigrams, self.N)

        self.case_split = re.compile(_expressions['camel_split'])

    def condProbWord(self, word, prev):
        try:
            return self.P2w[prev + NGRAM_SEP + word] / float(self.Pw[prev])
        except KeyError:
            return self.Pw(word)

    @staticmethod
    def unk_probability(key, total):
        return 10.0 / (total * 10 ** len(key))

    @staticmethod
    def combine(first, rem):
        (first_prob, first_word) = first
        (rem_prob, rem_words) = rem
        return first_prob + rem_prob, [first_word] + rem_words

    def splits(self, text):
        return [
            (text[: i + 1], text[i + 1:])
            for i in range(min(len(text), self.L))
        ]

    @lru_cache(maxsize=65536)
    def find_segment(self, text, prev='<S>'):
        if not text:
            return 0.0, []
        candidates = [
            self.combine(
                (log10(self.condProbWord(first, prev)), first),
                self.find_segment(rem, first),
            )
            for first, rem in self.splits(text)
        ]
        return max(candidates)

    @lru_cache(maxsize=65536)
    def _segment(self, word):
        if word.islower():
            return ' '.join(self.find_segment(word)[1])
        else:
            return self.case_split.sub(r' \1', word)

    @check_type
    def segment(self, strings: List[str]):
        """
        Segment strings.
        Example, "sayasygkan negarasaya" -> "saya sygkan negara saya"

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[str]
        """
        results = []
        for string in strings:
            string = string.split()
            result = []
            for word in string:
                result.append(self._segment(word))
            results.append(' '.join(result))
        return results


def viterbi(max_split_length: int = 20, **kwargs):
    """
    Load Segmenter class using viterbi algorithm.

    Parameters
    ----------
    max_split_length: int, (default=20)
        max length of words in a sentence to segment
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    result : malaya.segmentation.Segmenter class
    """

    check_file(PATH_PREPROCESSING[1], S3_PATH_PREPROCESSING[1], **kwargs)
    check_file(PATH_PREPROCESSING[2], S3_PATH_PREPROCESSING[2], **kwargs)
    return Segmenter(max_split_length=max_split_length)


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(_transformer_availability)


@check_type
def transformer(model: str = 'small', quantized: bool = False, **kwargs):
    """
    Load transformer encoder-decoder model to Segmentize.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'small'`` - Transformer SMALL parameters.
        * ``'base'`` - Transformer BASE parameters.
        * ``'super-tiny-t5'`` - T5 SUPER TINY parameters.
        * ``'super-super-tiny-t5'`` - T5 SUPER SUPER TINY parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.Segmentation class
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.segmentation.available_transformer()`.'
        )

    if 't5' in model:
        return t5_load.load(
            module='segmentation',
            model=model,
            model_class=T5_Segmentation,
            quantized=quantized,
            **kwargs,
        )
    else:
        return load_transformer.load(
            module='segmentation',
            model=model,
            encoder='yttm',
            model_class=Segmentation,
            quantized=quantized,
            **kwargs,
        )
