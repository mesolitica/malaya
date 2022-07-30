import re
import json
import ftfy
from functools import lru_cache
from malaya.text.rules import rules_normalizer
from malaya.text.regex import _expressions
from malaya.text.english.words import words as _english_words
from malaya.text.normalization import unpack_english_contractions
from malaya.tokenizer import Tokenizer
from malaya.function import validator
from typing import List, Callable
import logging

logger = logging.getLogger(__name__)

_annotate = [
    'hashtag',
    'allcaps',
    'elongated',
    'repeated',
    'emphasis',
    'censored',
]

_normalize = list(_expressions.keys())

rejected = ['<', '</', '>', '>']


def get_normalize():
    return _normalize


def get_annotate():
    return _annotate


def _case_of(text):
    return (
        str.upper
        if text.isupper()
        else str.lower
        if text.islower()
        else str.title
        if text.istitle()
        else str
    )


def _get_expression_dict():
    return {
        k.lower(): re.compile(_expressions[k]) for k, v in _expressions.items()
    }


class Preprocessing:
    def __init__(
        self,
        normalize=[
            'url',
            'email',
            'percent',
            'money',
            'phone',
            'user',
            'time',
            'date',
            'number',
        ],
        annotate=[
            'allcaps',
            'elongated',
            'repeated',
            'emphasis',
            'censored',
            'hashtag',
        ],
        lowercase=True,
        fix_unidecode=True,
        expand_english_contractions=True,
        translator=None,
        speller=None,
        segmenter=None,
        stemmer=None,
        demoji=None,
    ):
        self._fix_unidecode = fix_unidecode
        self._normalize = normalize
        self._annotate = annotate
        self._regexes = _get_expression_dict()
        self._tokenizer = Tokenizer(lowercase=lowercase).tokenize
        self._expand_contractions = expand_english_contractions
        self._translator = translator
        self._speller = speller
        self._segmenter = segmenter
        if self._segmenter:
            self._expand_hashtags = True
        else:
            self._expand_hashtags = False
        self._stemmer = stemmer
        self._demoji = demoji

    def _add_special_tag(self, m, tag, mode='single'):

        if isinstance(m, str):
            text = m
        else:
            text = m.group()

        if mode == 'single':
            return ' {} <{}> '.format(text, tag)
        elif mode == 'wrap':
            return ' '.join([' <{}> {} </{}> '.format(tag, text, tag)]) + ' '
        elif mode == 'every':
            tokens = text.split()
            processed = ' '.join([' {} <{}> '.format(t, tag) for t in tokens])
            return ' ' + processed + ' '

    @lru_cache(maxsize=65536)
    def _handle_hashtag_match(self, m):
        expanded = m.group()[1:]
        if self._expand_hashtags:
            expanded = self._segmenter(expanded)
            expanded = ' '.join(expanded.split('-'))
            expanded = ' '.join(expanded.split('_'))

        if 'hashtag' in self._annotate:
            expanded = self._add_special_tag(expanded, 'hashtag', mode='wrap')

        return expanded

    @lru_cache(maxsize=65536)
    def _handle_repeated_puncts(self, m):
        text = m.group()
        text = ''.join(sorted(set(text), reverse=True))

        if 'repeated' in self._annotate:
            text = self._add_special_tag(text, 'repeated', mode='wrap')

        return text

    @lru_cache(maxsize=65536)
    def _handle_generic_match(self, m, tag, mode='wrap'):
        text = m.group()
        text = self._add_special_tag(text, tag, mode=mode)

        return text

    def _handle_elongated_match(self, m):
        text = m.group()
        text = self._regexes['normalize_elong'].sub(r'\1\1', text)
        if self._speller and text.lower() not in _english_words:
            if hasattr(self._speller, 'normalize_elongated'):
                text = _case_of(text)(
                    self._speller.normalize_elongated(text.lower())
                )
            else:
                text = _case_of(text)(self._speller.correct(text.lower()))
        if 'elongated' in self._annotate:
            text = self._add_special_tag(text, 'elongated', mode='wrap')
        return text

    @lru_cache(maxsize=65536)
    def _handle_emphasis_match(self, m):
        text = m.group().replace('*', '')
        if 'emphasis' in self._annotate:
            text = self._add_special_tag(text, 'emphasis', mode='wrap')

        return text

    @lru_cache(maxsize=65536)
    def _handle_emphasis_emoji(self, m):
        text = m.group().replace('*', '')
        text = self._add_special_tag(text, 'emoji', mode='wrap')

        return text

    def _dict_replace(self, wordlist, _dict):
        return [_dict.get(w, w) for w in wordlist]

    @staticmethod
    def text(wordlist):
        in_hashtag = False
        _words = []
        for word in wordlist:
            if word == '<hashtag>':
                in_hashtag = True
            elif word == '</hashtag>':
                in_hashtag = False
            elif word in {'<allcaps>', '</allcaps>'} and in_hashtag:
                continue

            _words.append(word)

        return _words

    def process(self, text):
        logger.debug(f'early process: {text}')
        text = re.sub(r' +', ' ', text)
        if self._fix_unidecode:
            text = ftfy.fix_text(text)

        for item in self._normalize:
            text = self._regexes[item].sub(
                lambda m: ' ' + '<' + item + '>' + ' ', text
            )

        text = self._regexes['hashtag'].sub(
            lambda w: self._handle_hashtag_match(w), text
        )

        if 'allcaps' in self._annotate:
            text = self._regexes['allcaps'].sub(
                lambda w: self._handle_generic_match(
                    w, 'allcaps', mode='wrap'
                ),
                text,
            )
        if 'elongated' in self._annotate:
            text = self._regexes['elongated'].sub(
                lambda w: self._handle_elongated_match(w), text
            )
        if 'repeated' in self._annotate:
            text = self._regexes['repeat_puncts'].sub(
                lambda w: self._handle_repeated_puncts(w), text
            )
        if 'emphasis' in self._annotate:
            text = self._regexes['emphasis'].sub(
                lambda w: self._handle_emphasis_match(w), text
            )
        if 'censored' in self._annotate:
            text = self._regexes['censored'].sub(
                lambda w: self._handle_generic_match(w, 'censored'), text
            )
        if self._demoji is not None:
            text = self._demoji._compiled.sub(
                lambda w: self._handle_generic_match(w, 'emoji'), text
            )
        if self._expand_contractions:
            text = unpack_english_contractions(text)

        logger.debug(f'before self._tokenizer: {text}')
        text = re.sub(r' +', ' ', text)
        text = self.text(text.split())
        text = ' '.join(text)
        text = self._tokenizer(text)
        logger.debug(f'after self._tokenizer: {text}')

        logger.debug(f'before rules_normalizer: {text}')
        text = self._dict_replace(text, rules_normalizer)
        logger.debug(f'after rules_normalizer: {text}')
        if self._translator:
            logger.debug(f'before self._translator: {text}')
            text = [
                self._translator(w)
                if all([r not in w for r in rejected])
                else w
                for w in text
            ]
            logger.debug(f'after self._translator: {text}')
        if self._stemmer:
            logger.debug(f'before self._stemmer: {text}')

            text = [
                self._stemmer(w)
                if (
                    w not in _english_words
                    and all([r not in w for r in rejected])
                )
                else w
                for w in text
            ]
            logger.debug(f'after self._stemmer: {text}')

        text = [w for w in text if len(w) > 0]
        return text


class Demoji:
    def __init__(self, dictionary):
        self._dictionary = dictionary
        escp = (re.escape(c) for c in sorted(self._dictionary, key=len, reverse=True))
        self._compiled = re.compile(r'|'.join(escp))

    def demoji(self, string: str):
        """
        Find emojis with string representation.
        🔥 -> emoji api.

        Parameters
        ----------
        string: str

        Returns
        -------
        result: Dist[str]
        """

        results = self._compiled.findall(string)
        return {r: self._dictionary[r]['ms'] for r in results}


def preprocessing(
    normalize: List[str] = [
        'url',
        'email',
        'percent',
        'money',
        'phone',
        'user',
        'time',
        'date',
        'number',
    ],
    annotate: List[str] = [
        'allcaps',
        'elongated',
        'repeated',
        'emphasis',
        'censored',
        'hashtag',
    ],
    lowercase: bool = True,
    fix_unidecode: bool = True,
    expand_english_contractions: bool = True,
    translator: Callable = None,
    segmenter: Callable = None,
    stemmer: Callable = None,
    speller: Callable = None,
    demoji: Callable = None,
    **kwargs,
):
    """
    Load Preprocessing class.

    Parameters
    ----------
    normalize: List[str], optional (default=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'])
        normalizing tokens, can check all supported normalizing at `malaya.preprocessing.get_normalize()`.
    annotate: List[str], optional (default=['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored'])
        annonate tokens <open></open>,
        only accept ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored'].
    lowercase: bool, optional (default=True)
    fix_unidecode: bool, optional (default=True)
        fix unidecode using `ftfy.fix_text`.
    expand_english_contractions: bool, optional (default=True)
        expand english contractions.
    translator: Callable, optional (default=None)
        function to translate EN word to MS word.
    segmenter: Callable, optional (default=None)
        function to segmentize word.
        If provide, it will expand hashtags, #mondayblues == monday blues
    stemmer: Callable, optional (default=None)
        function to stem word.
    speller: object
        spelling correction object, need to have a method `correct` or `normalize_elongated`.
    demoji: object
        demoji object, need to have a method `demoji`.

    Returns
    -------
    result : malaya.preprocessing.Preprocessing class
    """

    if any([e not in _normalize for e in normalize]):
        raise ValueError(
            'normalize element not able to recognize, supported normalization can check at get_normalize()'
        )
    if any([e not in _annotate for e in annotate]):
        raise ValueError(
            f"annotate only accept {str(_annotate)}"
        )
    validator.validate_object_methods(
        speller, ['correct', 'normalize_elongated'], 'speller'
    )
    validator.validate_object_methods(
        demoji, ['demoji'], 'demoji'
    )

    return Preprocessing(
        normalize=normalize,
        annotate=annotate,
        lowercase=lowercase,
        fix_unidecode=fix_unidecode,
        expand_english_contractions=expand_english_contractions,
        translator=translator,
        speller=speller,
        segmenter=segmenter,
        stemmer=stemmer,
        demoji=demoji,
    )


def demoji():
    """
    Download latest emoji malay description from https://github.com/huseinzol05/malay-dataset/tree/master/dictionary/emoji

    Returns
    -------
    result : malaya.preprocessing.Demoji class
    """
    try:
        import requests
    except BaseException:
        raise ModuleNotFoundError(
            'requests not installed. Please install it by `pip3 install requests` and try again.'
        )
    r = requests.get('https://raw.githubusercontent.com/huseinzol05/malay-dataset/master/dictionary/emoji/demoji.json').json()
    return Demoji(dictionary=r)
