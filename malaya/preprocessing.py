import re
import json
import ftfy
import html
from functools import lru_cache
from unidecode import unidecode
from malaya.text.rules import rules_normalizer
from malaya.text.regex import _expressions, _money
from malaya.text.english.words import words as _english_words
from malaya.path import PATH_PREPROCESSING, S3_PATH_PREPROCESSING
from malaya.function import check_file, validator
from herpetologist import check_type
from typing import Tuple, List

_annotate = [
    'hashtag',
    'allcaps',
    'elongated',
    'repeated',
    'emphasis',
    'censored',
]

_normalize = list(_expressions.keys())


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


def unpack_english_contractions(text):
    """
    Replace *English* contractions in ``text`` str with their unshortened forms.
    N.B. The "'d" and "'s" forms are ambiguous (had/would, is/has/possessive),
    so are left as-is.
    Important Note: The function is taken from textacy (https://github.com/chartbeat-labs/textacy).
    """

    text = re.sub(
        r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't",
        r'\1\2 not',
        text,
    )
    text = re.sub(
        r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll",
        r'\1\2 will',
        text,
    )
    text = re.sub(
        r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r'\1\2 are', text
    )
    text = re.sub(
        r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve",
        r'\1\2 have',
        text,
    )
    text = re.sub(r"(\b)([Cc]a)n't", r'\1\2n not', text)
    text = re.sub(r"(\b)([Ii])'m", r'\1\2 am', text)
    text = re.sub(r"(\b)([Ll]et)'s", r'\1\2 us', text)
    text = re.sub(r"(\b)([Ww])on't", r'\1\2ill not', text)
    text = re.sub(r"(\b)([Ss])han't", r'\1\2hall not', text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r'\1\2ou all', text)
    return text


def _get_expression_dict():
    return {
        k.lower(): re.compile(_expressions[k]) for k, v in _expressions.items()
    }


class Tokenizer:
    def __init__(self, lowercase=False, **kwargs):
        """
        Load Tokenizer object.
        Check supported regex pattern at https://github.com/huseinzol05/Malaya/blob/master/malaya/text/regex.py#L85

        Parameters
        ----------
        lowercase: bool, optional (default=False)
            lowercase tokens.
        emojis: bool, optional (default=True)
            True to keep emojis.
        urls: bool, optional (default=True)
            True to keep urls.
        tags: bool, optional (default=True)
            True to keep tags: <tag>.
        emails: bool, optional (default=True)
            True to keep emails.
        users: bool, optional (default=True)
            True to keep users handles: @cbaziotis.
        hashtags: bool, optional (default=True)
            True to keep hashtags.
        phones: bool, optional (default=True)
            True to keep phones.
        percents: bool, optional (default=True)
            True to keep percents.
        money: bool, optional (default=True)
            True to keep money expressions.
        date: bool, optional (default=True)
            True to keep date expressions.
        time: bool, optional (default=True)
            True to keep time expressions.
        acronyms: bool, optional (default=True)
            True to keep acronyms.
        emoticons: bool, optional (default=True)
            True to keep emoticons.
        censored: bool, optional (default=True)
            True to keep censored words: f**k.
        emphasis: bool, optional (default=True)
            True to keep words with emphasis: *very* good.
        numbers: bool, optional (default=True)
            True to keep numbers.
        temperature: bool, optional (default=True)
            True to keep temperatures
        distance: bool, optional (default=True)
            True to keep distances.
        volume: bool, optional (default=True)
            True to keep volumes.
        duration: bool, optional (default=True)
            True to keep durations.
        weight: bool, optional (default=True)
            True to keep weights.
        hypen: bool, optional (default=True)
            True to keep hypens.
        """

        self.lowercase = lowercase
        pipeline = []
        self.regexes = _expressions

        emojis = kwargs.get('emojis', True)
        urls = kwargs.get('urls', True)
        tags = kwargs.get('tags', True)
        emails = kwargs.get('emails', True)
        users = kwargs.get('users', True)
        hashtags = kwargs.get('hashtags', True)
        cashtags = kwargs.get('cashtags', True)
        phones = kwargs.get('phones', True)
        percents = kwargs.get('percents', True)
        money = kwargs.get('money', True)
        date = kwargs.get('date', True)
        time = kwargs.get('time', True)
        acronyms = kwargs.get('acronyms', True)
        emoticons = kwargs.get('emoticons', True)
        censored = kwargs.get('censored', True)
        emphasis = kwargs.get('emphasis', True)
        numbers = kwargs.get('numbers', True)
        temperatures = kwargs.get('temperature', True)
        distances = kwargs.get('distance', True)
        volumes = kwargs.get('volume', True)
        durations = kwargs.get('duration', True)
        weights = kwargs.get('weight', True)
        hypens = kwargs.get('hypen', True)

        if urls:
            pipeline.append(self.regexes['url'])

        if tags:
            pipeline.append(self.regexes['tag'])

        if emails:
            pipeline.append(self.wrap_non_matching(self.regexes['email']))

        if users:
            pipeline.append(self.wrap_non_matching(self.regexes['user']))

        if hashtags:
            pipeline.append(self.wrap_non_matching(self.regexes['hashtag']))

        if cashtags:
            pipeline.append(self.wrap_non_matching(self.regexes['cashtag']))

        if phones:
            pipeline.append(self.wrap_non_matching(self.regexes['phone']))

        if percents:
            pipeline.append(self.wrap_non_matching(self.regexes['percent']))

        if money:
            pipeline.append(self.wrap_non_matching(self.regexes['money']))

        if date:
            pipeline.append(self.wrap_non_matching(self.regexes['date']))

        if time:
            pipeline.append(self.wrap_non_matching(self.regexes['time']))

        if acronyms:
            pipeline.append(self.wrap_non_matching(self.regexes['acronym']))

        if emoticons:
            pipeline.append(self.regexes['ltr_face'])
            pipeline.append(self.regexes['rtl_face'])

        if censored:
            pipeline.append(self.wrap_non_matching(self.regexes['censored']))

        if emphasis:
            pipeline.append(self.wrap_non_matching(self.regexes['emphasis']))

        if emoticons:
            pipeline.append(
                self.wrap_non_matching(self.regexes['rest_emoticons'])
            )

        if temperatures:
            pipeline.append(self.wrap_non_matching(self.regexes['temperature']))

        if distances:
            pipeline.append(self.wrap_non_matching(self.regexes['distance']))

        if volumes:
            pipeline.append(self.wrap_non_matching(self.regexes['volume']))

        if durations:
            pipeline.append(self.wrap_non_matching(self.regexes['duration']))

        if weights:
            pipeline.append(self.wrap_non_matching(self.regexes['weight']))

        if numbers:
            pipeline.append(self.regexes['number'])

        if emojis:
            pipeline.append(self.regexes['emoji'])

        if hypens:
            pipeline.append(self.regexes['hypen'])

        pipeline.append(self.regexes['word'])

        if emoticons:
            pipeline.append(
                self.wrap_non_matching(self.regexes['eastern_emoticons'])
            )

        # keep repeated puncts as one term
        # pipeline.append(r"")

        pipeline.append('(?:\\S)')  # CATCH ALL remaining terms

        self.tok = re.compile(r'({})'.format('|'.join(pipeline)))

    @staticmethod
    def wrap_non_matching(exp):
        return '(?:{})'.format(exp)

    def tokenize(self, text):
        """
        Tokenize string.

        Parameters
        ----------
        text : str

        Returns
        -------
        result: List[str]
        """
        escaped = html.unescape(text)
        tokenized = self.tok.findall(escaped)

        if self.lowercase:
            tokenized = [t.lower() for t in tokenized]

        return tokenized


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
    ):
        self._fix_unidecode = fix_unidecode
        self._normalize = normalize
        self._annotate = annotate
        self._regexes = _get_expression_dict()
        self._tokenizer = Tokenizer(lowercase=lowercase).tokenize
        self._expand_contractions = expand_english_contractions
        self._all_caps_tag = 'wrap'
        self._translator = translator
        self._speller = speller
        self._segmenter = segmenter
        if self._segmenter:
            self._expand_hashtags = True
        else:
            self._expand_hashtags = False
        self._stemmer = stemmer

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
            expanded = self._segmenter.segment([expanded])[0]
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
            text = self._add_special_tag(text, 'repeated')

        return text

    @lru_cache(maxsize=65536)
    def _handle_generic_match(self, m, tag, mode='every'):
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
            text = self._add_special_tag(text, 'elongated')
        return text

    @lru_cache(maxsize=65536)
    def _handle_emphasis_match(self, m):
        text = m.group().replace('*', '')
        if 'emphasis' in self._annotate:
            text = self._add_special_tag(text, 'emphasis')

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
                    w, 'allcaps', mode=self._all_caps_tag
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
        if self._expand_contractions:
            text = unpack_english_contractions(text)

        text = re.sub(r' +', ' ', text)
        text = self.text(text.split())
        text = ' '.join(text)
        text = self._tokenizer(text)
        text = self._dict_replace(text, rules_normalizer)
        if self._translator:
            text = self._dict_replace(text, self._translator)
        if self._stemmer:
            rejected = ['<', '</', '>', '>']
            text = [
                self._stemmer.stem(w)
                if (
                    w not in _english_words
                    and all([r not in w for r in rejected])
                )
                else w
                for w in text
            ]

        return text


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
    translate_english_to_bm: bool = True,
    speller=None,
    segmenter=None,
    stemmer=None,
    **kwargs,
):
    """
    Load Preprocessing class.

    Parameters
    ----------
    normalize: list
        normalizing tokens, can check all supported normalizing at `malaya.preprocessing.get_normalize()`.
    annotate: list
        annonate tokens <open></open>,
        only accept ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored'].
    lowercase: bool
    fix_unidecode: bool
    expand_english_contractions: bool
        expand english contractions
    translate_english_to_bm: bool
        translate english words to bahasa malaysia words
    speller: object
        spelling correction object, need to have a method `correct`
    segmenter: object
        segmentation object, need to have a method `segment`.
        If provide, it will expand hashtags, #mondayblues == monday blues
    stemmer: object
        stemmer object, need to have a method `stem`.
        If provide, it will stem or lemmatize the string.

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
            "annotate only accept ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored']"
        )
    validator.validate_object_methods(
        speller, ['correct', 'normalize_elongated'], 'speller'
    )
    validator.validate_object(segmenter, 'segment', 'segmenter')
    validator.validate_object(stemmer, 'stem', 'stemmer')

    if translate_english_to_bm:
        check_file(
            PATH_PREPROCESSING['english-malay'],
            S3_PATH_PREPROCESSING['english-malay'],
            **kwargs,
        )
        try:
            with open(PATH_PREPROCESSING['english-malay']['model']) as fopen:
                translator = json.load(fopen)
        except BaseException:
            raise Exception(
                "failed to load english-malay vocab, please run `malaya.utils.delete_cache('preprocessing/english-malay')`")
    else:
        translator = None

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
    )
