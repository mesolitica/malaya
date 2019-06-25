import re
import json
import ftfy
import html
import itertools
from functools import lru_cache
from math import log10
from unidecode import unidecode
from .texts._tatabahasa import _expressions, rules_normalizer, hujung
from .texts._english_words import _english_words
from ._utils._paths import PATH_PREPROCESSING, S3_PATH_PREPROCESSING
from ._utils._utils import check_file, check_available
from .stem import naive

_annotate = [
    'hashtag',
    'allcaps',
    'elongated',
    'repeated',
    'emphasis',
    'censored',
]

_normalize = list(_expressions.keys())

REGEX_TOKEN = re.compile(r'\b[a-z]{2,}\b')
NGRAM_SEP = '_'


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


def _is_number_regex(s):
    if re.match('^\d+?\.\d+?$', s) is None:
        return s.isdigit()
    return True


def _detect_money(word):
    if word[:2] == 'rm' and _is_number_regex(word[2:]):
        return True
    else:
        return False


def _naive_stem(word):
    hujung_result = [e for e in hujung if word.endswith(e)]
    if len(hujung_result):
        hujung_result = max(hujung_result, key = len)
        if len(hujung_result):
            word = word[: -len(hujung_result)]
    return word


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


class SocialTokenizer:
    def __init__(self, lowercase = False, **kwargs):
        """
        Args:
            lowercase (bool): set to True in order to lowercase the text
        Kwargs ():
            emojis (bool): True to keep emojis
            urls (bool): True to keep urls
            tags (bool): True to keep tags: <tag>
            emails (bool): True to keep emails
            users (bool): True to keep users handles: @cbaziotis
            hashtags (bool): True to keep hashtags
            cashtags (bool): True to keep cashtags
            phones (bool): True to keep phones
            percents (bool): True to keep percents
            money (bool): True to keep money expressions
            date (bool): True to keep date expressions
            time (bool): True to keep time expressions
            acronyms (bool): True to keep acronyms
            emoticons (bool): True to keep emoticons
            censored (bool): True to keep censored words: f**k
            emphasis (bool): True to keep words with emphasis: *very* good
            numbers (bool): True to keep numbers
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

        if numbers:
            pipeline.append(self.regexes['number'])

        if emojis:
            pipeline.append(self.regexes['emoji'])

        # any other word
        pipeline.append(self.regexes['word'])

        if emoticons:
            pipeline.append(
                self.wrap_non_matching(self.regexes['eastern_emoticons'])
            )

        # keep repeated puncts as one term
        # pipeline.append(r"")

        pipeline.append('(?:\S)')  # CATCH ALL remaining terms

        self.tok = re.compile(r'({})'.format('|'.join(pipeline)))

    @staticmethod
    def wrap_non_matching(exp):
        return '(?:{})'.format(exp)

    def tokenize(self, text):
        escaped = html.unescape(text)
        tokenized = self.tok.findall(escaped)

        if self.lowercase:
            tokenized = [t.lower() for t in tokenized]

        return tokenized


def _read_stats(gram = 1):
    with open(PATH_PREPROCESSING[gram]['model']) as fopen:
        return json.load(fopen)


class _Pdist(dict):
    @staticmethod
    def default_unk_func(key, total):
        return 1.0 / total

    def __init__(self, data = None, total = None, unk_func = None, **kwargs):
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


class _Segmenter:
    def __init__(self, max_split_length = 20):
        self.unigrams = _read_stats(1)
        self.bigrams = _read_stats(2)
        self.N = sum(self.unigrams.values())
        self.L = max_split_length

        self.Pw = _Pdist(self.unigrams, self.N, self.unk_probability)
        self.P2w = _Pdist(self.bigrams, self.N)

        self.case_split = _get_expression_dict()['camel_split']

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
            (text[: i + 1], text[i + 1 :])
            for i in range(min(len(text), self.L))
        ]

    @lru_cache(maxsize = 65536)
    def find_segment(self, text, prev = '<S>'):
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

    @lru_cache(maxsize = 65536)
    def segment(self, word):
        if word.islower():
            return ' '.join(self.find_segment(word)[1])
        else:
            return self.case_split.sub(r' \1', word).lower()


class _Preprocessing:
    def __init__(
        self,
        normalize = [
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
        annotate = [
            'allcaps',
            'elongated',
            'repeated',
            'emphasis',
            'censored',
            'hashtag',
        ],
        lowercase = True,
        fix_unidecode = True,
        expand_hashtags = True,
        expand_english_contractions = True,
        remove_postfix = True,
        maxlen_segmenter = 20,
        translator = None,
        speller = None,
    ):
        self._fix_unidecode = fix_unidecode
        self._normalize = normalize
        self._annotate = annotate
        self._remove_postfix = remove_postfix
        self._regexes = _get_expression_dict()
        self._expand_hashtags = expand_hashtags
        self._tokenizer = SocialTokenizer(lowercase = lowercase).tokenize
        if self._expand_hashtags:
            self._segmenter = _Segmenter(maxlen_segmenter)
        self._expand_contractions = expand_english_contractions
        self._all_caps_tag = 'wrap'
        self._translator = translator
        self._speller = speller

    def _add_special_tag(self, m, tag, mode = 'single'):

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

    @lru_cache(maxsize = 65536)
    def _handle_hashtag_match(self, m):
        expanded = m.group()[1:]
        if self._expand_hashtags:
            if expanded.islower():
                expanded = self._segmenter.segment(expanded)
                expanded = ' '.join(expanded.split('-'))
                expanded = ' '.join(expanded.split('_'))

            else:
                expanded = self._regexes['camel_split'].sub(r' \1', expanded)
                expanded = expanded.replace('-', '')
                expanded = expanded.replace('_', '')

        if 'hashtag' in self._annotate:
            expanded = self._add_special_tag(expanded, 'hashtag', mode = 'wrap')

        return expanded

    @lru_cache(maxsize = 65536)
    def _handle_repeated_puncts(self, m):
        text = m.group()
        text = ''.join(sorted(set(text), reverse = True))

        if 'repeated' in self._annotate:
            text = self._add_special_tag(text, 'repeated')

        return text

    @lru_cache(maxsize = 65536)
    def _handle_generic_match(self, m, tag, mode = 'every'):
        text = m.group()
        text = self._add_special_tag(text, tag, mode = mode)

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
                text = _case_of(text)(
                    self._speller.correct(text.lower(), debug = False)
                )
        if 'elongated' in self._annotate:
            text = self._add_special_tag(text, 'elongated')
        return text

    @lru_cache(maxsize = 65536)
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
                    w, 'allcaps', mode = self._all_caps_tag
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
        if self._remove_postfix:
            text = [
                _naive_stem(w) if w not in _english_words else w for w in text
            ]

        return text


def preprocessing(
    normalize = [
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
    annotate = [
        'allcaps',
        'elongated',
        'repeated',
        'emphasis',
        'censored',
        'hashtag',
    ],
    lowercase = True,
    fix_unidecode = True,
    expand_hashtags = True,
    expand_english_contractions = True,
    translate_english_to_bm = True,
    remove_postfix = True,
    maxlen_segmenter = 20,
    validate = True,
    speller = None,
):
    """
    Load Preprocessing class.

    Parameters
    ----------
    normalize: list
        normalizing tokens, can check all supported normalizing at malaya.preprocessing.get_normalize()
    annotate: list
        annonate tokens <open></open>, only accept ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored']
    lowercase: bool
    fix_unidecode: bool
    expand_hashtags: bool
        expand hashtags using Viterbi algorithm, #mondayblues == monday blues
    expand_english_contractions: bool
        expand english contractions
    translate_english_to_bm: bool
        translate english words to bahasa malaysia words
    remove_postfix: bool
        remove postfix from a word, faster way to get root word
    speller: object
        spelling correction object, need to have a method `correct`
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.


    Returns
    -------
    _Preprocessing : malaya.preprocessing._Preprocessing class
    """

    if any([e not in _normalize for e in normalize]):
        raise ValueError(
            'normalize element not able to recognize, supported normalization can check at get_normalize()'
        )
    if any([e not in _annotate for e in annotate]):
        raise ValueError(
            "annotate only accept ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored']"
        )
    if not isinstance(lowercase, bool):
        raise ValueError('lowercase must be a boolean')
    if not isinstance(fix_unidecode, bool):
        raise ValueError('fix_unidecode must be a boolean')
    if not isinstance(expand_hashtags, bool):
        raise ValueError('expand_hashtags must be a boolean')
    if not isinstance(expand_english_contractions, bool):
        raise ValueError('expand_english_contractions must be a boolean')
    if not isinstance(translate_english_to_bm, bool):
        raise ValueError('translate_english_to_bm must be a boolean')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')
    if not isinstance(remove_postfix, bool):
        raise ValueError('remove_postfix must be a boolean')
    if speller is not None:
        if not hasattr(speller, 'correct') and not hasattr(
            speller, 'normalize_elongated'
        ):
            raise ValueError(
                'speller must has `correct` or `normalize_elongated` method'
            )

    if expand_hashtags:
        if validate:
            check_file(PATH_PREPROCESSING[1], S3_PATH_PREPROCESSING[1])
        else:
            if not check_available(PATH_PREPROCESSING[1]):
                raise Exception(
                    'preprocessing is not available, please `validate = True`'
                )
        if validate:
            check_file(PATH_PREPROCESSING[2], S3_PATH_PREPROCESSING[2])
        else:
            if not check_available(PATH_PREPROCESSING[2]):
                raise Exception(
                    'preprocessing is not available, please `validate = True`'
                )

    if translate_english_to_bm:
        if validate:
            check_file(
                PATH_PREPROCESSING['english-malay'],
                S3_PATH_PREPROCESSING['english-malay'],
            )
        else:
            if not check_available(PATH_PREPROCESSING['english-malay']):
                raise Exception(
                    'translator english-malay is not available, please `validate = True`'
                )

        with open(PATH_PREPROCESSING['english-malay']['model']) as fopen:
            translator = json.load(fopen)
    else:
        translator = None

    return _Preprocessing(
        normalize = normalize,
        annotate = annotate,
        lowercase = lowercase,
        fix_unidecode = fix_unidecode,
        expand_hashtags = expand_hashtags,
        expand_english_contractions = expand_english_contractions,
        remove_postfix = remove_postfix,
        maxlen_segmenter = maxlen_segmenter,
        translator = translator,
        speller = speller,
    )


def segmenter(max_split_length = 20, validate = True):
    """
    Load Segmenter class.

    Parameters
    ----------
    max_split_length: int, (default=20)
        max length of words in a sentence to segment
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    _Segmenter : malaya.preprocessing._Segmenter class
    """

    if validate:
        check_file(PATH_PREPROCESSING[1], S3_PATH_PREPROCESSING[1])
    else:
        if not check_available(PATH_PREPROCESSING[1]):
            raise Exception(
                'preprocessing is not available, please `validate = True`'
            )
    if validate:
        check_file(PATH_PREPROCESSING[2], S3_PATH_PREPROCESSING[2])
    else:
        if not check_available(PATH_PREPROCESSING[2]):
            raise Exception(
                'preprocessing is not available, please `validate = True`'
            )
    return _Segmenter(max_split_length = max_split_length)


_tokenizer = SocialTokenizer().tokenize
_rejected = ['wkwk', 'http', 'https', 'lolol', 'hahaha']


def preprocessing_classification_index(string):
    string = ''.join(
        ''.join(s)[:2] for _, s in itertools.groupby(unidecode(string))
    )
    tokenized = _tokenizer(string)
    tokenized_dict = {no: w for no, w in enumerate(tokenized)}
    tokenized_indices = {w: -1 for w in tokenized}
    tokenized = [(no, naive(w)) for no, w in enumerate(tokenized)]
    tokenized = [(w[0], w[1].lower()) for w in tokenized if len(w[1]) > 1]
    tokenized = [
        w for w in tokenized if all([r not in w[1] for r in _rejected])
    ]
    tokenized = [(w[0], rules_normalizer.get(w[1], w[1])) for w in tokenized]
    tokenized = [
        (w[0], '<NUM>') if _is_number_regex(w[1]) else w for w in tokenized
    ]
    tokenized = [
        (w[0], '<MONEY>') if _detect_money(w[1]) else w for w in tokenized
    ]
    indices, words = zip(*tokenized)
    for no, index in enumerate(indices):
        tokenized_indices[tokenized_dict[index]] = no
    return tokenized_indices, words
