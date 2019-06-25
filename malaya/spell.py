import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from collections import Counter, defaultdict
from fuzzywuzzy import fuzz
import numpy as np
from functools import lru_cache
import json
from .texts._text_functions import normalizer_textcleaning, ENGLISH_WORDS
from .similarity import is_location
from .texts._tatabahasa import alphabet, consonants, vowels
from ._utils._paths import PATH_NGRAM, S3_PATH_NGRAM
from ._utils._utils import check_file, check_available


def _build_dicts(words):
    occurences = {}
    for l in alphabet:
        occurences[l] = defaultdict(lambda: 0)
    for word in words:
        occurences[word[0]][word] += 1
    return occurences


def _augment_vowel(string, selected = ['a', 'u', 'i']):
    pseudo = []
    for c in selected:
        pseudo.append(''.join([w + c for w in string[:-1]] + [string[-1]]))
    return pseudo


def _edit_normalizer(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    pseudo = _augment_vowel(word, vowels)
    fuzziness = []
    if len(word):
        if word[-1] == 'e':
            fuzziness.append(word[:-1] + 'a')
    if len(word):
        if word[-2:] == 'ak':
            fuzziness.append(word[:-1])
    if len(word) > 2:
        if (
            word[0] in consonants
            and word[1] in consonants
            and word[2] in vowels
        ):
            inner = word[0] + word[2] + word[1:]
            fuzziness.append(inner)
    if len(word) > 2:
        if (
            word[0] in vowels
            and word[-1] in consonants
            and word[2] in consonants
        ):
            inner = word[:-2] + word[0] + word[-1]
            fuzziness.append(inner)
    if len(word) > 2:
        if word[-1] == 'o' and word[-3] in vowels and word[-2] in consonants:
            inner = word[:-1] + 'ar'
            fuzziness.append(inner)
    if len(word):
        if word[0] == 'a' and word[1] in consonants:
            inner = 'h' + word
            fuzziness.append(inner)
            pseudo.extend(inner)
    return set(deletes + transposes + replaces + inserts + fuzziness + pseudo)


def _return_possible(word, dicts, edits):
    return set(e2 for e1 in edits(word) for e2 in edits(e1) if e2 in dicts)


def _return_known(word, dicts):
    return set(w for w in word if w in dicts)


class _SPELL:
    def __init__(self, corpus):
        self.corpus = corpus
        self.occurences = _build_dicts(self.corpus)
        self.corpus = Counter(corpus)

    def correct(self, string, first_char = True, debug = True, **kwargs):
        """
        Correct a word.

        Parameters
        ----------
        string: str
        first_char: bool, optional (default=True)
            If True, it will only pulled nearest words based on first character, faster but less accurate.
        debug : bool, optional (default=True)
            If true, it will print character similarity distances.

        Returns
        -------
        string: corrected string
        """
        if not (isinstance(string, str)) and not string.count(' '):
            raise ValueError('input must be a single word')
        if not len(string) > 1:
            raise ValueError('input must be long than 1 characters')
        if not isinstance(debug, bool):
            raise ValueError('debug must be a boolean')
        string = normalizer_textcleaning(string)
        if string.istitle():
            return string
        if not len(string):
            return string
        if string in ENGLISH_WORDS:
            return string
        if first_char:
            selected = self.occurences[string[0]]
        else:
            selected = self.corpus
        if len(string) > 2:
            if string[-2] in consonants and string[-1] == 'e':
                string = string[:-1] + 'a'
        candidates = (
            _return_known([string], selected)
            or _return_known(_edit_normalizer(string), selected)
            or _return_possible(string, selected, _edit_normalizer)
            or [string]
        )
        candidates = [
            (candidate, is_location(candidate))
            for candidate in list(candidates)
        ]
        if debug:
            print([(k, fuzz.ratio(string, k[0])) for k in candidates], '\n')
        strings = [fuzz.ratio(string, k[0]) for k in candidates]
        descending_sort = np.argsort(strings)[::-1]
        for index in descending_sort:
            if not candidates[index][1]:
                return candidates[index][0]
        return candidates[descending_sort[0]][0]


class _SpellCorrector:
    """
    The SpellCorrector extends the functionality of the Peter Norvig's
    spell-corrector in http://norvig.com/spell-correct.html
    And improve it using some algorithms from Normalization of noisy texts in Malaysian online reviews,
    https://www.researchgate.net/publication/287050449_Normalization_of_noisy_texts_in_Malaysian_online_reviews
    """

    def __init__(self, corpus):
        self._corpus = corpus
        self.WORDS = Counter(self._corpus)
        self.N = sum(self.WORDS.values())

    @staticmethod
    def tokens(text):
        return REGEX_TOKEN.findall(text.lower())

    def P(self, word):
        """
        Probability of `word`.
        """
        return self.WORDS[word] / self.N

    def most_probable(self, words):
        _known = self.known(words)
        if _known:
            return max(_known, key = self.P)
        else:
            return []

    @staticmethod
    def edit_step(word):
        """
        All edits that are one edit away from `word`.
        """
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in alphabet]
        inserts = [L + c + R for L, R in splits for c in alphabet]
        pseudo = _augment_vowel(word)
        fuzziness = []
        if len(word):
            if word[-1] == 'e':
                fuzziness.append(word[:-1] + 'a')
        if len(word):
            if word[-2:] == 'ak':
                fuzziness.append(word[:-1])
        if len(word) > 2:
            if (
                word[0] in consonants
                and word[1] in consonants
                and word[2] in vowels
            ):
                inner = word[0] + word[2] + word[1:]
                fuzziness.append(inner)
        if len(word) > 2:
            if (
                word[0] in vowels
                and word[-1] in consonants
                and word[2] in consonants
            ):
                inner = word[:-2] + word[0] + word[-1]
                fuzziness.append(inner)
        if len(word) > 2:
            if (
                word[-1] == 'o'
                and word[-3] in vowels
                and word[-2] in consonants
            ):
                inner = word[:-1] + 'ar'
                fuzziness.append(inner)
        if len(word):
            if word[0] == 'a' and word[1] in consonants:
                inner = 'h' + word
                fuzziness.append(inner)
                pseudo.extend(inner)
        return set(
            deletes + transposes + replaces + inserts + fuzziness + pseudo
        )

    def edits2(self, word):
        """
        All edits that are two edits away from `word`.
        """
        return (e2 for e1 in self.edit_step(word) for e2 in self.edit_step(e1))

    def known(self, words):
        """
        The subset of `words` that appear in the dictionary of WORDS.
        """
        return set(w for w in words if w in self.WORDS)

    def edit_candidates(self, word, assume_wrong = False, fast = True):
        """
        Generate possible spelling corrections for word.
        """

        if fast:
            if assume_wrong:
                return self.known(self.edit_step(word)) or [word]
            else:
                return (
                    self.known([word])
                    or self.known(self.edit_step(word))
                    or [word]
                )
        else:
            if assume_wrong:
                ttt = (
                    self.known(self.edit_step(word))
                    or self.known(self.edits2(word))
                    or {word}
                )
                return ttt
            else:
                return (
                    self.known([word])
                    or self.known(self.edit_step(word))
                    or self.known(self.edits2(word))
                    or [word]
                )

    @lru_cache(maxsize = 65536)
    def correct(self, word, assume_wrong = False, fast = False, **kwargs):
        """
        Most probable spelling correction for word.
        """
        return max(
            self.edit_candidates(
                word, assume_wrong = assume_wrong, fast = fast
            ),
            key = self.P,
        )

    def correct_text(self, text):
        """
        Correct all the words within a text, returning the corrected text."""

        return re.sub('[a-zA-Z]+', self.correct_match, text)

    def correct_match(self, match):
        """
        Spell-correct word in match, and preserve proper upper/lower/title case.
        """

        word = match.group()
        return self.case_of(word)(self.correct(word.lower()))

    def correct_word(self, word, assume_wrong = False, fast = False):
        """
        Spell-correct word in match, and preserve proper upper/lower/title case.
        """

        return self.case_of(word)(
            self.correct(word.lower(), assume_wrong = assume_wrong, fast = fast)
        )

    @staticmethod
    def case_of(text):
        """
        Return the case-function appropriate for text: upper, lower, title, or just str.
        """

        return (
            str.upper
            if text.isupper()
            else str.lower
            if text.islower()
            else str.title
            if text.istitle()
            else str
        )

    def elong_normalized_candidates(self, word, acc = None):
        if acc is None:
            acc = []
        candidates = [w for w in set(word) if word.count(w) > 1]
        for c in candidates:
            _w = word.replace(c + c, c)
            if _w in acc:
                continue
            acc.append(_w)
            self.elong_normalized_candidates(_w, acc)
        return acc + [word]

    def best_elong_candidate(self, word):
        candidates = self.elong_normalized_candidates(word)
        best = self.most_probable(candidates)
        return best or word

    def normalize_elongated(self, word):
        return self.case_of(word)(self.best_elong_candidate(word.lower()))


def fuzzy(corpus):
    """
    Train a fuzzy logic Spell Corrector.

    Parameters
    ----------
    corpus : list of strings
        Prefer to feed with malaya.load_malay_dictionary().

    Returns
    -------
    _SPELL: Trained malaya.spell._SPELL class
    """
    if not isinstance(corpus, list):
        raise ValueError('corpus must be a list')
    if not isinstance(corpus[0], str):
        raise ValueError('corpus must be list of strings')
    return _SPELL(corpus)


def probability(validate = True):
    """
    Train a Probability Spell Corrector.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    _SpellCorrector: Trained malaya.spell._SpellCorrector class
    """
    if validate:
        check_file(PATH_NGRAM[1], S3_PATH_NGRAM[1])
    else:
        if not check_available(PATH_NGRAM[1]):
            raise Exception(
                'preprocessing is not available, please `validate = True`'
            )
    with open(PATH_NGRAM[1]['model']) as fopen:
        corpus = json.load(fopen)
    return _SpellCorrector(corpus)
