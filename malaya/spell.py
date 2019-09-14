from collections import Counter, defaultdict
from .texts._distance import JaroWinkler
import numpy as np
import json
import re
from .texts._text_functions import ENGLISH_WORDS, MALAY_WORDS
from .texts._tatabahasa import (
    alphabet,
    consonants,
    vowels,
    rules_normalizer,
    permulaan,
    hujung,
)
from ._utils._paths import PATH_NGRAM, S3_PATH_NGRAM
from ._utils._utils import check_file, check_available


def _build_dicts(words):
    occurences = {}
    for l in alphabet:
        occurences[l] = defaultdict(lambda: 0)
    for word in words:
        occurences[word[0]][word] += 1
    return occurences


def _augment_vowel(
    string, selected = ['a', 'u', 'i', 'e'], included_end = True
):
    pseudo = []
    if included_end:
        end = string[-1]
    else:
        end = ''
    for c in selected:
        pseudo.append(''.join([w + c for w in string[:-1]] + [end]))
    return pseudo


def _return_possible(word, dicts, edits):
    return set(e2 for e1 in edits(word) for e2 in edits(e1) if e2 in dicts)


def _return_known(word, dicts):
    return set(w for w in word if w in dicts)


class _SPELL:
    def __init__(self, corpus, distancer):
        self._corpus = corpus
        self.WORDS = Counter(self._corpus)
        self.N = sum(self.WORDS.values())
        self._distancer = distancer()

    def correct(self, word, min_distance = 0.9, **kwargs):
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
        if not (isinstance(word, str)):
            raise ValueError('word must be a string')
        if word in ENGLISH_WORDS:
            return word
        if self._corpus.get(word, 0) > 5000:
            return word
        if word in MALAY_WORDS:
            return word
        hujung_result = [v for k, v in hujung.items() if word.endswith(k)]
        if len(hujung_result):
            hujung_result = max(hujung_result, key = len)
            if len(hujung_result):
                word = word[: -len(hujung_result)]
        permulaan_result = [
            v for k, v in permulaan.items() if word.startswith(k)
        ]
        if len(permulaan_result):
            permulaan_result = max(permulaan_result, key = len)
            if len(permulaan_result):
                word = word[len(permulaan_result) :]
        if len(word):
            if word in rules_normalizer:
                word = rules_normalizer[word]
            elif self._corpus.get(word, 0) > 1000:
                pass
            else:
                candidates = (
                    _return_known(_SpellCorrector.edit_step(word), self.WORDS)
                    or _return_possible(
                        word, self.WORDS, _SpellCorrector.edit_step
                    )
                    or {word}
                )
                candidates = {
                    i
                    for i in candidates
                    if len(i) > 3 and i not in ENGLISH_WORDS
                }
                candidates = list(candidates)
                strings = [
                    self._distancer.similarity(word, k) for k in candidates
                ]
                if np.where(np.array(strings) > min_distance)[0].shape[0]:
                    descending_sort = np.argsort(strings)[::-1]
                    word = candidates[descending_sort[0]]
                else:
                    return word
        if len(hujung_result) and not word.endswith(hujung_result):
            word = word + hujung_result
        if len(permulaan_result) and not word.startswith(permulaan_result):
            word = permulaan_result + word
        return word

    def correct_text(self, text):
        """
        Correct all the words within a text, returning the corrected text."""

        if not isinstance(text, str):
            raise ValueError('text must be a string')

        return re.sub('[a-zA-Z]+', self.correct_match, text)

    def correct_match(self, match):
        """
        Spell-correct word in match, and preserve proper upper/lower/title case.
        """

        word = match.group()
        if word[0].isupper():
            return word
        return _SpellCorrector.case_of(word)(self.correct(word.lower()))


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
        inserts = [L + c + R for L, R in splits for c in alphabet]
        pseudo = _augment_vowel(word)
        fuzziness = []
        if len(word):
            # berape -> berapa, mne -> mna
            if word[-1] == 'e':
                inner = word[:-1] + 'a'
                fuzziness.append(inner)
                pseudo.extend(_augment_vowel(inner, included_end = False))
            # pikir -> fikir
            if word[0] == 'p':
                inner = 'f' + word[1:]
                fuzziness.append(inner)
                pseudo.extend(_augment_vowel(inner))

        if len(word) > 2:
            # bapak -> bapa, mintak -> minta, mntak -> mnta
            if word[-2:] == 'ak':
                fuzziness.append(word[:-1])
                pseudo.extend(_augment_vowel(word[:-1], included_end = False))

            if (
                word[0] in consonants
                and word[1] in consonants
                and word[2] in vowels
            ):
                inner = word[0] + word[2] + word[1:]
                fuzziness.append(inner)
            if (
                word[0] in vowels
                and word[-1] in consonants
                and word[2] in consonants
            ):
                inner = word[:-2] + word[0] + word[-1]
                fuzziness.append(inner)

            if (
                word[-1] == 'o'
                and word[-3] in vowels
                and word[-2] in consonants
            ):
                inner = word[:-1] + 'ar'
                fuzziness.append(inner)

            if word[0] == 'a' and word[1] in consonants:
                inner = 'h' + word
                fuzziness.append(inner)
                pseudo.extend(_augment_vowel(inner))

        return set(deletes + transposes + inserts + fuzziness + pseudo)

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

    def edit_candidates(self, word, fast = True):
        """
        Generate possible spelling corrections for word.
        """

        if fast:
            ttt = self.known(self.edit_step(word)) or {word}
        else:
            ttt = (
                self.known(self.edit_step(word))
                or self.known(self.edits2(word))
                or {word}
            )
        ttt = {i for i in ttt if len(i) > 3 and i not in ENGLISH_WORDS}
        ttt = self.known([word]) | ttt
        if not len(ttt):
            ttt = {word}
        return ttt

    def correct(self, word, fast = False, **kwargs):
        """
        Most probable spelling correction for word.
        """
        if not isinstance(word, str):
            raise ValueError('word must be a string')

        if not isinstance(fast, bool):
            raise ValueError('fast must be a boolean')

        if word in ENGLISH_WORDS:
            return word
        if self._corpus.get(word, 0) > 5000:
            return word
        if word in MALAY_WORDS:
            return word
        hujung_result = [v for k, v in hujung.items() if word.endswith(k)]
        if len(hujung_result):
            hujung_result = max(hujung_result, key = len)
            if len(hujung_result):
                word = word[: -len(hujung_result)]
        permulaan_result = [
            v for k, v in permulaan.items() if word.startswith(k)
        ]
        if len(permulaan_result):
            permulaan_result = max(permulaan_result, key = len)
            if len(permulaan_result):
                word = word[len(permulaan_result) :]
        if len(word):
            if word in rules_normalizer:
                word = rules_normalizer[word]
            elif self._corpus.get(word, 0) > 1000:
                pass
            else:
                word = max(
                    self.edit_candidates(word, fast = fast), key = self.P
                )
        if len(hujung_result) and not word.endswith(hujung_result):
            word = word + hujung_result
        if len(permulaan_result) and not word.startswith(permulaan_result):
            word = permulaan_result + word
        return word

    def correct_text(self, text):
        """
        Correct all the words within a text, returning the corrected text."""

        if not isinstance(text, str):
            raise ValueError('text must be a string')

        return re.sub('[a-zA-Z]+', self.correct_match, text)

    def correct_match(self, match):
        """
        Spell-correct word in match, and preserve proper upper/lower/title case.
        """

        word = match.group()
        if word[0].isupper():
            return word
        return self.case_of(word)(self.correct(word.lower()))

    def correct_word(self, word, fast = False):
        """
        Spell-correct word in match, and preserve proper upper/lower/title case.
        """

        return self.case_of(word)(self.correct(word.lower(), fast = fast))

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


def distance(distancer = JaroWinkler, validate = True):
    """
    Train a String matching Spell Corrector.

    Parameters
    ----------
    distancer: object
        string matching object, default is malaya.texts._distance.JaroWinkler
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    _SPELL: Trained malaya.spell._SPELL class
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
    return _SPELL(corpus, distancer = distancer)


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
