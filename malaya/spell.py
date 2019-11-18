from collections import Counter, defaultdict
from itertools import product
import numpy as np
import json
import re
from .texts._jarowrinkler import JaroWinkler
from .texts._text_functions import ENGLISH_WORDS, MALAY_WORDS
from .texts._tatabahasa import (
    alphabet,
    consonants,
    vowels,
    rules_normalizer,
    permulaan,
    hujung,
    stopword_tatabahasa,
    quad_vowels,
    group_compound,
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


def _get_indices(string, c = 'a'):
    return [i for i in range(len(string)) if string[i] == c]


def _permutate(string, indices):
    p = [''.join(_set) for _set in product(list(vowels), repeat = len(indices))]
    p = [p_ for p_ in p if not all([a in p_ for a in quad_vowels])]
    mutate = []
    for p_ in p:
        s = list(string[:])
        for i in range(len(indices)):
            s[indices[i]] = p_[i]
        mutate.append(''.join(s))
    return mutate


def _augment_vowel_alternate(string):
    r = []
    # a flag to not duplicate
    last_time = False
    for i, c in enumerate(string[:-1], 1):
        last = i - 2
        if last < 0:
            last = 0

        # we only want to put a vowel after consonant if next that consonant if not a wovel
        if c in consonants and string[i] not in vowels:
            if c + string[i] in group_compound and not last_time:
                r.append(c + string[i])
                last_time = True
            elif string[last] + c in group_compound and not last_time:
                r.append(string[last] + c)
                last_time = True
            else:
                last_time = False
                if len(r):
                    # ['ng'] gg
                    if (
                        r[-1] in group_compound
                        and c + string[i] == r[-1][-1] * 2
                    ):
                        r.append('^')
                        continue
                    elif r[-1] in group_compound and c == r[-1][-1]:
                        if c + string[i] in group_compound:
                            continue
                        else:
                            r.append('a')
                            continue
                r.append(c + 'a')

        else:
            if len(r):
                if r[-1] in group_compound and c == r[-1][-1]:
                    continue
            r.append(c)

    if r[-1][-1] in vowels and string[-1] in consonants:
        r.append(string[-1])

    elif (
        r[-1] in group_compound
        and string[-2] in vowels
        and string[-1] in consonants
    ):
        r.append(string[-2:])

    left = ''.join(r).replace('^', '')
    right = left + 'a'
    return left, right


def _augment_vowel_prob(word):
    l, r = _augment_vowel_alternate(word)
    return list(
        set(_permutate(l, _get_indices(l)) + _permutate(r, _get_indices(r)))
    )


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


class _SpellCorrector:
    """
    The SpellCorrector extends the functionality of the Peter Norvig's
    spell-corrector in http://norvig.com/spell-correct.html
    And improve it using some algorithms from Normalization of noisy texts in Malaysian online reviews,
    https://www.researchgate.net/publication/287050449_Normalization_of_noisy_texts_in_Malaysian_online_reviews
    Added custom vowels augmentation
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
        pseudo.extend(_augment_vowel_prob(word))
        fuzziness = []
        if len(word):

            # berape -> berapa, mne -> mna
            if word[-1] == 'e':
                inner = word[:-1] + 'a'
                fuzziness.append(inner)
                pseudo.extend(_augment_vowel_prob(inner))

            # pikir -> fikir
            if word[0] == 'p':
                inner = 'f' + word[1:]
                fuzziness.append(inner)
                pseudo.extend(_augment_vowel_prob(inner))

        if len(word) > 2:
            # bapak -> bapa, mintak -> minta, mntak -> mnta
            if word[-2:] == 'ak':
                fuzziness.append(word[:-1])
                pseudo.extend(_augment_vowel_prob(word[:-1]))

            # hnto -> hantar, bako -> bkar, sabo -> sabar
            if (
                word[-1] == 'o'
                and word[-3] in vowels
                and word[-2] in consonants
            ):
                inner = word[:-1] + 'ar'
                fuzziness.append(inner)
                pseudo.extend(_augment_vowel_prob(inner))

            # antu -> hantu, antar -> hantar
            if word[0] == 'a' and word[1] in consonants:
                inner = 'h' + word
                fuzziness.append(inner)
                pseudo.extend(_augment_vowel(inner))
                pseudo.extend(_augment_vowel_prob(inner))

            # ptg -> ptng, dtg -> dtng
            if (
                word[-3] in consonants
                and word[-2] in consonants
                and word[-1] == 'g'
            ):
                inner = word[:-1] + 'ng'
                fuzziness.append(inner)
                pseudo.extend(_augment_vowel_prob(inner))

            # igt -> ingt
            if word[1] == 'g' and word[2] in consonants:
                inner = word[0] + 'n' + word[1:]
                fuzziness.append(inner)
                pseudo.extend(_augment_vowel_prob(inner))

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

    def edit_candidates(self, word):
        """
        Generate possible spelling corrections for word.
        """

        ttt = self.known(self.edit_step(word)) or {word}
        ttt = {i for i in ttt if len(i) > 3 and i not in ENGLISH_WORDS}
        ttt = self.known([word]) | ttt
        if not len(ttt):
            ttt = {word}
        return ttt

    def correct(self, word, **kwargs):
        """
        Most probable spelling correction for word.
        """
        if not isinstance(word, str):
            raise ValueError('word must be a string')

        if word in ENGLISH_WORDS:
            return word
        if self._corpus.get(word, 0) > 5000:
            return word
        if word in MALAY_WORDS:
            return word
        if word in stopword_tatabahasa:
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
                word = max(self.edit_candidates(word), key = self.P)
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

    def correct_word(self, word):
        """
        Spell-correct word in match, and preserve proper upper/lower/title case.
        """

        return self.case_of(word)(self.correct(word.lower()))

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


class _SymspellCorrector:
    """
    The SymspellCorrector extends the functionality of symspeller, https://github.com/mammothb/symspellpy
    And improve it using some algorithms from Normalization of noisy texts in Malaysian online reviews,
    https://www.researchgate.net/publication/287050449_Normalization_of_noisy_texts_in_Malaysian_online_reviews
    Added custom vowels augmentation
    """

    def __init__(self, model, verbosity, corpus, k = 10):
        self._model = model
        self._verbosity = verbosity
        self._corpus = corpus
        self.k = k

    def predict(self, word):
        max_edit_distance_lookup = 2
        suggestion_verbosity = self._verbosity
        suggestions = self._model.lookup(
            word, suggestion_verbosity, max_edit_distance_lookup
        )[: self.k]
        return suggestions

    def edit_step(self, word):
        result = list(_augment_vowel_alternate(word))

        if len(word):

            # berape -> berapa, mne -> mna
            if word[-1] == 'e':
                inner = word[:-1] + 'a'
                result.extend(list(_augment_vowel_alternate(inner)))

            # pikir -> fikir
            if word[0] == 'p':
                inner = 'f' + word[1:]
                result.extend(list(_augment_vowel_alternate(inner)))

        if len(word) > 2:
            # bapak -> bapa, mintak -> minta, mntak -> mnta
            if word[-2:] == 'ak':
                fuzziness.append(word[:-1])
                result.extend(list(_augment_vowel_alternate(word[:-1])))

            # hnto -> hantar, bako -> bkar, sabo -> sabar
            if (
                word[-1] == 'o'
                and word[-3] in vowels
                and word[-2] in consonants
            ):
                inner = word[:-1] + 'ar'
                result.extend(list(_augment_vowel_alternate(inner)))

            # antu -> hantu, antar -> hantar
            if word[0] == 'a' and word[1] in consonants:
                inner = 'h' + word
                result.extend(list(_augment_vowel_alternate(inner)))

            # ptg -> ptng, dtg -> dtng
            if (
                word[-3] in consonants
                and word[-2] in consonants
                and word[-1] == 'g'
            ):
                inner = word[:-1] + 'ng'
                result.extend(list(_augment_vowel_alternate(inner)))

            # igt -> ingt
            if word[1] == 'g' and word[2] in consonants:
                inner = word[0] + 'n' + word[1:]
                result.extend(list(_augment_vowel_alternate(inner)))

        words = {}
        for r in result:
            suggestions = self.predict(r)
            for s in suggestions:
                words[s.term] = words.get(s.term, 0) + (
                    s.count / (s.distance + 1)
                )

        return words

    def edit_candidates(self, word):
        ttt = self.edit_step(word) or {word: 10}
        ttt = {
            k: v
            for k, v in ttt.items()
            if len(k) > 3 and k not in ENGLISH_WORDS
        }
        ttt[word] = ttt.get(word, 0) + 10
        if not len(ttt):
            ttt = {word: 10}
        return ttt

    def correct(self, word, **kwargs):
        """
        Most probable spelling correction for word.
        """
        if not isinstance(word, str):
            raise ValueError('word must be a string')

        if word in ENGLISH_WORDS:
            return word
        if self._corpus.get(word, 0) > 5000:
            return word
        if word in MALAY_WORDS:
            return word
        if word in stopword_tatabahasa:
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
            else:
                stats = self.edit_candidates(word)
                word = max(stats, key = stats.get)
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


def probability(validate = True):
    """
    Train a Probability Spell Corrector.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    _SpellCorrector: malaya.spell._SpellCorrector class
    """

    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

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


def symspell(
    validate = True,
    max_edit_distance_dictionary = 2,
    prefix_length = 7,
    term_index = 0,
    count_index = 1,
    top_k = 10,
):
    """
    Train a symspell Spell Corrector.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    _SpellCorrector: malaya.spell._SymspellCorrector class
    """
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')
    if not isinstance(max_edit_distance_dictionary, int):
        raise ValueError('max_edit_distance_dictionary must be an integer')
    if not isinstance(prefix_length, int):
        raise ValueError('prefix_length must be an integer')
    if not isinstance(term_index, int):
        raise ValueError('term_index must be an integer')
    if not isinstance(count_index, int):
        raise ValueError('count_index must be an integer')

    if validate:
        check_file(PATH_NGRAM['symspell'], S3_PATH_NGRAM['symspell'])
        check_file(PATH_NGRAM[1], S3_PATH_NGRAM[1])
    else:
        if not check_available(PATH_NGRAM['symspell']):
            raise Exception(
                'preprocessing is not available, please `validate = True`'
            )
        if not check_available(PATH_NGRAM[1]):
            raise Exception(
                'preprocessing is not available, please `validate = True`'
            )
    try:
        from symspellpy.symspellpy import SymSpell, Verbosity
    except:
        raise Exception(
            'symspellpy not installed. Please install it and try again.'
        )
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    dictionary_path = PATH_NGRAM['symspell']['model']
    sym_spell.load_dictionary(dictionary_path, term_index, count_index)
    with open(PATH_NGRAM[1]['model']) as fopen:
        corpus = json.load(fopen)
    return _SymspellCorrector(sym_spell, Verbosity.ALL, corpus, k = top_k)
