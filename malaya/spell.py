import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from collections import Counter, defaultdict
from fuzzywuzzy import fuzz
import numpy as np
from .texts._text_functions import normalizer_textcleaning
from .topic_influencer import is_location
from .texts._tatabahasa import alphabet, consonants, vowels


def _build_dicts(words):
    occurences = {}
    for l in alphabet:
        occurences[l] = defaultdict(lambda: 0)
    for word in words:
        occurences[word[0]][word] += 1
    return occurences


def _edit_normalizer(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    pseudo = []
    for c in vowels:
        pseudo.append(''.join([w + c for w in word]))
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
            fuzziness.append(word[0] + word[2] + word[1:])
    if len(word) > 2:
        if (
            word[0] in vowels
            and word[-1] in consonants
            and word[2] in consonants
        ):
            fuzziness.append(word[:-2] + word[0] + word[-1])
    if len(word) > 2:
        if word[-1] == 'o' and word[-3] in vowels and word[-2] in consonants:
            fuzziness.append(word[:-1] + 'ar')
    if len(word):
        if word[0] == 'a' and word[1] in consonants:
            fuzziness.append('h' + word)
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

    def correct(self, string, first_char = True, debug = True):
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
        assert (isinstance(string, str)) and not string.count(
            ' '
        ), 'input must be a single word'
        assert len(string) > 1, 'input must be long than 1 characters'
        string = normalizer_textcleaning(string)
        if string.istitle():
            return string
        if not len(string):
            return string
        if first_char:
            candidates = (
                _return_known([string], self.occurences[string[0]])
                or _return_known(
                    _edit_normalizer(string), self.occurences[string[0]]
                )
                or _return_possible(
                    string, self.occurences[string[0]], _edit_normalizer
                )
                or [string]
            )
        else:
            candidates = (
                _return_known([string], self.corpus)
                or _return_known(_edit_normalizer(string), self.corpus)
                or _return_possible(string, self.corpus, _edit_normalizer)
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


def naive(corpus):
    """
    Train a fuzzy logic Spell Corrector

    Parameters
    ----------
    corpus : list of strings. Prefer to feed with malaya.load_malay_dictionary()

    Returns
    -------
    SPELL: Trained malaya.spell._SPELL class
    """
    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    return _SPELL(corpus)
