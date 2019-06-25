import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import numpy as np
import json
import re
from fuzzywuzzy import fuzz
from unidecode import unidecode
from ._utils._utils import (
    load_graph,
    check_file,
    check_available,
    generate_session,
)
from .texts._text_functions import (
    normalizer_textcleaning,
    stemmer_str_idx,
    pad_sentence_batch,
    ENGLISH_WORDS,
)
from .texts._tatabahasa import (
    rules_normalizer,
    consonants,
    vowels,
    sounds,
    GO,
    PAD,
    EOS,
    UNK,
    hujung_malaysian,
    calon_dictionary,
)
from .num2word import to_cardinal
from .word2num import word2num
from .preprocessing import _tokenizer

ignore_words = ['ringgit', 'sen']
ignore_postfix = ['adalah']


def _remove_postfix(word):
    if word in ignore_postfix:
        return word, ''
    for p in hujung_malaysian:
        if word.endswith(p):
            return word[: -len(p)], ' lah'
    return word, ''


def _normalize_ke(word):
    # kesebelas -> ke-sebelas
    # ke-21 -> ke-dua puluh satu
    if word.startswith('ke'):
        original = word
        word = word.replace('-', '')
        word = word.split('ke')
        try:
            num = word2num(word[1])
        except:
            pass
        try:
            num = int(word[1])
        except:
            return original
        return 'ke-' + to_cardinal(num)
    return word


def _normalize_title(word):
    if word.istitle() or word.isupper():
        return calon_dictionary.get(word, word)
    return word


def _is_number_regex(s):
    if re.match('^\d+?\.\d+?$', s) is None:
        return s.isdigit()
    return True


def _string_to_num(word):
    if '.' in word:
        return float(word)
    else:
        return int(word)


def _normalized_money(word):
    original = word
    word = word.lower()
    if word[:2] == 'rm' and _is_number_regex(word[2:]):
        return to_cardinal(_string_to_num(word[2:])) + ' ringgit'
    elif word[-3:] == 'sen':
        return to_cardinal(_string_to_num(word[:-3])) + ' sen'
    else:
        return original


class _SPELL_NORMALIZE:
    def __init__(self, speller):
        self._speller = speller

    def normalize(self, string, assume_wrong = True, check_english = True):
        """
        Normalize a string

        Parameters
        ----------
        string : str
        assume_wrong: bool, (default=True)
            force speller to predict.
        check_english: bool, (default=True)
            check a word in english dictionary.

        Returns
        -------
        string: normalized string
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(check_english, bool):
            raise ValueError('check_english must be a boolean')
        if not isinstance(assume_wrong, bool):
            raise ValueError('assume_wrong must be a boolean')

        result = []
        tokenized = _tokenizer(string)
        index = 0
        while index < len(tokenized):
            word = tokenized[index]
            if len(word) < 2 and word not in sounds:
                result.append(word)
                index += 1
                continue
            if word.lower() in ignore_words:
                result.append(word)
                index += 1
                continue
            if word.istitle() or word.isupper():
                result.append(_normalize_title(word))
                index += 1
                continue
            if check_english:
                if word.lower() in ENGLISH_WORDS:
                    result.append(word)
                    index += 1
                    continue
            if len(word) > 2:
                if word[-2] in consonants and word[-1] == 'e':
                    word = word[:-1] + 'a'
            if word[0] == 'x' and len(word) > 1:
                result_string = 'tak '
                word = word[1:]
            else:
                result_string = ''
            if word.lower() == 'ke' and index < (len(tokenized) - 2):
                if tokenized[index + 1] == '-' and _is_number_regex(
                    tokenized[index + 2]
                ):
                    result.append(
                        _normalize_ke(
                            word + tokenized[index + 1] + tokenized[index + 2]
                        )
                    )
                    index += 3
                    continue
            normalized_ke = _normalize_ke(word)
            if normalized_ke != word:
                result.append(normalized_ke)
                index += 1
                continue
            if _is_number_regex(word) and index < (len(tokenized) - 2):
                if tokenized[index + 1] == '-' and _is_number_regex(
                    tokenized[index + 2]
                ):
                    result.append(
                        to_cardinal(_string_to_num(word))
                        + ' hingga '
                        + to_cardinal(_string_to_num(tokenized[index + 2]))
                    )
                    index += 3
                    continue
            if word.lower() == 'pada' and index < (len(tokenized) - 3):
                if (
                    _is_number_regex(tokenized[index + 1])
                    and tokenized[index + 2] in '/-'
                    and _is_number_regex(tokenized[index + 3])
                ):
                    result.append(
                        'pada %s hari bulan %s'
                        % (
                            to_cardinal(_string_to_num(tokenized[index + 1])),
                            to_cardinal(_string_to_num(tokenized[index + 3])),
                        )
                    )
                    index += 4
                    continue
            money = _normalized_money(word)
            if money != word:
                result.append(money)
                index += 1
                continue

            word, end_result_string = _remove_postfix(word)
            if word in sounds:
                result.append(result_string + sounds[word] + end_result_string)
                index += 1
                continue
            if word in rules_normalizer:
                result.append(
                    result_string + rules_normalizer[word] + end_result_string
                )
                index += 1
                continue
            selected = self._speller.correct(
                word, debug = False, assume_wrong = assume_wrong
            )
            result.append(result_string + selected + end_result_string)
            index += 1
        return ' '.join(result)


class _FUZZY_NORMALIZE:
    def __init__(self, normalized, corpus):
        self.normalized = normalized
        self.corpus = corpus

    def normalize(self, string, fuzzy_ratio = 70, check_english = True):
        """
        Normalize a string

        Parameters
        ----------
        string : str
        fuzzy_ratio: int, (default=70)
            ratio of similar characters by positions, if 90, means 90%.
        check_english: bool, (default=True)
            check a word in english dictionary.

        Returns
        -------
        string: normalized string
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(check_english, bool):
            raise ValueError('check_english must be a boolean')
        if not isinstance(fuzzy_ratio, int):
            raise ValueError('fuzzy_ratio must be an integer')

        result = []
        tokenized = _tokenizer(string)
        index = 0
        while index < len(tokenized):
            word = tokenized[index]
            if len(word) < 2 and word not in sounds:
                result.append(word)
                index += 1
                continue
            if word.lower() in ignore_words:
                result.append(word)
                index += 1
                continue
            if word.istitle() or word.isupper():
                result.append(_normalize_title(word))
                index += 1
                continue
            if check_english:
                if word.lower() in ENGLISH_WORDS:
                    result.append(word)
                    index += 1
                    continue
            if len(word) > 2:
                if word[-2] in consonants and word[-1] == 'e':
                    word = word[:-1] + 'a'
            if word[0] == 'x' and len(word) > 1:
                result_string = 'tak '
                word = word[1:]
            else:
                result_string = ''
            if word.lower() == 'ke' and index < (len(tokenized) - 2):
                if tokenized[index + 1] == '-' and _is_number_regex(
                    tokenized[index + 2]
                ):
                    result.append(
                        _normalize_ke(
                            word + tokenized[index + 1] + tokenized[index + 2]
                        )
                    )
                    index += 3
                    continue
            normalized_ke = _normalize_ke(word)
            if normalized_ke != word:
                result.append(normalized_ke)
                index += 1
                continue
            if _is_number_regex(word) and index < (len(tokenized) - 2):
                if tokenized[index + 1] == '-' and _is_number_regex(
                    tokenized[index + 2]
                ):
                    result.append(
                        to_cardinal(_string_to_num(word))
                        + ' hingga '
                        + to_cardinal(_string_to_num(tokenized[index + 2]))
                    )
                    index += 3
                    continue
            if word.lower() == 'pada' and index < (len(tokenized) - 3):
                if (
                    _is_number_regex(tokenized[index + 1])
                    and tokenized[index + 2] in '/-'
                    and _is_number_regex(tokenized[index + 3])
                ):
                    result.append(
                        'pada %s hari bulan %s'
                        % (
                            to_cardinal(_string_to_num(tokenized[index + 1])),
                            to_cardinal(_string_to_num(tokenized[index + 3])),
                        )
                    )
                    index += 4
                    continue
            money = _normalized_money(word)
            if money != word:
                result.append(money)
                index += 1
                continue

            word, end_result_string = _remove_postfix(word)
            if word in sounds:
                result.append(result_string + sounds[word] + end_result_string)
                index += 1
                continue
            if word in rules_normalizer:
                result.append(
                    result_string + rules_normalizer[word] + end_result_string
                )
                index += 1
                continue
            results = []
            for i in range(len(self.normalized)):
                results.append(
                    np.mean([fuzz.ratio(word, k) for k in self.normalized[i]])
                )
            if len(np.where(np.array(results) > fuzzy_ratio)[0]) < 1:
                result.append(result_string + word + end_result_string)
                continue
            result.append(
                result_string
                + self.corpus[np.argmax(results)]
                + end_result_string
            )
            index += 1
        return ' '.join(result)


def fuzzy(corpus):
    """
    Train a fuzzy logic Normalizer

    Parameters
    ----------
    corpus : list of strings. Prefer to feed with malaya.load_malay_dictionary().

    Returns
    -------
    _FUZZY_NORMALIZE: Trained malaya.normalizer._FUZZY_NORMALIZE class
    """
    if not isinstance(corpus, list):
        raise ValueError('corpus must be a list')
    if not isinstance(corpus[0], str):
        raise ValueError('corpus must be list of strings')
    corpus = [unidecode(w) for w in corpus]
    transform = []
    for i in corpus:
        i = i.lower()
        result = []
        result.append(i)
        result.append(''.join(char for char in i if char not in vowels))
        if i[0] in consonants and i[-1] in consonants:
            inner = i[0] + i[-1]
            result.append(inner)
        if i[-1] == 'a':
            inner = i[:-1]
            result.append(i[:-1] + 'e')
            inner = i + 'k'
            result.append(inner)
        if i[1] in vowels and i[0] in consonants:
            inner = i[0] + i[2:]
            result.append(inner)
        if i[-2] in vowels and i[-1] in consonants:
            inner = i[:-2] + i[-1]
            result.append(inner)
        result.append(i[0] + i[-1])
        if i[-2:] == 'ar':
            result.append(i[:-2] + 'o')
        if i[:2] == 'ha':
            result.append(i[1:])
        result = filter(None, result)
        transform.append(list(set(result)))
    return _FUZZY_NORMALIZE(transform, corpus)


def spell(speller):
    """
    Train a Spelling Normalizer

    Parameters
    ----------
    speller : Malaya spelling correction object

    Returns
    -------
    _SPELL_NORMALIZE: malaya.normalizer._SPELL_NORMALIZE class
    """
    if not hasattr(speller, 'correct') and not hasattr(
        speller, 'normalize_elongated'
    ):
        raise ValueError(
            'speller must has `correct` or `normalize_elongated` method'
        )
    return _SPELL_NORMALIZE(speller)


def basic(string):
    """
    Use basic rules-based to normalize a string.

    Parameters
    ----------
    string: str

    Returns
    -------
    string: normalized string
    """
    if not isinstance(string, str):
        ValueError('input must be a string')
    result = []
    for word in normalizer_textcleaning(string).split():
        if word.istitle():
            result.append(word)
            continue
        if word in sounds:
            result.append(sounds[word])
        elif word[-1] == '2':
            result.append(word[:-1])
        else:
            result.append(word)
    return ' '.join(result)
