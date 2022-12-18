from malaya.path import PATH_AUGMENTATION, S3_PATH_AUGMENTATION
from malaya.augmentation.base import _make_upper, _replace
from malaya.text.tatabahasa import consonants, vowels
from malaya.text.function import case_of
from malaya.function import check_file
from herpetologist import check_type
from collections import defaultdict
import random
import json
from typing import Callable, Dict, List

_synonym_dict = None


def replace_synonym(string, threshold):
    for no, word in enumerate(string):
        if word in _synonym_dict and random.random() > threshold:
            w = random.choice(_synonym_dict[word])
            string[no] = w
    return string


@check_type
def synonym(
    string: str,
    threshold: float = 0.5,
    top_n=5,
    **kwargs
):
    """
    augmenting a string using synonym, https://github.com/huseinzol05/Malaya-Dataset#90k-synonym

    Parameters
    ----------
    string: str
        this string input assumed been properly tokenized and cleaned.
    threshold: float, optional (default=0.5)
        random selection for a word.
    top_n: int, (default=5)
        number of nearest neighbors returned. Length of returned result should as top_n.

    Returns
    -------
    result: List[str]
    """

    global _synonym_dict

    if _synonym_dict is None:
        path = check_file(
            PATH_AUGMENTATION['synonym'],
            S3_PATH_AUGMENTATION['synonym'],
            **kwargs
        )
        files = list(path.values())
        synonyms = defaultdict(list)
        for file in files:
            with open(file) as fopen:
                data = json.load(fopen)

            for i in data:
                if not len(i[1]):
                    continue
                synonyms[i[0]].extend(i[1])
                for r in i[1]:
                    synonyms[r].append(i[0])
        for k, v in synonyms.items():
            synonyms[k] = list(set(v))
        _synonym_dict = synonyms

    original_string = string
    string = string.split()

    augmented = []
    for i in range(top_n):
        string_ = replace_synonym(string, threshold)
        augmented.append(
            _make_upper(' '.join(string_), ' '.join(original_string))
        )
    return augmented


def replace_similar_consonants(
    word: str,
    threshold: float = 0.5,
    replace_consonants: Dict[str, List[str]] = {
        'n': ['m'],
        'r': ['t', 'q'],
        'g': ['h'],
        'j': ['k'],
        'k': ['l'],
        'd': ['s', 'f'],
        'g': ['f', 'h'],
        'b': ['n'],
        'f': ['p'],
    }
):
    """
    Naively replace consonants with another consonants to simulate typo or slang
    if after consonants is a vowel.

    Parameters
    ----------
    word: str
    threshold: float, optional (default=0.5)

    Returns
    -------
    result: List[str]
    """
    results = list(word)
    for no, c in enumerate(results[:-1]):
        if random.random() >= threshold and c in consonants and results[no + 1] in vowels:
            results[no] = random.choice(replace_consonants.get(c, [c]))

    if random.random() >= threshold and results[-1] in consonants and results[-2] in vowels and results[-3] in consonants:
        results[-1] = random.choice(replace_consonants.get(results[-1], [results[-1]]))

    return ''.join(results)


def replace_similar_vowels(
    word: str,
    threshold: float = 0.5,
    replace_vowels: Dict[str, List[str]] = {
        'u': ['o'],
        'a': ['o'],
        'i': ['o'],
        'o': ['u'],
    }
):
    """
    Naively replace vowels with another vowels to simulate typo or slang
    if after vowels is a consonant.

    Parameters
    ----------
    word: str
    threshold: float, optional (default=0.5)

    Returns
    -------
    result: str
    """

    results = list(word)
    for no, c in enumerate(results[:-1]):
        if random.random() >= threshold and c in vowels and results[no + 1] in consonants:
            results[no] = random.choice(replace_vowels.get(c, [c]))

    if random.random() >= threshold and results[-1] in vowels and results[-2] in consonants and results[-3] in vowels:
        results[-1] = random.choice(replace_vowels.get(results[-1], [results[-1]]))

    return ''.join(results)


@check_type
def socialmedia_form(word: str):
    """
    augmenting a word into socialmedia form.

    Parameters
    ----------
    word: str

    Returns
    -------
    result: List[str]
    """
    word_temp = word
    word = word.lower()

    if not len(word):
        raise ValueError('word is too short to augment shortform.')

    results = []

    if len(word) > 1:

        if word[-1] == 'a' and word[-2] in consonants:
            results.append(word[:-1] + 'e')

        if word[0] == 'f' and word[-1] == 'r':
            results.append('p' + word[1:])

        if word[-2] in consonants and word[-1] in vowels:
            results.append(word + 'k')

        if word[-2] in vowels and word[-1] == 'h':
            results.append(word[:-1])

    if len(word) > 2:
        if word[-3] in consonants and word[-2:] == 'ar':
            results.append(word[:-2] + 'o')

        if word[0] == 'h' and word[1] in vowels and word[2] in consonants:
            results.append(word[1:])

        if word[-3] in consonants and word[-2:] == 'ng':
            results.append(word[:-2] + 'g')

        if word[1:3] == 'ng':
            results.append(word[:1] + x[2:])

    results = list(set(results))
    results = [case_of(word_temp)(r) for r in results]
    return results


def vowel_alternate(word: str, threshold: float = 0.5):
    """
    augmenting a word into vowel alternate.

    vowel_alternate('singapore')
    -> sngpore

    vowel_alternate('kampung')
    -> kmpng

    vowel_alternate('ayam')
    -> aym

    Parameters
    ----------
    word: str
    threshold: float, optional (default=0.5)

    Returns
    -------
    result: str
    """
    word_temp = word
    word = word.lower()

    if not len(word):
        raise ValueError('word is too short to augment shortform.')

    word = list(word[:])
    i = 0
    while i < len(word) - 2:
        subword = word[i: i + 3]
        if subword[0] in consonants and subword[1] in vowels and subword[2] in consonants \
                and random.random() >= threshold:
            word.pop(i + 1)
        i += 1

    return case_of(word_temp)(''.join(word))


@check_type
def kelantanese_form(word: str):
    """
    augmenting a word into kelantanese form.
    `ayam` -> `ayom`
    `otak` -> `otok`
    `kakak` -> `kakok`

    `barang` -> `bare`
    `kembang` -> `kembe`
    `nyarang` -> `nyare`

    Parameters
    ----------
    word: str

    Returns
    -------
    result: List[str]
    """
    word_temp = word
    word = word.lower()

    if not len(word):
        raise ValueError('word is too short to augment shortform.')

    results = []
    if len(word) == 3:
        if word[0] in consonants and word[1] in 'a' and word[2] in consonants:
            results.append(word[0] + 'o' + word[2])

    if len(word) >= 4:
        if word[-1] in 'ao' and word[-2] in consonants and word[-3] in 'ae':
            results.append(word[:-1] + 'o')

        if word[-1] in consonants and word[-2] in 'au' and word[-3] in consonants and word[-4] in 'aou':
            results.append(word[:-2] + 'o' + word[-1])

        if word[-3:] == 'ang' and word[-4] in consonants:
            results.append(word[:-3] + 'e')

        if word[-2:] == 'ar' and word[-3] in consonants:
            results.append(word[:-2] + 'o')

        if word[-2] == 'an' and word[-3] in consonants:
            results.append(word[:-2] + 'e')

    results = list(set(results))
    results = [case_of(word_temp)(r) for r in results]
    return results
