import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import numpy as np
from fuzzywuzzy import fuzz
import pickle
import os
import json
import tensorflow as tf
from collections import Counter
from .utils import load_graph, download_file
from .num2word import to_cardinal
from .text_functions import (
    normalizer_textcleaning,
    stemmer_str_idx,
    pad_sentence_batch,
)
from .tatabahasa import (
    rules_normalizer,
    consonants,
    vowels,
    sounds,
    GO,
    PAD,
    EOS,
    UNK,
)
from .spell import return_possible, edit_normalizer, build_dicts, return_known
from .topics_influencers import is_location
from .paths import MALAY_TEXT, PATH_NORMALIZER, S3_PATH_NORMALIZER


def load_malay_dictionary():
    """
    load Pustaka dictionary for Spelling Corrector or anything.

    Returns
    -------
    list: list of strings
    """
    if not os.path.isfile(MALAY_TEXT):
        print('downloading Malay texts')
        download_file('v6/malay-text.txt', MALAY_TEXT)
    with open(MALAY_TEXT, 'r') as fopen:
        return [
            text.lower()
            for text in (list(filter(None, fopen.read().split('\n'))))
        ]


class DEEP_NORMALIZER:
    def __init__(self, x, logits, sess, dicts):
        self._sess = sess
        self._x = x
        self._logits = logits
        self._dicts = dicts
        self._dicts['rev_dictionary_to'] = {
            int(k): v for k, v in self._dicts['rev_dictionary_to'].items()
        }

    def normalize(self, string):
        """
        Normalize a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: normalized string
        """
        assert isinstance(string, str), 'input must be a string'
        token_strings = normalizer_textcleaning(string).split()
        idx = stemmer_str_idx(token_strings, self._dicts['dictionary_from'])
        predicted = self._sess.run(
            self._logits, feed_dict = {self._x: pad_sentence_batch(idx, PAD)[0]}
        )
        results = []
        for word in predicted:
            results.append(
                ''.join(
                    [
                        self._dicts['rev_dictionary_to'][c]
                        for c in word
                        if c not in [GO, PAD, EOS, UNK]
                    ]
                )
            )
        return ' '.join(results)


class SPELL_NORMALIZE:
    def __init__(self, corpus):
        self.corpus = Counter(corpus)

    def normalize(self, string, debug = True):
        """
        Normalize a string

        Parameters
        ----------
        string : str

        debug : bool, optional (default=True)
            If true, it will print character similarity distances.

        Returns
        -------
        string: normalized string
        """
        assert isinstance(string, str), 'input must be a string'
        result = []
        for word in normalizer_textcleaning(string).split():
            if word.istitle():
                result.append(word)
                continue
            if word[0] == 'x' and len(word) > 1:
                result_string = 'tak '
                word = word[1:]
            else:
                result_string = ''
            if word[-2:] == 'la':
                end_result_string = ' lah'
                word = word[:-2]
            elif word[-3:] == 'lah':
                end_result_string = ' lah'
                word = word[:-3]
            else:
                end_result_string = ''
            if word in sounds:
                result.append(result_string + sounds[word] + end_result_string)
                continue
            if word in rules_normalizer:
                result.append(
                    result_string + rules_normalizer[word] + end_result_string
                )
                continue
            if word in self.corpus:
                result.append(result_string + word + end_result_string)
                continue
            candidates = (
                return_known([word], self.corpus)
                or return_known(edit_normalizer(word), self.corpus)
                or return_possible(word, self.corpus, edit_normalizer)
                or [word]
            )
            candidates = list(candidates)
            candidates = [
                (candidate, is_location(candidate))
                for candidate in list(candidates)
            ]
            if debug:
                print([(k, fuzz.ratio(string, k[0])) for k in candidates], '\n')
            strings = [fuzz.ratio(string, k[0]) for k in candidates]
            descending_sort = np.argsort(strings)[::-1]
            selected = None
            for index in descending_sort:
                if not candidates[index][1]:
                    selected = candidates[index][0]
                    break
            selected = (
                candidates[descending_sort[0]][0] if not selected else selected
            )
            result.append(result_string + selected + end_result_string)
        return ' '.join(result)


class FUZZY_NORMALIZE:
    def __init__(self, normalized, corpus):
        self.normalized = normalized
        self.corpus = corpus

    def normalize(self, string):
        """
        Normalize a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: normalized string
        """
        assert isinstance(string, str), 'input must be a string'
        result = []
        for word in normalizer_textcleaning(string).split():
            if word.istitle():
                result.append(word)
                continue
            if word[0] == 'x' and len(word) > 1:
                result_string = 'tak '
                word = word[1:]
            else:
                result_string = ''
            if word[-2:] == 'la':
                end_result_string = ' lah'
                word = word[:-2]
            elif word[-3:] == 'lah':
                end_result_string = ' lah'
                word = word[:-3]
            else:
                end_result_string = ''
            if word in sounds:
                result.append(result_string + sounds[word] + end_result_string)
                continue
            if word in rules_normalizer:
                result.append(
                    result_string + rules_normalizer[word] + end_result_string
                )
                continue
            if word in self.corpus:
                result.append(result_string + word + end_result_string)
                continue
            results = []
            for i in range(len(self.normalized)):
                results.append(
                    np.mean([fuzz.ratio(word, k) for k in self.normalized[i]])
                )
            if len(np.where(np.array(results) > 70)[0]) < 1:
                result.append(result_string + word + end_result_string)
                continue
            result.append(
                result_string
                + self.corpus[np.argmax(results)]
                + end_result_string
            )
        return ' '.join(result)


def fuzzy_normalizer(corpus):
    """
    Train a fuzzy logic Normalizer

    Parameters
    ----------
    corpus : list of strings. Prefer to feed with malaya.load_malay_dictionary()

    Returns
    -------
    FUZZY_NORMALIZE: Trained malaya.normalizer.FUZZY_NORMALIZE class
    """
    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    transform = []
    for i in corpus:
        i = i.lower()
        result = []
        result.append(i)
        result.append(''.join(char for char in i if char not in vowels))
        if i[0] in consonants and i[-1] in consonants:
            result.append(i[0] + i[-1])
        if i[-1] == 'a':
            result.append(i[:-1] + 'e')
            result.append(i + 'k')
        if i[1] in vowels and i[0] in consonants:
            result.append(i[0] + i[2:])
        if i[-2] in vowels and i[-1] in consonants:
            result.append(i[:-2] + i[-1])
        result.append(i[0] + i[-1])
        if i[-2:] == 'ar':
            result.append(i[:-2] + 'o')
        if i[:2] == 'ha':
            result.append(i[1:])
        transform.append(list(set(result)))
    return FUZZY_NORMALIZE(transform, corpus)


def spell_normalizer(corpus):
    """
    Train a Spelling Normalizer

    Parameters
    ----------
    corpus : list of strings. Prefer to feed with malaya.load_malay_dictionary()

    Returns
    -------
    SPELL_NORMALIZE: Trained malaya.normalizer.SPELL_NORMALIZE class
    """
    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    return SPELL_NORMALIZE(corpus)


def basic_normalizer(string):
    """
    Use basic rules-based to normalize a string.

    Parameters
    ----------
    string: str

    Returns
    -------
    string: normalized string
    """
    assert isinstance(string, str), 'input must be a string'
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


def deep_normalizer():
    """
    Load deep-learning model to normalize a string. This model totally more sucks than fuzzy based, Husein still need to read more.

    Returns
    -------
    DEEP_NORMALIZER: malaya.normalizer.DEEP_NORMALIZER class

    """
    if not os.path.isfile(PATH_NORMALIZER['deep']['setting']):
        print('downloading JSON normalizer')
        download_file(
            S3_PATH_NORMALIZER['deep']['setting'],
            PATH_NORMALIZER['deep']['setting'],
        )
    with open(PATH_NORMALIZER['deep']['setting'], 'r') as fopen:
        dic_normalizer = json.load(fopen)
    if not os.path.isfile(PATH_NORMALIZER['deep']['model']):
        print('downloading normalizer graph')
        download_file(
            S3_PATH_NORMALIZER['deep']['model'],
            PATH_NORMALIZER['deep']['model'],
        )
    g = load_graph(PATH_NORMALIZER['deep']['model'])
    return DEEP_NORMALIZER(
        g.get_tensor_by_name('import/Placeholder:0'),
        g.get_tensor_by_name('import/logits:0'),
        tf.InteractiveSession(graph = g),
        dic_normalizer,
    )
