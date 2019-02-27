import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import numpy as np
import json
from fuzzywuzzy import fuzz
from unidecode import unidecode
from collections import Counter
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
)
from .spell import _return_possible, _edit_normalizer, _return_known
from .similarity import is_location
from ._utils._paths import MALAY_TEXT, PATH_NORMALIZER, S3_PATH_NORMALIZER


class _DEEP_NORMALIZER:
    def __init__(self, x, logits, sess, dicts, corpus):
        self._sess = sess
        self._x = x
        self._logits = logits
        self._dicts = dicts
        self._dicts['rev_dictionary_to'] = {
            int(k): v for k, v in self._dicts['rev_dictionary_to'].items()
        }
        self.corpus = corpus

    def normalize(self, string, check_english = True):
        """
        Normalize a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: normalized string
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(check_english, bool):
            raise ValueError('check_english must be a boolean')

        token_strings = normalizer_textcleaning(string).split()
        results, need_to_normalize = [], []
        for word in token_strings:
            if word.istitle():
                results.append(word)
                continue
            if check_english:
                if word in ENGLISH_WORDS:
                    results.append(word)
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
                results.append(result_string + sounds[word] + end_result_string)
                continue
            if word in rules_normalizer:
                results.append(
                    result_string + rules_normalizer[word] + end_result_string
                )
                continue
            if word in self.corpus:
                results.append(result_string + word + end_result_string)
                continue
            results.append('replace__me__')
            need_to_normalize.append(word)

        normalized = []
        if len(need_to_normalize):
            idx = stemmer_str_idx(
                need_to_normalize, self._dicts['dictionary_from']
            )
            predicted = self._sess.run(
                self._logits,
                feed_dict = {self._x: pad_sentence_batch(idx, PAD)[0]},
            )
            for word in predicted:
                normalized.append(
                    ''.join(
                        [
                            self._dicts['rev_dictionary_to'][c]
                            for c in word
                            if c not in [GO, PAD, EOS, UNK]
                        ]
                    )
                )
        cp_results, current_replace = [], 0
        for i in range(len(results)):
            if 'replace__me__' in results[i]:
                if current_replace < len(normalized):
                    results[i] = normalized[current_replace]
                    cp_results.append(results[i])
                    current_replace += 1
            else:
                cp_results.append(results[i])

        return ' '.join(cp_results)


class _SPELL_NORMALIZE:
    def __init__(self, corpus):
        self.corpus = Counter(corpus)

    def normalize(self, string, debug = True, check_english = True):
        """
        Normalize a string

        Parameters
        ----------
        string : str

        debug : bool, optional (default=True)
            If true, it will print character similarity distances.
        check_english: bool, (default=True)
            check a word in english dictionary

        Returns
        -------
        string: normalized string
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(debug, bool):
            raise ValueError('debug must be a boolean')
        if not isinstance(check_english, bool):
            raise ValueError('check_english must be a boolean')

        result = []
        for word in normalizer_textcleaning(string).split():
            if word.istitle():
                result.append(word)
                continue
            if check_english:
                if word in ENGLISH_WORDS:
                    result.append(word)
                    continue
            if len(word) > 2:
                if word[-2] in consonants and word[-1] == 'e':
                    word = word[:-1] + 'a'
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
                _return_known([word], self.corpus)
                or _return_known(_edit_normalizer(word), self.corpus)
                or _return_possible(word, self.corpus, _edit_normalizer)
                or [word]
            )
            candidates = list(candidates)
            candidates = [
                (candidate, is_location(candidate))
                for candidate in list(candidates)
            ]
            if debug:
                print([(k, fuzz.ratio(word, k[0])) for k in candidates], '\n')
            strings = [fuzz.ratio(word, k[0]) for k in candidates]
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


class _FUZZY_NORMALIZE:
    def __init__(self, normalized, corpus):
        self.normalized = normalized
        self.corpus = corpus

    def normalize(self, string, fuzzy_ratio = 70, check_english = True):
        """
        Normalize a string.

        Parameters
        ----------
        string : str
        fuzzy_ratio: int, (default=70)
            ratio of similar characters by positions, if 90, means 90%
        check_english: bool, (default=True)
            check a word in english dictionary

        Returns
        -------
        string: normalized string
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(fuzzy_ratio, int):
            raise ValueError('fuzzy_ratio must be an integer')
        if not isinstance(check_english, bool):
            raise ValueError('check_english must be a boolean')

        result = []
        for word in normalizer_textcleaning(string).split():
            if word.istitle():
                result.append(word)
                continue
            if check_english:
                if word in ENGLISH_WORDS:
                    result.append(word)
                    continue
            if len(word) > 2:
                if word[-2] in consonants and word[-1] == 'e':
                    word = word[:-1] + 'a'
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
            if len(np.where(np.array(results) > fuzzy_ratio)[0]) < 1:
                result.append(result_string + word + end_result_string)
                continue
            result.append(
                result_string
                + self.corpus[np.argmax(results)]
                + end_result_string
            )
        return ' '.join(result)


def fuzzy(corpus):
    """
    Train a fuzzy logic Normalizer

    Parameters
    ----------
    corpus : list of strings. Prefer to feed with malaya.load_malay_dictionary()

    Returns
    -------
    FUZZY_NORMALIZE: Trained malaya.normalizer._FUZZY_NORMALIZE class
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


def spell(corpus):
    """
    Train a Spelling Normalizer

    Parameters
    ----------
    corpus : list of strings. Prefer to feed with malaya.load_malay_dictionary()

    Returns
    -------
    SPELL_NORMALIZE: Trained malaya.normalizer._SPELL_NORMALIZE class
    """
    if not isinstance(corpus, list):
        raise ValueError('corpus must be a list')
    if not isinstance(corpus[0], str):
        raise ValueError('corpus must be list of strings')
    return _SPELL_NORMALIZE([unidecode(w) for w in corpus])


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


def available_deep_model():
    """
    List available deep learning stemming models.
    """
    return ['lstm', 'bahdanau', 'luong']


def deep_model(corpus, model = 'bahdanau', validate = True):
    """
    Load deep-learning model to normalize a string.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    DEEP_NORMALIZER: malaya.normalizer._DEEP_NORMALIZER class

    """
    if not isinstance(corpus, list):
        raise ValueError('corpus must be a list')
    if not isinstance(corpus[0], str):
        raise ValueError('corpus must be list of strings')
    if validate:
        check_file(PATH_NORMALIZER[model], S3_PATH_NORMALIZER[model])
    else:
        if not check_available(PATH_NORMALIZER[model]):
            raise Exception(
                'normalizer is not available, please `validate = True`'
            )
    try:
        with open(PATH_NORMALIZER[model]['setting'], 'r') as fopen:
            dic_normalizer = json.load(fopen)
        g = load_graph(PATH_NORMALIZER[model]['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('normalizer/%s') and try again"
            % (model)
        )
    return _DEEP_NORMALIZER(
        g.get_tensor_by_name('import/Placeholder:0'),
        g.get_tensor_by_name('import/logits:0'),
        generate_session(graph = g),
        dic_normalizer,
        [unidecode(w) for w in corpus],
    )
