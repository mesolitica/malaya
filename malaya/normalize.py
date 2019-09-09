import numpy as np
import json
import re
from unidecode import unidecode
from .texts._text_functions import ENGLISH_WORDS, MALAY_WORDS
from .texts._tatabahasa import (
    rules_normalizer,
    consonants,
    vowels,
    sounds,
    hujung_malaysian,
    calon_dictionary,
)
from .num2word import to_cardinal, to_ordinal
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


def _normalize_title(word):
    if word[0].isupper():
        return calon_dictionary.get(word.lower(), word)
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


def cardinal(x):
    try:
        if re.match('.*[A-Za-z]+.*', x):
            return x
        x = re.sub(',', '', x, count = 10)

        if re.match('.+\..*', x):
            x = to_cardinal(float(x))
        elif re.match('\..*', x):
            x = to_cardinal(float(x))
        else:
            x = to_cardinal(int(x))
        x = x.replace('kosong', 'o')
        x = re.sub('-', ' ', x, count = 10)
        x = re.sub(' dan', '', x, count = 10)
        return x
    except:
        return x


def digit(x):
    try:
        x = re.sub('[^0-9]', '', x)
        result_string = ''
        for i in x:
            result_string = result_string + cardinal(i) + ' '
        result_string = result_string.strip()
        return result_string
    except:
        return x


def letters(x):
    try:
        x = re.sub('[^a-zA-Z]', '', x)
        x = x.lower()
        result_string = ''
        for i in range(len(x)):
            result_string = result_string + x[i] + ' '
        return result_string.strip()
    except:
        return x


def rom_to_int(string):

    table = [
        ['M', 1000],
        ['CM', 900],
        ['D', 500],
        ['CD', 400],
        ['C', 100],
        ['XC', 90],
        ['L', 50],
        ['XL', 40],
        ['X', 10],
        ['IX', 9],
        ['V', 5],
        ['IV', 4],
        ['I', 1],
    ]
    returnint = 0
    for pair in table:

        continueyes = True

        while continueyes:
            if len(string) >= len(pair[0]):

                if string[0 : len(pair[0])] == pair[0]:
                    returnint += pair[1]
                    string = string[len(pair[0]) :]

                else:
                    continueyes = False
            else:
                continueyes = False

    return returnint


def ordinal(x):
    try:
        result_string = ''
        x = x.replace(',', '')
        x = x.replace('[\.]$', '')
        if re.match('^[0-9]+$', x):
            x = to_ordinal(int(x))
            return x
        if re.match('.*(V|X|I|L|D)', x):
            x = x.replace('-', '')
            if re.match('^ke.*', x):
                x = x[2:]
                x = rom_to_int(x)
                result_string = to_ordinal(x)
            else:
                x = rom_to_int(x)
                result_string = to_ordinal(x)
                result_string = 'yang ' + result_string
        elif re.match('^ke.*', x):
            x = x.replace('-', '')
            x = x[2:]
            result_string = to_ordinal(int(x))
        else:
            result_string = to_ordinal(int(x))
        return result_string
    except Exception as e:
        return x


def telephone(x):
    try:
        result_string = ''
        for i in range(0, len(x)):
            if re.match('[0-9]+', x[i]):
                result_string = result_string + cardinal(x[i]) + ' '
            else:
                result_string = result_string + 'sil '
        return result_string.strip()
    except:
        return x


def electronic(x):
    try:
        replacement = {
            '.': 'dot',
            ':': 'colon',
            '/': 'slash',
            '-': 'dash',
            '#': 'hash tag',
        }
        result_string = ''
        if re.match('.*[A-Za-z].*', x):
            for char in x:
                if re.match('[A-Za-z]', char):
                    result_string = result_string + letters(char) + ' '
                elif char in replacement:
                    result_string = result_string + replacement[char] + ' '
                elif re.match('[0-9]', char):
                    if char == 0:
                        result_string = result_string + 'o '
                    else:
                        number = cardinal(char)
                        for n in number:
                            result_string = result_string + n + ' '
            return result_string.strip()
        else:
            return x
    except:
        return x


def fraction(x):
    try:
        y = x.split('/')
        result_string = ''
        y[0] = cardinal(y[0])
        y[1] = cardinal(y[1])
        return '%s per %s' % (y[0], y[1])
    except:
        return x


def money(x):
    try:
        if re.match('^\$', x):
            x = x.replace('$', '')
            if len(x.split(' ')) == 1:
                if re.match('.*(M|m)$', x):
                    x = x.replace('M', '')
                    x = x.replace('m', '')
                    text = cardinal(x)
                    x = text + ' juta dollar'
                elif re.match('.*(b|B)$', x):
                    x = x.replace('B', '')
                    x = x.replace('b', '')
                    text = cardinal(x)
                    x = text + ' billion dollar'
                else:
                    text = cardinal(x)
                    x = text + ' dollar'
                return x.lower()
            elif len(x.split(' ')) == 2:
                text = cardinal(x.split(' ')[0])
                if x.split(' ')[1].lower() == 'million':
                    x = text + ' juta dollar'
                elif x.split(' ')[1].lower() == 'juta':
                    x = text + ' juta dollar'
                elif x.split(' ')[1].lower() == 'billion':
                    x = text + ' billion dollar'
                return x.lower()

        if re.match('^US\$', x):
            x = x.replace('US$', '')
            if len(x.split(' ')) == 1:
                if re.match('.*(M|m)$', x):
                    x = x.replace('M', '')
                    x = x.replace('m', '')
                    text = cardinal(x)
                    x = text + ' juta dollar'
                elif re.match('.*(b|B)$', x):
                    x = x.replace('b', '')
                    x = x.replace('B', '')
                    text = cardinal(x)
                    x = text + ' billion dollar'
                else:
                    text = cardinal(x)
                    x = text + ' dollar'
                return x.lower()
            elif len(x.split(' ')) == 2:
                text = cardinal(x.split(' ')[0])
                if x.split(' ')[1].lower() == 'million':
                    x = text + ' juta dollar'
                elif x.split(' ')[1].lower() == 'juta':
                    x = text + ' juta dollar'
                elif x.split(' ')[1].lower() == 'billion':
                    x = text + ' billion dollar'
                return x.lower()

        elif re.match('^£', x):
            x = x.replace('£', '')
            if len(x.split(' ')) == 1:
                if re.match('.*(M|m)$', x):
                    x = x.replace('M', '')
                    x = x.replace('m', '')
                    text = cardinal(x)
                    x = text + ' juta pound'
                elif re.match('.*(b|B)$', x):
                    x = x.replace('b', '')
                    x = x.replace('B', '')
                    text = cardinal(x)
                    x = text + ' billion pound'
                else:
                    text = cardinal(x)
                    x = text + ' pound'
                return x.lower()
            elif len(x.split(' ')) == 2:
                text = cardinal(x.split(' ')[0])
                if x.split(' ')[1].lower() == 'million':
                    x = text + ' juta pound'
                elif x.split(' ')[1].lower() == 'juta':
                    x = text + ' juta pound'
                elif x.split(' ')[1].lower() == 'billion':
                    x = text + ' billion pound'
                return x.lower()

        elif re.match('^€', x):
            x = x.replace('€', '')
            if len(x.split(' ')) == 1:
                if re.match('.*(M|m)$', x):
                    x = x.replace('M', '')
                    x = x.replace('m', '')
                    text = cardinal(x)
                    x = text + ' juta euro'
                elif re.match('.*(b|B)$', x):
                    x = x.replace('B', '')
                    x = x.replace('b', '')
                    text = cardinal(x)
                    x = text + ' billion euro'
                else:
                    text = cardinal(x)
                    x = text + ' euro'
                return x.lower()
            elif len(x.split(' ')) == 2:
                text = cardinal(x.split(' ')[0])
                if x.split(' ')[1].lower() == 'million':
                    x = text + ' juta euro'
                elif x.split(' ')[1].lower() == 'juta':
                    x = text + ' juta euro'
                elif x.split(' ')[1].lower() == 'billion':
                    x = text + ' billion euro'
                return x.lower()

        elif re.match('^RM', x) or re.match('^rm', x):
            x = x.lower()
            x = x.replace('rm', '')
            if len(x.split(' ')) == 1:
                if re.match('.*(M|m)$', x):
                    x = x.replace('M', '')
                    x = x.replace('m', '')
                    text = cardinal(x)
                    x = text + ' juta ringgit'
                elif re.match('.*(b|B)$', x):
                    x = x.replace('B', '')
                    x = x.replace('b', '')
                    text = cardinal(x)
                    x = text + ' billion ringgit'
                else:
                    text = cardinal(x)
                    x = text + ' ringgit'
                return x.lower()
            elif len(x.split(' ')) == 2:
                text = cardinal(x.split(' ')[0])
                if x.split(' ')[1].lower() == 'million':
                    x = text + ' juta ringgit'
                elif x.split(' ')[1].lower() == 'juta':
                    x = text + ' juta ringgit'
                elif x.split(' ')[1].lower() == 'billion':
                    x = text + ' billion ringgit'
                return x.lower()

        elif word[-3:] == 'sen':
            return to_cardinal(_string_to_num(word[:-3])) + ' sen'

    except:
        return x


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
            if word in '~@#$%^&*()_+{}|[:"\'];<>,.?/-':
                result.append(word)
                index += 1
                continue
            if word.lower() in ignore_words:
                result.append(word)
                index += 1
                continue
            if word[0].isupper():
                if word.upper() not in ['KE', 'PADA', 'RM', 'SEN', 'HINGGA']:
                    result.append(_normalize_title(word))
                    index += 1
                    continue
            if check_english:
                if word.lower() in ENGLISH_WORDS:
                    result.append(word)
                    index += 1
                    continue
            if word.lower() in MALAY_WORDS and word.lower() not in [
                'pada',
                'ke',
            ]:
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
                        ordinal(
                            word + tokenized[index + 1] + tokenized[index + 2]
                        )
                    )
                    index += 3
                    continue
                elif tokenized[index + 1] == '-' and re.match(
                    '.*(V|X|I|L|D)', tokenized[index + 2]
                ):
                    result.append(
                        ordinal(
                            word
                            + tokenized[index + 1]
                            + str(rom_to_int(tokenized[index + 2]))
                        )
                    )
                    index += 3
                    continue
                else:
                    result.append('ke')
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
                else:
                    result.append('pada')
                    index += 1
                    continue

            if _is_number_regex(word) and index < (len(tokenized) - 2):
                if tokenized[index + 1] == '/' and _is_number_regex(
                    tokenized[index + 2]
                ):
                    result.append(
                        fraction(
                            word + tokenized[index + 1] + tokenized[index + 2]
                        )
                    )
                    index += 3
                    continue
            if word.lower() == 'rm' and index < (len(tokenized) - 2):
                if (
                    _is_number_regex(tokenized[index + 1])
                    and tokenized[index + 2].lower() == 'sen'
                ):
                    result.append(money('rm' + tokenized[index + 1]))
                    index += 3
                    continue

            if word.lower() == 'rm' and index < (len(tokenized) - 1):
                if _is_number_regex(tokenized[index + 1]):
                    result.append(money('rm' + tokenized[index + 1]))
                    index += 2
                    continue

            if _is_number_regex(word) and index < (len(tokenized) - 1):
                if tokenized[index + 1].lower() == 'sen':
                    result.append(cardinal(word) + ' sen')
                    index += 2
                    continue

            money_ = money(word)
            if money_ != word:
                result.append(money_)
                index += 1
                continue

            cardinal_ = cardinal(word)
            if cardinal_ != word:
                result.append(cardinal_)
                index += 1
                continue

            normalized_ke = ordinal(word)
            if normalized_ke != word:
                result.append(normalized_ke)
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
