import re
import traceback
from malaya.num2word import to_cardinal, to_ordinal
from malaya.word2num import word2num
from malaya.text.tatabahasa import (
    hujung_malaysian,
    calon_dictionary,
)
from malaya.text.rules import rules_normalizer, rules_compound_normalizer
from malaya.text.function import ENGLISH_WORDS, MALAY_WORDS, case_of
import math
import logging

logger = logging.getLogger('malaya.text.normalization')

ignore_words = ['ringgit', 'sen']
ignore_postfix = ['adalah']
unit_mapping = {
    'kg': 'kilogram',
    'g': 'gram',
    'l': 'liter',
    'ml': 'milliliter',
    'c': 'celsius',
    'km': 'kilometer',
    'm': 'meter',
    'cm': 'sentimeter',
    'kilo': 'kilogram',
}

rules_compound_normalizer_regex = (
    '(?:' + '|'.join(list(rules_compound_normalizer.keys())) + ')'
)

rules_compound_normalizer_keys = list(rules_compound_normalizer.keys())


def _replace_compound(string):
    for k in rules_compound_normalizer_keys:
        results = [(m.start(0), m.end(0)) for m in re.finditer(k, string, flags=re.IGNORECASE)]
        for r in results:
            sub = string[r[0]: r[1]]
            replaced = rules_compound_normalizer.get(sub.lower())
            if replaced:
                if r[1] < len(string) and string[r[1]] != ' ':
                    continue
                if r[0] - 1 > len(string) and string[r[0] - 1] != ' ':
                    continue

                sub = case_of(sub)(replaced)
                string = string[:r[0]] + sub + string[r[1]:]
    return string


def _remove_postfix(word):
    if word in MALAY_WORDS or word in ENGLISH_WORDS or word in rules_normalizer:
        return word, ''
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


def _normalize_money(
    word, currency={'dollar': '$', 'ringgit': 'RM', 'pound': '£', 'euro': '€'}
):
    splitted = word.split()
    if splitted[-1] in ['dollar', 'ringgit', 'pound', 'euro']:
        v = word2num(' '.join(splitted[:-1]))
        return currency[splitted[-1]] + str(v)
    else:
        return word


def _is_number_regex(s):
    if re.match('^\\d+?\\.\\d+?$', s) is None:
        return s.isdigit()
    return True


def _string_to_num(word):
    if '.' in word:
        return float(word)
    else:
        return int(word)


def cardinal(x):
    cp_x = x[:]
    try:
        if re.match('.*[A-Za-z]+.*', x):
            return x
        x = re.sub(',', '', x, count=10)

        if re.match('.+\\..*', x):
            x = to_cardinal(float(x))
        elif re.match('\\..*', x):
            x = to_cardinal(float(x))
        else:
            x = to_cardinal(int(x))
        x = re.sub('-', ' ', x, count=10)
        return x
    except BaseException as e:
        logger.debug(traceback.format_exc())
        return cp_x


def split_currency(x):
    results = []
    for no, u in enumerate(x.split('.')):
        if no and len(u) == 1:
            u = u + '0'
        results.append(cardinal(u))
    return results


def digit(x):
    cp_x = x[:]
    try:
        x = re.sub('[^0-9]', '', x)
        result_string = ''
        for i in x:
            result_string = result_string + cardinal(i) + ' '
        result_string = result_string.strip()
        return result_string
    except BaseException as e:
        logger.debug(traceback.format_exc())
        return cp_x


def digit_unit(x):
    cp_x = x[:]
    try:
        n = re.sub('[^0-9.]', '', x)
        u = re.sub('[0-9. ]', '', x)
        u = unit_mapping.get(u.lower(), u)
        if '.' in n:
            n = float(n)
        else:
            n = int(n)
        n = to_cardinal(n)
        return f'{n} {u}'
    except Exception as e:
        logger.debug(traceback.format_exc())
        return cp_x


def letters(x):
    cp_x = x[:]
    try:
        x = re.sub('[^a-zA-Z]', '', x)
        x = x.lower()
        result_string = ''
        for i in range(len(x)):
            result_string = result_string + x[i] + ' '
        return result_string.strip()
    except BaseException as e:
        logger.debug(traceback.format_exc())
        return cp_x


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

    set_string = set(string)
    set_roman = set('MCDXLIV')
    if len(set_roman & set_string) < len(set_string):
        return -1

    returnint = 0
    for pair in table:

        continueyes = True

        while continueyes:
            if len(string) >= len(pair[0]):

                if string[0: len(pair[0])] == pair[0]:
                    returnint += pair[1]
                    string = string[len(pair[0]):]

                else:
                    continueyes = False
            else:
                continueyes = False

    return returnint


def ordinal(x):
    cp_x = x[:]
    try:
        result_string = ''
        x = x.replace(',', '')
        x = x.replace('[\\.]$', '')
        if re.match('^[0-9]+$', x):
            x = to_ordinal(int(x))
            return x
        if re.match('.*(V|X|I|L|D)', x):
            x = x.replace('-', '')
            if re.match('^ke.*', x):
                x = x[2:]
                x = rom_to_int(x)
                if x == -1:
                    return cp_x
                result_string = to_ordinal(x)
            else:
                x = rom_to_int(x)
                if x == -1:
                    return cp_x
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
        logger.debug(traceback.format_exc())
        return cp_x


def telephone(x):
    try:
        result_string = ''
        for i in range(0, len(x)):
            if re.match('[0-9]+', x[i]):
                result_string = result_string + cardinal(x[i]) + ' '
            else:
                result_string = result_string + 'sil '
        return result_string.strip()
    except BaseException as e:
        logger.debug(traceback.format_exc())
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
    except BaseException as e:
        logger.debug(traceback.format_exc())
        return x


def fraction(x):
    try:
        y = x.split('/')
        result_string = ''
        y[0] = cardinal(y[0])
        y[1] = cardinal(y[1])
        return '%s per %s' % (y[0], y[1])
    except BaseException as e:
        logger.debug(traceback.format_exc())
        return x


def combine_with_cent(
    x, currency='RM', currency_end='ringgit', cent='sen'
):
    text = split_currency(str(x))
    c = '%s%s' % (currency, str(x))
    if text[0] != 'kosong':
        x = '%s %s' % (text[0], currency_end)
    else:
        x = ''
    if len(text) == 2:
        if text[1] != 'kosong':
            x = '%s %s %s' % (x, text[1], cent)
    return x, c


def replace_ribu_juta(x):
    x = x.lower().replace('ribu', 'k')
    x = x.lower().replace('juta', 'm')
    return x


def money(x):
    try:
        if (
            re.match('^\\$', x)
            or x.lower().endswith('dollar')
            or x.lower().endswith('cent')
        ):
            x = x.lower()
            if not x.startswith('$') and x.endswith('cent'):
                cent = True
            else:
                cent = False
            x = x.replace('$', '').replace('dollar', '').replace('cent', '')
            x = re.sub(r'[ ]+', ' ', x).strip()
            x, n = re.split("(\\d+(?:[\\.,']\\d+)?)", x)[1:]
            n = replace_ribu_juta(n)
            x = re.sub(',', '', x, count=10)
            labels = []
            for c in n:
                if re.match('.*(M|m)$', c):
                    labels.append(1e6)
                elif re.match('.*(b|B)$', c):
                    labels.append(1e9)
                elif re.match('.*(k|K)$', c):
                    labels.append(1e3)
                elif re.match('.*(j|J)$', c):
                    labels.append(1e6)

            if cent:
                x = float(x)
                x = x / 100
            for l in labels:
                x = float(x)
                x = x * l

            if isinstance(x, float) and 1 - (x % 1) < 1e-5:
                x = math.ceil(x)

            x, c = combine_with_cent(
                x, currency='$', currency_end='dollar', cent='cent'
            )

            return re.sub(r'[ ]+', ' ', x.lower()).strip(), c

        elif (
            re.match('^US', x)
            or x.lower().endswith('dollar')
            or x.lower().endswith('cent')
            or x.lower().endswith('usd')
        ):
            x = x.lower()
            if not x.startswith('US') and x.endswith('cent'):
                cent = True
            else:
                cent = False
            x = (
                x.replace('US', '')
                .replace('usd', '')
                .replace('dollar', '')
                .replace('cent', '')
            )
            x = re.sub(r'[ ]+', ' ', x).strip()
            x, n = re.split("(\\d+(?:[\\.,']\\d+)?)", x)[1:]
            n = replace_ribu_juta(n)
            x = re.sub(',', '', x, count=10)
            labels = []
            for c in n:
                if re.match('.*(M|m)$', c):
                    labels.append(1e6)
                elif re.match('.*(b|B)$', c):
                    labels.append(1e9)
                elif re.match('.*(k|K)$', c):
                    labels.append(1e3)
                elif re.match('.*(j|J)$', c):
                    labels.append(1e6)

            if cent:
                x = float(x)
                x = x / 100
            for l in labels:
                x = float(x)
                x = x * l

            if isinstance(x, float) and 1 - (x % 1) < 1e-5:
                x = math.ceil(x)

            x, c = combine_with_cent(
                x, currency='$', currency_end='dollar', cent='cent'
            )

            return re.sub(r'[ ]+', ' ', x.lower()).strip(), c

        elif (
            re.match('^\\£', x)
            or x.lower().endswith('pound')
            or x.lower().endswith('penny')
        ):
            x = x.lower()
            if not x.startswith('£') and x.endswith('penny'):
                cent = True
            else:
                cent = False
            x = x.replace('£', '').replace('pound', '').replace('penny', '')
            x = re.sub(r'[ ]+', ' ', x).strip()
            x, n = re.split("(\\d+(?:[\\.,']\\d+)?)", x)[1:]
            n = replace_ribu_juta(n)
            x = re.sub(',', '', x, count=10)
            labels = []
            for c in n:
                if re.match('.*(M|m)$', c):
                    labels.append(1e6)
                elif re.match('.*(b|B)$', c):
                    labels.append(1e9)
                elif re.match('.*(k|K)$', c):
                    labels.append(1e3)
                elif re.match('.*(j|J)$', c):
                    labels.append(1e6)

            if cent:
                x = float(x)
                x = x / 100
            for l in labels:
                x = float(x)
                x = x * l

            if isinstance(x, float) and 1 - (x % 1) < 1e-5:
                x = math.ceil(x)

            x, c = combine_with_cent(
                x, currency='£', currency_end='pound', cent='cent'
            )
            return re.sub(r'[ ]+', ' ', x.lower()).strip(), c

        elif (
            re.match('^\\€', x)
            or x.lower().endswith('euro')
            or x.lower().endswith('cent')
        ):
            x = x.lower()
            if not x.startswith('€') and x.endswith('cent'):
                cent = True
            else:
                cent = False
            x = x.replace('€', '').replace('euro', '').replace('cent', '')
            x = re.sub(r'[ ]+', ' ', x).strip()
            x, n = re.split("(\\d+(?:[\\.,']\\d+)?)", x)[1:]
            n = replace_ribu_juta(n)
            x = re.sub(',', '', x, count=10)
            labels = []
            for c in n:
                if re.match('.*(M|m)$', c):
                    labels.append(1e6)
                elif re.match('.*(b|B)$', c):
                    labels.append(1e9)
                elif re.match('.*(k|K)$', c):
                    labels.append(1e3)
                elif re.match('.*(j|J)$', c):
                    labels.append(1e6)

            x = float(x)
            if cent:
                x = x / 100
            for l in labels:
                x = x * l

            if isinstance(x, float) and 1 - (x % 1) < 1e-5:
                x = math.ceil(x)

            x, c = combine_with_cent(
                x, currency='€', currency_end='euro', cent='cent'
            )
            return re.sub(r'[ ]+', ' ', x.lower()).strip(), c

        elif (
            re.match('^RM', x)
            or re.match('^rm', x)
            or x.lower().endswith('ringgit')
            or x.lower().endswith('sen')
        ):
            x = x.lower()
            if not x.startswith('rm') and x.endswith('sen'):
                cent = True
            else:
                cent = False

            x = x.replace('rm', '').replace('ringgit', '').replace('sen', '')
            x = re.sub(r'[ ]+', ' ', x).strip()
            x, n = re.split("(\\d+(?:[\\.,']\\d+)?)", x)[1:]
            n = replace_ribu_juta(n)
            x = re.sub(',', '', x, count=10)
            labels = []
            for c in n:
                if re.match('.*(M|m)$', c):
                    labels.append(1e6)
                elif re.match('.*(b|B)$', c):
                    labels.append(1e9)
                elif re.match('.*(k|K)$', c):
                    labels.append(1e3)
                elif re.match('.*(j|J)$', c):
                    labels.append(1e6)

            if cent:
                x = float(x)
                x = x / 100
            for l in labels:
                x = float(x)
                x = x * l

            if isinstance(x, float) and 1 - (x % 1) < 1e-5:
                x = math.ceil(x)

            x, c = combine_with_cent(x)
            return re.sub(r'[ ]+', ' ', x.lower()).strip(), c
        return x, None

    except Exception as e:
        logger.debug(traceback.format_exc())
        return x, None


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

    text = re.sub(r"(\b)([Cc]a)nt", r'\1\2n not', text)
    text = re.sub(r"(\b)([Ii])m", r'\1\2 am', text)
    text = re.sub(r"(\b)([Ll]et)s", r'\1\2 us', text)
    text = re.sub(r"(\b)([Ww])ont", r'\1\2ill not', text)

    return text
