import re
import traceback
from malaya import num2word
from malaya.word2num import word2num
from malaya.text.tatabahasa import (
    hujung_malaysian,
    calon_dictionary,
    hujung,
    permulaan,
    laughing,
    bulan,
    bulan_en,
)
from malaya.text.rules import rules_normalizer, rules_compound_normalizer
from malaya.text.function import case_of, MENGELUH, BETUL, GEMBIRA
from malaya.dictionary import ENGLISH_WORDS, MALAY_WORDS, is_malay, is_english
from malaya.text.regex import _expressions
from num2words import num2words
from datetime import time
import dateparser
import math
import logging

logger = logging.getLogger('malaya.text.normalization')

ignore_words = ['ringgit', 'sen', 'hmm', 'hhmm', 'hmmm']
ignore_postfix = ['adalah', 'allah']
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
unit_mapping_en = {
    'kg': 'kilogram',
    'g': 'gram',
    'l': 'liter',
    'ml': 'milliliter',
    'c': 'celcius',
    'km': 'kilometer',
    'm': 'meter',
    'cm': 'centimeter',
    'kilo': 'kilogram',
}

rules_compound_normalizer_regex = (
    r'\b(?:' + '|'.join(list(rules_compound_normalizer.keys())) + ')'
)

rules_compound_normalizer_keys = list(rules_compound_normalizer.keys())

digits = '0123456789'
sastrawi_stemmer = None

time_descriptors = {
    "am": "pagi", "a.m.": "pagi", "morning": "pagi", "pagi": "pagi", "pgi": "pagi",
    "pm": "petang", "p.m.": "petang", "petang": "petang", "ptg": "petang",
    "tengahari": "tengahari", "tngahari": "tengahari",
    "malam": "malam",
    "hours": "", "hour": "", "hrs": "", "jam": ""
}

time_pattern = re.compile(
    r'(?:pkul|pukul|kul)?\s*'
    r'(?P<hour>[0-2]?[0-9])'
    r'(?:[:.](?P<minute>[0-5][0-9]))?'
    r'(?:[:.](?P<second>[0-5][0-9]))?'
    r'(?:\s*(?P<desc>AM|PM|am|pm|a\.m\.|p\.m\.|pagi|pgi|morning|tengahari|tngahari|petang|ptg|malam|hours|hrs|jam))?',
    re.IGNORECASE
)

def to_cardinal(n, english=False):
    if english:
        s = num2words(n).replace('-', ' ')
        return re.sub(r'[ ]+', ' ', s).strip()
    else:
        return num2word.to_cardinal(n)

def to_ordinal(n, english=False):
    if english:
        s = num2words(n, to='ordinal').replace('-', ' ')
        return re.sub(r'[ ]+', ' ', s).strip()
    else:
        return num2word.to_ordinal(n)

def initialize_sastrawi():
    global sastrawi_stemmer
    if sastrawi_stemmer is None:
        from malaya.stem import sastrawi

        sastrawi_stemmer = sastrawi()


def _replace_compound(string):
    for k in rules_compound_normalizer_keys:
        results = [(m.start(0), m.end(0))
                   for m in re.finditer(r'\b' + k, string, flags=re.IGNORECASE)]
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


def get_stemmed(word, stemmed):
    if word == stemmed:
        return '', '', word

    index = [(m.start(0), m.end(0)) for m in re.finditer(stemmed, word, flags=re.IGNORECASE)]
    if len(index):
        permulaann = word[:index[0][0]]
        hujungan = word[index[-1][1]:]
        no_imbuhan = word[index[0][0]:index[-1][1]]

        permulaann = permulaan.get(permulaann, permulaann)
        hujungan = hujung.get(hujungan, hujungan)
    else:
        permulaann = ''
        hujungan = ''
        no_imbuhan = word

    return permulaann, hujungan, no_imbuhan


def get_hujung(word, stemmer=None):

    for p in ignore_postfix:
        if word.endswith(p):
            return word, ''

    if word[-1] in digits:
        digit = word[-1]
        word = word[:-1]
    else:
        digit = ''

    if stemmer is not None:
        stemmed = stemmer.stem_word(word)
        permulaann, hujungan, no_imbuhan = get_stemmed(word, stemmed)
        no_imbuhan = f'{permulaann}{no_imbuhan}'
        return no_imbuhan + digit, hujungan

    else:
        hujung_result = [(k, v) for k, v in hujung.items() if word.endswith(k)]
        if len(hujung_result):
            hujung_result = max(hujung_result, key=lambda x: len(x[1]))
            word = word[: -len(hujung_result[0])]
            hujung_result = hujung_result[1]
            return word + digit, hujung_result

    return word + digit, ''


def get_permulaan(word, stemmer=None):

    if stemmer is not None:
        stemmed = stemmer.stem_word(word)
        permulaann, hujungan, no_imbuhan = get_stemmed(word, stemmed)
        no_imbuhan = f'{no_imbuhan}{hujungan}'
        return no_imbuhan, permulaann

    else:
        permulaan_result = [
            (k, v) for k, v in permulaan.items() if word.startswith(k)
        ]
        if len(permulaan_result):
            permulaan_result = max(permulaan_result, key=lambda x: len(x[1]))
            word = word[len(permulaan_result[0]):]
            permulaan_result = permulaan_result[1]
            return word, permulaan_result

    return word, ''


def _remove_postfix(word, stemmer=None, validate_word=True):

    if validate_word and (is_malay(word) or is_english(word) or word in rules_normalizer):
        return word, ''
    for p in ignore_postfix:
        if word.endswith(p):
            return word, ''

    if word[-1] in digits:
        digit = word[-1]
        word = word[:-1]
    else:
        digit = ''

    if stemmer is not None:
        stemmed = stemmer.stem_word(word)
        permulaann, hujungan, no_imbuhan = get_stemmed(word, stemmed)
        no_imbuhan = f'{permulaann}{no_imbuhan}'
        if hujungan in hujung_malaysian:
            return no_imbuhan + digit, 'lah'

    else:
        for p in hujung_malaysian:
            if word.endswith(p):
                return word[: -len(p)] + digit, 'lah'

    return get_hujung(word + digit, stemmer=stemmer)


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


def _is_mandarin_char(s):
    return re.match(r'[\u4e00-\u9fff]+', s) is not None


def _is_number_regex(s):
    if re.match('^\\d+?\\.\\d+?$', s) is None:
        return s.isdigit()
    return True


def _string_to_num(word):
    if '.' in word:
        return float(word)
    else:
        return int(word)


def cardinal(x, english=False):
    cp_x = x[:]
    try:
        if re.match('.*[A-Za-z]+.*', x):
            return x
        x = re.sub(',', '', x, count=10)

        if re.match('.+\\..*', x):
            x = to_cardinal(float(x), english=english)
        elif re.match('\\..*', x):
            x = to_cardinal(float(x), english=english)
        else:
            x = to_cardinal(int(x), english=english)
        x = re.sub('-', ' ', x, count=10)
        return x
    except BaseException as e:
        logger.debug(traceback.format_exc())
        return cp_x


def split_currency(x, english=False):
    results = []
    for no, u in enumerate(x.split('.')):
        if no and len(u) == 1:
            u = u + '0'
        results.append(cardinal(u, english=english))
    return results


def digit(x, english=False):
    cp_x = x[:]
    try:
        x = re.sub('[^0-9]', '', x)
        result_string = ''
        for i in x:
            result_string = result_string + cardinal(i, english=english) + ' '
        result_string = result_string.strip()
        return result_string
    except BaseException as e:
        logger.debug(traceback.format_exc())
        return cp_x


def digit_unit(x, english=False):
    cp_x = x[:]
    try:
        n = re.sub('[^0-9.]', '', x)
        u = re.sub('[0-9. ]', '', x)
        if english:
            mapping = unit_mapping_en
        else:
            mapping = unit_mapping
        u = mapping.get(u.lower(), u)
        if '.' in n:
            n = float(n)
        else:
            n = int(n)
        n = to_cardinal(n, english=english)
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


def ordinal(x, english=False):
    cp_x = x[:]
    try:
        result_string = ''
        x = x.replace(',', '')
        x = x.replace('[\\.]$', '')
        if re.match('^[0-9]+$', x):
            x = to_ordinal(int(x), english=english)
            return x
        if re.match('.*(V|X|I|L|D)', x):
            x = x.replace('-', '')
            if re.match('^ke.*', x):
                x = x[2:]
                x = rom_to_int(x)
                if x == -1:
                    return cp_x
                result_string = to_ordinal(x, english=english)
            else:
                x = rom_to_int(x)
                if x == -1:
                    return cp_x
                result_string = to_ordinal(x, english=english)
                if english:
                    p = ''
                else:
                    p = 'yang '
                result_string = p + result_string
        elif re.match('^ke.*', x):
            x = x.replace('-', '')
            x = x[2:]
            result_string = to_ordinal(int(x), english=english)
        else:
            result_string = to_ordinal(int(x), english=english)
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


def fraction(x, english=False):
    try:
        y = x.split('/')
        result_string = ''
        y[0] = cardinal(y[0], english=english)
        y[1] = cardinal(y[1], english=english)
        if english:
            over = 'over'
        else:
            over = 'per'
        return '%s %s %s' % (y[0], over, y[1])
    except BaseException as e:
        logger.debug(traceback.format_exc())
        return x


def combine_with_cent(
    x, currency='RM', currency_end='ringgit', cent='sen', english=False,
):
    zeros = {'kosong', 'zero'}
    text = split_currency(str(x), english=english)
    c = '%s%s' % (currency, str(x))
    if text[0] not in zeros:
        x = '%s %s' % (text[0], currency_end)
    else:
        x = ''
    if len(text) == 2:
        if text[1] != zeros:
            x = '%s %s %s' % (x, text[1], cent)
    return x, c


def replace_ribu_juta(x):
    x = x.lower().replace('ribu', 'k').replace('thousand', 'k')
    x = x.lower().replace('juta', 'm').replace('million', 'm')
    return x


def normalize_numbers_with_shortform(x):
    x_ = x[:]
    try:
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

        for l in labels:
            x = float(x)
            x = x * l

        if isinstance(x, float) and 1 - (x % 1) < 1e-5:
            x = math.ceil(x)

        x = str(x)
        if x.endswith('.0'):
            x = x[:-2]
        return x
    except BaseException:
        return x_


def money(x, english=False):
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
                x, currency='$', currency_end='dollar', cent='cent', english=english
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
                x, currency='$', currency_end='dollar', cent='cent', english=english
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
                x, currency='£', currency_end='pound', cent='cent', english=english
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
                x, currency='€', currency_end='euro', cent='cent', english=english
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

            x, c = combine_with_cent(x, english=english)
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
        r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't\b",
        r'\1\2 not',
        text,
    )
    text = re.sub(
        r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll\b",
        r'\1\2 will',
        text,
    )
    text = re.sub(
        r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re\b", r'\1\2 are', text
    )
    text = re.sub(
        r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve\b",
        r'\1\2 have',
        text,
    )
    text = re.sub(r"(\b)([Cc]a)n't\b", r'\1\2n not', text)
    text = re.sub(r"(\b)([Ii])'m\b", r'\1\2 am', text)
    text = re.sub(r"(\b)([Ll]et)'s\b", r'\1\2 us', text)
    text = re.sub(r"(\b)([Ww])on't\b", r'\1\2ill not', text)
    text = re.sub(r"(\b)([Ss])han't\b", r'\1\2hall not', text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)\b", r'\1\2ou all', text)

    text = re.sub(r"(\b)([Cc]a)nt\b", r'\1\2n not', text)
    text = re.sub(r"(\b)([Ii])m\b", r'\1\2 am', text)
    text = re.sub(r"(\b)([Ll]et)s\b", r'\1\2 us', text)
    text = re.sub(r"(\b)([Ww])ont\b", r'\1\2ill not', text)

    return text


def repeat_word(word, repeat=1):
    """
    seadil2 -> seadil-adil
    terjadi2 -> terjadi-jadi
    """
    if word.startswith('se') or word.startswith('ter'):
        global sastrawi_stemmer
        if sastrawi_stemmer is None:
            from malaya.stem import sastrawi

            sastrawi_stemmer = sastrawi()

        stemmed = sastrawi_stemmer.stem(word)
    else:
        stemmed = word

    repeated = '-'.join([stemmed] * (repeat - 1))
    if len(repeated):
        return f'{word}-{repeated}'
    else:
        return word


def replace_any(string, lists, replace_with):
    result = []
    for word in string.split():
        word_lower = word.lower()

        if is_malay(word_lower):
            pass
        elif is_english(word_lower):
            pass
        elif (
            len(re.findall(_expressions['email'], word))
            and not len(re.findall(_expressions['url'], word))
            and not len(re.findall(_expressions['hashtag'], word))
            and not len(re.findall(_expressions['phone'], word))
            and not len(re.findall(_expressions['money'], word))
            and not len(re.findall(_expressions['date'], word))
            and not _is_number_regex(word)
        ):
            pass
        elif any([e in word_lower for e in lists]):
            word = case_of(word)(replace_with)

        result.append(word)
    return ' '.join(result)

def parse_time_string(text):
    matches = []
    
    for match in time_pattern.finditer(text):
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
        second = int(match.group("second") or 0)
        desc = (match.group("desc") or "").lower()

        desc = time_descriptors.get(desc, desc)

        if desc == "pagi":
            if hour == 12: hour = 0
        elif desc == "tengahari":
            if 1 <= hour <= 6: hour += 12
        elif desc == "petang":
            if hour < 12: hour += 12
        elif desc == "malam":
            if 1 <= hour < 12: hour += 12

        try:
            t = time(hour, minute, second)
            matches.append(t)
        except ValueError:
            continue
        
    return matches

def parse_date_string(
    word, 
    normalize_date=True, 
    dateparser_settings={'TIMEZONE': 'GMT+8'},
    english=False,
):
    if english:
        month_map = bulan_en
    else:
        month_map = bulan
    try:
        parsed = dateparser.parse(word, settings=dateparser_settings)
        if parsed:
            word = parsed.strftime('%d/%m/%Y')
            if normalize_date:
                day, month, year = word.split('/')
                day = cardinal(day, english=english)

                month = month_map[int(month)].title()
                year = cardinal(year, english=english)
                word = f'{day} {month} {year}'

    except Exception as e:
        logger.warning(str(e))
    return word

def fix_spacing(text):
    
    quote_pattern = r'"([^"]*)"'
    def fix_quotes(match):
        content = match.group(1).strip()
        return f'"{content}"'
    
    text = re.sub(quote_pattern, fix_quotes, text)
    paren_pattern = r'\(([^)]*)\)'
    
    def fix_parens(match):
        content = match.group(1).strip()
        return f'({content})'
    text = re.sub(paren_pattern, fix_parens, text)
    
    punct_pattern = r'\s+([,.?!])'
    text = re.sub(punct_pattern, r'\1', text)
    
    return text

def replace_laugh(string, replace_with='haha'):
    return replace_any(string, laughing, replace_with)

def replace_mengeluh(string, replace_with='aduh'):
    return replace_any(string, MENGELUH, replace_with)

def replace_betul(string, replace_with='betul'):
    return replace_any(string, BETUL, replace_with)

def replace_gembira(string, replace_with='gembira'):
    return replace_any(string, GEMBIRA, replace_with)
