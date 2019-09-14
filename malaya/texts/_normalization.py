import re
from ..num2word import to_cardinal, to_ordinal
from ..word2num import word2num
from ._tatabahasa import (
    rules_normalizer,
    consonants,
    vowels,
    sounds,
    hujung_malaysian,
    calon_dictionary,
)

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


def _normalize_money(
    word, currency = {'dollar': '$', 'ringgit': 'RM', 'pound': '£', 'euro': '€'}
):
    splitted = word.split()
    if splitted[-1] in ['dollar', 'ringgit', 'pound', 'euro']:
        v = word2num(' '.join(splitted[:-1]))
        return currency[splitted[-1]] + str(v)
    else:
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
        x = re.sub('-', ' ', x, count = 10)
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
        if (
            re.match('^\$', x)
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
            x, n = re.split("(\d+(?:[\.,']\d+)?)", x)[1:]
            x = re.sub(',', '', x, count = 10)
            labels = []
            for c in n:
                if re.match('.*(M|m)$', c):
                    labels.append(1e6)
                elif re.match('.*(b|B)$', c):
                    labels.append(1e9)
                elif re.match('.*(k|K)$', c):
                    labels.append(1e3)

            x = float(x)
            if cent:
                x = x / 100
            for l in labels:
                x = x * l

            text = cardinal(str(x))
            c = '$%s' % (str(x))
            x = '%s dollar' % (text)
            return re.sub(r'[ ]+', ' ', x.lower()).strip(), c

        elif (
            re.match('^US', x)
            or x.lower().endswith('dollar')
            or x.lower().endswith('cent')
        ):
            x = x.lower()
            if not x.startswith('US') and x.endswith('cent'):
                cent = True
            else:
                cent = False
            x = x.replace('US', '').replace('dollar', '').replace('cent', '')
            x = re.sub(r'[ ]+', ' ', x).strip()
            x, n = re.split("(\d+(?:[\.,']\d+)?)", x)[1:]
            x = re.sub(',', '', x, count = 10)
            labels = []
            for c in n:
                if re.match('.*(M|m)$', c):
                    labels.append(1e6)
                elif re.match('.*(b|B)$', c):
                    labels.append(1e9)
                elif re.match('.*(k|K)$', c):
                    labels.append(1e3)

            x = float(x)
            if cent:
                x = x / 100
            for l in labels:
                x = x * l

            text = cardinal(str(x))
            c = '$%s' % (str(x))
            x = '%s dollar' % (text)
            return re.sub(r'[ ]+', ' ', x.lower()).strip(), c

        elif (
            re.match('^\£', x)
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
            x, n = re.split("(\d+(?:[\.,']\d+)?)", x)[1:]
            x = re.sub(',', '', x, count = 10)
            labels = []
            for c in n:
                if re.match('.*(M|m)$', c):
                    labels.append(1e6)
                elif re.match('.*(b|B)$', c):
                    labels.append(1e9)
                elif re.match('.*(k|K)$', c):
                    labels.append(1e3)

            x = float(x)
            if cent:
                x = x / 100
            for l in labels:
                x = x * l

            text = cardinal(str(x))
            c = '£%s' % (str(x))
            x = '%s pound' % (text)
            return re.sub(r'[ ]+', ' ', x.lower()).strip(), c

        elif (
            re.match('^\€', x)
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
            x, n = re.split("(\d+(?:[\.,']\d+)?)", x)[1:]
            x = re.sub(',', '', x, count = 10)
            labels = []
            for c in n:
                if re.match('.*(M|m)$', c):
                    labels.append(1e6)
                elif re.match('.*(b|B)$', c):
                    labels.append(1e9)
                elif re.match('.*(k|K)$', c):
                    labels.append(1e3)

            x = float(x)
            if cent:
                x = x / 100
            for l in labels:
                x = x * l

            text = cardinal(str(x))
            c = '€%s' % (str(x))
            x = '%s euro' % (text)
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
            x, n = re.split("(\d+(?:[\.,']\d+)?)", x)[1:]
            x = re.sub(',', '', x, count = 10)
            labels = []
            for c in n:
                if re.match('.*(M|m)$', c):
                    labels.append(1e6)
                elif re.match('.*(b|B)$', c):
                    labels.append(1e9)
                elif re.match('.*(k|K)$', c):
                    labels.append(1e3)

            x = float(x)
            if cent:
                x = x / 100
            for l in labels:
                x = x * l

            text = cardinal(str(x))
            c = 'RM%s' % (str(x))
            x = '%s ringgit' % (text)
            return re.sub(r'[ ]+', ' ', x.lower()).strip(), c
        return x, None

    except Exception as e:
        return x, None
