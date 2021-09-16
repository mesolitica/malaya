import numpy as np
import json
import re
import dateparser
import itertools
from unidecode import unidecode
from malaya.num2word import to_cardinal, to_ordinal
from malaya.word2num import word2num
from malaya.text.function import (
    ENGLISH_WORDS,
    MALAY_WORDS,
    multireplace,
    case_of,
    replace_laugh,
    replace_mengeluh,
)
from malaya.text.regex import (
    _date,
    _past_date_string,
    _now_date_string,
    _future_date_string,
    _yesterday_tomorrow_date_string,
    _depan_date_string,
    _money,
    _expressions,
    _left_datetime,
    _right_datetime,
    _today_time,
    _left_datetodaytime,
    _right_datetodaytime,
    _left_yesterdaydatetime,
    _right_yesterdaydatetime,
    _left_yesterdaydatetodaytime,
    _right_yesterdaydatetodaytime,
)
from malaya.text.tatabahasa import (
    date_replace,
    consonants,
    sounds,
    hujung_malaysian,
)
from malaya.text.normalization import (
    _remove_postfix,
    _normalize_title,
    _is_number_regex,
    _string_to_num,
    _normalize_money,
    _replace_compoud,
    cardinal,
    digit,
    digit_unit,
    rom_to_int,
    ordinal,
    fraction,
    money,
    ignore_words,
)
from malaya.text.rules import rules_normalizer
from malaya.cluster import cluster_words
from malaya.function import validator
from herpetologist import check_type
import logging


def normalized_entity(normalized):
    money_ = re.findall(_money, normalized)
    money_ = [(s, money(s)[1]) for s in money_]
    dates_ = re.findall(_date, normalized)

    past_date_string_ = re.findall(_past_date_string, normalized)
    now_date_string_ = re.findall(_now_date_string, normalized)
    future_date_string_ = re.findall(_future_date_string, normalized)
    yesterday_date_string_ = re.findall(
        _yesterday_tomorrow_date_string, normalized
    )
    depan_date_string_ = re.findall(_depan_date_string, normalized)
    today_time_ = re.findall(_today_time, normalized)
    time_ = re.findall(_expressions['time'], normalized)

    left_datetime_ = [
        f'{i[0]} {i[1]}' for i in re.findall(_left_datetime, normalized)
    ]
    right_datetime_ = [
        f'{i[0]} {i[1]}' for i in re.findall(_right_datetime, normalized)
    ]
    today_left_datetime_ = [
        f'{i[0]} {i[1]}' for i in re.findall(_left_datetodaytime, normalized)
    ]
    today_right_datetime_ = [
        f'{i[0]} {i[1]}' for i in re.findall(_right_datetodaytime, normalized)
    ]
    left_yesterdaydatetime_ = [
        f'{i[0]} {i[1]}'
        for i in re.findall(_left_yesterdaydatetime, normalized)
    ]
    right_yesterdaydatetime_ = [
        f'{i[0]} {i[1]}'
        for i in re.findall(_right_yesterdaydatetime, normalized)
    ]
    left_yesterdaydatetodaytime_ = [
        f'{i[0]} {i[1]}'
        for i in re.findall(_left_yesterdaydatetodaytime, normalized)
    ]
    right_yesterdaydatetodaytime_ = [
        f'{i[0]} {i[1]}'
        for i in re.findall(_right_yesterdaydatetodaytime, normalized)
    ]

    dates_ = (
        dates_
        + past_date_string_
        + now_date_string_
        + future_date_string_
        + yesterday_date_string_
        + depan_date_string_
        + time_
        + today_time_
        + left_datetime_
        + right_datetime_
        + today_left_datetime_
        + today_right_datetime_
        + left_yesterdaydatetime_
        + right_yesterdaydatetime_
        + left_yesterdaydatetodaytime_
        + right_yesterdaydatetodaytime_
    )
    dates_ = [multireplace(s, date_replace) for s in dates_]
    dates_ = [re.sub(r'[ ]+', ' ', s).strip() for s in dates_]
    dates_ = cluster_words(dates_)
    dates_ = {s: dateparser.parse(s) for s in dates_}
    money_ = {s[0]: s[1] for s in money_}

    return dates_, money_


def check_repeat(word):
    if word[-1].isdigit():
        repeat = int(word[-1])
        word = word[:-1]
    else:
        repeat = 1

    if repeat < 1:
        repeat = 1
    return word, repeat


def groupby(string):
    results = []
    for word in string.split():
        if not (
            _is_number_regex(word)
            or re.findall(_expressions['url'], word)
            or re.findall(_money, word.lower())
        ):
            word = ''.join([''.join(s)[:2] for _, s in itertools.groupby(word)])
        results.append(word)
    return ' '.join(results)


def put_spacing_num(string):
    string = re.sub('[A-Za-z]+', lambda ele: ' ' + ele[0] + ' ', string).split()
    for i in range(len(string)):
        if _is_number_regex(string[i]):
            string[i] = ' '.join([to_cardinal(int(n)) for n in string[i]])
    string = ' '.join(string)
    return re.sub(r'[ ]+', ' ', string).strip()


class Normalizer:
    def __init__(self, tokenizer, speller=None):
        self._tokenizer = tokenizer
        self._speller = speller

    @check_type
    def normalize(
        self,
        string: str,
        check_english: bool = True,
        normalize_text: bool = True,
        normalize_entity: bool = True,
        normalize_url: bool = False,
        normalize_email: bool = False,
        normalize_year: bool = True,
        normalize_telephone: bool = True,
    ):
        """
        Normalize a string.

        Parameters
        ----------
        string : str
        check_english: bool, (default=True)
            check a word in english dictionary.
        normalize_text: bool, (default=True)
            if True, will try to replace shortforms with internal corpus.
        normalize_entity: bool, (default=True)
            normalize entities, only effect `date`, `datetime`, `time` and `money` patterns string only.
        normalize_url: bool, (default=False)
            if True, replace `://` with empty and `.` with `dot`.
            `https://huseinhouse.com` -> `https huseinhouse dot com`.
        normalize_email: bool, (default=False)
            if True, replace `@` with `di`, `.` with `dot`.
            `husein.zol05@gmail.com` -> `husein dot zol kosong lima di gmail dot com`.
        normalize_year: bool, (default=True)
            if True, `tahun 1987` -> `tahun sembilan belas lapan puluh tujuh`.
            if True, `1970-an` -> `sembilan belas tujuh puluh an`.
            if False, `tahun 1987` -> `tahun seribu sembilan ratus lapan puluh tujuh`.
        normalize_telephone: bool, (default=True)
            if True, `no 012-1234567` -> `no kosong satu dua, satu dua tiga empat lima enam tujuh`

        Returns
        -------
        string: normalized string
        """

        string = ' '.join(self._tokenizer(string))
        string = groupby(string)

        if normalize_text:
            string = replace_laugh(string)
            string = replace_mengeluh(string)
            string = _replace_compoud(string)

        if hasattr(self._speller, 'normalize_elongated'):
            string = [
                self._speller.normalize_elongated(word)
                if len(re.findall(r'(.)\1{1}', word))
                and not word[0].isupper()
                and not word.lower().startswith('ke-')
                and not _is_number_regex(word)
                else word
                for word in string.split()
            ]
            string = ' '.join(string)

        result, normalized = [], []

        tokenized = self._tokenizer(string)
        index = 0
        while index < len(tokenized):
            word = tokenized[index]
            word_lower = word.lower()
            word_upper = word.upper()
            first_c = word[0].isupper()

            s = f'index: {index}, word: {word}, queue: {result}'
            logging.debug(s)

            if word in '~@#$%^&*()_+{}|[:"\'];<>,.?/-':
                result.append(word)
                index += 1
                continue

            normalized.append(rules_normalizer.get(word_lower, word_lower))

            if word_lower in ignore_words:
                result.append(word)
                index += 1
                continue

            if first_c and not len(re.findall(_money, word_lower)):
                if word_lower in rules_normalizer and normalize_text:
                    result.append(case_of(word)(rules_normalizer[word_lower]))
                    index += 1
                    continue
                elif word_upper not in ['KE', 'PADA', 'RM', 'SEN', 'HINGGA']:
                    result.append(
                        _normalize_title(word) if normalize_text else word
                    )
                    index += 1
                    continue

            if check_english:
                if word_lower in ENGLISH_WORDS:
                    result.append(word)
                    index += 1
                    continue

            if word_lower in MALAY_WORDS and word_lower not in ['pada', 'ke']:
                result.append(word)
                index += 1
                continue

            if len(word) > 2 and normalize_text:
                if word[-2] in consonants and word[-1] == 'e':
                    word = word[:-1] + 'a'

            if word[0] == 'x' and len(word) > 1 and normalize_text:
                result_string = 'tak '
                word = word[1:]
            else:
                result_string = ''

            if word_lower == 'ke' and index < (len(tokenized) - 2):
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

            if word_lower == 'pada' and index < (len(tokenized) - 3):
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

            if (
                word_lower in ['tahun', 'thun']
                and index < (len(tokenized) - 1)
                and normalize_year
            ):
                if (
                    _is_number_regex(tokenized[index + 1])
                    and len(tokenized[index + 1]) == 4
                ):
                    t = tokenized[index + 1]
                    if t[1] != '0':
                        l = to_cardinal(int(t[:2]))
                        r = to_cardinal(int(t[2:]))
                        c = f'{l} {r}'
                    else:
                        c = to_cardinal(int(t))
                    if (
                        index < (len(tokenized) - 3)
                        and tokenized[index + 2] == '-'
                        and tokenized[index + 3].lower() == 'an'
                    ):
                        end = 'an'
                        plus = 4
                    else:
                        end = ''
                        plus = 2
                    result.append(f'tahun {c}{end}')
                    index += plus
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

                if (
                    tokenized[index + 1] == '-'
                    and tokenized[index + 2].lower() == 'an'
                    and normalize_year
                    and len(word) == 4
                ):
                    t = word
                    if t[1] != '0':
                        l = to_cardinal(int(t[:2]))
                        r = to_cardinal(int(t[2:]))
                        c = f'{l} {r}'
                    else:
                        c = to_cardinal(int(t))
                    result.append(f'{c}an')
                    index += 3
                    continue

            if re.findall(_money, word_lower):
                money_, _ = money(word)
                result.append(money_)
                if index < (len(tokenized) - 1):
                    if tokenized[index + 1].lower() in ('sen', 'cent'):
                        index += 2
                    else:
                        index += 1
                else:
                    index += 1
                continue

            if re.findall(_date, word_lower):
                word = word_lower
                word = multireplace(word, date_replace)
                word = re.sub(r'[ ]+', ' ', word).strip()
                parsed = dateparser.parse(word)
                if parsed:
                    result.append(parsed.strftime('%d/%m/%Y'))
                else:
                    result.append(word)
                index += 1
                continue

            if re.findall(_expressions['time'], word_lower):
                word = word_lower
                word = multireplace(word, date_replace)
                word = re.sub(r'[ ]+', ' ', word).strip()
                parsed = dateparser.parse(word)
                if parsed:
                    result.append(parsed.strftime('%H:%M:%S'))
                else:
                    result.append(word)
                index += 1
                continue

            if re.findall(_expressions['hashtag'], word_lower):
                result.append(word)
                index += 1
                continue

            if re.findall(_expressions['url'], word_lower):
                if normalize_url:
                    word = word.replace('://', ' ').replace('.', ' dot ')
                    word = put_spacing_num(word)
                result.append(word)
                index += 1
                continue

            if re.findall(_expressions['email'], word_lower):
                if normalize_email:
                    word = (
                        word.replace('://', ' ')
                        .replace('.', ' dot ')
                        .replace('@', ' di ')
                    )
                    word = put_spacing_num(word)
                result.append(word)
                index += 1
                continue

            if re.findall(_expressions['phone'], word_lower):
                if normalize_telephone:
                    splitted = word.split('-')
                    left = put_spacing_num(splitted[0])
                    right = put_spacing_num(splitted[1])
                    word = f'{left}, {right}'
                result.append(word)
                index += 1
                continue

            if re.findall(_expressions['user'], word_lower):
                result.append(word)
                index += 1
                continue

            if (
                re.findall(_expressions['temperature'], word_lower)
                or re.findall(_expressions['distance'], word_lower)
                or re.findall(_expressions['volume'], word_lower)
                or re.findall(_expressions['duration'], word_lower)
                or re.findall(_expressions['weight'], word_lower)
            ):
                word = word.replace(' ', '')
                result.append(digit_unit(word))
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
            word, repeat = check_repeat(word)

            if normalize_text:
                if word in sounds:
                    selected = sounds[word]
                elif word in rules_normalizer:
                    selected = rules_normalizer[word]
                elif self._speller:
                    selected = self._speller.correct(
                        word, string=' '.join(tokenized), index=index
                    )
                else:
                    selected = word

            else:
                selected = word

            selected = '-'.join([selected] * repeat)
            result.append(result_string + selected + end_result_string)
            index += 1

        result = ' '.join(result)
        normalized = ' '.join(normalized)

        if normalize_entity:
            dates_, money_ = normalized_entity(normalized)

        else:
            dates_, money_ = {}, {}
        return {'normalize': result, 'date': dates_, 'money': money_}


def normalizer(speller=None, **kwargs):
    """
    Load a Normalizer using any spelling correction model.

    Parameters
    ----------
    speller : spelling correction object, optional (default = None)

    Returns
    -------
    result: malaya.normalize.Normalizer class
    """

    validator.validate_object_methods(
        speller, ['correct', 'normalize_elongated'], 'speller'
    )

    from malaya.preprocessing import Tokenizer

    tokenizer = Tokenizer(**kwargs).tokenize
    return Normalizer(tokenizer, speller)
