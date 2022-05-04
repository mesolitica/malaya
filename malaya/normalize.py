import re
import dateparser
import itertools
from malaya.num2word import to_cardinal
from malaya.text.function import (
    is_english,
    is_malay,
    multireplace,
    case_of,
    replace_laugh,
    replace_mengeluh,
)
from malaya.text.regex import (
    _past_date_string,
    _now_date_string,
    _future_date_string,
    _yesterday_tomorrow_date_string,
    _depan_date_string,
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
    bulan,
)
from malaya.text.normalization import (
    _remove_postfix,
    _normalize_title,
    _is_number_regex,
    _string_to_num,
    _replace_compound,
    cardinal,
    digit_unit,
    rom_to_int,
    ordinal,
    fraction,
    money,
    ignore_words,
    digit,
)
from malaya.text.rules import rules_normalizer
from malaya.cluster import cluster_words
from malaya.function import validator
from herpetologist import check_type
from typing import Callable
import logging

logger = logging.getLogger('malaya.normalize')


def normalized_entity(normalized):
    money_ = re.findall(_expressions['money'], normalized)
    money_ = [(s, money(s)[1]) for s in money_]
    dates_ = re.findall(_expressions['date'], normalized)

    past_date_string_ = re.findall(_past_date_string, normalized)
    logger.debug(f'past_date_string_: {past_date_string_}')
    now_date_string_ = re.findall(_now_date_string, normalized)
    logger.debug(f'now_date_string_: {now_date_string_}')
    future_date_string_ = re.findall(_future_date_string, normalized)
    logger.debug(f'future_date_string_: {future_date_string_}')
    yesterday_date_string_ = re.findall(
        _yesterday_tomorrow_date_string, normalized
    )
    logger.debug(f'yesterday_date_string_: {yesterday_date_string_}')
    depan_date_string_ = re.findall(_depan_date_string, normalized)
    logger.debug(f'depan_date_string_: {depan_date_string_}')
    today_time_ = re.findall(_today_time, normalized)
    logger.debug(f'today_time_: {today_time_}')
    time_ = re.findall(_expressions['time'], normalized)
    logger.debug(f'time_: {time_}')

    left_datetime_ = [
        f'{i[0]} {i[1]}' for i in re.findall(_left_datetime, normalized)
    ]
    logger.debug(f'left_datetime_: {left_datetime_}')
    right_datetime_ = [
        f'{i[0]} {i[1]}' for i in re.findall(_right_datetime, normalized)
    ]
    logger.debug(f'right_datetime_: {left_datetime_}')
    today_left_datetime_ = [
        f'{i[0]} {i[1]}' for i in re.findall(_left_datetodaytime, normalized)
    ]
    logger.debug(f'today_left_datetime_: {today_left_datetime_}')
    today_right_datetime_ = [
        f'{i[0]} {i[1]}' for i in re.findall(_right_datetodaytime, normalized)
    ]
    logger.debug(f'today_right_datetime_: {today_right_datetime_}')
    left_yesterdaydatetime_ = [
        f'{i[0]} {i[1]}'
        for i in re.findall(_left_yesterdaydatetime, normalized)
    ]
    logger.debug(f'left_yesterdaydatetime_: {left_yesterdaydatetime_}')
    right_yesterdaydatetime_ = [
        f'{i[0]} {i[1]}'
        for i in re.findall(_right_yesterdaydatetime, normalized)
    ]
    logger.debug(f'right_yesterdaydatetime_: {right_yesterdaydatetime_}')
    left_yesterdaydatetodaytime_ = [
        f'{i[0]} {i[1]}'
        for i in re.findall(_left_yesterdaydatetodaytime, normalized)
    ]
    logger.debug(f'left_yesterdaydatetodaytime_: {left_yesterdaydatetodaytime_}')
    right_yesterdaydatetodaytime_ = [
        f'{i[0]} {i[1]}'
        for i in re.findall(_right_yesterdaydatetodaytime, normalized)
    ]
    logger.debug(f'right_yesterdaydatetodaytime_: {right_yesterdaydatetodaytime_}')

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
    dates_ = [d.replace('.', ':') for d in dates_ if not isinstance(d, tuple)]
    dates_ = [multireplace(s, date_replace) for s in dates_]
    dates_ = [re.sub(r'[ ]+', ' ', s).strip() for s in dates_]
    dates_ = cluster_words(dates_)
    dates_ = {s: dateparser.parse(s) for s in dates_}
    money_ = {s[0]: s[1] for s in money_}

    return dates_, money_


def check_repeat(word):
    if word[-1].isdigit() and not word[-2].isdigit():
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
            or re.findall(_expressions['money'], word.lower())
            or re.findall(_expressions['number'], word)
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
        normalize_text: bool = True,
        normalize_entity: bool = True,
        normalize_url: bool = False,
        normalize_email: bool = False,
        normalize_year: bool = True,
        normalize_telephone: bool = True,
        normalize_date: bool = True,
        normalize_time: bool = True,
        check_english_func=is_english,
        check_malay_func=is_malay,
        **kwargs,
    ):
        """
        Normalize a string.

        Parameters
        ----------
        string : str
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
        normalize_date: bool, (default=True)
            if True, `01/12/2001` -> `satu disember dua ribu satu`.
            if True, `Jun 2017` -> `satu Jun dua ribu tujuh belas`.
            if True, `2017 Jun` -> `satu Jun dua ribu tujuh belas`.
            if False, `2017 Jun` -> `01/06/2017`.
            if False, `Jun 2017` -> `01/06/2017`.
        normalize_time: bool, (default=True)
            if True, `pukul 2.30` -> `pukul dua tiga puluh minit`.
            if False, `pukul 2.30` -> `'02:00:00'`
        check_english_func: Callable, (default=malaya.text.is_english)
            function to check a word in english dictionary, default is malaya.text.is_english.
        check_malay_func: Callable, (default=malaya.text.is_malay)
            function to check a word in malay dictionary, default is malaya.text.is_malay.

        Returns
        -------
        string: {'normalize', 'date', 'money'}
        """
        tokenized = self._tokenizer(string)
        s = f'tokenized: {tokenized}'
        logger.debug(s)
        string = ' '.join(tokenized)
        string = groupby(string)

        if normalize_text:
            string = replace_laugh(string)
            string = replace_mengeluh(string)
            string = _replace_compound(string)

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
            logger.debug(s)

            if word in '~@#$%^&*()_+{}|[:"\'];<>,.?/-':
                s = f'index: {index}, word: {word}, condition punct'
                logger.debug(s)
                result.append(word)
                index += 1
                continue

            normalized.append(rules_normalizer.get(word_lower, word_lower))

            if word_lower in ignore_words:
                s = f'index: {index}, word: {word}, condition ignore words'
                logger.debug(s)
                result.append(word)
                index += 1
                continue

            if (
                first_c
                and not len(re.findall(_expressions['money'], word_lower))
                and not len(re.findall(_expressions['date'], word_lower))
            ):
                s = f'index: {index}, word: {word}, condition not in money and date'
                logger.debug(s)
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

            if check_english_func is not None:
                s = f'index: {index}, word: {word}, condition check english'
                logger.debug(s)
                if check_english_func(word_lower):
                    result.append(word)
                    index += 1
                    continue

            if check_malay_func is not None:
                s = f'index: {index}, word: {word}, condition check malay'
                logger.debug(s)
                if check_malay_func(word_lower) and word_lower not in ['pada', 'ke']:
                    result.append(word)
                    index += 1
                    continue

            if len(word) > 2 and normalize_text:
                s = f'index: {index}, word: {word}, condition len(word) > 2 and norm text'
                logger.debug(s)
                if word[-2] in consonants and word[-1] == 'e':
                    word = word[:-1] + 'a'

            if word[0] == 'x' and len(word) > 1 and normalize_text:
                s = f'index: {index}, word: {word}, condition word[0] == `x` and len(word) > 1 and norm text'
                logger.debug(s)
                result_string = 'tak '
                word = word[1:]
            else:
                s = f'index: {index}, word: {word}, condition else for (word[0] == `x` and len(word) > 1 and norm text)'
                logger.debug(s)
                result_string = ''

            if word_lower == 'ke' and index < (len(tokenized) - 2):
                s = f'index: {index}, word: {word}, condition ke'
                logger.debug(s)
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
                s = f'index: {index}, word: {word}, condition hingga'
                logger.debug(s)
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
                s = f'index: {index}, word: {word}, condition pada hari bulan'
                logger.debug(s)
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
                s = f'index: {index}, word: {word}, condition tahun'
                logger.debug(s)
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
                s = f'index: {index}, word: {word}, condition fraction'
                logger.debug(s)
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

            if re.findall(_expressions['money'], word_lower):
                s = f'index: {index}, word: {word}, condition money'
                logger.debug(s)
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

            if re.findall(_expressions['date'], word_lower):
                s = f'index: {index}, word: {word}, condition date'
                logger.debug(s)
                word = word_lower
                word = multireplace(word, date_replace)
                word = re.sub(r'[ ]+', ' ', word).strip()
                try:
                    s = f'index: {index}, word: {word}, parsing date'
                    logger.debug(s)
                    parsed = dateparser.parse(word)
                    if parsed:
                        word = parsed.strftime('%d/%m/%Y')
                        if normalize_date:
                            day, month, year = word.split('/')
                            day = cardinal(day)
                            month = bulan[int(month)].title()
                            year = cardinal(year)

                            word = f'{day} {month} {year}'
                except Exception as e:
                    logger.warning(str(e))
                result.append(word)

                index += 1
                continue

            if (
                re.findall(_expressions['time'], word_lower)
                or re.findall(_expressions['time_pukul'], word_lower)
            ):
                s = f'index: {index}, word: {word}, condition time'
                logger.debug(s)
                word = word_lower
                word = multireplace(word, date_replace)
                word = re.sub(r'[ ]+', ' ', word).strip()
                try:
                    s = f'index: {index}, word: {word}, parsing time'
                    logger.debug(s)
                    parsed = dateparser.parse(word.replace('.', ':'))
                    if parsed:
                        word = parsed.strftime('%H:%M:%S')
                        if normalize_time:
                            hour, minute, second = word.split(':')
                            hour = cardinal(hour)
                            if int(minute) > 0:
                                minute = cardinal(minute)
                                minute = f'{minute} minit'
                            else:
                                minute = ''
                            if int(second) > 0:
                                second = cardinal(second)
                                second = f'{second} saat'
                            else:
                                second = ''
                            word = f'pukul {hour} {minute} {second}'
                            word = re.sub(r'[ ]+', ' ', word).strip()
                except Exception as e:
                    logger.warning(str(e))
                result.append(word)
                index += 1
                continue

            if re.findall(_expressions['hashtag'], word_lower):
                s = f'index: {index}, word: {word}, condition hashtag'
                logger.debug(s)
                result.append(word)
                index += 1
                continue

            if re.findall(_expressions['url'], word_lower):
                s = f'index: {index}, word: {word}, condition url'
                logger.debug(s)
                if normalize_url:
                    word = word.replace('://', ' ').replace('.', ' dot ')
                    word = put_spacing_num(word)
                    word = word.replace('https', 'HTTPS').replace('http', 'HTTP').replace('www', 'WWW')
                result.append(word)
                index += 1
                continue

            if re.findall(_expressions['email'], word_lower):
                s = f'index: {index}, word: {word}, condition email'
                logger.debug(s)
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
                s = f'index: {index}, word: {word}, condition phone'
                logger.debug(s)
                if normalize_telephone:
                    splitted = word.split('-')
                    if len(splitted) == 2:
                        left = put_spacing_num(splitted[0])
                        right = put_spacing_num(splitted[1])
                        word = f'{left}, {right}'
                result.append(word)
                index += 1
                continue

            if re.findall(_expressions['user'], word_lower):
                s = f'index: {index}, word: {word}, condition user'
                logger.debug(s)
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
                s = f'index: {index}, word: {word}, condition units'
                logger.debug(s)
                word = word.replace(' ', '')
                result.append(digit_unit(word))
                index += 1
                continue

            if (
                re.findall(_expressions['percent'], word_lower)
            ):
                s = f'index: {index}, word: {word}, condition percent'
                logger.debug(s)
                word = word.replace('%', '')
                result.append(cardinal(word) + ' peratus')
                index += 1
                continue

            if re.findall(_expressions['ic'], word_lower):
                s = f'index: {index}, word: {word}, condition IC'
                logger.debug(s)
                result.append(digit(word))
                index += 1
                continue

            if (
                re.findall(_expressions['number'], word_lower)
                and word_lower[0] == '0'
                and '.' not in word_lower
            ):
                s = f'index: {index}, word: {word}, condition digit and word[0] == `0`'
                logger.debug(s)
                result.append(digit(word))
                index += 1
                continue

            cardinal_ = cardinal(word)
            if cardinal_ != word:
                s = f'index: {index}, word: {word}, condition cardinal'
                logger.debug(s)
                result.append(cardinal_)
                index += 1
                continue

            normalized_ke = ordinal(word)
            if normalized_ke != word:
                s = f'index: {index}, word: {word}, condition normalized ke'
                logger.debug(s)
                result.append(normalized_ke)
                index += 1
                continue

            word, end_result_string = _remove_postfix(word)
            if normalize_text:
                word, repeat = check_repeat(word)
            else:
                repeat = 1

            if normalize_text:
                s = f'index: {index}, word: {word}, condition normalize text'
                logger.debug(s)
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
