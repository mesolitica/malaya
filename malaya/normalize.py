import re
import dateparser
import itertools
import math
from malaya.num2word import to_cardinal
from malaya.text.function import (
    is_laugh,
    is_mengeluh,
    multireplace,
    case_of,
    PUNCTUATION,
)
from malaya.dictionary import is_english, is_malay, is_malaysia_location
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
    unpack_english_contractions,
    repeat_word,
    replace_laugh,
    replace_mengeluh,
    replace_betul,
    digits,
)
from malaya.text.rules import rules_normalizer, rules_normalizer_rev
from malaya.cluster import cluster_words
from malaya.function import validator
from malaya.preprocessing import Tokenizer, demoji
from herpetologist import check_type
from typing import Callable, List
import logging

logger = logging.getLogger(__name__)


def normalized_entity(normalized):

    normalized = re.sub(_expressions['ic'], '', normalized)
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
    if len(word) < 2:
        return word, 1

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
    def __init__(self, tokenizer, speller=None, stemmer=None):
        self._tokenizer = tokenizer
        self._speller = speller
        self._stemmer = stemmer
        self._demoji = None
        self._compiled = {
            k.lower(): re.compile(_expressions[k]) for k, v in _expressions.items()
        }

    @check_type
    def normalize(
        self,
        string: str,
        normalize_text: bool = True,
        normalize_url: bool = False,
        normalize_email: bool = False,
        normalize_year: bool = True,
        normalize_telephone: bool = True,
        normalize_date: bool = True,
        normalize_time: bool = True,
        normalize_emoji: bool = True,
        normalize_elongated: bool = True,
        normalize_hingga: bool = True,
        normalize_pada_hari_bulan: bool = True,
        normalize_fraction: bool = True,
        normalize_money: bool = True,
        normalize_units: bool = True,
        normalize_percent: bool = True,
        normalize_ic: bool = True,
        normalize_number: bool = True,
        normalize_x_kali: bool = True,
        normalize_cardinal: bool = True,
        normalize_ordinal: bool = True,
        normalize_entity: bool = True,
        expand_contractions: bool = True,
        check_english_func=is_english,
        check_malay_func=is_malay,
        translator: Callable = None,
        language_detection_word: Callable = None,
        acceptable_language_detection: List[str] = ['EN', 'CAPITAL', 'NOT_LANG'],
        segmenter: Callable = None,
        text_scorer: Callable = None,
        text_scorer_window: int = 2,
        not_a_word_threshold: float = 1e-4,
        **kwargs,
    ):
        """
        Normalize a string.

        Parameters
        ----------
        string : str
        normalize_text: bool, optional (default=True)
            if True, will try to replace shortforms with internal corpus.
        normalize_url: bool, optional (default=False)
            if True, replace `://` with empty and `.` with `dot`.
            `https://huseinhouse.com` -> `https huseinhouse dot com`.
        normalize_email: bool, optional (default=False)
            if True, replace `@` with `di`, `.` with `dot`.
            `husein.zol05@gmail.com` -> `husein dot zol kosong lima di gmail dot com`.
        normalize_year: bool, optional (default=True)
            if True, `tahun 1987` -> `tahun sembilan belas lapan puluh tujuh`.
            if True, `1970-an` -> `sembilan belas tujuh puluh an`.
            if False, `tahun 1987` -> `tahun seribu sembilan ratus lapan puluh tujuh`.
        normalize_telephone: bool, optional (default=True)
            if True, `no 012-1234567` -> `no kosong satu dua, satu dua tiga empat lima enam tujuh`
        normalize_date: bool, optional (default=True)
            if True, `01/12/2001` -> `satu disember dua ribu satu`.
            if True, `Jun 2017` -> `satu Jun dua ribu tujuh belas`.
            if True, `2017 Jun` -> `satu Jun dua ribu tujuh belas`.
            if False, `2017 Jun` -> `01/06/2017`.
            if False, `Jun 2017` -> `01/06/2017`.
        normalize_time: bool, optional (default=True)
            if True, `pukul 2.30` -> `pukul dua tiga puluh minit`.
            if False, `pukul 2.30` -> `'02:00:00'`
        normalize_emoji: bool, (default=True)
            if True, `🔥` -> `emoji api`
            Load from `malaya.preprocessing.demoji`.
        normalize_elongated: bool, optional (default=True)
            if True, `betuii` -> `betui`.
        normalize_hingga: bool, optional (default=True)
            if True, `2011 - 2019` -> `dua ribu sebelas hingga dua ribu sembilan belas`
        normalize_pada_hari_bulan: bool, optional (default=True)
            if True, `pada 10/4` -> `pada sepuluh hari bulan empat`
        normalize_fraction: bool, optional (default=True)
            if True, `10 /4` -> `sepuluh per empat`
        normalize_money: bool, optional (default=True)
            if True, `rm10.4m` -> `sepuluh juta empat ratus ribu ringgit`
        normalize_units: bool, optional (default=True)
            if True, `61.2 kg` -> `enam puluh satu perpuluhan dua kilogram`
        normalize_percent: bool, optional (default=True)
            if True, `0.8%` -> `kosong perpuluhan lapan peratus`
        normalize_ic: bool, optional (default=True)
            if True, `911111-01-1111` -> `sembilan satu satu satu satu satu sempang kosong satu sempang satu satu satu satu`
        normalize_number: bool, optional (default=True)
            if True `0123` -> `kosong satu dua tiga`
        normalize_x_kali: bool, optional (default=True)
            if True `10x` -> 'sepuluh kali'
        normalize_cardinal: bool, optional (default=True)
            if True, `123` -> `seratus dua puluh tiga`
        normalize_ordinal: bool, optional (default=True)
            if True, `ke-123` -> `keseratus dua puluh tiga`
        normalize_entity: bool, optional (default=True)
            normalize entities, only effect `date`, `datetime`, `time` and `money` patterns string only.
        expand_contractions: bool, optional (default=True)
            expand english contractions.
        check_english_func: Callable, optional (default=malaya.text.function.is_english)
            function to check a word in english dictionary, default is malaya.text.function.is_english.
            this parameter also will be use for malay text normalization.
        check_malay_func: Callable, optional (default=malaya.text.function.is_malay)
            function to check a word in malay dictionary, default is malaya.text.function.is_malay.
        translator: Callable, optional (default=None)
            function to translate EN word to MS word.
        language_detection_word: Callable, optional (default=None)
            function to detect language for each words to get better translation results.
        acceptable_language_detection: List[str], optional (default=['EN', 'CAPITAL', 'NOT_LANG'])
            only translate substrings if the results from `language_detection_word` is in `acceptable_language_detection`.
        segmenter: Callable, optional (default=None)
            function to segmentize word.
            If provide, it will expand a word, apaitu -> apa itu
        text_scorer: Callable, optional (default=None)
            function to validate upper word. 
            If lower case score is higher or equal than upper case score, will choose lower case.
        text_scorer_window: int, optional (default=2)
            size of lookback and lookforward to validate upper word.
        not_a_word_threshold: float, optional (default=1e-4)
            assume a word is not a human word if score lower than `not_a_word_threshold`.
            only usable if passed `text_scorer` parameter.

        Returns
        -------
        result: {'normalize', 'date', 'money'}
        """

        if normalize_emoji:
            if self._demoji is None:

                logger.info('caching malaya.preprocessing.demoji inside normalizer')
                self._demoji = demoji().demoji

            result_demoji = self._demoji(string)
        else:
            result_demoji = None

        if expand_contractions:
            logger.debug(f'before expand_contractions: {string}')
            string = unpack_english_contractions(string)
            logger.debug(f'after expand_contractions: {string}')

        tokenized = self._tokenizer(string)
        s = f'tokenized: {tokenized}'
        logger.debug(s)
        string = ' '.join(tokenized)
        string = groupby(string)

        if normalize_text:
            logger.debug(f'before normalize_text: {string}')
            string = replace_laugh(string)
            string = replace_mengeluh(string)
            string = replace_betul(string)
            string = _replace_compound(string)
            logger.debug(f'after normalize_text: {string}')

        if normalize_elongated:
            logger.debug(f'before normalize_elongated: {string}')
            normalized = []
            got_speller = hasattr(self._speller, 'normalize_elongated')
            for word in string.split():
                word_lower = word.lower()
                if (
                    len(re.findall(r'(.)\1{1}', word))
                    and not word[0].isupper()
                    and not word_lower.startswith('ke-')
                    and not len(re.findall(_expressions['email'], word))
                    and not len(re.findall(_expressions['url'], word))
                    and not len(re.findall(_expressions['hashtag'], word))
                    and not len(re.findall(_expressions['phone'], word))
                    and not len(re.findall(_expressions['money'], word))
                    and not len(re.findall(_expressions['date'], word))
                    and not len(re.findall(_expressions['ic'], word))
                    and not len(re.findall(_expressions['user'], word))
                    and not _is_number_regex(word)
                    and check_english_func is not None
                    and not check_english_func(word_lower)
                ):
                    word = self._compiled['normalize_elong'].sub(r'\1\1', word)
                    if got_speller:
                        word = self._speller.normalize_elongated(word)
                normalized.append(word)
            string = ' '.join(normalized)
            logger.debug(f'after normalize_elongated: {string}')

        result, normalized = [], []
        spelling_correction = {}
        spelling_correction_condition = {}

        tokenized = self._tokenizer(string)
        index = 0
        while index < len(tokenized):
            word = tokenized[index]
            word_lower = word.lower()
            word_upper = word.upper()
            word_title = word.title()
            first_c = word[0].isupper()

            s = f'index: {index}, word: {word}, queue: {result}'
            logger.debug(s)

            if word in PUNCTUATION:
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

            if re.findall(_expressions['ic'], word_lower):
                s = f'index: {index}, word: {word}, condition IC'
                logger.debug(s)
                if normalize_ic:
                    splitted = word.split('-')
                    ics = [digit(s) for s in splitted]
                    word = ' sempang '.join(ics)
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

            if normalize_emoji and word_lower in result_demoji:
                s = f'index: {index}, word: {word}, condition emoji'
                r = f'emoji {result_demoji[word_lower]}'
                if index - 1 >= 0:
                    if tokenized[index - 1] == '.':
                        r = r[0].upper() + r[1:]
                    elif len(result) and result[-1][-1] == ',':
                        pass
                    elif tokenized[index - 1] != ',':
                        r = f', {r}'

                if index + 1 < len(tokenized):
                    if tokenized[index + 1] == '.':
                        pass
                    elif tokenized[index + 1] != ',':
                        r = f'{r} ,'

                result.append(r)
                index += 1
                continue

            if text_scorer is not None:
                score = math.exp(text_scorer(word_lower))
                s = f'index: {index}, word: {word}, score: {score}, text_scorer is not None'
                logger.debug(s)
                if score <= not_a_word_threshold:
                    s = f'index: {index}, word: {word}, text_scorer(word_lower) <= not_a_word_threshold'
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

                    norm_title = _normalize_title(word) if normalize_text else word
                    if norm_title != word:
                        s = f'index: {index}, word: {word}, norm_title != word'
                        logger.debug(s)
                        result.append(norm_title)
                        index += 1
                        continue

                    titled = True
                    if len(word) > 1 and text_scorer is not None:
                        s = f'index: {index}, word: {word}, condition text_scorer is not None'
                        logger.debug(s)
                        l = ' '.join(result[-text_scorer_window:])
                        if len(l):
                            lower = f'{l} {word_lower}'
                            title = f'{l} {word_title}'
                            normal = f'{l} {word}'
                        else:
                            lower = word_lower
                            title = word_title
                            normal = word

                        if index + 1 < len(tokenized):
                            r = ' '.join(tokenized[index + 1: index + 1 + text_scorer_window])
                            if len(r):
                                lower = f'{lower} {r}'
                                title = f'{title} {r}'
                                normal = f'{normal} {r}'

                        lower_score = text_scorer(lower)
                        title_score = text_scorer(title)
                        normal_score = text_scorer(normal)
                        s = f'index: {index}, word: {word}, lower: {lower} , normal: {normal} , lower_score: {lower_score}, title_score: {title_score}, normal_score: {normal_score}'
                        logger.debug(s)
                        if lower_score > normal_score or title_score > normal_score:
                            s = f'index: {index}, word: {word}, condition lower_score > normal_score or title_score > normal_score'
                            logger.debug(s)
                            if lower_score > title_score:
                                word = word_lower
                                titled = False
                            else:
                                word = word_title

                    if titled:
                        s = f'index: {index}, word: {word}, condition titled'
                        logger.debug(s)
                        result.append(word)
                        index += 1
                        continue

            if check_english_func is not None and len(word) > 1:
                s = f'index: {index}, word: {word}, condition check english'
                logger.debug(s)
                found = False
                word_, repeat = check_repeat(word)
                word_lower_ = word_.lower()
                selected_word = word_
                if check_english_func(word_lower_):
                    found = True
                # suree -> sure -> detect
                elif len(word_lower_) > 1 and word_lower_[-1] == word_lower_[-2] and check_english_func(word_lower_[:-1]):
                    found = True
                    selected_word = word_[:-1]

                if found:
                    if translator is not None and language_detection_word is None:
                        s = f'index: {index}, word: {word_}, condition to translate inside checking'
                        logger.debug(s)
                        translated = translator(selected_word)
                        if len(translated) >= len(selected_word) * 3:
                            logger.debug(f'reject translation, {selected_word} -> {translated}')
                        elif ', United States' in translated:
                            logger.debug(f'reject translation, {word_} -> {translated}')
                        else:
                            selected_word = translated

                    result.append(repeat_word(case_of(word)(selected_word), repeat))
                    index += 1
                    continue

            if check_malay_func is not None and len(word) > 1:
                s = f'index: {index}, word: {word}, condition check malay'
                logger.debug(s)
                if word_lower not in ['pada', 'ke', 'tahun', 'thun']:
                    if check_malay_func(word_lower):
                        result.append(word)
                        index += 1
                        continue
                    # kenapaa -> kenapa -> detect
                    elif len(word_lower) > 1 and word_lower[-1] == word_lower[-2] and check_malay_func(word_lower[:-1]):
                        result.append(word[:-1])
                        index += 1
                        continue

            if is_malaysia_location(word):
                s = f'index: {index}, word: {word}, is_malaysia_location'
                logger.debug(s)
                result.append(word_lower.title())
                index += 1
                continue

            if word_lower in rules_normalizer and normalize_text:
                s = f'index: {index}, word: {word}, condition in early rules normalizer'
                logger.debug(s)
                result.append(case_of(word)(rules_normalizer[word_lower]))
                index += 1
                continue

            if len(word) > 2 and normalize_text and check_english_func is not None and not check_english_func(word):
                s = f'index: {index}, word: {word}, condition len(word) > 2 and norm text'
                logger.debug(s)
                if word[-2] in consonants and word[-1] == 'e':
                    word = word[:-1] + 'a'

            if word[0] == 'x' and len(
                    word) > 1 and normalize_text and check_english_func is not None and not check_english_func(word):
                s = f'index: {index}, word: {word}, condition word[0] == `x` and len(word) > 1 and norm text'
                logger.debug(s)
                result_string = 'tak '
                word = word[1:]
            else:
                s = f'index: {index}, word: {word}, condition else for (word[0] == `x` and len(word) > 1 and norm text)'
                logger.debug(s)
                result_string = ''

            if normalize_ordinal and word_lower == 'ke' and index < (len(tokenized) - 2):
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

            if normalize_hingga and _is_number_regex(word) and index < (len(tokenized) - 2):
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

            if normalize_pada_hari_bulan and word_lower == 'pada' and index < (len(tokenized) - 3):
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

            if normalize_fraction and _is_number_regex(word) and index < (len(tokenized) - 2):
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
                if normalize_money:
                    money_, _ = money(word)
                    result.append(money_)
                    if index < (len(tokenized) - 1):
                        if tokenized[index + 1].lower() in ('sen', 'cent'):
                            index += 2
                        else:
                            index += 1
                    else:
                        index += 1
                else:
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
                if normalize_units:
                    word = word.replace(' ', '')
                    word = digit_unit(word)
                result.append(word)
                index += 1
                continue

            if re.findall(_expressions['percent'], word_lower):
                s = f'index: {index}, word: {word}, condition percent'
                logger.debug(s)
                if normalize_percent:
                    word = word.replace('%', '')
                    word = cardinal(word) + ' peratus'
                result.append(word)
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
                if index - 1 >= 0 and tokenized[index - 1].lower() in ['pkul', 'pukul', 'pkl']:
                    prefix = ''
                else:
                    prefix = 'pukul '
                try:
                    s = f'index: {index}, word: {word}, parsing time'
                    logger.debug(s)
                    parsed = dateparser.parse(word.replace('.', ':'))
                    if parsed:
                        word = parsed.strftime('%H:%M:%S')
                        hour, minute, second = word.split(':')
                        if normalize_time:
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
                            word = f'{prefix}{hour} {minute} {second}'
                        else:
                            pukul = f'{prefix}{hour}'
                            if int(minute) > 0:
                                pukul = f'{pukul}.{minute}'
                            if int(second) > 0:
                                pukul = f'{pukul}:{second}'
                            word = pukul
                        word = re.sub(r'[ ]+', ' ', word).strip()
                except Exception as e:
                    logger.warning(str(e))
                result.append(word)
                index += 1
                continue

            if (
                re.findall(_expressions['number'], word_lower)
                and word_lower[0] == '0'
                and '.' not in word_lower
            ):
                s = f'index: {index}, word: {word}, condition digit and word[0] == `0`'
                logger.debug(s)
                if normalize_number:
                    word = digit(word)
                result.append(word)
                index += 1
                continue

            if (
                len(word_lower) >= 2
                and word_lower[-1] == 'x'
                and re.findall(_expressions['number'], word_lower[:-1])
                and '.' not in word_lower
            ):
                s = f'index: {index}, word: {word}, condition x kali'
                logger.debug(s)
                word = word[:-1]
                if normalize_x_kali:
                    word = cardinal(word)
                word = f'{word} kali'
                result.append(word)
                index += 1
                continue

            if len(re.findall(_expressions['number'], word)):
                s = f'index: {index}, word: {word}, condition is number'
                logger.debug(s)
                result.append(word)
                index += 1
                continue

            if normalize_cardinal:
                cardinal_ = cardinal(word)
                if cardinal_ != word:
                    s = f'index: {index}, word: {word}, condition cardinal'
                    logger.debug(s)
                    result.append(cardinal_)
                    index += 1
                    continue

            if normalize_ordinal:
                normalized_ke = ordinal(word)
                if normalized_ke != word:
                    s = f'index: {index}, word: {word}, condition ordinal'
                    logger.debug(s)
                    result.append(normalized_ke)
                    index += 1
                    continue

            if segmenter is not None:
                s = f'index: {index}, word: {word}, condition to segment'
                logger.debug(s)
                if word[-1] in digits:
                    word_ = word[:-1]
                    d = word[-1]
                else:
                    word_ = word
                    d = ''
                segmentized = segmenter(word_) + d
                words = segmentized.split()
            else:
                words = [word]

            for no_word, word in enumerate(words):
                if self._stemmer is not None:
                    s = f'index: {index}, word: {word}, self._stemmer is not None'
                    word, end_result_string = _remove_postfix(word, stemmer=self._stemmer)
                    if len(end_result_string) and end_result_string[0] in digits:
                        word = word + end_result_string[0]
                        end_result_string = end_result_string[1:]
                else:
                    end_result_string = ''

                if normalize_text:
                    word, repeat = check_repeat(word)
                else:
                    repeat = 1

                s = f'index: {index}, word: {word}, end_result_string: {end_result_string}, repeat: {repeat}'
                logger.debug(s)

                if normalize_text:
                    s = f'index: {index}, word: {word}, condition normalize text'
                    logger.debug(s)
                    if word in sounds:
                        selected = sounds[word]
                    elif word in rules_normalizer:
                        selected = rules_normalizer[word]
                    # betuii -> betui -> betul
                    elif len(word) > 1 and word[-1] == word[-2] and word[:-1] in rules_normalizer:
                        selected = rules_normalizer[word[:-1]]
                    # betuii -> betui -> betul
                    elif len(word) > 1 and word[-1] == word[-2] and word[:-1] in rules_normalizer_rev:
                        selected = word[:-1]
                    else:
                        selected = word
                        if translator is not None and language_detection_word is None:
                            s = f'index: {index}, word: {word}, condition to translate'
                            logger.debug(s)
                            translated = translator(word)
                            if len(translated) >= len(word) * 3:
                                logger.debug(f'reject translation, {word} -> {translated}')
                            elif ', United States' in translated:
                                logger.debug(f'reject translation, {word} -> {translated}')
                            elif translated in PUNCTUATION:
                                logger.debug(f'reject translation, {word} -> {translated}')
                            else:
                                selected = translated

                        if selected == word and self._speller:
                            s = f'index: {index}, word: {word}, condition to spelling correction'
                            logger.debug(s)
                            spelling_correction[len(result)] = selected

                else:
                    selected = word

                selected = repeat_word(selected, repeat)
                spelling_correction_condition[len(result)] = [repeat, result_string, end_result_string]
                result.append(result_string + selected + end_result_string)

            index += 1

        for index, selected in spelling_correction.items():
            logger.debug(f'spelling correction, index: {index}, selected: {selected}')
            selected = self._speller.correct(
                selected, string=result, index=index, **kwargs
            )
            repeat, result_string, end_result_string = spelling_correction_condition[index]
            selected = repeat_word(selected, repeat)
            selected = result_string + selected + end_result_string
            result[index] = selected

        result = ' '.join(result)
        normalized = ' '.join(normalized)

        result = re.sub(r'[ ]+', ' ', result).strip()
        normalized = re.sub(r'[ ]+', ' ', normalized).strip()

        if translator is not None and language_detection_word is not None:
            splitted = result.split()
            result_langs = language_detection_word(splitted)

            logger.debug(f'condition translator and language_detection_word, {result_langs}')

            new_result, temp, temp_lang = [], [], []
            for no_r, r in enumerate(result_langs):
                s = f'index: {no_r}, label: {r}, word: {splitted[no_r]}, queue: {new_result}'
                logger.debug(s)
                if r in acceptable_language_detection and not is_laugh(splitted[no_r]) and not is_mengeluh(splitted[no_r]):
                    temp.append(splitted[no_r])
                    temp_lang.append(r)
                else:
                    if len(temp):
                        if 'EN' in temp_lang:
                            logger.debug(f'condition len(temp) and EN in temp_lang, {temp}, {temp_lang}')
                            translated = translator(' '.join(temp))
                            new_result.extend(translated.split())
                        else:
                            logger.debug(f'condition len(temp) and EN not in temp_lang, {temp}, {temp_lang}')
                            new_result.extend(temp)
                        temp = []
                        temp_lang = []
                    new_result.append(splitted[no_r])

            if len(temp):
                if 'EN' in temp_lang:
                    logger.debug(f'condition len(temp) and EN in temp_lang, {temp}, {temp_lang}')
                    translated = translator(' '.join(temp))
                    new_result.extend(translated.split())
                else:
                    logger.debug(f'condition len(temp) and EN not in temp_lang, {temp}, {temp_lang}')
                    new_result.extend(temp)

            result = ' '.join(new_result)

        if normalize_entity:
            dates_, money_ = normalized_entity(normalized)

        else:
            dates_, money_ = {}, {}
        return {'normalize': result, 'date': dates_, 'money': money_}


def normalizer(
    speller: Callable = None,
    stemmer: Callable = None,
    **kwargs,
):
    """
    Load a Normalizer using any spelling correction model.

    Parameters
    ----------
    speller: Callable, optional (default=None)
        function to correct spelling, must have `correct` or `normalize_elongated` method.
    stemmer: Callable, optional (default=None)
        function to stem, must have `stem_word` method.
        If provide stemmer, will accurately to stem kata imbuhan akhir.

    Returns
    -------
    result: malaya.normalize.Normalizer class
    """

    validator.validate_object_methods(
        speller, ['correct', 'normalize_elongated'], 'speller'
    )
    if stemmer is not None:
        if not hasattr(stemmer, 'stem_word'):
            raise ValueError('stemmer must have `stem_word` method')

    tokenizer = Tokenizer(**kwargs).tokenize
    return Normalizer(tokenizer=tokenizer, speller=speller, stemmer=stemmer)
