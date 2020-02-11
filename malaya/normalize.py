import numpy as np
import json
import re
import dateparser
from unidecode import unidecode
from .num2word import to_cardinal, to_ordinal
from .word2num import word2num
from .texts._text_functions import ENGLISH_WORDS, MALAY_WORDS, multireplace
from .texts._regex import (
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
from .texts._tatabahasa import (
    rules_normalizer,
    date_replace,
    consonants,
    sounds,
    hujung_malaysian,
)
from .texts._normalization import (
    _remove_postfix,
    _normalize_title,
    _is_number_regex,
    _string_to_num,
    _normalize_money,
    cardinal,
    digit,
    rom_to_int,
    ordinal,
    fraction,
    money,
    ignore_words,
)
from .cluster import cluster_words
from .preprocessing import _tokenizer
from herpetologist import check_type


class _SPELL_NORMALIZE:
    def __init__(self, speller):
        self._speller = speller

    @check_type
    def normalize(self, string: str, check_english: bool = True):
        """
        Normalize a string

        Parameters
        ----------
        string : str
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

        result, normalized = [], []
        tokenized = _tokenizer(string)
        index = 0
        while index < len(tokenized):
            word = tokenized[index]
            if word in '~@#$%^&*()_+{}|[:"\'];<>,.?/-':
                result.append(word)
                index += 1
                continue
            normalized.append(rules_normalizer.get(word.lower(), word.lower()))
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

            if re.findall(_money, word.lower()):
                money_, _ = money(word)
                result.append(money_)
                index += 1
                continue

            if re.findall(_date, word.lower()):
                word = word.lower()
                word = multireplace(word, date_replace)
                word = re.sub(r'[ ]+', ' ', word).strip()
                parsed = dateparser.parse(word)
                if parsed:
                    result.append(parsed.strftime('%d/%m/%Y'))
                else:
                    result.append(word)
                index += 1
                continue

            if re.findall(_expressions['time'], word.lower()):
                word = word.lower()
                word = multireplace(word, date_replace)
                word = re.sub(r'[ ]+', ' ', word).strip()
                parsed = dateparser.parse(word)
                if parsed:
                    result.append(parsed.strftime('%H:%M:%S'))
                else:
                    result.append(word)
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
                word, string = ' '.join(tokenized), index = index
            )
            result.append(result_string + selected + end_result_string)
            index += 1

        result = ' '.join(result)
        normalized = ' '.join(normalized)
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
            '%s %s' % (i[0], i[1])
            for i in re.findall(_left_datetime, normalized)
        ]
        right_datetime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_right_datetime, normalized)
        ]
        today_left_datetime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_left_datetodaytime, normalized)
        ]
        today_right_datetime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_right_datetodaytime, normalized)
        ]
        left_yesterdaydatetime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_left_yesterdaydatetime, normalized)
        ]
        right_yesterdaydatetime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_right_yesterdaydatetime, normalized)
        ]
        left_yesterdaydatetodaytime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_left_yesterdaydatetodaytime, normalized)
        ]
        right_yesterdaydatetodaytime_ = [
            '%s %s' % (i[0], i[1])
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
        return {'normalize': result, 'date': dates_, 'money': money_}


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
