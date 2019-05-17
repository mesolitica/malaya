from __future__ import print_function, unicode_literals

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

malaysian_number_system = {
    'kosong': 0,
    'satu': 1,
    'dua': 2,
    'tiga': 3,
    'empat': 4,
    'lima': 5,
    'enam': 6,
    'tujuh': 7,
    'lapan': 8,
    'sembilan': 9,
    'sepuluh': 10,
    'seribu': 1000,
    'sejuta': 1000000,
    'seratus': 100,
    'sebelas': 11,
    'ratus': 100,
    'ribu': 1000,
    'juta': 1000000,
    'bilion': 1000000000,
    'perpuluhan': '.',
    'negatif': -1,
    'belas': 10,
    'puluh': 10,
    'pertama': 1,
}

decimal_words = [
    'kosong',
    'satu',
    'dua',
    'tiga',
    'empat',
    'lima',
    'enam',
    'tujuh',
    'lapan',
    'sembilan',
]


def _get_decimal_sum(decimal_digit_words):
    decimal_number_str = []
    for dec_word in decimal_digit_words:
        if dec_word not in decimal_words:
            return 0
        else:
            decimal_number_str.append(malaysian_number_system[dec_word])
    final_decimal_string = '0.' + ''.join(map(str, decimal_number_str))
    return float(final_decimal_string)


def _number_formation(number_words):
    numbers = []
    belas = False
    for number_word in number_words:
        if number_word in ['belas', 'sebelas']:
            belas = True
        numbers.append(malaysian_number_system[number_word])
    if len(numbers) == 5:
        return (
            (numbers[0] * numbers[1]) + (numbers[2] * numbers[3]) + numbers[4]
        )
    elif len(numbers) == 4:
        if numbers[0] == 100:
            return numbers[0] + (numbers[1] * numbers[2]) + numbers[3]
        return (numbers[0] * numbers[1]) + numbers[2] + numbers[3]
    elif len(numbers) == 3:
        if belas:
            return numbers[0] + numbers[1] + numbers[2]
        return numbers[0] * numbers[1] + numbers[2]
    elif len(numbers) == 2:
        if 100 in numbers or 10 in numbers:
            if belas:
                return numbers[0] + numbers[1]
            return numbers[0] * numbers[1]
        else:
            return numbers[0] + numbers[1]
    else:
        return numbers[0]


def word2num(string):
    if not isinstance(string, str):
        raise ValueError('input must be a string')

    string = string.replace('-', ' ')
    string = string.replace('ke', '')
    string = string.replace('dan', '')
    string = string.lower()

    if string.isdigit():
        return int(string)

    split_words = string.strip().split()

    clean_numbers = []
    clean_decimal_numbers = []

    for word in split_words:
        if word in malaysian_number_system:
            clean_numbers.append(word)

    if not len(clean_numbers):
        raise ValueError(
            'No valid number words found! Please enter a valid number word'
        )

    if (
        clean_numbers.count('ribu') > 1
        or clean_numbers.count('juta') > 1
        or clean_numbers.count('bilion') > 1
        or clean_numbers.count('perpuluhan') > 1
        or clean_numbers.count('negatif') > 1
        or clean_numbers.count('seribu') > 1
        or clean_numbers.count('sejuta') > 1
    ):
        raise ValueError(
            'Redundant number word! Please enter a valid number word'
        )

    negative = False
    if clean_numbers[0] == 'negatif':
        negative = True
        clean_numbers = clean_numbers[1:]

    if clean_numbers.count('perpuluhan') == 1:
        clean_decimal_numbers = clean_numbers[
            clean_numbers.index('perpuluhan') + 1 :
        ]
        clean_numbers = clean_numbers[: clean_numbers.index('perpuluhan')]

    billion_index = (
        clean_numbers.index('bilion') if 'bilion' in clean_numbers else -1
    )
    million_index = (
        clean_numbers.index('juta') if 'juta' in clean_numbers else -1
    )
    thousand_index = (
        clean_numbers.index('ribu') if 'ribu' in clean_numbers else -1
    )

    if (
        thousand_index > -1
        and (thousand_index < million_index or thousand_index < billion_index)
    ) or (million_index > -1 and million_index < billion_index):
        raise ValueError('Malformed number! Please enter a valid number word')

    total_sum = 0

    if len(clean_numbers) > 0:
        if len(clean_numbers) == 1:
            total_sum += malaysian_number_system[clean_numbers[0]]
        else:
            if billion_index > -1:
                billion_multiplier = _number_formation(
                    clean_numbers[0:billion_index]
                )
                total_sum += billion_multiplier * 1000000000

            if million_index > -1:
                if billion_index > -1:
                    million_multiplier = _number_formation(
                        clean_numbers[billion_index + 1 : million_index]
                    )
                else:
                    million_multiplier = _number_formation(
                        clean_numbers[0:million_index]
                    )
                total_sum += million_multiplier * 1000000

            if thousand_index > -1:
                if million_index > -1:
                    thousand_multiplier = _number_formation(
                        clean_numbers[million_index + 1 : thousand_index]
                    )
                elif billion_index > -1 and million_index == -1:
                    thousand_multiplier = _number_formation(
                        clean_numbers[billion_index + 1 : thousand_index]
                    )
                else:
                    thousand_multiplier = _number_formation(
                        clean_numbers[0:thousand_index]
                    )
                total_sum += thousand_multiplier * 1000

            if thousand_index > -1 and thousand_index != len(clean_numbers) - 1:
                hundreds = _number_formation(
                    clean_numbers[thousand_index + 1 :]
                )
            elif million_index > -1 and million_index != len(clean_numbers) - 1:
                hundreds = _number_formation(clean_numbers[million_index + 1 :])
            elif billion_index > -1 and billion_index != len(clean_numbers) - 1:
                hundreds = _number_formation(clean_numbers[billion_index + 1 :])
            elif (
                thousand_index == -1
                and million_index == -1
                and billion_index == -1
            ):
                hundreds = _number_formation(clean_numbers)
            else:
                hundreds = 0
            total_sum += hundreds

    if len(clean_decimal_numbers) > 0:
        decimal_sum = _get_decimal_sum(clean_decimal_numbers)
        total_sum += decimal_sum

    return total_sum * -1 if negative else total_sum
