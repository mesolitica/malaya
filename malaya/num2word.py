from __future__ import print_function, unicode_literals

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

BASE = {
    0: [],
    1: ['satu'],
    2: ['dua'],
    3: ['tiga'],
    4: ['empat'],
    5: ['lima'],
    6: ['enam'],
    7: ['tujuh'],
    8: ['lapan'],
    9: ['sembilan'],
}

TENS_TO = {
    3: 'ribu',
    6: 'juta',
    9: 'billion',
    12: 'trillion',
    15: 'quadrillion',
    18: 'quintillion',
    21: 'sextillion',
    24: 'septillion',
    27: 'oktillion',
    30: 'nonillion',
    33: 'decillion',
}

errmsg_floatord = 'Cannot treat float number as ordinal'
errmsg_negord = 'Cannot treat negative number as ordinal'
errmsg_toobig = 'Too large'
max_num = 10 ** 36


def verify_ordinal(value):
    if not value == int(value):
        raise TypeError(errmsg_floatord % value)
    if not abs(value) == value:
        raise TypeError(errmsg_negord % value)


def split_by_koma(number):
    return str(number).split('.')


def ratus(number):
    if number == '1':
        return ['seratus']
    elif number == '0':
        return []
    else:
        return BASE[int(number)] + ['ratus']


def puluh(number):
    if number[0] == '1':
        if number[1] == '0':
            return ['sepuluh']
        elif number[1] == '1':
            return ['sebelas']
        else:
            return BASE[int(number[1])] + ['belas']
    elif number[0] == '0':
        return BASE[int(number[1])]
    else:
        return BASE[int(number[0])] + ['puluh'] + BASE[int(number[1])]


def split_by_3(number):
    blocks = ()
    length = len(number)
    if length < 3:
        blocks += ((number,),)
    else:
        len_of_first_block = length % 3
        if len_of_first_block > 0:
            blocks += (number[0:len_of_first_block],)
        for i in range(len_of_first_block, length, 3):
            blocks += ((number[i : i + 3],),)
    return blocks


def spell(blocks):
    word_blocks = ()
    first_block = blocks[0]
    if len(first_block[0]) == 1:
        if first_block[0] == '0':
            spelling = ['nol']
        else:
            spelling = BASE[int(first_block[0])]
    elif len(first_block[0]) == 2:
        spelling = puluh(first_block[0])
    else:
        spelling = ratus(first_block[0][0]) + puluh(first_block[0][1:3])
    word_blocks += ((first_block[0], spelling),)
    for block in blocks[1:]:
        spelling = ratus(block[0][0]) + puluh(block[0][1:3])
        block += (spelling,)
        word_blocks += (block,)
    return word_blocks


def spell_float(float_part):
    word_list = []
    for n in float_part:
        if n == '0':
            word_list += ['nol']
            continue
        word_list += BASE[int(n)]
    return ' '.join(['', 'perpuluhan'] + word_list)


def join(word_blocks, float_part):
    word_list = []
    length = len(word_blocks) - 1
    first_block = (word_blocks[0],)
    start = 0

    if length == 1 and first_block[0][0] == '1':
        word_list += ['seribu']
        start = 1

    for i in range(start, length + 1, 1):
        word_list += word_blocks[i][1]
        if not word_blocks[i][1]:
            continue
        if i == length:
            break
        word_list += [TENS_TO[(length - i) * 3]]

    return ' '.join(word_list) + float_part


def to_cardinal(number):
    """
    Translate from number input to cardinal text representation

    Parameters
    ----------
    number: int

    Returns
    -------
    string: cardinal representation
    """
    if number >= max_num:
        raise OverflowError(errmsg_toobig % (number, max_num))
    minus = ''
    if number < 0:
        minus = 'negatif '
    float_word = ''
    n = split_by_koma(abs(number))
    if len(n) == 2:
        float_word = spell_float(n[1])
    return minus + join(spell(split_by_3(n[0])), float_word)


def to_ordinal(number):
    """
    Translate from number input to ordinal text representation

    Parameters
    ----------
    number: int

    Returns
    -------
    string: ordinal representation
    """

    verify_ordinal(number)
    out_word = to_cardinal(number)
    if out_word == 'satu':
        return 'pertama'
    return 'ke' + out_word


def to_ordinal_num(number):
    """
    Translate from number input to ordinal numering text representation

    Parameters
    ----------
    number: int

    Returns
    -------
    string: ordinal numering representation
    """

    verify_ordinal(number)
    return 'ke-' + str(number)


def to_currency(value):
    """
    Translate from number input to cardinal currency text representation

    Parameters
    ----------
    number: int

    Returns
    -------
    string: cardinal currency representation
    """

    return to_cardinal(value) + ' ringgit'


def to_year(value):
    """
    Translate from number input to cardinal year text representation

    Parameters
    ----------
    number: int

    Returns
    -------
    string: cardinal year representation
    """
    return to_cardinal(value)
