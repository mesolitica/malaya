# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

from . import abstractive
from . import extractive

import re
from typing import List


def gather_capitalize(
    string,
    acceptable_lowercase: List[str] = ['dan'],
    acceptable_regex: str = '\d+',
):
    """
    This function only pull capitalized substrings.
    'saya tak suka Ayam Goreng Pakjo dan Ikan Goreng sangat pun' -> ['Ayam Goreng Pakjo', 'Ikan Goreng']

    Parameters
    ----------
    string: str
        assumed `string` been properly tokenized.
    acceptable_lowercase: List[str], optional (default=['dan'])
        acceptable lowercase, `Ayam Goreng dan Pakjo`.
    acceptable_regex: str, optional (default='\d+')

    Returns
    -------
    result: List[str]
    """
    splitted = string.split()
    results, temp = [], []
    for s in splitted:
        if s.isupper() or s.istitle() or s in acceptable_lowercase or len(re.findall(acceptable_regex, s)):
            temp.append(s)
        else:
            if len(temp):
                results.append(' '.join(temp))
                temp = []

    if len(temp):
        results.append(' '.join(temp))

    return results
