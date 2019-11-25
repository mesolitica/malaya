import re
import emoji
from gensim.utils import deaccent
from collections import Counter
from collections import ChainMap
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import itertools
from rules import *

re_3986_enhanced = re.compile(
    r"""
        # Parse and capture RFC-3986 Generic URI components.
        ^                                    # anchor to beginning of string
        (?:  (?P<scheme>    [^:/?#\s]+):// )?  # capture optional scheme
        (?:(?P<authority>  [^/?#\s]*)  )?  # capture optional authority
             (?P<path>        [^?#\s]*)      # capture required path
        (?:\?(?P<query>        [^#\s]*)  )?  # capture optional query
        (?:\#(?P<fragment>      [^\s]*)  )?  # capture optional fragment
        $                                    # anchor to end of string
        """,
    re.MULTILINE | re.VERBOSE,
)

re_domain = re.compile(
    r"""
        # Pick out top two levels of DNS domain from authority.
        (?P<domain>[^.]+\.[A-Za-z]{2,6})  # $domain: top two domain levels.
        (?::[0-9]*)?                      # Optional port number.
        $                                 # Anchor to end of string.
        """,
    re.MULTILINE | re.VERBOSE,
)


def domain_search(text):
    try:
        return re_domain.search(
            re_3986_enhanced.match(text).group('authority')
        ).group('domain')
    except:
        return 'url'


def make_cleaning(s, c_dict):
    s = s.translate(c_dict)
    return s


def make_dict_cleaning(s, w_dict):
    s = w_dict.get(s, s)
    return s


def cleaning(string):
    string = ' '.join(
        [make_cleaning(w, normalized_chars) for w in string.split()]
    )
    string = re.sub('\(dot\)', '.', string)
    string = deaccent(string)

    # remove href
    string = (
        re.sub(re.findall(r'\<a(.*?)\>', string)[0], '', string)
        if (len(re.findall(r'\<a (.*?)\>', string)) > 0)
        and ('href' in re.findall(r'\<a (.*?)\>', string)[0])
        else string
    )

    string = re.sub(
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', string
    )
    string = re.sub(r'http\S+|www.\S+', '', string)
    string = re.sub(r'[ ]+', ' ', string).strip().split()
    string = [w for w in string if w[0] != '@']

    return ' '.join(string)


def cleaning_strings(strings):
    for i in tqdm(range(len(strings))):
        strings[i] = cleaning(strings[i])
    return strings


def clean_chars(strings):
    global_chars_list = list(set([c for line in strings for c in line]))
    chars = ''.join(
        [
            c
            for c in global_chars_list
            if (c not in emoji.UNICODE_EMOJI) and (c not in white_list_chars)
        ]
    )
    chars_dict = {}

    for char in chars:
        try:
            new_char = unicodedata.name(char).split()[-1:][0].lower()
            if len(new_char) == 1:
                chars_dict[ord(char)] = new_char
            else:
                chars_dict[ord(char)] = ''
        except:
            chars_dict[ord(char)] = ''

    for i in tqdm(range(len(strings))):
        strings[i] = ' '.join(
            [make_cleaning(w, chars_dict) for w in strings[i].split()]
        )

    return strings


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def chunks_multiple(l, n, dict):
    for i in range(0, len(l), n):
        yield (l[i : i + n], dict)


def multiprocessing(strings, function, cores = 16, list_mode = True):
    df_split = chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()
    if list_mode:
        return list(itertools.chain(*pooled))
    else:
        return dict(ChainMap(*pooled))


def multiprocessing_multiple(
    strings, dict, function, cores = 16, list_mode = True
):
    df_split = chunks_multiple(strings, len(strings) // cores, dict)
    pool = Pool(cores)
    pooled = pool.starmap(function, df_split)
    pool.close()
    pool.join()
    if list_mode:
        return list(itertools.chain(*pooled))
    else:
        return dict(ChainMap(*pooled))


def unique_words(strings):
    return list(set([c for line in strings for c in line.split()]))


def duplicate_dots_marks_exclamations(strings):
    temp_dict = {}
    for word in strings:
        new_word = word
        if (
            (Counter(word)['.'] > 1)
            or (Counter(word)['!'] > 1)
            or (Counter(word)['?'] > 1)
            or (Counter(word)[','] > 1)
        ):
            if Counter(word)['.'] > 1:
                new_word = re.sub('\.\.+', ' . . . ', new_word)
            if Counter(word)['!'] > 1:
                new_word = re.sub('\!\!+', ' ! ! ! ', new_word)
            if Counter(word)['?'] > 1:
                new_word = re.sub('\?\?+', ' ? ? ? ', new_word)
            if Counter(word)[','] > 1:
                new_word = re.sub('\,\,+', ' , , , ', new_word)
            temp_dict[word] = new_word
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    return temp_dict


def remove_underscore(strings):
    temp_dict = {}
    for word in strings:
        if (
            len(re.compile("[a-zA-Z0-9\-\.\,\/']").sub('', word)) / len(word)
            > 0.6
        ) and ('_' in word):
            temp_dict[word] = re.sub('_', '', word)
    return temp_dict


def isolate_spamchars(strings):
    temp_dict = {}
    for word in strings:
        if (
            (
                len(re.compile("[a-zA-Z0-9\-\.\,\/']").sub('', word))
                / len(word)
                > 0.6
            )
            and (len(Counter(word)) == 1)
            and (len(word) > 2)
        ):
            temp_dict[word] = ' '.join(
                [' ' + next(iter(Counter(word).keys())) + ' ' for i in range(3)]
            )
    return temp_dict


def break_short_words(strings):
    strings = [k for k in strings if len(k) <= 20]
    temp_dict = {}
    for word in strings:
        if '/' in word:
            temp_dict[word] = re.sub('/', ' / ', word)
    return temp_dict


def break_long_words(strings):
    strings = [k for k in strings if len(k) > 20]
    temp_dict = {}
    for word in strings:
        if '_' in word:
            temp_dict[word] = re.sub('_', ' ', word)
        elif '/' in word:
            temp_dict[word] = re.sub('/', ' / ', word)
        elif len(' '.join(word.split('-')).split()) > 2:
            temp_dict[word] = re.sub('-', ' ', word)
    return temp_dict


def remove_ending_underscore(strings):
    strings = [k for k in strings if '_' in k]
    temp_dict = {}
    for word in strings:
        new_word = word
        if word[len(word) - 1] == '_':
            for i in range(len(word), 0, -1):
                if word[i - 1] != '_':
                    new_word = word[:i]
                    temp_dict[word] = new_word
                    break
    return temp_dict


def remove_starting_underscore(strings):
    strings = [k for k in strings if '_' in k]
    temp_dict = {}
    for word in strings:
        new_word = word
        if word[0] == '_':
            for i in range(len(word)):
                if word[i] != '_':
                    new_word = word[i:]
                    temp_dict[word] = new_word
                    break
    return temp_dict


def end_punct(strings):
    strings = [k for k in strings if not k[len(k) - 1].isalnum()]
    temp_dict = {}
    for word in strings:
        new_word = word
        for i in range(len(word), 0, -1):
            if word[i - 1].isalnum():
                new_word = word[:i] + ' ' + word[i:]
                break
        temp_dict[word] = new_word
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    return temp_dict


def start_punct(strings):
    strings = [k for k in strings if not k[0].isalnum()]
    temp_dict = {}
    for word in strings:
        new_word = word
        for i in range(len(word)):
            if word[i].isalnum():
                new_word = word[:i] + ' ' + word[i:]
                break
        temp_dict[word] = new_word
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    return temp_dict


def join_dashes(strings):
    temp_dict = {}
    for word in strings:
        temp_dict[word] = re.sub('\-\-+', '-', word)
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    return temp_dict


def string_dict_cleaning(strings, dict):
    for i in tqdm(range(len(strings))):
        strings[i] = ' '.join(
            [make_dict_cleaning(w, dict) for w in strings[i].split()]
        )
    return strings
