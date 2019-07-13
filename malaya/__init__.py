# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import os
from shutil import rmtree
from pathlib import Path

home = os.path.join(str(Path.home()), 'Malaya')
version = '2.6'
bump_version = '2.6.1'
version_path = os.path.join(home, 'version')


def _delete_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))


def _delete_macos():
    macos = os.path.join(home, '__MACOSX')
    if os.path.exists(macos):
        rmtree(macos)


from ._utils._paths import MALAY_TEXT, MALAY_TEXT_200K
from ._utils._utils import DisplayablePath, download_file

try:
    if not os.path.exists(home):
        os.makedirs(home)
except:
    raise Exception(
        'Malaya cannot make directory for caching. Please check your '
        + str(Path.home())
    )

_delete_macos()

if not os.path.isfile(version_path):
    print('not found any version, deleting previous version models..')
    _delete_folder(home)
    with open(version_path, 'w') as fopen:
        fopen.write(version)
else:
    with open(version_path, 'r') as fopen:
        cached_version = fopen.read()
    try:
        if float(cached_version) < 1:
            print(
                'Found old version of Malaya, deleting previous version models..'
            )
            _delete_folder(home)
            with open(version_path, 'w') as fopen:
                fopen.write(version)
    except:
        print('Found old version of Malaya, deleting previous version models..')
        _delete_folder(home)
        with open(version_path, 'w') as fopen:
            fopen.write(version)


def print_cache(location = None):
    """
    Print cached data, this will print entire cache folder if let location = None
    """
    path = os.path.join(home, location) if location else home
    paths = DisplayablePath.make_tree(Path(path))
    for path in paths:
        print(path.displayable())


def clear_all_cache():
    """
    Remove cached data, this will delete entire cache folder
    """
    _delete_macos()
    try:
        print('clearing cached models..')
        _delete_folder(home)
        with open(version_path, 'w') as fopen:
            fopen.write(version)
        return True
    except:
        print(
            'failed to clear cached models. Please make sure %s is able to overwrite from Malaya'
            % (home)
        )


def clear_cache(location):
    """
    Remove selected cached data, please run malaya.print_cache() to get path
    """
    if not isinstance(location, str):
        raise ValueError('location must be a string')
    location = os.path.join(home, location)
    if not os.path.exists(location):
        raise Exception(
            'folder not exist, please check path from malaya.print_cache()'
        )
    if not os.path.isdir(location):
        raise Exception(
            'Please use parent directory, please check path from malaya.print_cache()'
        )
    _delete_folder(location)
    return True


def load_malay_dictionary():
    """
    load Pustaka dictionary for Spelling Corrector and Normalizer.

    Returns
    -------
    list: list of strings
    """
    if not os.path.isfile(MALAY_TEXT):
        print('downloading Malay texts')
        download_file('v6/malay-text.txt', MALAY_TEXT)
    try:
        with open(MALAY_TEXT, 'r') as fopen:
            results = [
                text.lower()
                for text in (list(filter(None, fopen.read().split('\n'))))
            ]
            if len(results) < 20000:
                raise Exception(
                    "model corrupted due to some reasons, please run malaya.clear_cache('dictionary') and try again"
                )
            return results
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('dictionary') and try again"
        )


def load_200k_malay_dictionary():
    """
    load 200k words dictionary for Spelling Corrector and Normalizer.

    Returns
    -------
    list: list of strings
    """

    if not os.path.isfile(MALAY_TEXT_200K):
        print('downloading 200k Malay texts')
        download_file('v6/malay-text.txt', MALAY_TEXT_200K)
    try:
        with open(MALAY_TEXT_200K, 'r') as fopen:
            results = json.load(fopen)
            if len(results) < 200000:
                raise Exception(
                    "model corrupted due to some reasons, please run malaya.clear_cache('dictionary-200k') and try again"
                )
            return results
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('dictionary-200k') and try again"
        )


def describe_pos_malaya():
    """
    Describe Malaya Part-Of-Speech supported (deprecated, use describe_pos() instead)
    """
    print(
        'This classes are deprecated, we prefer to use `malaya.describe_pos()`'
    )
    print('KT - Kata Tanya')
    print('KJ - Kata Kerja')
    print('KP - Kata Perintah')
    print('KPA - Kata Pangkal')
    print('KB - Kata Bantu')
    print('KPENGUAT - Kata Penguat')
    print('KPENEGAS - Kata Penegas')
    print('NAFI - Kata Nafi')
    print('KPEMERI - Kata Pemeri')
    print('KS - Kata Sendi')
    print('KPEMBENAR - Kata Pembenar')
    print('NAFI - Kata Nafi')
    print('NO - Numbers')
    print('SUKU - Suku Bilangan')
    print('PISAHAN - Kata Pisahan')
    print('KETERANGAN - Kata Keterangan')
    print('ARAH - Kata Arah')
    print('KH - Kata Hubung')
    print('GN - Ganti Nama')
    print('KA - Kata Adjektif')
    print('O - not related, out scope')


def describe_pos():
    """
    Describe Part-Of-Speech supported
    """
    print('ADJ - Adjective, kata sifat')
    print('ADP - Adposition')
    print('ADV - Adverb, kata keterangan')
    print('ADX - Auxiliary verb, kata kerja tambahan')
    print('CCONJ - Coordinating conjuction, kata hubung')
    print('DET - Determiner, kata penentu')
    print('NOUN - Noun, kata nama')
    print('NUM - Number, nombor')
    print('PART - Particle')
    print('PRON - Pronoun, kata ganti')
    print('PROPN - Proper noun, kata ganti nama khas')
    print('SCONJ - Subordinating conjunction')
    print('SYM - Symbol')
    print('VERB - Verb, kata kerja')
    print('X - Other')


def describe_entities_malaya():
    """
    Describe Malaya Entities supported (deprecated, use describe_entities() instead)
    """
    print(
        'This classes are deprecated, we prefer to use `malaya.describe_entities()`'
    )
    print('PRN - person, group of people, believes, etc')
    print('LOC - location')
    print('NORP - Military, police, government, Parties, etc')
    print('ORG - Organization, company')
    print('LAW - related law document, etc')
    print('ART - art of work, special names, etc')
    print('EVENT - event happen, etc')
    print('FAC - facility, hospitals, clinics, etc')
    print('TIME - date, day, time, etc')
    print('O - not related, out scope')


def describe_entities():
    """
    Describe Entities supported
    """
    print('OTHER - Other')
    print('law - law, regulation, related law documents, documents, etc')
    print('location - location, place')
    print('organization - organization, company, government, facilities, etc')
    print('person - person, group of people, believes, etc')
    print('quantity - numbers, quantity')
    print('time - date, day, time, etc')
    print('event - unique event happened, etc')


def describe_dependency():
    """
    Describe Dependency supported
    """
    print('acl - clausal modifier of noun')
    print('advcl - adverbial clause modifier')
    print('advmod - adverbial modifier')
    print('amod - adjectival modifier')
    print('appos - appositional modifier')
    print('aux - auxiliary')
    print('case - case marking')
    print('ccomp - clausal complement')
    print('compound - compound')
    print('compound:plur - plural compound')
    print('conj - conjunct')
    print('cop - cop')
    print('csubj - clausal subject')
    print('dep - dependent')
    print('det - determiner')
    print('fixed - multi-word expression')
    print('flat - name')
    print('iobj - indirect object')
    print('mark - marker')
    print('nmod - nominal modifier')
    print('nsubj - nominal subject')
    print('obj - direct object')
    print('parataxis - parataxis')
    print('root - root')
    print('xcomp - open clausal complement')
    print(
        'you can read more from https://universaldependencies.org/en/dep/xcomp.html'
    )


from . import cluster
from . import dependency
from . import elmo
from . import emotion
from . import entity
from . import fast_text
from . import language_detection
from . import normalize
from . import num2word
from . import pos
from . import preprocessing
from . import relevancy
from . import sentiment
from . import similarity
from . import spell
from . import stack
from . import stem
from . import subjective
from . import summarize
from . import topic_model
from . import toxic
from . import word_mover
from . import word2num
from . import word2vec
from .texts import vectorizer
