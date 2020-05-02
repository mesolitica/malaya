# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

import os
from shutil import rmtree
from pathlib import Path

home = os.path.join(str(Path.home()), 'Malaya')
version = '3.4'
bump_version = '3.4.2'
version_path = os.path.join(home, 'version')


def _delete_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))


def _delete_macos():
    macos = os.path.join(home, '__MACOSX')
    if os.path.exists(macos):
        rmtree(macos)


try:
    if not os.path.exists(home):
        os.makedirs(home)
except:
    raise Exception(
        'Malaya cannot make directory for caching. Please check your '
        + str(Path.home())
    )

_delete_macos()
from malaya.function import DisplayablePath, download_file

if not os.path.isfile(version_path):
    _delete_folder(home)
    with open(version_path, 'w') as fopen:
        fopen.write(version)
else:
    with open(version_path, 'r') as fopen:
        cached_version = fopen.read()
    try:
        if float(cached_version) < 1:
            _delete_folder(home)
            with open(version_path, 'w') as fopen:
                fopen.write(version)
    except:
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
        raise Exception(
            f'failed to clear cached models. Please make sure {home} is able to overwrite from Malaya'
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


from . import cluster
from . import dependency
from . import emotion
from . import entity
from . import language_detection
from . import lexicon
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
from . import transformer
from . import word2num
from . import wordvector
