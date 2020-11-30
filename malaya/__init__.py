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
import logging


home = os.path.join(str(Path.home()), 'Malaya')
version = '4.0'
bump_version = '4.0.5'
version_path = os.path.join(home, 'version')
__version__ = bump_version
path = os.path.dirname(__file__)


def available_gpu():
    """
    Get list of GPUs from `nvidia-smi`.

    Returns
    -------
    result : List[str]
    """
    percent = []
    try:
        ns = os.popen('nvidia-smi')
        lines_ns = ns.readlines()
        for line in lines_ns:
            if line.find('%') != -1:
                percent.append(int(line.split('%')[-2][-3:]))
        percent = [f'/device:GPU:{i}' for i in range(len(percent))]
    except:
        pass
    return percent


def check_malaya_gpu():
    import pkg_resources

    return 'malaya-gpu' in [p.project_name for p in pkg_resources.working_set]


if check_malaya_gpu():
    __gpu__ = available_gpu()
else:
    __gpu__ = []


def gpu_available():
    """
    Check Malaya is GPU version.

    Returns
    -------
    result : bool
    """

    return len(__gpu__) > 0


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
    Print cached data, this will print entire cache folder if let location = None.

    Parameters
    ----------
    location : str, (default=None)
        if location is None, will print entire cache directory.

    """

    from malaya.function import DisplayablePath

    path = os.path.join(home, location) if location else home
    paths = DisplayablePath.make_tree(Path(path))
    for path in paths:
        print(path.displayable())


def clear_all_cache():
    """
    Remove cached data, this will delete entire cache folder.
    """
    _delete_macos()
    try:
        logging.info('clearing cached models..')
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
    Remove selected cached data, please run malaya.print_cache() to get path.

    Parameters
    ----------
    location : str

    Returns
    -------
    result : boolean
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


def clear_session(model):
    """
    Clear session from a model to prevent any out-of-memory or segmentation fault issues.

    Parameters
    ----------
    model : malaya object.

    Returns
    -------
    result : boolean
    """

    success = False
    try:
        if hasattr(model, 'sess'):
            model.sess.close()
            success = True
        elif hasattr(model, '_sess'):
            model._sess.close()
            success = True
    except Exception as e:
        logging.warning(e)
    return success


from . import augmentation
from . import cluster
from . import constituency
from . import dependency
from . import emotion
from . import entity
from . import generator
from . import keyword_extraction
from . import language_detection
from . import lexicon
from . import normalize
from . import nsfw
from . import num2word
from . import paraphrase
from . import pos
from . import preprocessing
from . import relevancy
from . import sentiment
from . import similarity
from . import spell
from . import stack
from . import stem
from . import subjectivity
from . import summarization
from . import topic_model
from . import toxicity
from . import transformer
from . import true_case
from . import translation
from . import word2num
from . import wordvector
from . import zero_shot
