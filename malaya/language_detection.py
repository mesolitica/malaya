import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import pickle
import os
from .utils import download_file
from .sklearn_model import LANGUAGE_DETECTION
from .paths import PATH_LANG_DETECTION, S3_PATH_LANG_DETECTION

lang_labels = {0: 'OTHER', 1: 'ENGLISH', 2: 'INDONESIA', 3: 'MALAY'}


def get_language_labels():
    """
    Return language labels dictionary.
    """
    return lang_labels


def multinomial_detect_languages():
    """
    Load multinomial language detection model.

    Returns
    -------
    LANGUAGE_DETECTION : malaya.sklearn_model.LANGUAGE_DETECTION class
    """
    if not os.path.isfile(PATH_LANG_DETECTION['multinomial']['vector']):
        print('downloading LANGUAGE-DETECTION pickled vectorizer')
        download_file(
            S3_PATH_LANG_DETECTION['multinomial']['vector'],
            PATH_LANG_DETECTION['multinomial']['vector'],
        )
    if not os.path.isfile(PATH_LANG_DETECTION['multinomial']['model']):
        print('downloading LANGUAGE-DETECTION pickled multinomial model')
        download_file(
            S3_PATH_LANG_DETECTION['multinomial']['model'],
            PATH_LANG_DETECTION['multinomial']['model'],
        )
    with open(PATH_LANG_DETECTION['multinomial']['vector'], 'rb') as fopen:
        vector = pickle.load(fopen)
    with open(PATH_LANG_DETECTION['multinomial']['model'], 'rb') as fopen:
        model = pickle.load(fopen)
    return LANGUAGE_DETECTION(model, lang_labels, vector)


def sgd_detect_languages():
    """
    Load SGD language detection model.

    Returns
    -------
    LANGUAGE_DETECTION : malaya.sklearn_model.LANGUAGE_DETECTION class
    """
    if not os.path.isfile(PATH_LANG_DETECTION['sgd']['vector']):
        print('downloading LANGUAGE-DETECTION pickled vectorizer')
        download_file(
            S3_PATH_LANG_DETECTION['sgd']['vector'],
            PATH_LANG_DETECTION['sgd']['vector'],
        )
    if not os.path.isfile(PATH_LANG_DETECTION['sgd']['model']):
        print('downloading LANGUAGE-DETECTION pickled SGD model')
        download_file(
            S3_PATH_LANG_DETECTION['sgd']['model'],
            PATH_LANG_DETECTION['sgd']['model'],
        )
    with open(PATH_LANG_DETECTION['sgd']['vector'], 'rb') as fopen:
        vector = pickle.load(fopen)
    with open(PATH_LANG_DETECTION['sgd']['model'], 'rb') as fopen:
        model = pickle.load(fopen)
    return LANGUAGE_DETECTION(model, lang_labels, vector)


def xgb_detect_languages():
    """
    Load XGB language detection model.

    Returns
    -------
    LANGUAGE_DETECTION : malaya.sklearn_model.LANGUAGE_DETECTION class
    """
    if not os.path.isfile(PATH_LANG_DETECTION['xgb']['vector']):
        print('downloading LANGUAGE-DETECTION pickled vectorizer')
        download_file(
            S3_PATH_LANG_DETECTION['xgb']['vector'],
            PATH_LANG_DETECTION['xgb']['vector'],
        )
    if not os.path.isfile(PATH_LANG_DETECTION['xgb']['model']):
        print('downloading LANGUAGE-DETECTION pickled XGB model')
        download_file(
            S3_PATH_LANG_DETECTION['xgb']['model'],
            PATH_LANG_DETECTION['xgb']['model'],
        )
    with open(PATH_LANG_DETECTION['xgb']['vector'], 'rb') as fopen:
        vector = pickle.load(fopen)
    with open(PATH_LANG_DETECTION['xgb']['model'], 'rb') as fopen:
        model = pickle.load(fopen)
    return LANGUAGE_DETECTION(model, lang_labels, vector, mode = 'xgb')
