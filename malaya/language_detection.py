import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import pickle
import os
import xgboost as xgb
import numpy as np
from .text_functions import simple_textcleaning
from .utils import download_file
from .sklearn_model import USER_XGB, USER_BAYES
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
    USER_BAYES : malaya.sklearn_model.USER_BAYES class
    """
    if not os.path.isfile(PATH_LANG_DETECTION['multinomial']['vector']):
        print('downloading LANGUAGE-DETECTION pickled bag-of-word multinomial')
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
        BOW = pickle.load(fopen)
    with open(PATH_LANG_DETECTION['multinomial']['model'], 'rb') as fopen:
        MULTINOMIAL = pickle.load(fopen)
    return USER_BAYES(MULTINOMIAL, lang_labels, BOW)


def xgb_detect_languages():
    """
    Load XGB language detection model.

    Returns
    -------
    USER_XGB : malaya.sklearn_model.USER_XGB class
    """
    if not os.path.isfile(PATH_LANG_DETECTION['xgb']['vector']):
        print('downloading LANGUAGE-DETECTION pickled bag-of-word XGB')
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
        BOW = pickle.load(fopen)
    with open(PATH_LANG_DETECTION['xgb']['model'], 'rb') as fopen:
        XGB = pickle.load(fopen)
    return USER_XGB(XGB, lang_labels, BOW)
