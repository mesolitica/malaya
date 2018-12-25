import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import pickle
from ._utils._utils import check_file
from ._models._sklearn_model import LANGUAGE_DETECTION
from ._utils._paths import PATH_LANG_DETECTION, S3_PATH_LANG_DETECTION

lang_labels = {0: 'OTHER', 1: 'ENGLISH', 2: 'INDONESIA', 3: 'MALAY'}


def label():
    """
    Return language labels dictionary.
    """
    return lang_labels


def multinomial():
    """
    Load multinomial language detection model.

    Returns
    -------
    LANGUAGE_DETECTION : malaya._models._sklearn_model.LANGUAGE_DETECTION class
    """
    check_file(
        PATH_LANG_DETECTION['multinomial'],
        S3_PATH_LANG_DETECTION['multinomial'],
    )
    try:
        with open(PATH_LANG_DETECTION['multinomial']['vector'], 'rb') as fopen:
            vector = pickle.load(fopen)
        with open(PATH_LANG_DETECTION['multinomial']['model'], 'rb') as fopen:
            model = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('language-detection/multinomial') and try again"
        )
    return LANGUAGE_DETECTION(model, lang_labels, vector)


def sgd():
    """
    Load SGD language detection model.

    Returns
    -------
    LANGUAGE_DETECTION : malaya._models._sklearn_model.LANGUAGE_DETECTION class
    """
    check_file(PATH_LANG_DETECTION['sgd'], S3_PATH_LANG_DETECTION['sgd'])
    try:
        with open(PATH_LANG_DETECTION['sgd']['vector'], 'rb') as fopen:
            vector = pickle.load(fopen)
        with open(PATH_LANG_DETECTION['sgd']['model'], 'rb') as fopen:
            model = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('language-detection/sgd') and try again"
        )
    return LANGUAGE_DETECTION(model, lang_labels, vector)


def xgb():
    """
    Load XGB language detection model.

    Returns
    -------
    LANGUAGE_DETECTION : malaya._models._sklearn_model.LANGUAGE_DETECTION class
    """
    check_file(PATH_LANG_DETECTION['xgb'], S3_PATH_LANG_DETECTION['xgb'])
    try:
        with open(PATH_LANG_DETECTION['xgb']['vector'], 'rb') as fopen:
            vector = pickle.load(fopen)
        with open(PATH_LANG_DETECTION['xgb']['model'], 'rb') as fopen:
            model = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('language-detection/xgb') and try again"
        )
    return LANGUAGE_DETECTION(model, lang_labels, vector, mode = 'xgb')
