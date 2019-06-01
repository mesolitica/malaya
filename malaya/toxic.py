import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import json
import pickle
import os
from ._utils._utils import (
    check_file,
    load_graph,
    check_available,
    generate_session,
)
from . import home
from ._utils._paths import PATH_TOXIC, S3_PATH_TOXIC
from ._models._sklearn_model import TOXIC
from ._models._tensorflow_model import SIGMOID


def available_deep_model():
    """
    List available deep learning toxicity analysis models.
    """
    return ['self-attention', 'bahdanau', 'luong']


def multinomial(validate = True):
    """
    Load multinomial toxic model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    TOXIC : malaya._models._sklearn_model.TOXIC class
    """
    if validate:
        check_file(PATH_TOXIC['multinomial'], S3_PATH_TOXIC['multinomial'])
    else:
        if not check_available(PATH_TOXIC['multinomial']):
            raise Exception(
                'toxic/multinomial is not available, please `validate = True`'
            )
    try:
        with open(PATH_TOXIC['multinomial']['model'], 'rb') as fopen:
            multinomial = pickle.load(fopen)
        with open(PATH_TOXIC['multinomial']['vector'], 'rb') as fopen:
            vectorize = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('toxic/multinomial') and try again"
        )
    return TOXIC(multinomial, vectorize)


def logistic(validate = True):
    """
    Load logistic toxic model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    TOXIC : malaya._models._sklearn_model.TOXIC class
    """
    if validate:
        check_file(PATH_TOXIC['logistic'], S3_PATH_TOXIC['logistic'])
    else:
        if not check_available(PATH_TOXIC['logistic']):
            raise Exception(
                'toxic/logistic is not available, please `validate = True`'
            )
    try:
        with open(PATH_TOXIC['logistic']['model'], 'rb') as fopen:
            logistic = pickle.load(fopen)
        with open(PATH_TOXIC['logistic']['vector'], 'rb') as fopen:
            vectorize = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('toxic/logistic') and try again"
        )
    return TOXIC(logistic, vectorize)


def deep_model(model = 'luong', validate = True):
    """
    Load deep learning sentiment analysis model.

    Parameters
    ----------
    model : str, optional (default='luong')
        Model architecture supported. Allowed values:

        * ``'self-attention'`` - Fast-text architecture, embedded and logits layers only with self attention.
        * ``'bahdanau'`` - LSTM with bahdanau attention architecture.
        * ``'luong'`` - LSTM with luong attention architecture.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    TOXIC: malaya._models._tensorflow_model.SIGMOID class
    """
    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')
    model = model.lower()
    if model not in available_deep_model():
        raise Exception(
            'model is not supported, please check supported models from malaya.toxic.available_deep_model()'
        )
    if validate:
        check_file(PATH_TOXIC[model], S3_PATH_TOXIC[model])
    else:
        if not check_available(PATH_TOXIC[model]):
            raise Exception(
                'toxic/%s is not available, please `validate = True`' % (model)
            )
    try:
        with open(PATH_TOXIC[model]['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        g = load_graph(PATH_TOXIC[model]['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('toxic/%s') and try again"
            % (model)
        )

    return SIGMOID(
        g.get_tensor_by_name('import/Placeholder:0'),
        g.get_tensor_by_name('import/logits:0'),
        g.get_tensor_by_name('import/logits_seq:0'),
        g.get_tensor_by_name('import/alphas:0'),
        generate_session(graph = g),
        dictionary,
    )
