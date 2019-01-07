import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import json
import pickle
import os
import tensorflow as tf
from ._utils._utils import check_file, load_graph
from . import home
from ._utils._paths import PATH_TOXIC, S3_PATH_TOXIC
from ._models._sklearn_model import TOXIC
from ._models._tensorflow_model import SIGMOID


def available_deep_model():
    """
    List available deep learning toxicity analysis models.
    """
    return ['bahdanau', 'hierarchical', 'luong', 'fast-text', 'entity-network']


def multinomial():
    """
    Load multinomial toxic model.

    Returns
    -------
    TOXIC : malaya._models._sklearn_model.TOXIC class
    """
    check_file(PATH_TOXIC['multinomial'], S3_PATH_TOXIC['multinomial'])
    with open(PATH_TOXIC['multinomial']['model'], 'rb') as fopen:
        multinomial = pickle.load(fopen)
    with open(PATH_TOXIC['multinomial']['vector'], 'rb') as fopen:
        vectorize = pickle.load(fopen)
    return TOXIC(multinomial, vectorize)


def logistic():
    """
    Load logistic toxic model.

    Returns
    -------
    TOXIC : malaya._models._sklearn_model.TOXIC class
    """
    check_file(PATH_TOXIC['logistic'], S3_PATH_TOXIC['logistic'])
    with open(PATH_TOXIC['logistic']['model'], 'rb') as fopen:
        logistic = pickle.load(fopen)
    with open(PATH_TOXIC['logistic']['vector'], 'rb') as fopen:
        vectorize = pickle.load(fopen)
    return TOXIC(logistic, vectorize)


def deep_model(model = 'luong'):
    """
    Load deep learning sentiment analysis model.

    Parameters
    ----------
    model : str, optional (default='luong')
        Model architecture supported. Allowed values:

        * ``'fast-text'`` - Fast-text architecture, embedded and logits layers only
        * ``'hierarchical'`` - LSTM with hierarchical attention architecture
        * ``'bahdanau'`` - LSTM with bahdanau attention architecture
        * ``'luong'`` - LSTM with luong attention architecture
        * ``'entity-network'`` - Recurrent Entity-Network architecture

    Returns
    -------
    TOXIC: malaya._models._tensorflow_model.SIGMOID class
    """
    assert isinstance(model, str), 'model must be a string'
    model = model.lower()
    if model == 'fast-text':
        check_file(PATH_TOXIC['fast-text'], S3_PATH_TOXIC['fast-text'])
        with open(PATH_TOXIC['fast-text']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        with open(PATH_TOXIC['fast-text']['pickle'], 'rb') as fopen:
            ngram = pickle.load(fopen)
        g = load_graph(PATH_TOXIC['fast-text']['model'])
        return SIGMOID(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            ngram = ngram,
        )
    elif model == 'hierarchical':
        check_file(PATH_TOXIC['hierarchical'], S3_PATH_TOXIC['hierarchical'])
        with open(PATH_TOXIC['hierarchical']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        g = load_graph(PATH_TOXIC['hierarchical']['model'])
        return SIGMOID(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            alphas = g.get_tensor_by_name('import/alphas:0'),
        )
    elif model in ['bahdanau', 'luong']:
        check_file(PATH_TOXIC[model], S3_PATH_TOXIC[model])
        with open(PATH_TOXIC[model]['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        g = load_graph(PATH_TOXIC[model]['model'])
        return SIGMOID(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            alphas = g.get_tensor_by_name('import/alphas:0'),
        )
    elif model == 'entity-network':
        check_file(
            PATH_TOXIC['entity-network'], S3_PATH_TOXIC['entity-network']
        )
        with open(PATH_TOXIC['entity-network']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)
        g = load_graph(PATH_TOXIC['entity-network']['model'])
        return SIGMOID(
            g.get_tensor_by_name('import/Placeholder_question:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            dropout_keep_prob = g.get_tensor_by_name(
                'import/Placeholder_dropout_keep_prob:0'
            ),
            story = g.get_tensor_by_name('import/Placeholder_story:0'),
        )
    else:
        raise Exception(
            'model sentiment not supported, please check supported models from malaya.toxic.available_deep_model()'
        )
