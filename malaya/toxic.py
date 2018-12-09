import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import json
import pickle
import os
import tensorflow as tf
from .utils import download_file, load_graph
from . import home
from .paths import PATH_TOXIC, S3_PATH_TOXIC
from .sklearn_model import TOXIC
from .tensorflow_model import DEEP_TOXIC


def get_available_toxic_models():
    """
    List available deep learning toxicity analysis models.
    """
    return ['bahdanau', 'hierarchical', 'luong', 'fast-text', 'entity-network']


def multinomial_detect_toxic():
    """
    Load multinomial toxic model.

    Returns
    -------
    TOXIC : malaya.sklearn_model.TOXIC class
    """
    if not os.path.isfile(PATH_TOXIC['multinomial']['model']):
        print('downloading TOXIC pickled multinomial model')
        download_file(
            S3_PATH_TOXIC['multinomial']['model'],
            PATH_TOXIC['multinomial']['model'],
        )
    if not os.path.isfile(PATH_TOXIC['multinomial']['vector']):
        print('downloading TOXIC pickled tfidf vectorizations')
        download_file(
            S3_PATH_TOXIC['multinomial']['vector'],
            PATH_TOXIC['multinomial']['vector'],
        )
    with open(PATH_TOXIC['multinomial']['model'], 'rb') as fopen:
        multinomial = pickle.load(fopen)
    with open(PATH_TOXIC['multinomial']['vector'], 'rb') as fopen:
        vectorize = pickle.load(fopen)
    return TOXIC(multinomial, vectorize)


def logistics_detect_toxic():
    """
    Load logistic toxic model.

    Returns
    -------
    TOXIC : malaya.sklearn_model.TOXIC class
    """
    if not os.path.isfile(PATH_TOXIC['logistic']['model']):
        print('downloading TOXIC pickled logistics model')
        download_file(
            S3_PATH_TOXIC['logistic']['model'], PATH_TOXIC['logistic']['model']
        )
    if not os.path.isfile(PATH_TOXIC['logistic']['vector']):
        print('downloading TOXIC pickled tfidf vectorizations')
        download_file(
            S3_PATH_TOXIC['logistic']['vector'],
            PATH_TOXIC['logistic']['vector'],
        )
    with open(PATH_TOXIC['logistic']['model'], 'rb') as fopen:
        logistic = pickle.load(fopen)
    with open(PATH_TOXIC['logistic']['vector'], 'rb') as fopen:
        vectorize = pickle.load(fopen)
    return TOXIC(logistic, vectorize)


def deep_toxic(model = 'luong'):
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
    TOXIC: malaya.tensorflow_model.DEEP_TOXIC class
    """
    assert isinstance(model, str), 'model must be a string'
    model = model.lower()
    if model == 'fast-text':
        if not os.path.isfile(PATH_TOXIC['fast-text']['model']):
            print('downloading TOXIC frozen fast-text model')
            download_file(
                S3_PATH_TOXIC['fast-text']['model'],
                PATH_TOXIC['fast-text']['model'],
            )
        if not os.path.isfile(PATH_TOXIC['fast-text']['setting']):
            print('downloading TOXIC fast-text dictionary')
            download_file(
                S3_PATH_TOXIC['fast-text']['setting'],
                PATH_TOXIC['fast-text']['setting'],
            )
        if not os.path.isfile(PATH_TOXIC['fast-text']['pickle']):
            print('downloading TOXIC fast-text bigrams')
            download_file(
                S3_PATH_TOXIC['fast-text']['pickle'],
                PATH_TOXIC['fast-text']['pickle'],
            )
        with open(PATH_TOXIC['fast-text']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        with open(PATH_TOXIC['fast-text']['pickle'], 'rb') as fopen:
            ngram = pickle.load(fopen)
        g = load_graph(PATH_TOXIC['fast-text']['model'])
        return DEEP_TOXIC(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            ngram = ngram,
        )
    elif model == 'hierarchical':
        if not os.path.isfile(PATH_TOXIC['hierarchical']['model']):
            print('downloading TOXIC frozen hierarchical model')
            download_file(
                S3_PATH_TOXIC['hierarchical']['model'],
                PATH_TOXIC['hierarchical']['model'],
            )
        if not os.path.isfile(PATH_TOXIC['hierarchical']['setting']):
            print('downloading TOXIC hierarchical dictionary')
            download_file(
                S3_PATH_TOXIC['hierarchical']['setting'],
                PATH_TOXIC['hierarchical']['setting'],
            )
        with open(PATH_TOXIC['hierarchical']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        g = load_graph(PATH_TOXIC['hierarchical']['model'])
        return DEEP_TOXIC(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            alphas = g.get_tensor_by_name('import/alphas:0'),
        )
    elif model == 'bahdanau':
        if not os.path.isfile(PATH_TOXIC['bahdanau']['model']):
            print('downloading TOXIC frozen bahdanau model')
            download_file(
                S3_PATH_TOXIC['bahdanau']['model'],
                PATH_TOXIC['bahdanau']['model'],
            )
        if not os.path.isfile(PATH_TOXIC['bahdanau']['setting']):
            print('downloading TOXIC bahdanau dictionary')
            download_file(
                S3_PATH_TOXIC['bahdanau']['setting'],
                PATH_TOXIC['bahdanau']['setting'],
            )
        with open(PATH_TOXIC['bahdanau']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        g = load_graph(PATH_TOXIC['bahdanau']['model'])
        return DEEP_TOXIC(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            alphas = g.get_tensor_by_name('import/alphas:0'),
        )
    elif model == 'luong':
        if not os.path.isfile(PATH_TOXIC['luong']['model']):
            print('downloading TOXIC frozen luong model')
            download_file(
                S3_PATH_TOXIC['luong']['model'], PATH_TOXIC['luong']['model']
            )
        if not os.path.isfile(PATH_TOXIC['luong']['setting']):
            print('downloading TOXIC luong dictionary')
            download_file(
                S3_PATH_TOXIC['luong']['setting'],
                PATH_TOXIC['luong']['setting'],
            )
        with open(PATH_TOXIC['luong']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)
        g = load_graph(PATH_TOXIC['luong']['model'])
        return DEEP_TOXIC(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            alphas = g.get_tensor_by_name('import/alphas:0'),
        )
    elif model == 'entity-network':
        if not os.path.isfile(PATH_TOXIC['entity-network']['model']):
            print('downloading TOXIC frozen entity-network model')
            download_file(
                S3_PATH_TOXIC['entity-network']['model'],
                PATH_TOXIC['entity-network']['model'],
            )
        if not os.path.isfile(PATH_TOXIC['entity-network']['setting']):
            print('downloading TOXIC entity-network dictionary')
            download_file(
                S3_PATH_TOXIC['entity-network']['setting'],
                PATH_TOXIC['entity-network']['setting'],
            )
        with open(PATH_TOXIC['entity-network']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)
        g = load_graph(PATH_TOXIC['entity-network']['model'])
        return DEEP_TOXIC(
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
            'model sentiment not supported, please check supported models from malaya.get_available_toxic_models()'
        )
