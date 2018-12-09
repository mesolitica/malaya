import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import pickle
import os
import json
import tensorflow as tf
from .utils import download_file, load_graph
from .text_functions import entities_textcleaning
from .paths import PATH_ENTITIES, S3_PATH_ENTITIES
from .tensorflow_model import TAGGING
from .sklearn_model import CRF


def get_available_entities_models():
    """
    List available deep learning entities models, ['concat', 'bahdanau', 'luong']
    """
    return ['concat', 'bahdanau', 'luong']


def crf_entities():
    """
    Load CRF Entities Recognition model.

    Returns
    -------
    CRF : malaya.sklearn_model.CRF class
    """
    if not os.path.isfile(PATH_ENTITIES['crf']['model']):
        print('downloading ENTITIES frozen CRF model')
        download_file(
            S3_PATH_ENTITIES['crf']['model'], PATH_ENTITIES['crf']['model']
        )
    with open(PATH_ENTITIES['crf']['model'], 'rb') as fopen:
        model = pickle.load(fopen)
    return CRF(model)


def deep_entities(model = 'bahdanau'):
    """
    Load deep learning NER model.

    Parameters
    ----------
    model : str, optional (default='bahdanau')
        Model architecture supported. Allowed values:

        * ``'concat'`` - Concating character and word embedded for BiLSTM
        * ``'bahdanau'`` - Concating character and word embedded including Bahdanau Attention for BiLSTM
        * ``'luong'`` - Concating character and word embedded including Luong Attention for BiLSTM

    Returns
    -------
    TAGGING: malaya.tensorflow_model.TAGGING class
    """
    assert isinstance(model, str), 'model must be a string'
    model = model.lower()
    if model in ['concat', 'bahdanau', 'luong']:
        if not os.path.isfile(PATH_ENTITIES[model]['model']):
            print('downloading ENTITIES frozen %s model' % (model))
            download_file(
                S3_PATH_ENTITIES[model]['model'], PATH_ENTITIES[model]['model']
            )
        if not os.path.isfile(PATH_ENTITIES[model]['setting']):
            print('downloading ENTITIES %s dictionary' % (model))
            download_file(
                S3_PATH_ENTITIES[model]['setting'],
                PATH_ENTITIES[model]['setting'],
            )
        with open(PATH_ENTITIES[model]['setting'], 'r') as fopen:
            nodes = json.loads(fopen.read())
        g = load_graph(PATH_ENTITIES[model]['model'])
        return TAGGING(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/Placeholder_1:0'),
            g.get_tensor_by_name('import/logits:0'),
            nodes,
            tf.InteractiveSession(graph = g),
        )

    else:
        raise Exception(
            'model not supported, please check supported models from malaya.get_available_entities_models()'
        )
