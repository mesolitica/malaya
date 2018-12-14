import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import os
import re
import pickle
import json
import tensorflow as tf
from .tatabahasa import tatabahasa_dict, hujung, permulaan
from .utils import download_file, load_graph
from .text_functions import entities_textcleaning
from .sklearn_model import CRF
from .tensorflow_model import TAGGING
from .paths import PATH_POS, S3_PATH_POS


def get_available_pos_models():
    """
    List available deep learning entities models, ['concat', 'bahdanau', 'luong','entity-network']
    """
    return ['concat', 'bahdanau', 'luong', 'entity-network']


def naive_POS_word(word):
    for key, vals in tatabahasa_dict.items():
        if word in vals:
            return (key, word)
    try:
        if len(re.findall(r'^(.*?)(%s)$' % ('|'.join(hujung[:1])), i)[0]) > 1:
            return ('KJ', word)
    except:
        pass
    try:
        if (
            len(re.findall(r'^(.*?)(%s)' % ('|'.join(permulaan[:-4])), word)[0])
            > 1
        ):
            return ('KJ', word)
    except Exception as e:
        pass
    if len(word) > 2:
        return ('KN', word)
    else:
        return ('', word)


def naive_pos(string):
    """
    Recognize POS in a string using Regex.

    Parameters
    ----------
    string: str

    Returns
    -------
    string : tokenized string with POS related
    """
    assert isinstance(string, str), 'input must be a string'
    string = string.lower()
    results = []
    for i in string.split():
        results.append(naive_POS_word(i))
    return results


def crf_pos():
    """
    Load CRF POS Recognition model.

    Returns
    -------
    CRF : malaya.sklearn_model.CRF class
    """
    if not os.path.isfile(PATH_POS['crf']['model']):
        print('downloading POS frozen CRF model')
        download_file(S3_PATH_POS['crf']['model'], PATH_POS['crf']['model'])
    with open(PATH_POS['crf']['model'], 'rb') as fopen:
        model = pickle.load(fopen)
    return CRF(model, is_lower = False)


def deep_pos(model = 'concat'):
    """
    Load deep learning POS Recognition model.

    Parameters
    ----------
    model : str, optional (default='bahdanau')
        Model architecture supported. Allowed values:

        * ``'concat'`` - Concating character and word embedded for BiLSTM
        * ``'bahdanau'`` - Concating character and word embedded including Bahdanau Attention for BiLSTM
        * ``'luong'`` - Concating character and word embedded including Luong Attention for BiLSTM
        * ``'entity-network'`` - Concating character and word embedded on hybrid Entity-Network and RNN

    Returns
    -------
    TAGGING: malaya.tensorflow_model.TAGGING class
    """
    assert isinstance(model, str), 'model must be a string'
    model = model.lower()
    if model in ['concat', 'bahdanau', 'luong', 'entity-network']:
        if not os.path.isfile(PATH_POS[model]['model']):
            print('downloading POS frozen %s model' % (model))
            download_file(S3_PATH_POS[model]['model'], PATH_POS[model]['model'])
        if not os.path.isfile(PATH_POS[model]['setting']):
            print('downloading POS %s dictionary' % (model))
            download_file(
                S3_PATH_POS[model]['setting'], PATH_POS[model]['setting']
            )
        with open(PATH_POS[model]['setting'], 'r') as fopen:
            nodes = json.loads(fopen.read())
        g = load_graph(PATH_POS[model]['model'])
        if model == 'entity-network':
            return TAGGING(
                g.get_tensor_by_name('import/question:0'),
                g.get_tensor_by_name('import/char_ids:0'),
                g.get_tensor_by_name('import/logits:0'),
                nodes,
                tf.InteractiveSession(graph = g),
                model,
                is_lower = False,
                story = g.get_tensor_by_name('import/story:0'),
            )
        else:
            return TAGGING(
                g.get_tensor_by_name('import/Placeholder:0'),
                g.get_tensor_by_name('import/Placeholder_1:0'),
                g.get_tensor_by_name('import/logits:0'),
                nodes,
                tf.InteractiveSession(graph = g),
                model,
                is_lower = False,
            )
    else:
        raise Exception(
            'model not supported, please check supported models from malaya.get_available_pos_models()'
        )
