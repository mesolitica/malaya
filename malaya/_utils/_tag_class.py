import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import pickle
import json
import tensorflow as tf
from ._utils import check_file, load_graph
from ..texts._text_functions import entities_textcleaning
from .._models._tensorflow_model import TAGGING
from .._models._sklearn_model import CRF


def crf(path, s3_path, class_name):
    check_file(path['crf'], s3_path['crf'])
    try:
        with open(path['crf']['model'], 'rb') as fopen:
            model = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/crf') and try again"
            % (class_name)
        )
    return CRF(model)


def deep_model(path, s3_path, class_name, model = 'bahdanau'):
    """
    Load deep learning NER model.

    Parameters
    ----------
    model : str, optional (default='bahdanau')
        Model architecture supported. Allowed values:

        * ``'concat'`` - Concating character and word embedded for BiLSTM
        * ``'bahdanau'`` - Concating character and word embedded including Bahdanau Attention for BiLSTM
        * ``'luong'`` - Concating character and word embedded including Luong Attention for BiLSTM
        * ``'entity-network'`` - Concating character and word embedded on hybrid Entity-Network and RNN
        * ``'attention'`` - Concating character and word embedded with self-attention for BiLSTM

    Returns
    -------
    TAGGING: malaya._models._tensorflow_model.TAGGING class
    """
    assert isinstance(model, str), 'model must be a string'
    model = model.lower()
    if model in ['concat', 'bahdanau', 'luong', 'entity-network', 'attention']:
        check_file(path[model], s3_path[model])
        try:
            with open(path[model]['setting'], 'r') as fopen:
                nodes = json.loads(fopen.read())
            g = load_graph(path[model]['model'])
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
        if model == 'entity-network':
            return TAGGING(
                g.get_tensor_by_name('import/question:0'),
                g.get_tensor_by_name('import/char_ids:0'),
                g.get_tensor_by_name('import/logits:0'),
                nodes,
                tf.InteractiveSession(graph = g),
                model,
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
            )

    else:
        raise Exception(
            'model not supported, please check supported models from malaya.%s.available_deep_model()'
            % (class_name)
        )
