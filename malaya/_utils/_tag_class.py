import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import pickle
import json
from ._utils import check_file, load_graph, check_available, generate_session
from .._models._tensorflow_model import TAGGING
from .._models._sklearn_model import CRF


def crf(path, s3_path, class_name, is_lower = True, validate = True):
    if validate:
        check_file(path['crf'], s3_path['crf'])
    else:
        if not check_available(path['crf']):
            raise Exception(
                '%s/crf is not available, please `validate = True`'
                % (class_name)
            )
    try:
        with open(path['crf']['model'], 'rb') as fopen:
            model = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/crf') and try again"
            % (class_name)
        )
    return CRF(model, is_lower = is_lower)


def deep_model(
    path,
    s3_path,
    class_name,
    model = 'bahdanau',
    is_lower = True,
    validate = True,
):
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
    if not isinstance(model, str):
        raise ValueError('model must be a string')

    model = model.lower()
    if model in ['concat', 'bahdanau', 'luong', 'entity-network', 'attention']:
        if validate:
            check_file(path[model], s3_path[model])
        else:
            if not check_available(path[model]):
                raise Exception(
                    '%s/%s is not available, please `validate = True`'
                    % (class_name, model)
                )
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
                generate_session(graph = g),
                model,
                g.get_tensor_by_name('import/transitions:0'),
                g.get_tensor_by_name('import/Variable:0'),
                is_lower = is_lower,
                story = g.get_tensor_by_name('import/story:0'),
            )
        elif model in ['bahdanau', 'luong']:
            return TAGGING(
                g.get_tensor_by_name('import/Placeholder:0'),
                g.get_tensor_by_name('import/Placeholder_1:0'),
                g.get_tensor_by_name('import/logits:0'),
                nodes,
                generate_session(graph = g),
                model,
                g.get_tensor_by_name('import/transitions:0'),
                g.get_tensor_by_name('import/Variable:0'),
                is_lower = is_lower,
                tags_state_fw = g.get_tensor_by_name('import/state_fw:0'),
                tags_state_bw = g.get_tensor_by_name('import/state_bw:0'),
            )
        else:
            return TAGGING(
                g.get_tensor_by_name('import/Placeholder:0'),
                g.get_tensor_by_name('import/Placeholder_1:0'),
                g.get_tensor_by_name('import/logits:0'),
                nodes,
                generate_session(graph = g),
                model,
                g.get_tensor_by_name('import/transitions:0'),
                g.get_tensor_by_name('import/Variable:0'),
                is_lower = is_lower,
            )

    else:
        raise Exception(
            'model not supported, please check supported models from malaya.%s.available_deep_model()'
            % (class_name)
        )
