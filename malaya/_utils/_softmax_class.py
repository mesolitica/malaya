import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import json
import os
import pickle
from ._utils import check_file, load_graph, check_available, generate_session
from ..stem import _classification_textcleaning_stemmer
from .._models._sklearn_model import USER_XGB, USER_BAYES
from .._models._tensorflow_model import SOFTMAX, SPARSE_SOFTMAX


def sparse_deep_model(
    path,
    s3_path,
    class_name,
    label,
    output_size,
    embedded_size = 128,
    model = 'fast-text-char',
    validate = True,
):

    if not isinstance(model, str):
        raise ValueError('model must be a string')

    model = model.lower()
    if model == 'fast-text-char':
        if validate:
            check_file(path[model], s3_path[model])
        else:
            if not check_available(path[model]):
                raise Exception(
                    '%s/%s is not available, please `validate = True`'
                    % (class_name, model)
                )
        try:
            with open(path[model]['vector'], 'rb') as fopen:
                vector = pickle.load(fopen)

            return SPARSE_SOFTMAX(
                os.path.dirname(path[model]['model']),
                vector,
                label,
                output_size,
                embedded_size,
                len(vector.vocabulary_),
            )
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
    else:
        raise Exception(
            'model subjectivity not supported, please check supported models from malaya.%s.available_sparse_deep_model()'
            % (class_name)
        )


def deep_model(
    path, s3_path, class_name, label, model = 'luong', validate = True
):
    if not isinstance(model, str):
        raise ValueError('model must be a string')

    model = model.lower()
    if model in ['bahdanau', 'luong', 'hierarchical']:
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
                dictionary = json.load(fopen)['dictionary']
            g = load_graph(path[model]['model'])
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
        return SOFTMAX(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            generate_session(graph = g),
            model,
            dictionary,
            alphas = g.get_tensor_by_name('import/alphas:0'),
            label = label,
        )
    elif model in ['bidirectional', 'fast-text']:
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
                dictionary = json.load(fopen)['dictionary']
            g = load_graph(path[model]['model'])
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
        return SOFTMAX(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            generate_session(graph = g),
            model,
            dictionary,
            label = label,
        )
    elif model == 'bert':
        if validate:
            check_file(path['bert'], s3_path['bert'])
        else:
            if not check_available(path['bert']):
                raise Exception(
                    '%s/%s is not available, please `validate = True`'
                    % (class_name, model)
                )
        try:
            with open(path['bert']['setting'], 'r') as fopen:
                dictionary = json.load(fopen)
            g = load_graph(path['bert']['model'])
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
        return SOFTMAX(
            g.get_tensor_by_name('import/Placeholder_input_ids:0'),
            g.get_tensor_by_name('import/loss/logits:0'),
            generate_session(graph = g),
            model,
            dictionary,
            input_mask = g.get_tensor_by_name(
                'import/Placeholder_input_mask:0'
            ),
            segment_ids = g.get_tensor_by_name(
                'import/Placeholder_segment_ids:0'
            ),
            is_training = g.get_tensor_by_name(
                'import/Placeholder_is_training:0'
            ),
            label = label,
        )
    elif model == 'entity-network':
        if validate:
            check_file(path['entity-network'], s3_path['entity-network'])
        else:
            if not check_available(path['entity-network']):
                raise Exception(
                    '%s/%s is not available, please `validate = True`'
                    % (class_name, model)
                )
        try:
            with open(path['entity-network']['setting'], 'r') as fopen:
                dictionary = json.load(fopen)
            g = load_graph(path['entity-network']['model'])
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
        maxlen = 100 if 'sentiment' in class_name else 80
        return SOFTMAX(
            g.get_tensor_by_name('import/Placeholder_question:0'),
            g.get_tensor_by_name('import/logits:0'),
            generate_session(graph = g),
            model,
            dictionary,
            dropout_keep_prob = g.get_tensor_by_name(
                'import/Placeholder_dropout_keep_prob:0'
            ),
            story = g.get_tensor_by_name('import/Placeholder_story:0'),
            label = label,
            maxlen = maxlen,
        )
    else:
        raise Exception(
            'model subjectivity not supported, please check supported models from malaya.%s.available_deep_model()'
            % (class_name)
        )


def multinomial(path, s3_path, class_name, label, validate = True):
    if validate:
        check_file(path['multinomial'], s3_path['multinomial'])
    else:
        if not check_available(path['multinomial']):
            raise Exception(
                '%s/multinomial is not available, please `validate = True`'
                % (class_name)
            )
    try:
        with open(path['multinomial']['model'], 'rb') as fopen:
            multinomial = pickle.load(fopen)
        with open(path['multinomial']['vector'], 'rb') as fopen:
            vectorize = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/multinomial') and try again"
            % (class_name)
        )
    return USER_BAYES(
        multinomial,
        label,
        vectorize,
        cleaning = _classification_textcleaning_stemmer,
    )


def xgb(path, s3_path, class_name, label, validate = True):
    if validate:
        check_file(path['xgb'], s3_path['xgb'])
    else:
        if not check_available(path['xgb']):
            raise Exception(
                '%s/xgb is not available, please `validate = True`'
                % (class_name)
            )
    try:
        with open(path['xgb']['model'], 'rb') as fopen:
            xgb = pickle.load(fopen)
        with open(path['xgb']['vector'], 'rb') as fopen:
            vectorize = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/xgb') and try again"
            % (class_name)
        )
    return USER_XGB(
        xgb, label, vectorize, cleaning = _classification_textcleaning_stemmer
    )
