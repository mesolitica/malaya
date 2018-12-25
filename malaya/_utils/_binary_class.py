import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
import json
import pickle
from ._utils import check_file, load_graph
from ..stem import _classification_textcleaning_stemmer
from .._models._sklearn_model import USER_XGB, USER_BAYES
from .._models._tensorflow_model import SENTIMENT


def deep_model(path, s3_path, class_name, model = 'luong'):
    assert isinstance(model, str), 'model must be a string'
    model = model.lower()
    if model == 'fast-text':
        check_file(path['fast-text'], s3_path['fast-text'])
        try:
            with open(path['fast-text']['setting'], 'r') as fopen:
                dictionary = json.load(fopen)['dictionary']
            with open(path['fast-text']['pickle'], 'rb') as fopen:
                ngram = pickle.load(fopen)
            g = load_graph(path['fast-text']['model'])
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            ngram = ngram,
        )
    elif model == 'hierarchical':
        check_file(path['hierarchical'], s3_path['hierarchical'])
        try:
            with open(path['hierarchical']['setting'], 'r') as fopen:
                dictionary = json.load(fopen)['dictionary']
            g = load_graph(path['hierarchical']['model'])
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            alphas = g.get_tensor_by_name('import/alphas:0'),
        )
    elif model in ['bahdanau', 'luong']:
        check_file(path[model], s3_path[model])
        try:
            with open(path[model]['setting'], 'r') as fopen:
                dictionary = json.load(fopen)['dictionary']
            g = load_graph(path[model]['model'])
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            alphas = g.get_tensor_by_name('import/alphas:0'),
        )
    elif model == 'bidirectional':
        check_file(path['bidirectional'], s3_path['bidirectional'])
        try:
            with open(path['bidirectional']['setting'], 'r') as fopen:
                dictionary = json.load(fopen)
            g = load_graph(path['bidirectional']['model'])
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
        )
    elif model == 'bert':
        check_file(path['bert'], s3_path['bert'])
        try:
            with open(path['bert']['setting'], 'r') as fopen:
                dictionary = json.load(fopen)
            g = load_graph(path['bert']['model'])
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder_input_ids:0'),
            g.get_tensor_by_name('import/loss/logits:0'),
            tf.InteractiveSession(graph = g),
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
        )
    elif model == 'entity-network':
        check_file(path['entity-network'], s3_path['entity-network'])
        try:
            with open(path['entity-network']['setting'], 'r') as fopen:
                dictionary = json.load(fopen)
            g = load_graph(path['entity-network']['model'])
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
                % (class_name, model)
            )
        return SENTIMENT(
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
            'model subjectivity not supported, please check supported models from malaya.%s.available_deep_model()'
            % (class_name)
        )


def multinomial(path, s3_path, class_name):
    check_file(path['multinomial'], s3_path['multinomial'])
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
        ['negative', 'positive'],
        vectorize,
        cleaning = _classification_textcleaning_stemmer,
    )


def xgb(path, s3_path, class_name):
    check_file(path['xgb'], s3_path['xgb'])
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
        xgb,
        ['negative', 'positive'],
        vectorize,
        cleaning = _classification_textcleaning_stemmer,
    )
