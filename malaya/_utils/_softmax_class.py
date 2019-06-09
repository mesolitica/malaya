import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import json
import os
import pickle
from ._utils import check_file, load_graph, check_available, generate_session
from ..stem import _classification_textcleaning_stemmer
from .._models._sklearn_model import (
    BINARY_XGB,
    BINARY_BAYES,
    MULTICLASS_XGB,
    MULTICLASS_BAYES,
)
from .._models._tensorflow_model import (
    BINARY_SOFTMAX,
    MULTICLASS_SOFTMAX,
    SPARSE_SOFTMAX,
    BINARY_BERT,
    MULTICLASS_BERT,
)


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
                path = os.path.dirname(path[model]['model']),
                vectorizer = vector,
                label = label,
                output_size = output_size,
                embedded_size = embedded_size,
                vocab_size = len(vector.vocabulary_),
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

    if len(label) > 2:
        selected_class = MULTICLASS_SOFTMAX
    else:
        selected_class = BINARY_SOFTMAX

    return selected_class(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        logits_seq = g.get_tensor_by_name('import/logits_seq:0'),
        alphas = g.get_tensor_by_name('import/alphas:0'),
        sess = generate_session(graph = g),
        dictionary = dictionary,
        class_name = class_name,
        label = label,
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

    if len(label) > 2:
        selected_class = MULTICLASS_BAYES
    else:
        selected_class = BINARY_BAYES
    return selected_class(
        multinomial = multinomial,
        label = label,
        vectorize = vectorize,
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
    if len(label) > 2:
        selected_class = MULTICLASS_XGB
    else:
        selected_class = BINARY_XGB

    return selected_class(
        xgb = xgb,
        label = label,
        vectorize = vectorize,
        cleaning = _classification_textcleaning_stemmer,
    )


def bert(path, s3_path, class_name, label, validate = True):
    try:
        from bert import tokenization
    except:
        raise Exception(
            'bert-tensorflow not installed. Please install it using `pip3 install bert-tensorflow` and try again.'
        )
    if validate:
        check_file(path['bert'], s3_path['bert'])
    else:
        if not check_available(path['bert']):
            raise Exception(
                '%s/bert is not available, please `validate = True`'
                % (class_name)
            )

    tokenization.validate_case_matches_checkpoint(False, '')
    tokenizer = tokenization.FullTokenizer(
        vocab_file = path['bert']['vocab'], do_lower_case = False
    )
    try:
        g = load_graph(path['bert']['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/bert') and try again"
            % (class_name)
        )

    if len(label) > 2:
        selected_class = MULTICLASS_BERT
    else:
        selected_class = BINARY_BERT

    return selected_class(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
        input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        sess = generate_session(graph = g),
        tokenizer = tokenizer,
        maxlen = 100,
        label = label,
    )
