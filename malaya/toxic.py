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
from ._models._tensorflow_model import SIGMOID, SPARSE_SIGMOID, SIGMOID_BERT


_label_toxic = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate',
]


def available_sparse_deep_model():
    """
    List available sparse deep learning toxicity analysis models.
    """
    return ['fast-text-char']


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
    SIGMOID: malaya._models._tensorflow_model.SIGMOID class
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


def bert(validate = True):
    """
    Load BERT toxicity model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    SIGMOID_BERT : malaya._models._tensorflow_model.SIGMOID_BERT class
    """
    try:
        from bert import tokenization
    except:
        raise Exception(
            'bert-tensorflow not installed. Please install it using `pip3 install bert-tensorflow` and try again.'
        )
    if validate:
        check_file(PATH_TOXIC['bert'], S3_PATH_TOXIC['bert'])
    else:
        if not check_available(PATH_TOXIC['bert']):
            raise Exception(
                'toxic/bert is not available, please `validate = True`'
            )

    tokenization.validate_case_matches_checkpoint(False, '')
    tokenizer = tokenization.FullTokenizer(
        vocab_file = PATH_TOXIC['bert']['vocab'], do_lower_case = False
    )
    try:
        g = load_graph(PATH_TOXIC['bert']['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('toxic/bert') and try again"
        )

    return SIGMOID_BERT(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
        input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        sess = generate_session(graph = g),
        tokenizer = tokenizer,
        maxlen = 100,
        label = _label_toxic,
    )


def sparse_deep_model(model = 'fast-text-char', validate = True):
    """
    Load deep learning sentiment analysis model.

    Parameters
    ----------
    model : str, optional (default='luong')
        Model architecture supported. Allowed values:

        * ``'fast-text-char'`` - Fast-text architecture for character based n-grams, embedded and logits layers only.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    SPARSE_SIGMOID: malaya._models._tensorflow_model.SPARSE_SIGMOID class
    """

    if not isinstance(model, str):
        raise ValueError('model must be a string')

    model = model.lower()
    if model == 'fast-text-char':
        if validate:
            check_file(PATH_TOXIC[model], S3_PATH_TOXIC[model])
        else:
            if not check_available(PATH_TOXIC[model]):
                raise Exception(
                    'toxic/%s is not available, please `validate = True`'
                    % (model)
                )
        try:
            with open(PATH_TOXIC[model]['vector'], 'rb') as fopen:
                vector = pickle.load(fopen)

            return SPARSE_SIGMOID(
                path = os.path.dirname(PATH_TOXIC[model]['model']),
                vectorizer = vector,
                label = _label_toxic,
                output_size = len(_label_toxic),
                embedded_size = 128,
                vocab_size = len(vector.vocabulary_),
            )
        except:
            raise Exception(
                "model corrupted due to some reasons, please run malaya.clear_cache('toxic/%s') and try again"
                % (model)
            )
    else:
        raise Exception(
            'model subjectivity not supported, please check supported models from malaya.toxic.available_sparse_deep_model()'
        )
