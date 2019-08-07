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
from ._models._tensorflow_model import SIGMOID
from ._models._bert_model import SIGMOID_BERT

from .bert import _extract_attention_weights_import, bert_num_layers


_label_toxic = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate',
]


def available_deep_model():
    """
    List available deep learning toxicity analysis models.
    """
    return ['self-attention', 'bahdanau', 'luong']


def available_bert_model():
    """
    List available bert toxicity analysis models.
    """
    return ['multilanguage', 'base', 'small']


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
    Load deep learning toxicity analysis model.

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


def bert(model = 'base', validate = True):
    """
    Load BERT toxicity model.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'multilanguage'`` - bert multilanguage released by Google, trained on toxicity analysis.
        * ``'base'`` - base bert-bahasa released by Malaya, trained on toxicity analysis.
        * ``'small'`` - small bert-bahasa released by Malaya, trained on toxicity analysis.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    SIGMOID_BERT : malaya._models._tensorflow_model.SIGMOID_BERT class
    """

    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

    model = model.lower()
    if model not in available_bert_model():
        raise Exception(
            'model is not supported, please check supported models from malaya.toxic.available_bert_model()'
        )

    if validate:
        check_file(PATH_TOXIC[model], S3_PATH_TOXIC[model])
    else:
        if not check_available(PATH_TOXIC[model]):
            raise Exception(
                'toxic/%s is not available, please `validate = True`' % (model)
            )

    if model == 'multilanguage':
        from bert import tokenization

        tokenizer = tokenization.FullTokenizer(
            vocab_file = PATH_TOXIC[model]['vocab'], do_lower_case = False
        )
        cls = '[CLS]'
        sep = '[SEP]'
    else:
        import sentencepiece as spm
        from .texts._text_functions import SentencePieceTokenizer

        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(PATH_TOXIC[model]['tokenizer'])

        with open(PATH_TOXIC[model]['vocab']) as fopen:
            v = fopen.read().split('\n')[:-1]
        v = [i.split('\t') for i in v]
        v = {i[0]: i[1] for i in v}
        tokenizer = SentencePieceTokenizer(v, sp_model)
        cls = '<cls>'
        sep = '<sep>'

    try:
        g = load_graph(PATH_TOXIC[model]['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('toxic/%s') and try again"
            % (model)
        )

    return SIGMOID_BERT(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        segment_ids = None,
        input_masks = None,
        logits = g.get_tensor_by_name('import/logits:0'),
        logits_seq = g.get_tensor_by_name('import/logits_seq:0'),
        sess = generate_session(graph = g),
        tokenizer = tokenizer,
        label = _label_toxic,
        cls = cls,
        sep = sep,
        attns = _extract_attention_weights_import(bert_num_layers[model], g),
        class_name = 'toxicity',
    )
