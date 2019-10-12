import json
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

from ._transformer._bert import (
    _extract_attention_weights_import,
    bert_num_layers,
)


_label_toxic = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate',
]


_availability = {
    'bert': ['base', 'small'],
    'xlnet': ['base'],
    'albert': ['base'],
}


def available_transformer_model():
    """
    List available transformer toxicity analysis models.
    """
    return _availability


def multinomial(validate = True):
    """
    Load multinomial toxicity model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    BAYES : malaya._models._sklearn_model.MULTILABEL_BAYES class
    """


def transformer(model = 'xlnet', size = 'base', validate = True):
    """
    Load Transformer emotion model.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - BERT architecture from google.
        * ``'xlnet'`` - XLNET architecture from google.
        * ``'albert'`` - ALBERT architecture from google.
    size : str, optional (default='base')
        Model size supported. Allowed values:

        * ``'base'`` - BASE size.
        * ``'small'`` - SMALL size.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    BERT : malaya._models._bert_model.BINARY_BERT class
    """
    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(size, str):
        raise ValueError('size must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

    model = model.lower()
    size = size.lower()
    if model not in _availability['model']:
        raise Exception(
            'model not supported, please check supported models from malaya.sentiment.available_transformer_model()'
        )
    if size not in _availability['size']:
        raise Exception(
            'size not supported, please check supported models from malaya.sentiment.available_transformer_model()'
        )
    return _softmax_class.bert(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'sentiment',
        ['negative', 'positive'],
        model = model,
        validate = validate,
    )
