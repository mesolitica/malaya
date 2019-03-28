import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from ._utils import _softmax_class
from ._models._sklearn_model import USER_BAYES
from ._utils._paths import PATH_SENTIMENTS, S3_PATH_SENTIMENTS


def available_sparse_deep_model():
    """
    List available sparse deep learning sentiment analysis models.
    """
    return ['fast-text-char']


def available_deep_model():
    """
    List available deep learning sentiment analysis models.
    """
    return [
        'fast-text',
        'hierarchical',
        'bahdanau',
        'luong',
        'bidirectional',
        'bert',
        'entity-network',
    ]


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
    SPARSE_SOFTMAX: malaya._models._tensorflow_model.SPARSE_SOFTMAX class
    """
    return _softmax_class.sparse_deep_model(
        PATH_SENTIMENTS,
        S3_PATH_SENTIMENTS,
        'sentiment',
        ['negative', 'positive'],
        2,
        model = model,
        validate = validate,
    )


def deep_model(model = 'luong', validate = True):
    """
    Load deep learning sentiment analysis model.

    Parameters
    ----------
    model : str, optional (default='luong')
        Model architecture supported. Allowed values:

        * ``'fast-text'`` - Fast-text architecture, embedded and logits layers only.
        * ``'hierarchical'`` - LSTM with hierarchical attention architecture.
        * ``'bahdanau'`` - LSTM with bahdanau attention architecture.
        * ``'bidirectional'`` - LSTM with Bidirectional RNN architecture.
        * ``'luong'`` - LSTM with luong attention architecture.
        * ``'bert'`` - Deep Bidirectional transformers architecture.
        * ``'entity-network'`` - Recurrent Entity-Network architecture.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    SOFTMAX: malaya._models._tensorflow_model.SOFTMAX class
    """
    return _softmax_class.deep_model(
        PATH_SENTIMENTS,
        S3_PATH_SENTIMENTS,
        'sentiment',
        ['negative', 'positive'],
        model = model,
        validate = validate,
    )


def multinomial(validate = True):
    """
    Load multinomial sentiment model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    USER_BAYES : malaya._models._sklearn_model.USER_BAYES class
    """
    return _softmax_class.multinomial(
        PATH_SENTIMENTS,
        S3_PATH_SENTIMENTS,
        'sentiment',
        ['negative', 'positive'],
        validate = validate,
    )


def xgb(validate = True):
    """
    Load XGB sentiment model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    USER_XGB : malaya._models._sklearn_model.USER_XGB class
    """
    return _softmax_class.xgb(
        PATH_SENTIMENTS,
        S3_PATH_SENTIMENTS,
        'sentiment',
        ['negative', 'positive'],
        validate = validate,
    )
