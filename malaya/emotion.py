import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from ._utils import _softmax_class
from ._utils._paths import PATH_EMOTION, S3_PATH_EMOTION

_emotion_label = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']


def available_sparse_deep_model():
    """
    List available sparse deep learning emotion analysis models.
    """
    return ['fast-text-char']


def available_deep_model():
    """
    List available deep learning emotion analysis models.
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


def sparse_deep_model(model = 'fast-text-char'):
    """
    Load deep learning emotion analysis model.

    Parameters
    ----------
    model : str, optional (default='luong')
        Model architecture supported. Allowed values:

        * ``'fast-text-char'`` - Fast-text architecture for character based n-grams, embedded and logits layers only

    Returns
    -------
    SPARSE_SOFTMAX: malaya._models._tensorflow_model.SPARSE_SOFTMAX class
    """
    return _softmax_class.sparse_deep_model(
        PATH_EMOTION,
        S3_PATH_EMOTION,
        'emotion',
        _emotion_label,
        len(_emotion_label),
        model = model,
    )


def deep_model(model = 'luong'):
    """
    Load deep learning emotion analysis model.

    Parameters
    ----------
    model : str, optional (default='luong')
        Model architecture supported. Allowed values:

        * ``'fast-text'`` - Fast-text architecture, embedded and logits layers only
        * ``'hierarchical'`` - LSTM with hierarchical attention architecture
        * ``'bahdanau'`` - LSTM with bahdanau attention architecture
        * ``'bidirectional'`` - LSTM with Bidirectional RNN architecture
        * ``'luong'`` - LSTM with luong attention architecture
        * ``'bert'`` - Deep Bidirectional transformers architecture
        * ``'entity-network'`` - Recurrent Entity-Network architecture

    Returns
    -------
    SOFTMAX: malaya._models._tensorflow_model.SOFTMAX class
    """
    return _softmax_class.deep_model(
        PATH_EMOTION, S3_PATH_EMOTION, 'emotion', _emotion_label, model = model
    )


def multinomial():
    """
    Load multinomial emotion model.

    Returns
    -------
    USER_BAYES : malaya._models._sklearn_model.USER_BAYES class
    """
    return _softmax_class.multinomial(
        PATH_EMOTION, S3_PATH_EMOTION, 'emotion', _emotion_label
    )


def xgb():
    """
    Load XGB emotion model.

    Returns
    -------
    USER_XGB : malaya._models._sklearn_model.USER_XGB class
    """
    return _softmax_class.xgb(
        PATH_EMOTION, S3_PATH_EMOTION, 'emotion', _emotion_label
    )
