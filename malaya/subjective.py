import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from ._utils import _binary_class
from ._utils._paths import PATH_SUBJECTIVE, S3_PATH_SUBJECTIVE


def available_deep_model():
    """
    List available deep learning subjectivity analysis models.
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


def deep_model(model = 'luong'):
    """
    Load deep learning subjectivity analysis model.

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
    SENTIMENT: malaya._models._tensorflow_model.SENTIMENT class
    """
    return _binary_class.deep_model(
        PATH_SUBJECTIVE, S3_PATH_SUBJECTIVE, 'subjective', model = model
    )


def multinomial():
    """
    Load multinomial subjectivity model.

    Returns
    -------
    USER_BAYES : malaya._models._sklearn_model.USER_BAYES class
    """
    return _binary_class.multinomial(
        PATH_SUBJECTIVE, S3_PATH_SUBJECTIVE, 'subjective'
    )


def xgb():
    """
    Load XGB subjectivity model.

    Returns
    -------
    USER_XGB : malaya._models._sklearn_model.USER_XGB class
    """
    return _binary_class.xgb(PATH_SUBJECTIVE, S3_PATH_SUBJECTIVE, 'subjective')
