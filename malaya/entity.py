import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from ._utils import _tag_class
from ._utils._paths import PATH_ENTITIES, S3_PATH_ENTITIES


def available_deep_model():
    """
    List available deep learning entities models, ['concat', 'bahdanau', 'luong', 'entity-network', 'attention']
    """
    return ['concat', 'bahdanau', 'luong', 'entity-network', 'attention']


def crf(validate = True):
    """
    Load CRF Entities Recognition model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    CRF : malaya._models._sklearn_model.CRF class
    """
    return _tag_class.crf(
        PATH_ENTITIES, S3_PATH_ENTITIES, 'entity', validate = validate
    )


def deep_model(model = 'bahdanau', validate = True):
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
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    TAGGING: malaya._models._tensorflow_model.TAGGING class
    """
    return _tag_class.deep_model(
        PATH_ENTITIES,
        S3_PATH_ENTITIES,
        'entity',
        model = model,
        validate = validate,
    )
