import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from ._utils import _tag_class
from ._utils._paths import (
    PATH_ENTITIES,
    S3_PATH_ENTITIES,
    PATH_ENTITIES_SENSITIVE,
    S3_PATH_ENTITIES_SENSITIVE,
)


def available_deep_model():
    """
    List available deep learning entities models, ['concat', 'bahdanau', 'luong', 'entity-network', 'attention']
    """
    return ['concat', 'bahdanau', 'luong', 'entity-network', 'attention']


def crf(validate = True, sensitive = False):
    """
    Load CRF Entities Recognition model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.
    sensitive: bool, optional (default=False)
        if True, the entities recognition model will become case sensitive.

    Returns
    -------
    CRF : malaya._models._sklearn_model.CRF class
    """
    if sensitive:
        return _tag_class.crf(
            PATH_ENTITIES_SENSITIVE,
            S3_PATH_ENTITIES_SENSITIVE,
            'entity-sensitive',
            is_lower = False,
            validate = validate,
        )
    else:
        return _tag_class.crf(
            PATH_ENTITIES, S3_PATH_ENTITIES, 'entity', validate = validate
        )


def deep_model(model = 'bahdanau', sensitive = False, validate = True):
    """
    Load deep learning NER model.

    Parameters
    ----------
    model : str, optional (default='bahdanau')
        Model architecture supported. Allowed values:

        * ``'concat'`` - Concating character and word embedded for BiLSTM.
        * ``'bahdanau'`` - Concating character and word embedded including Bahdanau Attention for BiLSTM.
        * ``'luong'`` - Concating character and word embedded including Luong Attention for BiLSTM.
        * ``'entity-network'`` - Concating character and word embedded on hybrid Entity-Network and RNN.
        * ``'attention'`` - Concating character and word embedded with self-attention for BiLSTM.
    sensitive: bool, optional (default=False)
        if True, the entities recognition model will become case sensitive.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    TAGGING: malaya._models._tensorflow_model.TAGGING class
    """
    if sensitive:
        return _tag_class.deep_model(
            PATH_ENTITIES_SENSITIVE,
            S3_PATH_ENTITIES_SENSITIVE,
            'entity-sensitive',
            model = model,
            is_lower = False,
            validate = validate,
        )
    else:
        return _tag_class.deep_model(
            PATH_ENTITIES,
            S3_PATH_ENTITIES,
            'entity',
            model = model,
            validate = validate,
        )
