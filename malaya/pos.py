import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import re
from .texts._tatabahasa import tatabahasa_dict, hujung, permulaan
from ._utils import _tag_class
from ._utils._paths import PATH_POS, S3_PATH_POS


def available_deep_model():
    """
    List available deep learning entities models, ['concat', 'bahdanau', 'luong', 'entity-network', 'attention'].
    """
    return ['concat', 'bahdanau', 'luong', 'entity-network', 'attention']


def _naive_POS_word(word):
    for key, vals in tatabahasa_dict.items():
        if word in vals:
            return (key, word)
    try:
        if len(re.findall(r'^(.*?)(%s)$' % ('|'.join(hujung[:1])), i)[0]) > 1:
            return ('KJ', word)
    except:
        pass
    try:
        if (
            len(re.findall(r'^(.*?)(%s)' % ('|'.join(permulaan[:-4])), word)[0])
            > 1
        ):
            return ('KJ', word)
    except Exception as e:
        pass
    if len(word) > 2:
        return ('KN', word)
    else:
        return ('', word)


def naive(string):
    """
    Recognize POS in a string using Regex.

    Parameters
    ----------
    string: str

    Returns
    -------
    string : tokenized string with POS related
    """
    if not isinstance(string, str):
        raise ValueError('input must be a string')
    string = string.lower()
    results = []
    for i in string.split():
        results.append(_naive_POS_word(i))
    return results


def crf(validate = True):
    """
    Load CRF POS Recognition model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    CRF : malaya.sklearn_model.CRF class
    """
    return _tag_class.crf(
        PATH_POS, S3_PATH_POS, 'pos', is_lower = False, validate = validate
    )


def deep_model(model = 'concat', validate = True):
    """
    Load deep learning POS Recognition model.

    Parameters
    ----------
    model : str, optional (default='bahdanau')
        Model architecture supported. Allowed values:

        * ``'concat'`` - Concating character and word embedded for BiLSTM.
        * ``'bahdanau'`` - Concating character and word embedded including Bahdanau Attention for BiLSTM.
        * ``'luong'`` - Concating character and word embedded including Luong Attention for BiLSTM.
        * ``'entity-network'`` - Concating character and word embedded on hybrid Entity-Network and RNN.
        * ``'attention'`` - Concating character and word embedded with self-attention for BiLSTM.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    TAGGING: malaya.tensorflow_model.TAGGING class
    """
    return _tag_class.deep_model(
        PATH_POS,
        S3_PATH_POS,
        'pos',
        model = model,
        is_lower = False,
        validate = validate,
    )
