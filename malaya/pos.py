import re
from .texts._tatabahasa import tatabahasa_dict, hujung, permulaan
from ._utils import _tag_class
from ._utils._paths import PATH_POS, S3_PATH_POS


def available_deep_model():
    """
    List available deep learning entities models, ['concat', 'bahdanau', 'luong'].
    """
    return ['concat', 'bahdanau', 'luong']


def available_bert_model():
    """
    List available bert entities models, ['multilanguage', 'base', 'small']
    """
    return ['multilanguage', 'base', 'small']


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
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    TAGGING: malaya.tensorflow_model.TAGGING class
    """

    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

    model = model.lower()
    if model not in available_deep_model():
        raise Exception(
            'model not supported, please check supported models from malaya.entity.available_deep_model()'
        )

    return _tag_class.deep_model(
        PATH_POS, S3_PATH_POS, 'pos', model = model, validate = validate
    )


def bert(model = 'base', validate = True):

    """
    Load BERT POS model.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'multilanguage'`` - bert multilanguage released by Google, trained on POS.
        * ``'base'`` - base bert-bahasa released by Malaya, trained on POS.
        * ``'small'`` - small bert-bahasa released by Malaya, trained on POS.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    TAGGING_BERT: malaya._models._tensorflow_model.TAGGING_BERT class
    """

    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

    model = model.lower()
    if model not in available_bert_model():
        raise Exception(
            'model not supported, please check supported models from malaya.pos.available_bert_model()'
        )

    return _tag_class.bert(
        PATH_POS, S3_PATH_POS, 'pos', model = model, validate = validate
    )
