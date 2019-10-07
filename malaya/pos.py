import re
from .texts._tatabahasa import tatabahasa_dict, hujung, permulaan
from ._utils import _tag_class
from ._utils._paths import PATH_POS, S3_PATH_POS


_availability = {'model': ['bert', 'xlnet'], 'size': ['base', 'small']}


def available_transformer_model():
    """
    List available transformer Entity Tagging models.
    """
    return _availability


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


def transformer(model = 'xlnet', size = 'base', validate = True):
    """
    Load Transformer Entity Tagging model, transfer learning Transformer + CRF.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - BERT architecture from google.
        * ``'xlnet'`` - XLNET architecture from google.
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
