import re
from malaya.text.tatabahasa import tatabahasa_dict, hujung, permulaan
from malaya.supervised import tag
from malaya.path import PATH_POS, S3_PATH_POS
from herpetologist import check_type

label = [
    'ADJ',
    'ADP',
    'ADV',
    'ADX',
    'CCONJ',
    'DET',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'SCONJ',
    'SYM',
    'VERB',
    'X',
]

_availability = {
    'bert': ['426.4 MB', 'accuracy: 0.952'],
    'tiny-bert': ['57.7 MB', 'accuracy: 0.953'],
    'albert': ['48.7 MB', 'accuracy: 0.951'],
    'tiny-albert': ['22.4 MB', 'accuracy: 0.933'],
    'xlnet': ['446.6 MB', 'accuracy: 0.954'],
    'alxlnet': ['46.8 MB', 'accuracy: 0.951'],
}


def describe():
    """
    Describe Part-Of-Speech supported.
    """
    print('ADJ - Adjective, kata sifat')
    print('ADP - Adposition')
    print('ADV - Adverb, kata keterangan')
    print('ADX - Auxiliary verb, kata kerja tambahan')
    print('CCONJ - Coordinating conjuction, kata hubung')
    print('DET - Determiner, kata penentu')
    print('NOUN - Noun, kata nama')
    print('NUM - Number, nombor')
    print('PART - Particle')
    print('PRON - Pronoun, kata ganti')
    print('PROPN - Proper noun, kata ganti nama khas')
    print('SCONJ - Subordinating conjunction')
    print('SYM - Symbol')
    print('VERB - Verb, kata kerja')
    print('X - Other')


def available_transformer():
    """
    List available transformer Part-Of-Speech Tagging models.
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


@check_type
def naive(string: str):
    """
    Recognize POS in a string using Regex.

    Parameters
    ----------
    string: str

    Returns
    -------
    string : tokenized string with POS related
    """
    string = string.lower()
    results = []
    for i in string.split():
        results.append(_naive_POS_word(i))
    return results


@check_type
def transformer(model: str = 'xlnet', **kwargs):
    """
    Load Transformer Entity Tagging model, transfer learning Transformer + CRF.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - BERT architecture from google.
        * ``'tiny-bert'`` - BERT architecture from google with smaller parameters.
        * ``'albert'`` - ALBERT architecture from google.
        * ``'tiny-albert'`` - ALBERT architecture from google with smaller parameters.
        * ``'xlnet'`` - XLNET architecture from google.
        * ``'alxlnet'`` - XLNET architecture from google + Malaya.

    Returns
    -------
    result : malaya.supervised.tag.transformer
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.pos.available_transformer()'
        )
    return tag.transformer(
        PATH_POS, S3_PATH_POS, 'pos', model = model, **kwargs
    )
