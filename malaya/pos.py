import re
from malaya.text.tatabahasa import tatabahasa_dict, hujung, permulaan
from malaya.supervised import tag
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

_transformer_availability = {
    'bert': {
        'Size (MB)': 426.4,
        'Quantized Size (MB)': 111,
        'macro precision': 0.93280,
        'macro recall': 0.93129,
        'macro f1-score': 0.93181,
    },
    'tiny-bert': {
        'Size (MB)': 57.7,
        'Quantized Size (MB)': 15.4,
        'macro precision': 0.92810,
        'macro recall': 0.92649,
        'macro f1-score': 0.92704,
    },
    'albert': {
        'Size (MB)': 48.7,
        'Quantized Size (MB)': 12.8,
        'macro precision': 0.93199,
        'macro recall': 0.91948,
        'macro f1-score': 0.92547,
    },
    'tiny-albert': {
        'Size (MB)': 22.4,
        'Quantized Size (MB)': 5.98,
        'macro precision': 0.90579,
        'macro recall': 0.89501,
        'macro f1-score': 0.90002,
    },
    'xlnet': {
        'Size (MB)': 446.6,
        'Quantized Size (MB)': 118,
        'macro precision': 0.93303,
        'macro recall': 0.93222,
        'macro f1-score': 0.93236,
    },
    'alxlnet': {
        'Size (MB)': 46.8,
        'Quantized Size (MB)': 13.3,
        'macro precision': 0.92732,
        'macro recall': 0.93046,
        'macro f1-score': 0.92819,
    },
}


def describe():
    """
    Describe Part-Of-Speech supported.
    """
    d = [
        {'Tag': 'ADJ', 'Description': 'Adjective, kata sifat'},
        {'Tag': 'ADP', 'Description': 'Adposition'},
        {'Tag': 'ADV', 'Description': 'Adverb, kata keterangan'},
        {'Tag': 'ADX', 'Description': 'Auxiliary verb, kata kerja tambahan'},
        {'Tag': 'CCONJ', 'Description': 'Coordinating conjuction, kata hubung'},
        {'Tag': 'DET', 'Description': 'Determiner, kata penentu'},
        {'Tag': 'NOUN', 'Description': ' Noun, kata nama'},
        {'Tag': 'NUM', 'Description': 'Number, nombor'},
        {'Tag': 'PART', 'Description': 'Particle'},
        {'Tag': 'PRON', 'Description': 'Pronoun, kata ganti'},
        {'Tag': 'PROPN', 'Description': 'Proper noun, kata ganti nama khas'},
        {'Tag': 'SCONJ', 'Description': 'Subordinating conjunction'},
        {'Tag': 'SYM', 'Description': 'Symbol'},
        {'Tag': 'VERB', 'Description': 'Verb, kata kerja'},
        {'Tag': 'X', 'Description': 'Other'},
    ]

    from malaya.function import describe_availability

    return describe_availability(d, transpose=False)


def available_transformer():
    """
    List available transformer Part-Of-Speech Tagging models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 20% test set.'
    )


def _naive_POS_word(word):
    for key, vals in tatabahasa_dict.items():
        if word in vals:
            return (key, word)
    try:
        if len(re.findall(r'^(.*?)(%s)$' % ('|'.join(hujung[:1])), i)[0]) > 1:
            return ('KJ', word)
    except BaseException:
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
    string : List[Tuple[str, str]]
    """
    string = string.lower()
    results = []
    for i in string.split():
        results.append(_naive_POS_word(i))
    return results


@check_type
def transformer(model: str = 'xlnet', quantized: bool = False, **kwargs):
    """
    Load Transformer POS Tagging model, transfer learning Transformer + CRF.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - Google BERT BASE parameters.
        * ``'tiny-bert'`` - Google BERT TINY parameters.
        * ``'albert'`` - Google ALBERT BASE parameters.
        * ``'tiny-albert'`` - Google ALBERT TINY parameters.
        * ``'xlnet'`` - Google XLNET BASE parameters.
        * ``'alxlnet'`` - Malaya ALXLNET BASE parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.model.bert.TaggingBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.TaggingXLNET`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.pos.available_transformer()`.'
        )
    return tag.transformer(
        module='pos', model=model, quantized=quantized, **kwargs
    )
