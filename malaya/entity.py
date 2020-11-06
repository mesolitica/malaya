from malaya.supervised import tag
from malaya.path import PATH_ENTITIES, S3_PATH_ENTITIES
from malaya.text.entity import ENTITY_REGEX
from herpetologist import check_type

label = {
    'PAD': 0,
    'X': 1,
    'OTHER': 2,
    'law': 3,
    'location': 4,
    'organization': 5,
    'person': 6,
    'quantity': 7,
    'time': 8,
    'event': 9,
}
label_ontonotes5 = {
    'PAD': 0,
    'X': 1,
    'PERSON': 2,
    'NORP': 3,
    'FAC': 4,
    'ORG': 5,
    'GPE': 6,
    'LOC': 7,
    'PRODUCT': 8,
    'EVENT': 9,
    'WORK_OF_ART': 10,
    'LAW': 11,
    'LANGUAGE': 12,
    'DATE': 13,
    'TIME': 14,
    'PERCENT': 15,
    'MONEY': 16,
    'QUANTITY': 17,
    'ORDINAL': 18,
    'CARDINAL': 19,
}
_transformer_availability = {
    'bert': {'Size (MB)': 425.4, 'Quantized Size (MB)': 111, 'Accuracy': 0.994},
    'tiny-bert': {
        'Size (MB)': 57.7,
        'Quantized Size (MB)': 15.4,
        'Accuracy': 0.986,
    },
    'albert': {
        'Size (MB)': 48.6,
        'Quantized Size (MB)': 12.8,
        'Accuracy': 0.984,
    },
    'tiny-albert': {
        'Size (MB)': 22.4,
        'Quantized Size (MB)': 5.98,
        'Accuracy': 0.971,
    },
    'xlnet': {
        'Size (MB)': 446.6,
        'Quantized Size (MB)': 118,
        'Accuracy': 0.992,
    },
    'alxlnet': {
        'Size (MB)': 46.8,
        'Quantized Size (MB)': 13.3,
        'Accuracy': 0.993,
    },
}


def describe():
    """
    Describe Entities supported.
    """
    d = [
        {'Tag': 'OTHER', 'Description': 'other'},
        {
            'Tag': 'law',
            'Description': 'law, regulation, related law documents, documents, etc',
        },
        {'Tag': 'location', 'Description': 'location, place'},
        {
            'Tag': 'organization',
            'Description': 'organization, company, government, facilities, etc',
        },
        {
            'Tag': 'person',
            'Description': 'person, group of people, believes, unique arts (eg; food, drink), etc',
        },
        {'Tag': 'quantity', 'Description': 'numbers, quantity'},
        {'Tag': 'time', 'Description': 'date, day, time, etc'},
        {'Tag': 'event', 'Description': 'unique event happened, etc'},
    ]

    from malaya.function import describe_availability

    return describe_availability(d, transpose = False)


def describe_ontonotes5():
    """
    Describe OntoNotes5 Entities supported. https://spacy.io/api/annotation#named-entities
    """
    d = [
        {'Tag': 'PERSON', 'Description': 'People, including fictional.'},
        {
            'Tag': 'NORP',
            'Description': 'Nationalities or religious or political groups.',
        },
        {
            'Tag': 'FAC',
            'Description': 'Buildings, airports, highways, bridges, etc.',
        },
        {
            'Tag': 'ORG',
            'Description': 'Companies, agencies, institutions, etc.',
        },
        {'Tag': 'GPE', 'Description': 'Countries, cities, states.'},
        {
            'Tag': 'LOC',
            'Description': 'Non-GPE locations, mountain ranges, bodies of water.',
        },
        {
            'Tag': 'PRODUCT',
            'Description': 'Objects, vehicles, foods, etc. (Not services.)',
        },
        {
            'Tag': 'EVENT',
            'Description': 'Named hurricanes, battles, wars, sports events, etc.',
        },
        {'Tag': 'WORK_OF_ART', 'Description': 'Titles of books, songs, etc.'},
        {'Tag': 'LAW', 'Description': 'Named documents made into laws.'},
        {'Tag': 'LANGUAGE', 'Description': 'Any named language.'},
        {
            'Tag': 'DATE',
            'Description': 'Absolute or relative dates or periods.',
        },
        {'Tag': 'TIME', 'Description': 'Times smaller than a day.'},
        {'Tag': 'PERCENT', 'Description': 'Percentage, including "%".'},
        {'Tag': 'MONEY', 'Description': 'Monetary values, including unit.'},
        {
            'Tag': 'QUANTITY',
            'Description': 'Measurements, as of weight or distance.',
        },
        {'Tag': 'ORDINAL', 'Description': '"first", "second", etc.'},
        {
            'Tag': 'CARDINAL',
            'Description': 'Numerals that do not fall under another type.',
        },
    ]

    from malaya.function import describe_availability

    return describe_availability(d, transpose = False)


def available_transformer():
    """
    List available transformer Entity Tagging models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 20% test set.'
    )


@check_type
def transformer(model: str = 'xlnet', quantized: bool = False, **kwargs):
    """
    Load Transformer Entity Tagging model, transfer learning Transformer + CRF.

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
    result : malaya.supervised.tag.transformer function
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.entity.available_transformer()`.'
        )
    return tag.transformer(
        PATH_ENTITIES,
        S3_PATH_ENTITIES,
        'entity',
        model = model,
        quantized = quantized,
        **kwargs
    )


def general_entity(model = None):
    """
    Load Regex based general entities tagging along with another supervised entity tagging model.

    Parameters
    ----------
    model : object
        model must has `predict` method. 
        Make sure the `predict` method returned [(string, label), (string, label)].

    Returns
    -------
    result: malaya.text.entity.ENTITY_REGEX class
    """
    if not hasattr(model, 'predict') and model is not None:
        raise ValueError('model must has `predict` method')
    return ENTITY_REGEX(model = model)
