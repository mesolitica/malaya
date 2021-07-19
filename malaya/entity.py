from malaya.supervised import tag
from malaya.text.entity import EntityRegex
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
    'OTHER': 2,
    'ADDRESS': 3,
    'PERSON': 4,
    'NORP': 5,
    'FAC': 6,
    'ORG': 7,
    'GPE': 8,
    'LOC': 9,
    'PRODUCT': 10,
    'EVENT': 11,
    'WORK_OF_ART': 12,
    'LAW': 13,
    'LANGUAGE': 14,
    'DATE': 15,
    'TIME': 16,
    'PERCENT': 17,
    'MONEY': 18,
    'QUANTITY': 19,
    'ORDINAL': 20,
    'CARDINAL': 21,
}
_transformer_availability = {
    'bert': {
        'Size (MB)': 425.4,
        'Quantized Size (MB)': 111,
        'macro precision': 0.99291,
        'macro recall': 0.97864,
        'macro f1-score': 0.98537,
    },
    'tiny-bert': {
        'Size (MB)': 57.7,
        'Quantized Size (MB)': 15.4,
        'macro precision': 0.98151,
        'macro recall': 0.94754,
        'macro f1-score': 0.96134,
    },
    'albert': {
        'Size (MB)': 48.6,
        'Quantized Size (MB)': 12.8,
        'macro precision': 0.98026,
        'macro recall': 0.95332,
        'macro f1-score': 0.96492,
    },
    'tiny-albert': {
        'Size (MB)': 22.4,
        'Quantized Size (MB)': 5.98,
        'macro precision': 0.96100,
        'macro recall': 0.90363,
        'macro f1-score': 0.92374,
    },
    'xlnet': {
        'Size (MB)': 446.6,
        'Quantized Size (MB)': 118,
        'macro precision': 0.99344,
        'macro recall': 0.98154,
        'macro f1-score': 0.98725,
    },
    'alxlnet': {
        'Size (MB)': 46.8,
        'Quantized Size (MB)': 13.3,
        'macro precision': 0.99215,
        'macro recall': 0.97575,
        'macro f1-score': 0.98337,
    },
}
_transformer_ontonotes5_availability = {
    'bert': {
        'Size (MB)': 425.4,
        'Quantized Size (MB)': 111,
        'macro precision': 0.94460,
        'macro recall': 0.93244,
        'macro f1-score': 0.93822,
    },
    'tiny-bert': {
        'Size (MB)': 57.7,
        'Quantized Size (MB)': 15.4,
        'macro precision': 0.91908,
        'macro recall': 0.91635,
        'macro f1-score': 0.91704,
    },
    'albert': {
        'Size (MB)': 48.6,
        'Quantized Size (MB)': 12.8,
        'macro precision': 0.93010,
        'macro recall': 0.92341,
        'macro f1-score': 0.92636,
    },
    'tiny-albert': {
        'Size (MB)': 22.4,
        'Quantized Size (MB)': 5.98,
        'macro precision': 0.90298,
        'macro recall': 0.88251,
        'macro f1-score': 0.89145,
    },
    'xlnet': {
        'Size (MB)': 446.6,
        'Quantized Size (MB)': 118,
        'macro precision': 0.93814,
        'macro recall': 0.95021,
        'macro f1-score': 0.94388,
    },
    'alxlnet': {
        'Size (MB)': 46.8,
        'Quantized Size (MB)': 13.3,
        'macro precision': 0.93244,
        'macro recall': 0.92942,
        'macro f1-score': 0.93047,
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

    return describe_availability(d, transpose=False)


def describe_ontonotes5():
    """
    Describe OntoNotes5 Entities supported. https://spacy.io/api/annotation#named-entities
    """
    d = [
        {'Tag': 'OTHER', 'Description': 'other'},
        {'Tag': 'ADDRESS', 'Description': 'Address of physical location.'},
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

    return describe_availability(d, transpose=False)


def available_transformer():
    """
    List available transformer Entity Tagging models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 20% test set.'
    )


def available_transformer_ontonotes5():
    """
    List available transformer Entity Tagging models trained on Ontonotes 5 Bahasa.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_ontonotes5_availability, text='tested on 20% test set.'
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
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.model.bert.TaggingBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.TaggingXLNET`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.entity.available_transformer()`.'
        )
    return tag.transformer(
        module='entity', model=model, quantized=quantized, **kwargs
    )


@check_type
def transformer_ontonotes5(
    model: str = 'xlnet', quantized: bool = False, **kwargs
):
    """
    Load Transformer Entity Tagging model trained on Ontonotes 5 Bahasa, transfer learning Transformer + CRF.

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
    if model not in _transformer_ontonotes5_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.entity.available_transformer_ontonotes5()`.'
        )
    return tag.transformer(
        module='entity-ontonotes5',
        model=model,
        quantized=quantized,
        **kwargs
    )


def general_entity(model=None):
    """
    Load Regex based general entities tagging along with another supervised entity tagging model.

    Parameters
    ----------
    model : object
        model must have `predict` method.
        Make sure the `predict` method returned [(string, label), (string, label)].

    Returns
    -------
    result: malaya.text.entity.EntityRegex class
    """
    if not hasattr(model, 'predict') and model is not None:
        raise ValueError('model must have `predict` method')
    return EntityRegex(model=model)
