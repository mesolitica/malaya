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
_transformer_availability = {
    'bert': {'Size (MB)': 425.4, 'Accuracy': 0.994},
    'tiny-bert': {'Size (MB)': 57.7, 'Accuracy': 0.986},
    'albert': {'Size (MB)': 48.6, 'Accuracy': 0.984},
    'tiny-albert': {'Size (MB)': 22.4, 'Accuracy': 0.971},
    'xlnet': {'Size (MB)': 446.6, 'Accuracy': 0.992},
    'alxlnet': {'Size (MB)': 46.8, 'Accuracy': 0.993},
}


def describe():
    """
    Describe Entities supported
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


def available_transformer():
    """
    List available transformer Entity Tagging models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 20% test set.'
    )


@check_type
def transformer(model: str = 'xlnet', **kwargs):
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

    Returns
    -------
    result : malaya.supervised.tag.transformer function
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from malaya.entity.available_transformer()'
        )
    return tag.transformer(
        PATH_ENTITIES, S3_PATH_ENTITIES, 'entity', model = model, **kwargs
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
