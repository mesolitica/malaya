from malaya.supervised import tag
from malaya.path import PATH_ENTITIES, S3_PATH_ENTITIES
from malaya.text.entity import _Entity_regex
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
_availability = {
    'bert': ['426.4 MB', 'accuracy: 0.994'],
    'tiny-bert': ['57.7 MB', 'accuracy: 0.986'],
    'albert': ['48.6 MB', 'accuracy: 0.984'],
    'tiny-albert': ['22.4 MB', 'accuracy: 0.971'],
    'xlnet': ['446.6 MB', 'accuracy: 0.992'],
    'alxlnet': ['46.8 MB', 'accuracy: 0.993'],
}


def describe():
    """
    Describe Entities supported
    """
    print('OTHER - Other')
    print('law - law, regulation, related law documents, documents, etc')
    print('location - location, place')
    print('organization - organization, company, government, facilities, etc')
    print(
        'person - person, group of people, believes, unique arts (eg; food, drink), etc'
    )
    print('quantity - numbers, quantity')
    print('time - date, day, time, etc')
    print('event - unique event happened, etc')


def available_transformer():
    """
    List available transformer Entity Tagging models.
    """
    return _availability


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
    result : Transformer class
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
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
        model must has `predict` method. Make sure the `predict` method returned [(string, label), (string, label)].

    Returns
    -------
    result: malaya.text.entity._Entity_regex class
    """
    if not hasattr(model, 'predict') and model is not None:
        raise ValueError('model must has `predict` method')
    return _Entity_regex(model = model)
