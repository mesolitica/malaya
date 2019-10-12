from ._utils import _tag_class
from ._utils._paths import PATH_ENTITIES, S3_PATH_ENTITIES
from .texts._entity import _Entity_regex


_availability = {
    'bert': ['base', 'small'],
    'xlnet': ['base'],
    'albert': ['base'],
}


def available_transformer_model():
    """
    List available transformer Entity Tagging models.
    """
    return _availability


def transformer(model = 'xlnet', size = 'base', validate = True):
    """
    Load Transformer Entity Tagging model, transfer learning Transformer + CRF.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - BERT architecture from google.
        * ``'xlnet'`` - XLNET architecture from google.
        * ``'albert'`` - ALBERT architecture from google.
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
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.entity.available_transformer_model()'
        )
    if size not in _availability[model]:
        raise Exception(
            'size not supported, please check supported models from malaya.entity.available_transformer_model()'
        )
    return _softmax_class.bert(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'entity',
        ['negative', 'positive'],
        model = model,
        validate = validate,
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
    _Entity_regex: malaya.texts._entity._Entity_regex class
    """
    if not hasattr(model, 'predict') and model is not None:
        raise ValueError('model must has `predict` method')
    return _Entity_regex(model = model)
