from ._utils import _softmax_class
from ._utils._paths import PATH_RELEVANCY, S3_PATH_RELEVANCY

_availability = {'bert': ['base'], 'xlnet': ['base'], 'albert': ['base']}


def available_transformer_model():
    """
    List available transformer relevancy analysis models.
    """
    return _availability


def transformer(model = 'xlnet', size = 'base', validate = True):
    """
    Load Transformer emotion model.

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
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.relevancy.available_transformer_model()'
        )
    if size not in _availability[model]:
        raise Exception(
            'size not supported, please check supported models from malaya.relevancy.available_transformer_model()'
        )
    return _softmax_class.bert(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'relevancy',
        ['negative', 'positive'],
        model = model,
        validate = validate,
    )
