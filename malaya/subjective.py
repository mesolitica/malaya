from ._utils import _softmax_class
from ._utils._paths import PATH_SUBJECTIVE, S3_PATH_SUBJECTIVE

_availability = {
    'bert': ['base', 'small'],
    'xlnet': ['base'],
    'albert': ['base'],
}


def available_transformer_model():
    """
    List available transformer subjective analysis models.
    """
    return _availability


def multinomial(validate = True):
    """
    Load multinomial subjectivity model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    BAYES : malaya._models._sklearn_model.BAYES class
    """
    return _softmax_class.multinomial(
        PATH_SUBJECTIVE,
        S3_PATH_SUBJECTIVE,
        'subjective',
        ['negative', 'positive'],
        validate = validate,
    )


def transformer(model = 'xlnet', size = 'base', validate = True):
    """
    Load Transformer emotion model.

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
            'model not supported, please check supported models from malaya.subjective.available_transformer_model()'
        )
    if size not in _availability[model]:
        raise Exception(
            'size not supported, please check supported models from malaya.subjective.available_transformer_model()'
        )
    return _softmax_class.bert(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'subjective',
        ['negative', 'positive'],
        model = model,
        validate = validate,
    )
