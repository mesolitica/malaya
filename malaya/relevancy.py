from ._utils import _softmax_class
from ._utils._paths import PATH_RELEVANCY, S3_PATH_RELEVANCY


def available_deep_model():
    """
    List available deep learning relevancy analysis models.
    """
    return ['self-attention', 'dilated-cnn']


def available_bert_model():
    """
    List available bert relevancy analysis models.
    """
    return ['multilanguage', 'base']


def deep_model(model = 'self-attention', validate = True):
    """
    Load deep learning relevancy analysis model.

    Parameters
    ----------
    model : str, optional (default='luong')
        Model architecture supported. Allowed values:

        * ``'self-attention'`` - Fast-text architecture, embedded and logits layers only with self attention.
        * ``'dilated-cnn'`` - Stack dilated CNN with self attention.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    SOFTMAX: malaya._models._tensorflow_model.SOFTMAX class
    """
    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')
    model = model.lower()
    if model not in available_deep_model():
        raise Exception(
            'model is not supported, please check supported models from malaya.relevancy.available_deep_model()'
        )

    return _softmax_class.deep_model(
        PATH_RELEVANCY,
        S3_PATH_RELEVANCY,
        'relevancy',
        ['positive', 'negative'],
        model = model,
        validate = validate,
    )


def bert(model = 'base', validate = True):
    """
    Load BERT relevancy model.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'multilanguage'`` - bert multilanguage released by Google, trained on relevancy analysis.
        * ``'base'`` - base bert-bahasa released by Malaya, trained on relevancy analysis.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    BERT : malaya._models._bert_model.MULTICLASS_BERT class
    """
    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

    model = model.lower()
    if model not in available_bert_model():
        raise Exception(
            'model is not supported, please check supported models from malaya.relevancy.available_bert_model()'
        )
    return _softmax_class.bert(
        PATH_RELEVANCY,
        S3_PATH_RELEVANCY,
        'relevancy',
        ['positive', 'negative'],
        model = model,
        validate = validate,
    )
