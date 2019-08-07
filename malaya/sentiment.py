from ._utils import _softmax_class
from ._utils._paths import PATH_SENTIMENT, S3_PATH_SENTIMENT


def available_deep_model():
    """
    List available deep learning sentiment analysis models.
    """
    return ['self-attention', 'bahdanau', 'luong']


def available_bert_model():
    """
    List available bert sentiment analysis models.
    """
    return ['multilanguage', 'base', 'small']


def deep_model(model = 'luong', validate = True):
    """
    Load deep learning sentiment analysis model.

    Parameters
    ----------
    model : str, optional (default='luong')
        Model architecture supported. Allowed values:

        * ``'self-attention'`` - Fast-text architecture, embedded and logits layers only with self attention.
        * ``'bahdanau'`` - LSTM with bahdanau attention architecture.
        * ``'luong'`` - LSTM with luong attention architecture.
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
            'model is not supported, please check supported models from malaya.sentiment.available_deep_model()'
        )

    return _softmax_class.deep_model(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'sentiment',
        ['negative', 'positive'],
        model = model,
        validate = validate,
    )


def multinomial(validate = True):
    """
    Load multinomial sentiment model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    BAYES : malaya._models._sklearn_model.BAYES class
    """
    return _softmax_class.multinomial(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'sentiment',
        ['negative', 'positive'],
        validate = validate,
    )


def xgb(validate = True):
    """
    Load XGB sentiment model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    XGB : malaya._models._sklearn_model.XGB class
    """
    return _softmax_class.xgb(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'sentiment',
        ['negative', 'positive'],
        validate = validate,
    )


def bert(model = 'base', validate = True):
    """
    Load BERT sentiment model.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'multilanguage'`` - bert multilanguage released by Google, trained on sentiment analysis.
        * ``'base'`` - base bert-bahasa released by Malaya, trained on sentiment analysis.
        * ``'small'`` - small bert-bahasa released by Malaya, trained on sentiment analysis.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    BERT : malaya._models._bert_model.BINARY_BERT class
    """
    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

    model = model.lower()
    if model not in available_bert_model():
        raise Exception(
            'model is not supported, please check supported models from malaya.sentiment.available_bert_model()'
        )
    return _softmax_class.bert(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'sentiment',
        ['negative', 'positive'],
        model = model,
        validate = validate,
    )
