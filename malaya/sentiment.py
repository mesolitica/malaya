from malaya.supervised import softmax
from malaya.path import PATH_SENTIMENT, S3_PATH_SENTIMENT
from herpetologist import check_type

_availability = {
    'bert': ['base', 'small'],
    'xlnet': ['base'],
    'albert': ['base'],
}


def available_transformer_model():
    """
    List available transformer sentiment analysis models.
    """
    return _availability


def multinomial(**kwargs):
    """
    Load multinomial sentiment model.

    Returns
    -------
    BAYES : malaya._models._sklearn_model.BAYES class
    """
    return _softmax_class.multinomial(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'sentiment',
        ['negative', 'positive'],
        **kwargs
    )


@check_type
def transformer(model: str = 'xlnet', size: str = 'base', **kwargs):
    """
    Load Transformer sentiment model.

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

    Returns
    -------
    MODEL : Transformer class
    """

    model = model.lower()
    size = size.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.sentiment.available_transformer_model()'
        )
    if size not in _availability[model]:
        raise Exception(
            'size not supported, please check supported models from malaya.sentiment.available_transformer_model()'
        )
    return _softmax_class.transformer(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'sentiment',
        ['negative', 'positive'],
        model = model,
        size = size,
        **kwargs
    )
