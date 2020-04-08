from malaya.supervised import softmax
from malaya.path import PATH_SENTIMENT, S3_PATH_SENTIMENT
from herpetologist import check_type

_availability = [
    'bert',
    'tiny-bert',
    'albert',
    'tiny-albert',
    'xlnet',
    'alxlnet',
]


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
    BAYES : malaya.model.ml.BAYES class
    """
    return softmax.multinomial(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'sentiment',
        ['negative', 'positive'],
        **kwargs
    )


@check_type
def transformer(model: str = 'bert', **kwargs):
    """
    Load Transformer sentiment model.

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
    MODEL : Transformer class
    """

    model = model.lower()
    size = size.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.sentiment.available_transformer_model()'
        )
    return softmax.transformer(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'sentiment',
        ['negative', 'positive'],
        model = model,
        **kwargs
    )
