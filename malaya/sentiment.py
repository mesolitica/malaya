from malaya.supervised import softmax
from malaya.path import PATH_SENTIMENT, S3_PATH_SENTIMENT
from herpetologist import check_type

label = ['negative', 'positive']

_availability = {
    'bert': ['425.6 MB', 'accuracy: 0.993'],
    'tiny-bert': ['57.4 MB', 'accuracy: 0.987'],
    'albert': ['48.6 MB', 'accuracy: 0.992'],
    'tiny-albert': ['22.4 MB', 'accuracy: 0.985'],
    'xlnet': ['446.5 MB', 'accuracy: 0.993'],
    'alxlnet': ['46.8 MB', 'accuracy: 0.991'],
}


def available_transformer():
    """
    List available transformer sentiment analysis models.
    """
    return _availability


def multinomial(**kwargs):
    """
    Load multinomial sentiment model.

    Returns
    -------
    result : malaya.model.ml.BAYES class
    """
    return softmax.multinomial(
        PATH_SENTIMENT, S3_PATH_SENTIMENT, 'sentiment', label, **kwargs
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
    result : Transformer class
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.sentiment.available_transformer()'
        )
    return softmax.transformer(
        PATH_SENTIMENT,
        S3_PATH_SENTIMENT,
        'sentiment',
        label,
        model = model,
        **kwargs
    )
