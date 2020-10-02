from malaya.supervised import softmax
from malaya.path import PATH_SENTIMENT, S3_PATH_SENTIMENT
from herpetologist import check_type

label = ['negative', 'positive']

_transformer_availability = {
    'bert': {'Size (MB)': 425.6, 'Accuracy': 0.993},
    'tiny-bert': {'Size (MB)': 57.4, 'Accuracy': 0.987},
    'albert': {'Size (MB)': 48.6, 'Accuracy': 0.992},
    'tiny-albert': {'Size (MB)': 22.4, 'Accuracy': 0.985},
    'xlnet': {'Size (MB)': 446.6, 'Accuracy': 0.993},
    'alxlnet': {'Size (MB)': 46.8, 'Accuracy': 0.991},
}


def available_transformer():
    """
    List available transformer sentiment analysis models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 20% test set.'
    )


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

        * ``'bert'`` - Google BERT BASE parameters.
        * ``'tiny-bert'`` - Google BERT TINY parameters.
        * ``'albert'`` - Google ALBERT BASE parameters.
        * ``'tiny-albert'`` - Google ALBERT TINY parameters.
        * ``'xlnet'`` - Google XLNET BASE parameters.
        * ``'alxlnet'`` - Malaya ALXLNET BASE parameters.

    Returns
    -------
    result : malaya.supervised.softmax.transformer function
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
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
