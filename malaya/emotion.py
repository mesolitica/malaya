from malaya.supervised import softmax
from malaya.path import PATH_EMOTION, S3_PATH_EMOTION
from herpetologist import check_type

label = ['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']
_availability = {
    'bert': ['425.6 MB', 'accuracy: 0.992'],
    'tiny-bert': ['57.4 MB', 'accuracy: 0.988'],
    'albert': ['48.6 MB', 'accuracy: 0.997'],
    'tiny-albert': ['22.4 MB', 'accuracy: 0.981'],
    'xlnet': ['446.5 MB', 'accuracy: 0.990'],
    'alxlnet': ['46.8 MB', 'accuracy: 0.989'],
}


def available_transformer():
    """
    List available transformer emotion analysis models.
    """
    return _availability


def multinomial(**kwargs):
    """
    Load multinomial emotion model.

    Returns
    -------
    result : malaya.model.ml.BAYES class
    """
    return softmax.multinomial(
        PATH_EMOTION, S3_PATH_EMOTION, 'emotion', label, **kwargs
    )


@check_type
def transformer(model: str = 'xlnet', **kwargs):
    """
    Load Transformer emotion model.

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
            'model not supported, please check supported models from malaya.emotion.available_transformer()'
        )
    return softmax.transformer(
        PATH_EMOTION, S3_PATH_EMOTION, 'emotion', label, model = model, **kwargs
    )
