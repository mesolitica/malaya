from malaya.supervised import softmax
from malaya.path import PATH_EMOTION, S3_PATH_EMOTION
from herpetologist import check_type

_emotion_label = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
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
    List available transformer emotion analysis models.
    """
    return _availability


def multinomial(**kwargs):
    """
    Load multinomial emotion model.

    Returns
    -------
    BAYES : malaya._models._sklearn_model.BAYES class
    """
    return softmax.multinomial(
        PATH_EMOTION, S3_PATH_EMOTION, 'emotion', _emotion_label, **kwargs
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
    MODEL : Transformer class
    """

    model = model.lower()
    size = size.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.emotion.available_transformer_model()'
        )
    return softmax.transformer(
        PATH_EMOTION,
        S3_PATH_EMOTION,
        'emotion',
        _emotion_label,
        model = model,
        size = size,
        validate = validate,
    )
