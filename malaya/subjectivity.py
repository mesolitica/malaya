from malaya.supervised import softmax
from malaya.path import PATH_SUBJECTIVE, S3_PATH_SUBJECTIVE
from herpetologist import check_type

label = ['negative', 'positive']
_availability = {
    'bert': ['425.6 MB', 'accuracy: 0.916'],
    'tiny-bert': ['57.4 MB', 'accuracy: 0.903'],
    'albert': ['48.6 MB', 'accuracy: 0.903'],
    'tiny-albert': ['22.4 MB', 'accuracy: 0.894'],
    'xlnet': ['446.5 MB', 'accuracy: 0.917'],
    'alxlnet': ['46.8 MB', 'accuracy: 0.908'],
}


def available_transformer():
    """
    List available transformer subjective analysis models.
    """
    return _availability


def multinomial(**kwargs):
    """
    Load multinomial subjectivity model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    result : malaya.model.ml.BAYES class
    """
    return softmax.multinomial(
        PATH_SUBJECTIVE, S3_PATH_SUBJECTIVE, 'subjective', label, **kwargs
    )


@check_type
def transformer(model: str = 'xlnet', **kwargs):
    """
    Load Transformer subjectivity model.

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
            'model not supported, please check supported models from malaya.subjective.available_transformer()'
        )
    return softmax.transformer(
        PATH_SUBJECTIVE,
        S3_PATH_SUBJECTIVE,
        'subjective',
        label,
        model = model,
        **kwargs
    )
