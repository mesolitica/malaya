from malaya.supervised import softmax
from malaya.path import PATH_SUBJECTIVE, S3_PATH_SUBJECTIVE
from herpetologist import check_type

label = ['negative', 'positive']

_transformer_availability = {
    'bert': {'Size (MB)': 425.6, 'Accuracy': 0.916},
    'tiny-bert': {'Size (MB)': 57.4, 'Accuracy': 0.903},
    'albert': {'Size (MB)': 48.6, 'Accuracy': 0.903},
    'tiny-albert': {'Size (MB)': 22.4, 'Accuracy': 0.894},
    'xlnet': {'Size (MB)': 446.6, 'Accuracy': 0.917},
    'alxlnet': {'Size (MB)': 46.8, 'Accuracy': 0.908},
}


def available_transformer():
    """
    List available transformer subjective analysis models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 20% test set.'
    )


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
