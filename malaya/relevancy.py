from malaya.supervised import softmax
from malaya.path import PATH_RELEVANCY, S3_PATH_RELEVANCY
from herpetologist import check_type

label = ['not relevant', 'relevant']

_transformer_availability = {
    'bert': {'Size (MB)': 425.6, 'Accuracy': 0.872},
    'tiny-bert': {'Size (MB)': 57.4, 'Accuracy': 0.656},
    'albert': {'Size (MB)': 48.6, 'Accuracy': 0.871},
    'tiny-albert': {'Size (MB)': 22.4, 'Accuracy': 0.843},
    'xlnet': {'Size (MB)': 446.6, 'Accuracy': 0.885},
    'alxlnet': {'Size (MB)': 46.8, 'Accuracy': 0.874},
}


def available_transformer():
    """
    List available transformer relevancy analysis models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 20% test set.'
    )


@check_type
def transformer(model: str = 'xlnet', **kwargs):
    """
    Load Transformer emotion model.

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
            'model not supported, please check supported models from malaya.relevancy.available_transformer()'
        )
    return softmax.transformer(
        PATH_RELEVANCY,
        S3_PATH_RELEVANCY,
        'relevancy',
        label,
        model = model,
        **kwargs
    )
