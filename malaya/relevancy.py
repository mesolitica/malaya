from malaya.supervised import softmax
from malaya.path import PATH_RELEVANCY, S3_PATH_RELEVANCY
from herpetologist import check_type

label = ['not relevant', 'relevant']

_availability = {
    'bert': ['425.6 MB', 'accuracy: 0.872'],
    'tiny-bert': ['57.4 MB', 'accuracy: 0.656'],
    'albert': ['48.6 MB', 'accuracy: 0.871'],
    'tiny-albert': ['22.4 MB', 'accuracy: 0.843'],
    'xlnet': ['446.5 MB', 'accuracy: 0.885'],
    'alxlnet': ['46.8 MB', 'accuracy: 0.874'],
}


def available_transformer():
    """
    List available transformer relevancy analysis models.
    """
    return _availability


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
    result : Transformer class
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.relevancy.available_transformer()()'
        )
    return softmax.transformer(
        PATH_RELEVANCY,
        S3_PATH_RELEVANCY,
        'relevancy',
        label,
        model = model,
        **kwargs
    )
