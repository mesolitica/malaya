from ._utils import _softmax_class
from ._utils._paths import PATH_RELEVANCY, S3_PATH_RELEVANCY
from herpetologist import check_type

_availability = {'bert': ['base'], 'xlnet': ['base'], 'albert': ['base']}


def available_transformer_model():
    """
    List available transformer relevancy analysis models.
    """
    return _availability


@check_type
def transformer(
    model: str = 'xlnet', size: str = 'base', validate: bool = True
):
    """
    Load Transformer relevancy model.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - BERT architecture from google.
        * ``'xlnet'`` - XLNET architecture from google.
    size : str, optional (default='base')
        Model size supported. Allowed values:

        * ``'base'`` - BASE size.
        * ``'small'`` - SMALL size.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    MODEL : Transformer class
    """

    model = model.lower()
    size = size.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.relevancy.available_transformer_model()'
        )
    if size not in _availability[model]:
        raise Exception(
            'size not supported, please check supported models from malaya.relevancy.available_transformer_model()'
        )
    return _softmax_class.transformer(
        PATH_RELEVANCY,
        S3_PATH_RELEVANCY,
        'relevancy',
        ['negative', 'positive'],
        model = model,
        size = size,
        validate = validate,
    )
