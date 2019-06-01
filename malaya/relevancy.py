import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from ._utils import _softmax_class
from ._utils._paths import PATH_RELEVANCY, S3_PATH_RELEVANCY


def available_deep_model():
    """
    List available deep learning relevancy analysis models.
    """
    return ['self-attention', 'dilated-cnn']


def deep_model(model = 'luong', validate = True):
    """
    Load deep learning relevancy analysis model.

    Parameters
    ----------
    model : str, optional (default='luong')
        Model architecture supported. Allowed values:

        * ``'self-attention'`` - Fast-text architecture, embedded and logits layers only with self attention.
        * ``'dilated-cnn'`` - Stack dilated CNN with self attention.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    SOFTMAX: malaya._models._tensorflow_model.SOFTMAX class
    """
    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')
    model = model.lower()
    if model not in available_deep_model():
        raise Exception(
            'model is not supported, please check supported models from malaya.relevancy.available_deep_model()'
        )

    return _softmax_class.deep_model(
        PATH_RELEVANCY,
        S3_PATH_RELEVANCY,
        'relevancy',
        ['positive', 'negative'],
        model = model,
        validate = validate,
    )
