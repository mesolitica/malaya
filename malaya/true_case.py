from malaya.model.tf import TRUE_CASE
from malaya.path import PATH_TRUE_CASE, S3_PATH_TRUE_CASE
from malaya.supervised import transformer as load_transformer
from herpetologist import check_type

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.1,
        'Sequence Accuracy': 0.347,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 63.8,
        'Sequence Accuracy': 0.696,
    },
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(_transformer_availability)


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load transformer encoder-decoder model to True Case.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'small'`` - Transformer SMALL parameters.
        * ``'base'`` - Transformer BASE parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.TRUE_CASE class
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya.true_case.available_transformer()`.'
        )

    return load_transformer.load(
        PATH_TRUE_CASE,
        S3_PATH_TRUE_CASE,
        model,
        'yttm',
        TRUE_CASE,
        quantized = quantized,
        **kwargs
    )
