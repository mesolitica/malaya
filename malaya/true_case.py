from malaya.model.tf import TrueCase
from malaya.supervised import transformer as load_transformer
from malaya.supervised import t5 as t5_load
from malaya.model.t5 import TrueCase as T5_TrueCase
from herpetologist import check_type

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.1,
        'CER': 0.0246012,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 63.8,
        'CER': 0.0146193,
        'Suggested length': 256,
    },
    'super-tiny-t5': {
        'Size (MB)': 81.8,
        'Quantized Size (MB)': 27.1,
        'CER': 0.0254679,
        'Suggested length': 256,
    },
    'super-super-tiny-t5': {
        'Size (MB)': 39.6,
        'Quantized Size (MB)': 12,
        'CER': 0.02533658,
        'Suggested length': 256,
    }
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
        * ``'super-tiny-t5'`` - T5 SUPER TINY parameters.
        * ``'super-super-tiny-t5'`` - T5 SUPER SUPER TINY parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.TrueCase class
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.true_case.available_transformer()`.'
        )

    if 't5' in model:
        return t5_load.load(
            module='true-case',
            model=model,
            model_class=T5_TrueCase,
            quantized=quantized,
            **kwargs,
        )
    else:
        return load_transformer.load(
            module='true-case',
            model=model,
            encoder='yttm',
            model_class=TrueCase,
            quantized=quantized,
            **kwargs,
        )
