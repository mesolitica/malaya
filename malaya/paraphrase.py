from malaya.supervised import t5 as t5_load
from malaya.supervised import transformer as transformer_load
from malaya.model.t5 import Paraphrase as T5_Paraphrase
from malaya.model.tf import Paraphrase as TF_Paraphrase
from herpetologist import check_type


_transformer_availability = {
    't5': {'Size (MB)': 1250, 'Quantized Size (MB)': 481, 'BLEU': 0.60890377},
    'small-t5': {
        'Size (MB)': 355.6,
        'Quantized Size (MB)': 195,
        'BLEU': 0.6174561,
    },
    'tiny-t5': {
        'Size (MB)': 208,
        'Quantized Size (MB)': 103,
        'BLEU': 0.46032128,
    },
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on ParaSCI test set.'
    )


@ check_type
def transformer(model: str = 'small-t5', quantized: bool = False, **kwargs):
    """
    Load Malaya transformer encoder-decoder model to generate a paraphrase given a string.

    Parameters
    ----------
    model : str, optional (default='small-t5')
        Model architecture supported. Allowed values:

        * ``'t5'`` - T5 BASE parameters.
        * ``'small-t5'`` - T5 SMALL parameters.
        * ``'tiny-t5'`` - T5 TINY parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `t5` in model, will return `malaya.model.t5.Paraphrase`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.paraphrase.available_transformer()`.'
        )
    return t5_load.load(
        module='paraphrase-v2',
        model=model,
        model_class=T5_Paraphrase,
        quantized=quantized,
        **kwargs,
    )
