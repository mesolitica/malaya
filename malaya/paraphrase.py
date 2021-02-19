from malaya.supervised import t5 as t5_load
from malaya.supervised import transformer as transformer_load
from malaya.model.t5 import Paraphrase as T5_Paraphrase
from malaya.model.tf import Paraphrase as TF_Paraphrase
from herpetologist import check_type


_transformer_availability = {
    't2t': {'Size (MB)': 832, 'Quantized Size (MB)': 279, 'BLEU': 0.59612},
    'small-t2t': {
        'Size (MB)': 379,
        'Quantized Size (MB)': 120,
        'BLEU': 0.65849,
    },
    't5': {'Size (MB)': 1250, 'Quantized Size (MB)': 481, 'BLEU': 0.86698},
    'small-t5': {
        'Size (MB)': 355.6,
        'Quantized Size (MB)': 195,
        'BLEU': 0.81801,
    },
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 1k paraphrase texts.'
    )


@check_type
def transformer(model: str = 't2t', quantized: bool = False, **kwargs):
    """
    Load Malaya transformer encoder-decoder model to generate a paraphrase given a string.

    Parameters
    ----------
    model : str, optional (default='t2t')
        Model architecture supported. Allowed values:

        * ``'t2t'`` - Malaya Transformer BASE parameters.
        * ``'small-t2t'`` - Malaya Transformer SMALL parameters.
        * ``'t5'`` - T5 BASE parameters.
        * ``'small-t5'`` - T5 SMALL parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:
        
        * if `t2t` in model, will return `malaya.model.tf.Paraphrase`.
        * if `t5` in model, will return `malaya.model.t5.Paraphrase`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.paraphrase.available_transformer()`.'
        )

    if 't2t' in model:
        return transformer_load.load_lm(
            module = 'paraphrase',
            model = model,
            model_class = TF_Paraphrase,
            quantized = quantized,
            **kwargs,
        )
    if 't5' in model:
        return t5_load.load(
            module = 'paraphrase',
            model = model,
            model_class = T5_Paraphrase,
            quantized = quantized,
            **kwargs,
        )
