from malaya.supervised import t5 as t5_load
from malaya.supervised import transformer as transformer_load
from malaya.supervised import bigbird as bigbird_load
from malaya.model.tf import Summarization as TF_Summarization
from malaya.model.t5 import Summarization as T5_Summarization
from malaya.model.bigbird import Summarization as BigBird_Summarization
from herpetologist import check_type
import os

_transformer_availability = {
    't2t': {
        'Size (MB)': 832,
        'Quantized Size (MB)': 279,
        'ROUGE-1': 0.33209,
        'ROUGE-2': 0.13622,
        'ROUGE-L': 0.23348,
        'Suggested length': 768,
    },
    'small-t2t': {
        'Size (MB)': 379,
        'Quantized Size (MB)': 120,
        'ROUGE-1': 0.33,
        'ROUGE-2': 0.13417,
        'ROUGE-L': 0.23059,
        'Suggested length': 768,
    },
    't5': {
        'Size (MB)': 1250,
        'Quantized Size (MB)': 481,
        'ROUGE-1': 0.34103,
        'ROUGE-2': 0.14994,
        'ROUGE-L': 0.23655,
        'Suggested length': 1024,
    },
    'small-t5': {
        'Size (MB)': 355.6,
        'Quantized Size (MB)': 195,
        'ROUGE-1': 0.33854,
        'ROUGE-2': 0.14588,
        'ROUGE-L': 0.23528,
        'Suggested length': 1024,
    },
    'bigbird': {
        'Size (MB)': 910,
        'Quantized Size (MB)': 230,
        'ROUGE-1': 0.3258,
        'ROUGE-2': 0.13534,
        'ROUGE-L': 0.2228,
        'Suggested length': 2048,
    },
    'small-bigbird': {
        'Size (MB)': 303.0,
        'Quantized Size (MB)': 77.3,
        'ROUGE-1': 0.3219,
        'ROUGE-2': 0.1338,
        'ROUGE-L': 0.2198,
        'Suggested length': 2048,
    },
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 5k CNN test set.'
    )


@check_type
def transformer(model: str = 't2t', quantized: bool = False, **kwargs):

    """
    Load Malaya transformer encoder-decoder model to generate a summary given a string.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'t2t'`` - Malaya Transformer BASE parameters.
        * ``'small-t2t'`` - Malaya Transformer SMALL parameters.
        * ``'t5'`` - T5 BASE parameters.
        * ``'small-t5'`` - T5 SMALL parameters.
        * ``'bigbird'`` - BigBird + Pegasus BASE parameters.
        * ``'small-bigbird'`` - BigBird + Pegasus SMALL parameters.
    
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:
        
        * if `t2t` in model, will return `malaya.model.tf.Summarization`.
        * if `t5` in model, will return `malaya.model.t5.Summarization`.
        * if `bigbird` in model, will return `malaya.model.bigbird.Summarization`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.summarization.abstractive.available_transformer()`.'
        )

    if 't2t' in model:
        return transformer_load.load_lm(
            module = 'abstractive-summarization',
            model = model,
            model_class = TF_Summarization,
            quantized = quantized,
            **kwargs,
        )

    if 't5' in model:
        return t5_load.load(
            module = 'abstractive-summarization',
            model = model,
            model_class = T5_Summarization,
            quantized = quantized,
            **kwargs,
        )

    if 'bigbird' in model:
        return bigbird_load.load_lm(
            module = 'abstractive-summarization',
            model = model,
            model_class = BigBird_Summarization,
            maxlen = _transformer_availability[model]['Suggested length'],
            quantized = quantized,
            **kwargs,
        )
