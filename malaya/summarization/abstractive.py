from malaya.supervised import t5 as t5_load
from malaya.supervised import transformer as transformer_load
from malaya.supervised import bigbird as bigbird_load
from malaya.supervised import pegasus as pegasus_load
from malaya.model.tf import Summarization as TF_Summarization
from malaya.model.t5 import Summarization as T5_Summarization
from malaya.model.bigbird import Summarization as BigBird_Summarization
from malaya.model.pegasus import Summarization as Pegasus_Summarization
from herpetologist import check_type
import os

_transformer_availability = {
    't5': {
        'Size (MB)': 1250,
        'Quantized Size (MB)': 481,
        'ROUGE-1': 0.3717403,
        'ROUGE-2': 0.18471429,
        'ROUGE-L': 0.2582724,
        'Suggested length': 512,
    },
    'small-t5': {
        'Size (MB)': 355.6,
        'Quantized Size (MB)': 195,
        'ROUGE-1': 0.36696956,
        'ROUGE-2': 0.1773298,
        'ROUGE-L': 0.2546696,
        'Suggested length': 512,
    },
    'tiny-t5': {
        'Size (MB)': 208,
        'Quantized Size (MB)': 103,
        'ROUGE-1': 0.302676,
        'ROUGE-2': 0.11932098,
        'ROUGE-L': 0.20291817,
        'Suggested length': 512,
    },
    'pegasus': {
        'Size (MB)': 894,
        'Quantized Size (MB)': 225,
        'ROUGE-1': 0.25109342,
        'ROUGE-2': 0.06678945,
        'ROUGE-L': 0.15590666,
        'Suggested length': 512,
    },
    'small-pegasus': {
        'Size (MB)': 293,
        'Quantized Size (MB)': 74.2,
        'ROUGE-1': 0.29012334,
        'ROUGE-2': 0.11878814,
        'ROUGE-L': 0.19232224,
        'Suggested length': 512,
    },
    'bigbird': {
        'Size (MB)': 910,
        'Quantized Size (MB)': 230,
        'ROUGE-1': 0.2673456,
        'ROUGE-2': 0.07239062,
        'ROUGE-L': 0.16132586,
        'Suggested length': 1536,
    },
    'small-bigbird': {
        'Size (MB)': 303.0,
        'Quantized Size (MB)': 77.3,
        'ROUGE-1': 0.24620333,
        'ROUGE-2': 0.05896076,
        'ROUGE-L': 0.15158954,
        'Suggested length': 1536,
    },
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 12k CNN + DailyNews test set.'
    )


@check_type
def transformer(model: str = 'small-t5', quantized: bool = False, **kwargs):
    """
    Load Malaya transformer encoder-decoder model to generate a summary given a string.

    Parameters
    ----------
    model : str, optional (default='small-t5')
        Model architecture supported. Allowed values:

        * ``'t5'`` - T5 BASE parameters.
        * ``'small-t5'`` - T5 SMALL parameters.
        * ``'tiny-t5'`` - T5 TINY parameters.
        * ``'pegasus'`` - Pegasus BASE parameters.
        * ``'small-pegasus'`` - Pegasus SMALL parameters.
        * ``'bigbird'`` - BigBird + Pegasus BASE parameters.
        * ``'small-bigbird'`` - BigBird + Pegasus SMALL parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `t5` in model, will return `malaya.model.t5.Summarization`.
        * if `bigbird` in model, will return `malaya.model.bigbird.Summarization`.
        * if `pegasus` in model, will return `malaya.model.pegasus.Summarization`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.summarization.abstractive.available_transformer()`.'
        )

    if 't5' in model:
        return t5_load.load(
            module='abstractive-summarization-v2',
            model=model,
            model_class=T5_Summarization,
            quantized=quantized,
            **kwargs,
        )

    if 'bigbird' in model:
        return bigbird_load.load_pegasus(
            module='abstractive-summarization-v2',
            model=model,
            model_class=BigBird_Summarization,
            maxlen=_transformer_availability[model]['Suggested length'],
            quantized=quantized,
            **kwargs,
        )

    if 'pegasus' in model:
        return pegasus_load.load(
            module='abstractive-summarization-v2',
            model=model,
            model_class=Pegasus_Summarization,
            quantized=quantized,
            **kwargs,
        )
