from malaya.supervised import t2t
from malaya.supervised.settings import jawi_left, jawi_right
from malaya.text.function import rumi_jawi_textcleaning
from herpetologist import check_type
from malaya.model.tf import RumiJawi
from malaya.function import describe_availability
from typing import List
import logging

logger = logging.getLogger(__name__)

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.1,
        'CER': 0.0246012,
        'WER': 0,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 63.8,
        'CER': 0.0146193,
        'WER': 0,
        'Suggested length': 256,
    },
}


def available_transformer():
    """
    List available transformer models.
    """
    logger.info('tested on first 10k Rumi-Jawi test set, dataset at https://huggingface.co/datasets/mesolitica/rumi-jawi')

    return describe_availability(_transformer_availability)


def transformer(model='base', quantized: bool = False, **kwargs):
    """
    Load transformer encoder-decoder model to convert rumi to jawi.

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
    result: malaya.model.tf.RumiJawi class
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.rumi_jawi.available_transformer()`.'
        )

    return load_transformer.load(
        module='rumi-jawi',
        model=model,
        encoder='yttm',
        model_class=RumiJawi,
        quantized=quantized,
        **kwargs,
    )
