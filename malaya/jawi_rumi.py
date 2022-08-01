from malaya.supervised import transformer as load_transformer
from malaya.model.tf import JawiRumi
from malaya.function import describe_availability
from herpetologist import check_type
from typing import List
import logging

logger = logging.getLogger(__name__)

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.1,
        'CER': 0.004477071098945696,
        'WER': 0.013642122192393089,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 63.8,
        'CER': 0.0007639846016540444,
        'WER': 0.003042474270655385,
        'Suggested length': 256,
    },
}


def available_transformer():
    """
    List available transformer models.
    """

    logger.info('tested on first 10k Jawi-Rumi test set, dataset at https://huggingface.co/datasets/mesolitica/rumi-jawi')

    return describe_availability(_transformer_availability)


def transformer(model='base', quantized: bool = False, **kwargs):
    """
    Load transformer encoder-decoder model to convert jawi to rumi.

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
    result: malaya.model.tf.JawiRumi class
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.jawi_rumi.available_transformer()`.'
        )

    return load_transformer.load(
        module='jawi-rumi',
        model=model,
        encoder='yttm',
        model_class=JawiRumi,
        quantized=quantized,
        **kwargs,
    )
