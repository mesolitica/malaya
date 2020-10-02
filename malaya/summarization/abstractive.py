from malaya.path import PATH_SUMMARIZE, S3_PATH_SUMMARIZE
from malaya.supervised import t5 as t5_load
from malaya.supervised import transformer as transformer_load
from herpetologist import check_type
import os

_t5_availability = {
    'small': {
        'Size (MB)': 122,
        'Uncompressed Size (MB)': 355.6,
        'ROUGE-1': 0.33854,
        'ROUGE-2': 0.14588,
        'ROUGE-L': 0.23528,
    },
    'base': {
        'Size (MB)': 448,
        'Uncompressed Size (MB)': 1300,
        'ROUGE-1': 0.34103,
        'ROUGE-2': 0.14994,
        'ROUGE-L': 0.23655,
    },
}

_transformer_availability = {
    'base': {
        'Size (MB)': 832,
        'ROUGE-1': 0.31863,
        'ROUGE-2': 0.12150,
        'ROUGE-L': 0.22023,
    },
    'small': {
        'Size (MB)': 379,
        'ROUGE-1': 0.32215,
        'ROUGE-2': 0.12741,
        'ROUGE-L': 0.23528,
    },
}


def available_t5():
    """
    List available T5 models.
    """

    from malaya.function import describe_availability

    return describe_availability(
        _t5_availability, text = 'tested on 5k CNN test set.'
    )


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 5k CNN test set.'
    )


@check_type
def t5(model: str = 'base', compressed: bool = True, **kwargs):

    """
    Load T5 model to generate a summary given a string.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'base'`` - T5 BASE parameters.
        * ``'small'`` - T5 SMALL parameters.

    compressed: bool, optional (default=True)
        Load compressed model, but this not able to utilize malaya-gpu function. 
        This only compressed model size, but when loaded into VRAM / RAM, size uncompressed and compressed are the same.
        We prefer un-compressed model due to compressed model prone to error.

    Returns
    -------
    result: malaya.model.t5.SUMMARIZATION class
    """

    model = model.lower()
    if model not in _t5_availability:
        raise ValueError(
            'model not supported, please check supported models from malaya.summarization.abstractive.available_t5()'
        )

    from malaya.model.t5 import SUMMARIZATION

    return t5_load.load(
        path = PATH_SUMMARIZE,
        s3_path = S3_PATH_SUMMARIZE,
        model = model,
        model_class = SUMMARIZATION,
        compressed = compressed,
        **kwargs,
    )


@check_type
def transformer(model: str = 'base', **kwargs):
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from malaya.summarization.abstractive.available_transformer()'
        )
    from malaya.model.tf import SUMMARIZATION

    return transformer_load.load_lm(
        path = PATH_SUMMARIZE['transformer'],
        s3_path = S3_PATH_SUMMARIZE['transformer'],
        model = model,
        model_class = SUMMARIZATION,
        **kwargs,
    )
