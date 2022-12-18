from malaya.supervised import t5 as t5_load
from malaya.model.t5 import Generator
from malaya.function import describe_availability
from malaya.supervised import huggingface as load_huggingface
import logging
import warnings

logger = logging.getLogger(__name__)

_transformer_availability = {
    'small-t5': {
        'Size (MB)': 355.6,
        'ROUGE-1': 0.3717403,
        'ROUGE-2': 0.18471429,
        'Maximum Length': 1024,
    },
    't5': {
        'Size (MB)': 1250,
        'ROUGE-1': 0.3717403,
        'ROUGE-2': 0.18471429,
        'Maximum Length': 1024,
    },
}

_huggingface_availability = {
    'mesolitica/finetune-isi-penting-generator-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'ROUGE-1': 0.24620333,
        'ROUGE-2': 0.05896076,
        'ROUGE-L': 0.15158954,
        'Suggested length': 1024,
    },
    'mesolitica/finetune-isi-penting-generator-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'ROUGE-1': 0.24620333,
        'ROUGE-2': 0.05896076,
        'ROUGE-L': 0.15158954,
        'Suggested length': 1024,
    },
}


def _describe():
    logger.info('tested on semisupervised summarization on unseen AstroAwani 20 news, https://github.com/huseinzol05/malay-dataset/tree/master/summarization/semisupervised-astroawani')
    logger.info('each news compared ROUGE with 5 different generated texts.')


def available_transformer():
    """
    List available transformer models.
    """

    warnings.warn(
        '`malaya.generator.isi_penting.available_transformer` is deprecated, use `malaya.generator.isi_penting.available_huggingface` instead', DeprecationWarning)

    _describe()
    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available huggingface models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


def transformer(model: str = 't5', quantized: bool = False, **kwargs):
    """
    Load Transformer model to generate a string given a isu penting.

    Parameters
    ----------
    model: str, optional (default='base')
        Check available models at `malaya.generator.isi_penting.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.t5.Generator class
    """

    warnings.warn(
        '`malaya.generator.isi_penting.transformer` is deprecated, use `malaya.generator.isi_penting.huggingface` instead', DeprecationWarning)

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.generator.isi_penting.available_transformer()`.'
        )

    return t5_load.load(
        module='generator',
        model=model,
        model_class=Generator,
        quantized=quantized,
        **kwargs,
    )


def huggingface(
    model: str = 'mesolitica/finetune-isi-penting-generator-t5-base-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to generate text based on isi penting.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-isi-penting-generator-t5-base-standard-bahasa-cased')
        Check available models at `malaya.generator.isi_penting.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.IsiPentingGenerator
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.generator.isi_penting.available_huggingface()`.'
        )
    return load_huggingface.load_isi_penting(model=model, **kwargs)
