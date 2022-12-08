from malaya.supervised import t5 as t5_load
from malaya.model.t5 import Paraphrase as T5_Paraphrase
from malaya.supervised import huggingface as load_huggingface
from herpetologist import check_type
from malaya.function import describe_availability
import logging
import warnings

logger = logging.getLogger(__name__)

_transformer_availability = {
    't5': {
        'Size (MB)': 1250,
        'Quantized Size (MB)': 481,
        'BLEU': 35.86211127195363,
        'SacreBLEU Verbose': '61.1/40.6/31.3/25.2 (BP = 0.959 ratio = 0.960 hyp_len = 96986 ref_len = 101064)',
        'Suggested length': 256,
    },
    'small-t5': {
        'Size (MB)': 355.6,
        'Quantized Size (MB)': 195,
        'BLEU': 37.24731076156855,
        'SacreBLEU Verbose': '61.6/41.7/32.5/26.3 (BP = 0.968 ratio = 0.968 hyp_len = 97840 ref_len = 101064)',
        'Suggested length': 256,
    },
    'tiny-t5': {
        'Size (MB)': 208,
        'Quantized Size (MB)': 103,
        'BLEU': 13.253918978157994,
        'SacreBLEU Verbose': '38.6/16.3/9.4/5.8 (BP = 0.973 ratio = 0.974 hyp_len = 98419 ref_len = 101064)',
        'Suggested length': 256,
    },
}

_huggingface_availability = {
    'mesolitica/finetune-paraphrase-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 36.92696648298233,
        'SacreBLEU Verbose': '62.5/42.3/33.0/26.9 (BP = 0.943 ratio = 0.945 hyp_len = 95496 ref_len = 101064)',
        'Suggested length': 256,
    },
    'mesolitica/finetune-paraphrase-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 37.598729045833316,
        'SacreBLEU Verbose': '62.6/42.5/33.2/27.0 (BP = 0.957 ratio = 0.958 hyp_len = 96781 ref_len = 101064)',
        'Suggested length': 256,
    },
    'mesolitica/finetune-paraphrase-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 35.95965899952292,
        'SacreBLEU Verbose': '61.7/41.3/32.0/25.8 (BP = 0.944 ratio = 0.946 hyp_len = 95593 ref_len = 101064)',
        'Suggested length': 256,
    },
}


def _describe():
    logger.info('tested on MRPC validation set, https://huggingface.co/datasets/mesolitica/translated-MRPC')
    logger.info('tested on ParaSCI ARXIV test set, https://huggingface.co/datasets/mesolitica/translated-paraSCI')


def available_transformer():
    """
    List available transformer models.
    """

    warnings.warn(
        '`malaya.paraphrase.available_transformer` is deprecated, use `malaya.paraphrase.available_huggingface` instead', DeprecationWarning)

    _describe()
    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available huggingface models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


@check_type
def transformer(model: str = 'small-t5', quantized: bool = False, **kwargs):
    """
    Load Malaya transformer encoder-decoder model to paraphrase.

    Parameters
    ----------
    model: str, optional (default='small-t5')
        Check available models at `malaya.paraphrase.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `t5` in model, will return `malaya.model.t5.Paraphrase`.
    """
    warnings.warn(
        '`malaya.paraphrase.transformer` is deprecated, use `malaya.paraphrase.huggingface` instead', DeprecationWarning)

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


def huggingface(
    model: str = 'mesolitica/finetune-paraphrase-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to paraphrase.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-paraphrase-t5-small-standard-bahasa-cased')
        Check available models at `malaya.paraphrase.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Paraphrase
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.paraphrase.available_huggingface()`.'
        )
    return load_huggingface.load_paraphrase(model=model, initial_text='parafrasa: ', **kwargs)
