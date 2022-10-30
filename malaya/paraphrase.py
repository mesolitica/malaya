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
        'BLEU': 63.43349716445256,
        'SacreBLEU Verbose': '86.1/69.9/58.4/49.2 (BP = 0.983 ratio = 0.984 hyp_len = 138679 ref_len = 141004)',
        'Suggested length': 256,
    },
    'small-t5': {
        'Size (MB)': 355.6,
        'Quantized Size (MB)': 195,
        'BLEU': 61.47064047142362,
        'SacreBLEU Verbose': '85.0/67.7/55.3/45.6 (BP = 0.996 ratio = 0.996 hyp_len = 140439 ref_len = 141004)',
        'Suggested length': 256,
    },
    'tiny-t5': {
        'Size (MB)': 208,
        'Quantized Size (MB)': 103,
        'BLEU': 44.21090693563815,
        'SacreBLEU Verbose': '69.3/49.2/38.0/30.2 (BP = 0.994 ratio = 0.994 hyp_len = 140115 ref_len = 141004)',
        'Suggested length': 256,
    },
}

_huggingface_availability = {
    'mesolitica/finetune-paraphrase-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 61.06784273649806,
        'SacreBLEU Verbose': '86.1/68.4/55.8/45.9 (BP = 0.980 ratio = 0.980 hyp_len = 138209 ref_len = 141004)',
        'Suggested length': 256,
    },
    'mesolitica/finetune-paraphrase-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 61.559202822392486,
        'SacreBLEU Verbose': '86.0/68.4/56.1/46.3 (BP = 0.984 ratio = 0.984 hyp_len = 138806 ref_len = 141004)',
        'Suggested length': 256,
    },
    'mesolitica/finetune-paraphrase-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 58.764876478744064,
        'SacreBLEU Verbose': '84.5/65.8/53.0/43.1 (BP = 0.984 ratio = 0.985 hyp_len = 138828 ref_len = 141004)',
        'Suggested length': 256,
    },
}


def _describe():
    logger.info('tested on MRPC validation set, https://huggingface.co/datasets/mesolitica/translated-MRPC')
    logger.info('tested on PAWS test set, https://huggingface.co/datasets/mesolitica/translated-PAWS')


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


def huggingface(model='mesolitica/finetune-paraphrase-t5-small-standard-bahasa-cased', **kwargs):
    """
    Load HuggingFace model to paraphrase.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-paraphrase-t5-small-standard-bahasa-cased')
        Check available models at `malaya.paraphrase.available_huggingface()`.

    Returns
    -------
    result: malaya.torch_model.huggingface.Paraphrase
    """
    if model not in _huggingface_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.paraphrase.available_huggingface()`.'
        )
    return load_huggingface.load_paraphrase(model=model, initial_text='parafrasa: ', **kwargs)
