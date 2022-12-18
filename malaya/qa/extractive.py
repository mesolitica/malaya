from malaya.supervised import qa
from malaya.supervised import huggingface as load_huggingface
from herpetologist import check_type
from malaya.function import describe_availability
import logging
import warnings

logger = logging.getLogger(__name__)

_transformer_availability = {
    'tiny-bert': {
        'Size (MB)': 60.9,
        'Quantized Size (MB)': 15.3,
        'exact': 53.45758,
        'f1': 56.79821,
        'total': 11858,
    },
    'bert': {
        'Size (MB)': 452,
        'Quantized Size (MB)': 113,
        'exact': 57.1681,
        'f1': 61.4874,
        'total': 11858,
    },
    'albert': {
        'Size (MB)': 58.1,
        'Quantized Size (MB)': 14.6,
        'exact': 58.97284,
        'f1': 63.12757,
        'total': 11858,
    },
    'tiny-albert': {
        'Size (MB)': 24.8,
        'Quantized Size (MB)': 6.35,
        'exact': 50.00843,
        'f1': 50.00843,
        'total': 11858,
    },
    'xlnet': {
        'Size (MB)': 478,
        'Quantized Size (MB)': 120,
        'exact': 62.74245,
        'f1': 66.56101,
        'total': 11858,
    },
    'alxlnet': {
        'Size (MB)': 58.4,
        'Quantized Size (MB)': 15.6,
        'exact': 61.97503,
        'f1': 65.89765,
        'total': 11858,
    },
}

_huggingface_availability = {
    'mesolitica/finetune-extractive-qa-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'exact-ms': 0.426955861,
        'f1-ms': 0.5113033923,
        'exact-en': 0.4692567567,
        'f1-en': 0.541063384,
    },
    'mesolitica/finetune-extractive-qa-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'exact-ms': 0.530340113,
        'f1-ms': 0.61693299,
        'exact-en': 0.55878378,
        'f1-en': 0.6292999233,
    },
    'mesolitica/finetune-extractive-qa-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'exact-ms': 0.570354729,
        'f1-ms': 0.64136968,
        'exact-en': 0.570354729,
        'f1-en': 0.64136968144,
    },
    'mesolitica/finetune-extractive-qa-flan-t5-base': {
        'Size (MB)': 990,
        'exact-ms': 0.552198497,
        'f1-ms': 0.62477981389,
        'exact-en': 0.709290540,
        'f1-en': 0.77995760453,
    },
}


def _describe():
    logger.info('tested on translated SQUAD V2 Dev set, https://github.com/huseinzol05/malay-dataset/tree/master/question-answer/squad')


def available_transformer():
    """
    List available Transformer Span models.
    """

    warnings.warn(
        '`malaya.qa.extractive.available_transformer` is deprecated, use `malaya.qa.extractive.available_huggingface` instead', DeprecationWarning)

    _describe()
    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available huggingface models.
    """

    _describe()
    logger.info('tested on SQUAD V2 Dev set, https://rajpurkar.github.io/SQuAD-explorer/')
    return describe_availability(_huggingface_availability)


@check_type
def transformer(model: str = 'xlnet', quantized: bool = False, **kwargs):
    """
    Load Transformer Span model trained on SQUAD V2 dataset.

    Parameters
    ----------
    model: str, optional (default='xlnet')
        Check available models at `malaya.qa.extractive.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.SQUAD class
    """

    warnings.warn(
        '`malaya.qa.extractive.transformer` is deprecated, use `malaya.qa.extractive.huggingface` instead', DeprecationWarning)

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.qa.extractive.available_transformer()`.'
        )
    return qa.transformer_squad(
        module='qa-squad', model=model, quantized=quantized, **kwargs
    )


def huggingface(
    model: str = 'mesolitica/finetune-qa-extractive-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to answer extractive question answers.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-extractive-qa-t5-small-standard-bahasa-cased')
        Check available models at `malaya.qa.extractive.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.ExtractiveQA
    """

    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.qa.extractive.available_huggingface()`.'
        )

    return load_huggingface.load_extractive_qa(model=model, **kwargs)
