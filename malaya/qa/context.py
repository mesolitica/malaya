from malaya.supervised import qa
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
    'mesolitica/finetune-qa-context-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'WER': 0.1345597,
        'Suggested length': 256,
    },
    'mesolitica/finetune-qa-context-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'WER': 0.1345597,
        'Suggested length': 256,
    },
    'mesolitica/finetune-qa-context-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'WER': 0.1345597,
        'Suggested length': 256,
    },
}


def _describe():
    logger.info('testes on translated SQUAD V2 Dev set at https://github.com/huseinzol05/malay-dataset/tree/master/question-answer/squad')


def available_transformer():
    """
    List available Transformer Span models.
    """

    warnings.warn(
        '`malaya.qa.context.available_transformer` is deprecated, use `malaya.qa.context.available_huggingface` instead', DeprecationWarning)

    _describe()
    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available huggingface models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


@check_type
def transformer(model: str = 'xlnet', quantized: bool = False, **kwargs):
    """
    Load Transformer Span model trained on SQUAD V2 dataset.

    Parameters
    ----------
    model: str, optional (default='xlnet')
        Check available models at `malaya.qa.context.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.SQUAD class
    """

    warnings.warn(
        '`malaya.qa.context.transformer` is deprecated, use `malaya.qa.context.huggingface` instead', DeprecationWarning)

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.qa.context.available_transformer()`.'
        )
    return qa.transformer_squad(
        module='qa-squad', model=model, quantized=quantized, **kwargs
    )


def huggingface(model: str = 'mesolitica/finetune-qa-context-t5-small-standard-bahasa-cased', **kwargs):
    """
    Load HuggingFace model to answer questions based on context.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-qa-context-t5-small-standard-bahasa-cased')
        Check available models at `malaya.qa.context.available_huggingface()`.

    Returns
    -------
    result: malaya.torch_model.huggingface.QAContext
    """

    if model not in _huggingface_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.qa.context.available_huggingface()`.'
        )
