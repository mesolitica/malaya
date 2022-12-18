from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
from herpetologist import check_type
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/finetune-keyword-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'f1': 0.3291554473802324,
        'Suggested length': 1024,
    },
    'mesolitica/finetune-keyword-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'f1': 0.3367989506031038,
        'Suggested length': 1024,
    },
}


def available_huggingface():
    """
    List available huggingface models.
    """

    logger.info('tested on test set, https://huggingface.co/datasets/51la5/keyword-extraction/tree/main')
    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/finetune-keyword-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to abstractive keyword.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-keyword-t5-small-standard-bahasa-cased')
        Check available models at `malaya.keyword.abstractive.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Keyword
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.keyword.abstractive.available_huggingface()`.'
        )
    return load_huggingface.load_keyword(model=model, **kwargs)
