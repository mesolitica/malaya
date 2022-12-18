from herpetologist import check_type
from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/finetune-zeroshot-ner-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'exactly-match': 0.920265194,
        'f1': 0.9508079,
    },
    'mesolitica/finetune-zeroshot-ner-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'exactly-match': 0.920894092,
        'f1': 0.951281162,
    },
    'mesolitica/finetune-zeroshot-ner-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'exactly-match': 0.8917532135,
        'f1': 0.934447887,
    },
}


def available_huggingface():
    """
    List available huggingface models.
    """

    logger.info('tested on test set, https://huggingface.co/datasets/mesolitica/zeroshot-NER')
    logger.warning(
        '`exactly-match` and `f1` scores based on generated test set, it does not represent accuracy on actual human texts.')
    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/finetune-zeroshot-ner-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to zeroshot NER.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-zeroshot-ner-t5-small-standard-bahasa-cased')
        Check available models at `malaya.zero_shot.entity.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.ZeroShotNER
    """

    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.zero_shot.entity.available_huggingface()`.'
        )
    return load_huggingface.load_zeroshot_ner(model=model, **kwargs)
