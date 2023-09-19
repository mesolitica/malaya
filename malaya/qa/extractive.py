from malaya.function import describe_availability
from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import ExtractiveQA
import logging
import warnings

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/finetune-qa-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'exact-ms': 0.426955861,
        'f1-ms': 0.5113033923,
        'exact-en': 0.4692567567,
        'f1-en': 0.541063384,
    },
    'mesolitica/finetune-qa-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'exact-ms': 0.530340113,
        'f1-ms': 0.61693299,
        'exact-en': 0.55878378,
        'f1-en': 0.6292999233,
    },
    'mesolitica/finetune-qa-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'exact-ms': 0.570354729,
        'f1-ms': 0.64136968,
        'exact-en': 0.570354729,
        'f1-en': 0.64136968144,
    },
}


def available_huggingface():
    """
    List available huggingface models.
    """

    logger.info('tested on translated SQUAD V2 Dev set, https://github.com/huseinzol05/malay-dataset/tree/master/question-answer/squad')
    logger.info('tested on SQUAD V2 Dev set, https://rajpurkar.github.io/SQuAD-explorer/')
    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/finetune-qa-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to answer extractive question answers.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-qa-t5-small-standard-bahasa-cased')
        Check available models at `malaya.qa.extractive.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.ExtractiveQA
    """
    return load(
        model=model,
        class_model=ExtractiveQA,
        availability=_huggingface_availability,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
