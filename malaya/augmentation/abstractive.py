from malaya.function import describe_availability
from malaya.supervised import huggingface as load_huggingface
from typing import List
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/finetune-noisy-translation-t5-tiny-bahasa-cased-v3': {
        'Size (MB)': 139,
        'Suggested length': 1024,
        'ms-pasar ms chrF2++': 52.16,
        'en-pasar ms chrF2++': 49.97,
        'from lang': ['en', 'ms'],
        'to lang': ['pasar ms'],
        'old model': True,
    },
    'mesolitica/finetune-noisy-translation-t5-small-bahasa-cased-v3': {
        'Size (MB)': 242,
        'Suggested length': 1024,
        'ms-pasar ms chrF2++': 51.62,
        'en-pasar ms chrF2++': 50.30,
        'from lang': ['en', 'ms'],
        'to lang': ['pasar ms'],
        'old model': True,
    },
    'mesolitica/finetune-noisy-translation-t5-base-bahasa-cased-v3': {
        'Size (MB)': 892,
        'Suggested length': 1024,
        'ms-pasar ms chrF2++': 51.06,
        'en-pasar ms chrF2++': 50.63,
        'from lang': ['en', 'ms'],
        'to lang': ['pasar ms'],
        'old model': True,
    },
    'mesolitica/translation-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'Suggested length': 1536,
        'ms-pasar ms chrF2++': 49.01,
        'en-pasar ms chrF2++': 45.29,
        'ms-manglish chrF2++': 37.55,
        'en-manglish chrF2++': 44.32,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn'],
        'to lang': ['manglish', 'pasar ms'],
        'old model': False,
    },
    'mesolitica/translation-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'Suggested length': 1536,
        'ms-pasar ms chrF2++': 54.30,
        'en-pasar ms chrF2++': 51.88,
        'ms-manglish chrF2++': 39.98,
        'en-manglish chrF2++': 44.58,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn'],
        'to lang': ['manglish', 'pasar ms'],
        'old model': False,
    },
    'mesolitica/translation-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'Suggested length': 1536,
        'ms-pasar ms chrF2++': 50.25,
        'en-pasar ms chrF2++': 49.26,
        'ms-manglish chrF2++': 38.41,
        'en-manglish chrF2++': 43.38,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn'],
        'to lang': ['manglish', 'pasar ms'],
        'old model': False,
    },
}


def available_huggingface():
    """
    List available huggingface models.
    """

    logger.info(
        'tested on noisy twitter google translation, https://huggingface.co/datasets/mesolitica/augmentation-test-set')
    return describe_availability(_huggingface_availability)


def _huggingface(
    availability,
    model: str = 'mesolitica/translation-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    from_lang: List[str] = None,
    to_lang: List[str] = None,
    old_model: bool = False,
    path=__name__,
    **kwargs,
):
    if model not in availability:
        if force_check:
            raise ValueError(
                f'model not supported, please check supported models from `{path}.available_huggingface()`.'
            )

    else:
        from_lang = from_lang or availability[model].get('from lang')
        to_lang = to_lang or availability[model].get('to lang')
        old_model = old_model or 'mesolitica/finetune' in model

    return load_huggingface.load_translation(
        model=model,
        from_lang=from_lang,
        to_lang=to_lang,
        old_model=old_model,
        **kwargs,
    )


def huggingface(
    model: str = 'mesolitica/translation-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    from_lang: List[str] = None,
    to_lang: List[str] = None,
    old_model: bool = False,
    **kwargs,
):
    """
    Load HuggingFace model to abstractive text augmentation.

    Parameters
    ----------
    model: str, optional (default='mesolitica/translation-t5-small-standard-bahasa-cased')
        Check available models at `malaya.translation.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Translation
    """
    return _huggingface(
        availability=_huggingface_availability,
        model=model,
        force_check=force_check,
        from_lang=from_lang,
        to_lang=to_lang,
        old_model=old_model,
        path=__name__,
        **kwargs,
    )
