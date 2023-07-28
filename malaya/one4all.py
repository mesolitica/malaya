from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/nanot5-tiny-malaysian-cased-one4all': {
        'Size (MB)': 358,
        'Suggested length': 1536,
    },
    'mesolitica/nanot5-small-malaysian-cased-one4all': {
        'Size (MB)': 358,
        'Suggested length': 1536,
    },
    'mesolitica/nanot5-base-malaysian-cased-one4all': {
        'Size (MB)': 892,
        'Suggested length': 1536,
    },
}


def available_huggingface():
    """
    List available huggingface models.
    """
    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/nanot5-small-malaysian-cased-one4all',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model for One for All tasks.

    Parameters
    ----------
    model: str, optional (default='mesolitica/nanot5-small-malaysian-cased-one4all')
        Check available models at `malaya.one4all.available_huggingface()`.
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
