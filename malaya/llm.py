from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability, check_file
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
}


def available_huggingface():
    """
    List available HuggingFace models.
    """

    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/llama-7b',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to translate.

    Parameters
    ----------
    model: str, optional (default='mesolitica/llama-7b')
        Check available models at `malaya.llm.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.LLM
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.llm.available_huggingface()`.'
        )
    return load_huggingface.load_llm
