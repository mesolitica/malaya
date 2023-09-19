from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/llama2-7b-lora-instruct-32768-no-alignment': {
        'base model': 'https://huggingface.co/mesolitica/llama-7b-hf-32768-fpf',
        'Size (GB)': 13.85,
        'context length': 32768,
        'trained on': [
            'https://huggingface.co/datasets/mesolitica/google-translate-ultrachat',
            'https://huggingface.co/datasets/mesolitica/chatgpt-msa-common-crawl-qa',
            'https://huggingface.co/datasets/mesolitica/chatgpt-malaysia-hansard-qa',
        ]
    },
    'mesolitica/llama2-13b-lora-instruct-32768-no-alignment': {
        'base model': 'https://huggingface.co/mesolitica/llama-13b-hf-32768-fpf',
        'Size (GB)': 13.85,
        'context length': 32768,
        'trained on': [
            'https://huggingface.co/datasets/mesolitica/google-translate-ultrachat',
            'https://huggingface.co/datasets/mesolitica/chatgpt-msa-common-crawl-qa',
            'https://huggingface.co/datasets/mesolitica/chatgpt-malaysia-hansard-qa',
        ]
    },
}


def available_huggingface():
    """
    List available HuggingFace models.
    """

    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/llama2-7b-ft-instruct-16384-packing',
    force_check: bool = True,
    **kwargs,
):
    """
    Load LLM HuggingFace model.

    Parameters
    ----------
    model: str, optional (default='mesolitica/llama2-7b-ft-instruct-16384-packing')
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

    return load_huggingface.load_llm(
        model=model,
        **kwargs,
    )
