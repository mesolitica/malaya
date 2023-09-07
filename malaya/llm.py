from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/llama2-7b-ft-instruct-2048-packing-forum': {
        'base model': 'https://huggingface.co/mesolitica/llama-7b-hf-16384-fpf',
        'Size (GB)': 13.85,
        'context length': 2048,
        'packing': True,
        'trained on': [
            'https://huggingface.co/datasets/mesolitica/google-translate-OIG',
            'https://huggingface.co/datasets/mesolitica/google-translate-sharegpt',
        ]
    },
    'mesolitica/llama2-7b-ft-instruct-16384-packing': {
        'base model': 'https://huggingface.co/mesolitica/llama-7b-hf-16384-fpf',
        'Size (GB)': 13.85,
        'context length': 16384,
        'packing': True,
        'trained on': [
            'https://huggingface.co/datasets/mesolitica/google-translate-oasst1',
            'https://huggingface.co/datasets/mesolitica/chatgpt-code-context',
            'https://huggingface.co/datasets/mesolitica/google-translate-OIG',
            'https://huggingface.co/datasets/mesolitica/google-translate-NSText2SQL',
            'https://huggingface.co/datasets/mesolitica/chatgpt-msa-common-crawl-qa',
            'https://huggingface.co/datasets/mesolitica/chatgpt-malaysia-hansard-qa',
            'https://huggingface.co/datasets/mesolitica/google-translate-sharegpt',
        ]
    },
    'malaysia-ai/llama2-13b-ft-instruct-2048-packing': {
        'base model': 'https://huggingface.co/meta-llama/Llama-2-13b-hf',
        'Size (GB)': 26.03,
        'context length': 2048,
        'packing': True,
        'trained on': [
            'https://huggingface.co/datasets/mesolitica/google-translate-oasst1',
            'https://huggingface.co/datasets/mesolitica/chatgpt-code-context',
            'https://huggingface.co/datasets/mesolitica/google-translate-OIG',
            'https://huggingface.co/datasets/mesolitica/google-translate-NSText2SQL',
            'https://huggingface.co/datasets/mesolitica/chatgpt-msa-common-crawl-qa',
            'https://huggingface.co/datasets/mesolitica/chatgpt-malaysia-hansard-qa',
            'https://huggingface.co/datasets/mesolitica/google-translate-sharegpt',
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
