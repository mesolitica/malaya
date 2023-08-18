from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/llama-7b-hf-1536-ms-qlora': {
        'base model': 'https://huggingface.co/meta-llama/Llama-2-7b-hf',
        'Size (GB)': 13.85,
        'context length': 1536,
        'packing': False,
        'trained on': [
            'https://huggingface.co/datasets/mesolitica/google-translate-oasst1',
            'https://huggingface.co/datasets/mesolitica/chatgpt-code-context',
            'https://huggingface.co/datasets/mesolitica/google-translate-OIG',
            'https://huggingface.co/datasets/mesolitica/google-translate-NSText2SQL',
            'https://huggingface.co/datasets/mesolitica/chatgpt-msa-common-crawl-qa',
            'https://huggingface.co/datasets/mesolitica/chatgpt-malaysia-hansard-qa',
            'https://huggingface.co/datasets/mesolitica/chatgpt-gpt4all/blob/main/translated-gpt4all-code.json',
            'https://huggingface.co/datasets/mesolitica/google-translate-vicuna-sharegpt',
        ]
    },
    'mesolitica/llama-7b-hf-1024-ms-qlora-packing': {
        'base model': 'https://huggingface.co/meta-llama/Llama-2-7b-hf',
        'Size (GB)': 13.85,
        'context length': 1024,
        'packing': True,
        'trained on': [
            'https://huggingface.co/datasets/mesolitica/google-translate-oasst1',
            'https://huggingface.co/datasets/mesolitica/chatgpt-code-context',
            'https://huggingface.co/datasets/mesolitica/google-translate-OIG',
            'https://huggingface.co/datasets/mesolitica/google-translate-NSText2SQL',
            'https://huggingface.co/datasets/mesolitica/chatgpt-msa-common-crawl-qa',
            'https://huggingface.co/datasets/mesolitica/chatgpt-malaysia-hansard-qa',
            'https://huggingface.co/datasets/mesolitica/chatgpt-gpt4all/blob/main/translated-gpt4all-code.json',
            'https://huggingface.co/datasets/mesolitica/google-translate-vicuna-sharegpt',
        ]
    },
    'mesolitica/llama-7b-hf-1024-ms-qlora-packing-meleis': {
        'base model': 'https://huggingface.co/meta-llama/Llama-2-7b-hf',
        'Size (GB)': 13.85,
        'context length': 1024,
        'packing': True,
        'trained on': [
            'https://huggingface.co/datasets/mesolitica/google-translate-oasst1',
            'https://huggingface.co/datasets/mesolitica/google-translate-OIG',
            'https://huggingface.co/datasets/mesolitica/google-translate-vicuna-sharegpt',
        ]
    },
}


def available_huggingface():
    """
    List available HuggingFace models.
    """

    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/llama-7b-hf-1024-ms-qlora',
    force_check: bool = True,
    **kwargs,
):
    """
    Load LLM HuggingFace model.

    Parameters
    ----------
    model: str, optional (default='mesolitica/llama-7b-hf-1024-ms-qlora')
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
