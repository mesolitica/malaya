from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/gptj6b-finetune': {
        'base model': 'EleutherAI/gpt-j-6b',
        'Size (GB)': 24.2,
        'sharded': False,
    },
    'mesolitica/pythia-6.9b-finetune': {
        'base model': 'EleutherAI/pythia-6.9b',
        'Size (GB)': 13.85,
        'sharded': True,
    },
    'mesolitica/pythia-2.8b-finetune': {
        'base model': 'EleutherAI/pythia-2.8b',
        'Size (GB)': 5.68,
        'sharded': True,
    }
}


def available_huggingface():
    """
    List available HuggingFace models.
    """

    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/pythia-2.8b-finetune',
    force_check: bool = True,
    **kwargs,
):
    """
    Load LLM HuggingFace model.

    Parameters
    ----------
    model: str, optional (default='mesolitica/pythia-2.8b-finetune')
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

    if not _huggingface_availability[model]['sharded']:
        logger.warning(
            f'`{model}` is not sharded, this can caused OOM during loading the machine, make sure you have enough memory to load it at the first place.')

    return load_huggingface.load_llm
