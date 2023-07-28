from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/Llama-2-7b-finetune-qlora': {
        'base model': 'https://huggingface.co/meta-llama/Llama-2-7b-hf',
        'Size (GB)': 13.85,
        'sharded': False,
        'lora': True,
        'context length': 512,
    },
    'mesolitica/Llama-2-7b-finetune-qlora': {
        'base model': 'https://huggingface.co/meta-llama/Llama-2-7b-hf',
        'Size (GB)': 13.85,
        'sharded': False,
        'lora': True,
        'context length': 1024,
    },
}


def available_huggingface():
    """
    List available HuggingFace models.
    """

    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica//Llama-2-7b-finetune-qlora',
    force_check: bool = True,
    proceed_non_shard: bool = False,
    **kwargs,
):
    """
    Load LLM HuggingFace model.

    Parameters
    ----------
    model: str, optional (default='mesolitica//Llama-2-7b-finetune-qlora')
        Check available models at `malaya.llm.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.
    proceed_non_shard: bool, optional (default=False)
        if False and use non sharded model, will throw a warning and stop.
        else, will proceed.

    Returns
    -------
    result: malaya.torch_model.huggingface.LLM
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.llm.available_huggingface()`.'
        )

    if not _huggingface_availability[model]['sharded'] and not proceed_non_shard:
        logger.warning(
            f'`{model}` is not sharded, this can caused OOM during loading the machine, make sure you have enough memory to load it at the first place.')
        logger.warning(
            'If you are aware of non sharded model memory usage, pass `proceed_non_shard=False` to proceed.')

    return load_huggingface.load_llm(
        base_model=_huggingface_availability[model]['base model'],
        model=model,
        lora=_huggingface_availability[model]['lora'],
        **kwargs,
    )
