from malaya.supervised.huggingface import load
from malaya.torch_model.llm import LLM

available_huggingface = {
    'mesolitica/mallam-1.1b-20k-instructions-v2': {
        'base model': 'https://huggingface.co/mesolitica/mallam-1.1B-4096',
        'Size (GB)': 2.25,
        'context length': 20480,
    },
    'mesolitica/mallam-5b-20k-instructions-v2': {
        'base model': 'https://huggingface.co/mesolitica/mallam-5B-4096',
        'Size (GB)': 10.0,
        'context length': 20480,
    },
    'mesolitica/malaysian-tinyllama-1.1b-16k-instructions-v2': {
        'base model': 'https://huggingface.co/mesolitica/mallam-5B-4096',
        'Size (GB)': 2.2,
        'context length': 16384,
    },
    'mesolitica/malaysian-mistral-7b-32k-instructions-v3': {
        'base model': 'https://huggingface.co/mesolitica/mallam-5B-4096',
        'Size (GB)': 2.25,
        'context length': 20480,
    },
}


def huggingface(
    model: str = 'mesolitica/malaysian-llama2-7b-32k-instructions',
    force_check: bool = True,
    **kwargs,
):
    """
    Load LLM HuggingFace model.

    Parameters
    ----------
    model: str, optional (default='mesolitica/malaysian-llama2-7b-32k-instructions')
        Check available models at `malaya.llm.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.LLM
    """
    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.llm.available_huggingface`.'
        )

    return load(
        model=model,
        class_model=LLM,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
