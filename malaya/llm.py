from malaya.supervised.huggingface import load
from malaya.torch_model.llm import LLM

available_huggingface = {
    'mesolitica/malaysian-llama2-7b-32k-instructions': {
        'base model': 'https://huggingface.co/mesolitica/llama-7b-hf-32768-fpf',
        'Size (GB)': 13.85,
        'context length': 32768,
        'trained on': [
            'https://huggingface.co/datasets/mesolitica/google-translate-ultrachat',
        ]
    },
    'mesolitica/malaysian-llama2-13b-32k-instructions': {
        'base model': 'https://huggingface.co/mesolitica/llama-13b-hf-32768-fpf',
        'Size (GB)': 26.03,
        'context length': 32768,
        'trained on': [
            'https://huggingface.co/datasets/mesolitica/google-translate-ultrachat',
        ]
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
