from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Translation

available_huggingface = {
    'mesolitica/jawi-nanot5-tiny-malaysian-cased': {
        'Size (MB)': 205,
        'Suggested length': 2048,
        'jawi-rumi chrF2++': 67.62,
        'rumi-jawi chrF2++': 64.41,
        'from lang': ['jawi', 'rumi'],
        'to lang': ['jawi', 'rumi'],
    },
    'mesolitica/jawi-nanot5-small-malaysian-cased': {
        'Size (MB)': 358,
        'Suggested length': 2048,
        'jawi-rumi chrF2++': 67.62,
        'rumi-jawi chrF2++': 64.41,
        'from lang': ['jawi', 'rumi'],
        'to lang': ['jawi', 'rumi'],
    },
}


info = """
tested on first 10k Rumi-Jawi test set, dataset at https://huggingface.co/datasets/mesolitica/rumi-jawi
""".strip()


def huggingface(
    model: str = 'mesolitica/jawi-nanot5-small-malaysian-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to translate.

    Parameters
    ----------
    model: str, optional (default='mesolitica/jawi-nanot5-small-malaysian-cased')
        Check available models at `malaya.jawi.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Translation
    """
    return load(
        model=model,
        class_model=Translation,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
