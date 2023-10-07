from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Embedding

available_huggingface = {
    'mesolitica/nanot5-small-embedding': {
        'Size (MB)': 2170,
        'embedding size': 1024,
        'Suggested length': 2048,
        'b.cari.com.my': 0,
        'c.cari.com.my': 0,
        'malay-news': 0,
        'twitter': 0,
    },
    'mesolitica/nanot5-base-embedding': {
        'Size (MB)': 2170,
        'embedding size': 1024,
        'Suggested length': 2048,
        'b.cari.com.my': 0,
        'c.cari.com.my': 0,
        'malay-news': 0,
        'twitter': 0,
    },
    'mesolitica/llama2-embedding-600m-8k': {
        'Size (MB)': 2170,
        'embedding size': 1536,
        'Suggested length': 32768,
        'b.cari.com.my': 0,
        'c.cari.com.my': 0,
        'malay-news': 0,
        'twitter': 0,
    },
    'mesolitica/llama2-embedding-1b-8k': {
        'Size (MB)': 3790,
        'embedding size': 1536,
        'Suggested length': 32768,
        'b.cari.com.my': 0,
        'c.cari.com.my': 0,
        'malay-news': 0,
        'twitter': 0,
    },
}


def huggingface(
    model: str = 'mesolitica/nanot5-small-embedding',
    **kwargs,
):
    """
    Load HuggingFace model to translate.

    Parameters
    ----------
    model: str, optional (default='mesolitica/nanot5-small-embedding')
        Check available models at `malaya.embedding.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Embedding
    """
    return load(
        model=model,
        class_model=Embedding,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
