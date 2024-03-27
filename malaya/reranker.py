from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Reranker

available_huggingface = {
    'mesolitica/reranker-malaysian-mistral-64M-32k': {
        'Size (MB)': 95.7,
        'Suggested length': 32768,
    },
    'mesolitica/reranker-malaysian-mistral-191M-32k': {
        'Size (MB)': 332,
        'Suggested length': 32768,
    },
    'mesolitica/reranker-malaysian-mistral-474M-32k': {
        'Size (MB)': 884,
        'Suggested length': 32768,
    }
}


def huggingface(
    model: str = 'mesolitica/reranker-malaysian-mistral-64M-32k',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model for reranking task.

    Parameters
    ----------
    model: str, optional (default='mesolitica/reranker-malaysian-mistral-64M-32k')
        Check available models at `malaya.reranker.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Reranker
    """
    return load(
        model=model,
        class_model=Reranker,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
