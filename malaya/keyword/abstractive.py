from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Keyword

available_huggingface = {
    'mesolitica/finetune-keyword-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'f1': 0.3291554473802324,
        'Suggested length': 1024,
    },
    'mesolitica/finetune-keyword-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'f1': 0.3367989506031038,
        'Suggested length': 1024,
    },
}

info = """
tested on test set, https://huggingface.co/datasets/51la5/keyword-extraction/tree/main
""".strip()


def huggingface(
    model: str = 'mesolitica/finetune-keyword-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to abstractive keyword.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-keyword-t5-small-standard-bahasa-cased')
        Check available models at `malaya.keyword.abstractive.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Keyword
    """

    return load(
        model=model,
        class_model=Keyword,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
