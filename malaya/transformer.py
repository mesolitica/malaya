from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Transformer

available_huggingface = {
    'mesolitica/roberta-base-bahasa-cased': {
        'Size (MB)': 443,
    },
    'mesolitica/roberta-tiny-bahasa-cased': {
        'Size (MB)': 66.1,
    },
    'mesolitica/bert-base-standard-bahasa-cased': {
        'Size (MB)': 443,
    },
    'mesolitica/bert-tiny-standard-bahasa-cased': {
        'Size (MB)': 66.1,
    },
    'mesolitica/roberta-base-standard-bahasa-cased': {
        'Size (MB)': 443,
    },
    'mesolitica/roberta-tiny-standard-bahasa-cased': {
        'Size (MB)': 66.1,
    },
    'mesolitica/electra-base-generator-bahasa-cased': {
        'Size (MB)': 140,
    },
    'mesolitica/electra-small-generator-bahasa-cased': {
        'Size (MB)': 19.3,
    },
}


def huggingface(
    model: str = 'mesolitica/electra-base-generator-bahasa-cased',
    **kwargs,
):
    """
    Load transformer model.

    Parameters
    ----------
    model: str, optional (default='mesolitica/electra-base-generator-bahasa-cased')
        Check available models at `malaya.transformer.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.
    """

    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.transformer.available_huggingface`.'
        )

    return load(
        model=model,
        class_model=Transformer,
        available_huggingface=available_huggingface,
        path=__name__,
        **kwargs,
    )
