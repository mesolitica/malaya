from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Translation
from typing import List

available_huggingface = {
    'mesolitica/translation-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'Suggested length': 1536,
        'ms-pasar ms chrF2++': 49.01,
        'en-pasar ms chrF2++': 45.29,
        'ms-manglish chrF2++': 37.55,
        'en-manglish chrF2++': 44.32,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn'],
        'to lang': ['manglish', 'pasar ms'],
    },
    'mesolitica/translation-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'Suggested length': 1536,
        'ms-pasar ms chrF2++': 54.30,
        'en-pasar ms chrF2++': 51.88,
        'ms-manglish chrF2++': 39.98,
        'en-manglish chrF2++': 44.58,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn'],
        'to lang': ['manglish', 'pasar ms'],
    },
    'mesolitica/translation-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'Suggested length': 1536,
        'ms-pasar ms chrF2++': 50.25,
        'en-pasar ms chrF2++': 49.26,
        'ms-manglish chrF2++': 38.41,
        'en-manglish chrF2++': 43.38,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn'],
        'to lang': ['manglish', 'pasar ms'],
    },
    'mesolitica/translation-nanot5-tiny-malaysian-cased': {
        'Size (MB)': 205,
        'Suggested length': 2048,
        'ms-pasar ms chrF2++': 53.36,
        'en-pasar ms chrF2++': 49.31,
        'ms-manglish chrF2++': 37.22,
        'en-manglish chrF2++': 43.50,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn', 'manglish', 'pasar ms', 'mandarin', 'pasar mandarin'],
        'to lang': ['manglish', 'pasar ms', 'pasar mandarin'],
    },
    'mesolitica/translation-nanot5-small-malaysian-cased': {
        'Size (MB)': 358,
        'Suggested length': 2048,
        'ms-pasar ms chrF2++': 55.09,
        'en-pasar ms chrF2++': 53.01,
        'ms-manglish chrF2++': 40.19,
        'en-manglish chrF2++': 45.69,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn', 'manglish', 'pasar ms', 'mandarin', 'pasar mandarin'],
        'to lang': ['manglish', 'pasar ms', 'pasar mandarin'],
    },
    'mesolitica/translation-nanot5-base-malaysian-cased': {
        'Size (MB)': 990,
        'Suggested length': 2048,
        'ms-pasar ms chrF2++': 55.57,
        'en-pasar ms chrF2++': 55.02,
        'ms-manglish chrF2++': 40.17,
        'en-manglish chrF2++': 43.44,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn', 'manglish', 'pasar ms', 'mandarin', 'pasar mandarin'],
        'to lang': ['manglish', 'pasar ms', 'pasar mandarin'],
    },
}


def huggingface(
    model: str = 'mesolitica/translation-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to abstractive text augmentation.

    Parameters
    ----------
    model: str, optional (default='mesolitica/translation-t5-small-standard-bahasa-cased')
        Check available models at `malaya.augmentation.abstractive.available_huggingface`.
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
        path=__name__,
        **kwargs,
    )
