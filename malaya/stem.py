from malaya.model.stem import Naive, Sastrawi

available_huggingface = {
    'mesolitica/stemming-nanot5-super-tiny-malaysian-cased': {
        'Size (MB)': 205,
        'Suggested length': 2048,
        'CER': 0.02138838,
        'WER': 0.04952738,
    },
    'mesolitica/stemming-nanot5-tiny-malaysian-cased': {
        'Size (MB)': 358,
        'Suggested length': 2048,
        'CER': 0.02138838,
        'WER': 0.04952738,
    },
}


def naive():
    """
    Load stemming model using startswith and endswith naively using regex patterns.

    Returns
    -------
    result : malaya.stem.Naive class
    """

    return Naive()


def sastrawi():
    """
    Load stemming model using Sastrawi, this also include lemmatization.

    Returns
    -------
    result: malaya.stem.Sastrawi class
    """

    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    except BaseException:
        raise ModuleNotFoundError(
            'PySastrawi not installed. Please install it by `pip install PySastrawi` and try again.'
        )
    return Sastrawi(StemmerFactory())
