from malaya.model.stem import Naive, Sastrawi
from malaya.supervised.rnn import load
from malaya.torch_model.rnn import Stem

available_huggingface = {
    'mesolitica/stem-lstm-512': {
        'Size (MB)': 35.2,
        'hidden size': 512,
        'CER': 0.02549779186652238,
        'WER': 0.05448552235248484,
    },
}

info = """
Trained on train set and tested on test set, https://github.com/huseinzol05/malay-dataset/tree/master/normalization/stemmer
""".strip()


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


def huggingface(
    model: str = 'mesolitica/stem-lstm-512',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to stem and lemmatization.

    Parameters
    ----------
    model: str, optional (default='mesolitica/stem-lstm-512')
        Check available models at `malaya.stem.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.rnn.Stem
    """

    return load(
        model=model,
        class_model=Stem,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
