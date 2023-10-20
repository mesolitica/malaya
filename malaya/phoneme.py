
from malaya.model.stem import Tokenizer
from malaya.supervised.rnn import load
from malaya.torch_model.rnn import Phoneme

available_huggingface_dbp = {
    'mesolitica/syllable-lstm': {
        'Size (MB)': 35.2,
        'hidden size': 512,
        'CER': 0.011996584781229728,
        'WER': 0.06915983606557377,
    },
}

available_huggingface_ipa = {
    'mesolitica/syllable-lstm': {
        'Size (MB)': 35.2,
        'hidden size': 512,
        'CER': 0.011996584781229728,
        'WER': 0.06915983606557377,
    },
}


def huggingface_dbp(
    model: str = 'mesolitica/syllable-lstm',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model for syllable tokenizer.

    Parameters
    ----------
    model: str, optional (default='mesolitica/syllable-lstm')
        Check available models at `malaya.syllable.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.rnn.Syllable
    """

    return load(
        model=model,
        class_model=Phoneme,
        available_huggingface=available_huggingface_dbp,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )


def huggingface_ipa(
    model: str = 'mesolitica/syllable-lstm',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model for syllable tokenizer.

    Parameters
    ----------
    model: str, optional (default='mesolitica/syllable-lstm')
        Check available models at `malaya.syllable.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.rnn.Syllable
    """

    return load(
        model=model,
        class_model=Phoneme,
        available_huggingface=available_huggingface_ipa,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
