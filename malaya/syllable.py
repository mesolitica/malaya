from malaya.model.stem import Tokenizer
from malaya.supervised.rnn import load
from malaya.torch_model.rnn import Syllable


available_huggingface = {
    'mesolitica/syllable-lstm': {
        'Size (MB)': 35.2,
        'hidden size': 512,
        'CER': 0.011996584781229728,
        'WER': 0.06915983606557377,
    },
}

info = """
trained on 95% dataset, tested on another 5% test set, dataset at https://github.com/huseinzol05/malay-dataset/tree/master/tokenization/syllable
""".strip()


def rules(**kwargs):
    """
    Load rules based syllable tokenizer.
    originally from https://github.com/fahadh4ilyas/syllable_splitter/blob/master/SyllableSplitter.py
    - improved `cuaca` double vocal `ua` based on https://en.wikipedia.org/wiki/Comparison_of_Indonesian_and_Standard_Malay#Syllabification
    - improved `rans` double consonant `ns` based on https://www.semanticscholar.org/paper/Syllabification-algorithm-based-on-syllable-rules-Musa-Kadir/a819f255f066ae0fd7a30b3534de41da37d04ea1
    - improved `au` and `ai` double vocal.

    Returns
    -------
    result: malaya.syllable.Tokenizer class
    """
    return Tokenizer()


def huggingface(
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
        class_model=Syllable,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
