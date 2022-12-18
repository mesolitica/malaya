from malaya.alignment.en_ms import _eflomal, available_huggingface, huggingface as load_huggingface
from typing import Callable


def eflomal(preprocessing_func: Callable = None, **kwargs):
    """
    load eflomal word alignment for MS-EN. Model size around ~300MB.

    Parameters
    ----------
    preprocessing_func: Callable, optional (default=None)
        preprocessing function to call during loading prior file.
        Using `malaya.text.function.replace_punct` able to reduce ~30% of memory usage.

    Returns
    -------
    result: malaya.model.alignment.Eflomal
    """

    return _eflomal(preprocessing_func=preprocessing_func, file='ms-en', **kwargs)


def huggingface(
    model: str = 'mesolitica/finetuned-bert-base-multilingual-cased-noisy-en-ms',
    force_check: bool = True,
    **kwargs,
):
    """
    Load huggingface BERT model word alignment for MS-EN, Required Tensorflow >= 2.0.

    Parameters
    ----------
    model : str, optional (default='mesolitica/finetuned-bert-base-multilingual-cased-noisy-en-ms')
        Check available models at `malaya.alignment.ms_en.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.model.alignment.HuggingFace
    """
    return load_huggingface(model=model, **kwargs)
