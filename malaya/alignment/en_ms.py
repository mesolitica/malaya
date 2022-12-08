from malaya.function import check_file
from malaya.model.alignment import Eflomal, HuggingFace
from malaya_boilerplate import frozen_graph
from malaya_boilerplate.utils import check_tf2
import tensorflow as tf
from typing import Callable

_huggingface_availability = {
    'mesolitica/finetuned-bert-base-multilingual-cased-noisy-en-ms': {
        'Size (MB)': 599,
    },
    'bert-base-multilingual-cased': {
        'Size (MB)': 714,
    },
}


def _eflomal(preprocessing_func, file, **kwargs):
    path = check_file(
        file=file,
        module='eflomal-alignment',
        keys={'model': 'model.priors'},
        quantized=False,
        **kwargs,
    )
    return Eflomal(preprocessing_func=preprocessing_func, priors_filename=path['model'])


def available_huggingface():
    """
    List available HuggingFace models.
    """
    from malaya.function import describe_availability

    return describe_availability(_huggingface_availability)


def eflomal(preprocessing_func: Callable = None, **kwargs):
    """
    load eflomal word alignment for EN-MS. Model size around ~300MB.

    Parameters
    ----------
    preprocessing_func: Callable, optional (default=None)
        preprocessing function to call during loading prior file.
        Using `malaya.text.function.replace_punct` able to reduce ~30% of memory usage.

    Returns
    -------
    result: malaya.model.alignment.Eflomal
    """

    try:
        from eflomal import read_text, write_text, align
    except BaseException:
        raise ModuleNotFoundError(
            'eflomal not installed. Please install it from https://github.com/robertostling/eflomal for Linux / Windows or https://github.com/huseinzol05/maceflomal for Mac and try again.'
        )
    return _eflomal(preprocessing_func=preprocessing_func, file='en-ms', **kwargs)


@check_tf2
def huggingface(
    model: str = 'mesolitica/finetuned-bert-base-multilingual-cased-noisy-en-ms',
    force_check: bool = True,
    **kwargs,
):
    """
    Load huggingface BERT model word alignment for EN-MS, Required Tensorflow >= 2.0.

    Parameters
    ----------
    model : str, optional (default='mesolitica/finetuned-bert-base-multilingual-cased-noisy-en-ms')
        Check available models at `malaya.alignment.en_ms.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.model.alignment.HuggingFace
    """

    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.alignment.en_ms.available_huggingface()`.'
        )

    tokenizer = BertTokenizer.from_pretrained(model)
    device = frozen_graph.get_device(**kwargs)
    with tf.device(device):
        model = TFBertModel.from_pretrained(model)

    return HuggingFace(model=model, tokenizer=tokenizer)
