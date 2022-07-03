from malaya.function import check_file
from malaya.model.alignment import Eflomal, HuggingFace
from malaya_boilerplate import frozen_graph
import tensorflow as tf

_huggingface_availability = {
    'mesolitica/finetuned-bert-base-multilingual-cased-noisy-en-ms': {
        'Size (MB)': 599,
    },
    'bert-base-multilingual-cased': {
        'Size (MB)': 714,
    },
}


def _eflomal(file, **kwargs):
    path = check_file(
        file=file,
        module='eflomal-alignment',
        keys={'model': 'model.priors'},
        quantized=False,
        **kwargs,
    )
    return Eflomal(priors_filename=path['model'])


def available_huggingface():
    """
    List available HuggingFace models.
    """
    from malaya.function import describe_availability

    return describe_availability(_huggingface_availability)


def eflomal(**kwargs):
    """
    load eflomal word alignment for EN-MS. Model size around ~300MB.

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
    return _eflomal(file='en-ms', **kwargs)


def huggingface(model: str = 'mesolitica/finetuned-bert-base-multilingual-cased-noisy-en-ms', **kwargs):
    """
    Load huggingface BERT model word alignment for EN-MS, Required Tensorflow >= 2.0.

    Parameters
    ----------
    model : str, optional (default='mesolitica/finetuned-bert-base-multilingual-cased-noisy-en-ms')
        Model architecture supported. Allowed values:

        * ``'mesolitica/finetuned-bert-base-multilingual-cased-noisy-en-ms'`` - finetuned BERT multilanguage on noisy EN-MS.
        * ``'bert-base-multilingual-cased'`` - pretrained BERT multilanguage.

    Returns
    -------
    result: malaya.model.alignment.HuggingFace
    """

    model = model.lower()
    if model not in _huggingface_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.alignment.en_ms.available_huggingface()`.'
        )

    from malaya_boilerplate.utils import check_tf2_huggingface

    check_tf2_huggingface()

    try:
        from transformers import TFBertModel, BertTokenizer
    except BaseException:
        raise ModuleNotFoundError(
            'transformers not installed. Please install it by `pip3 install transformers` and try again.'
        )

    tokenizer = BertTokenizer.from_pretrained(model)
    device = frozen_graph.get_device(**kwargs)
    with tf.device(device):
        model = TFBertModel.from_pretrained(model)

    return HuggingFace(model=model, tokenizer=tokenizer)
