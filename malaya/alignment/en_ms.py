from malaya.function import check_file
from malaya.model.alignment import Eflomal, HuggingFace
from malaya_boilerplate import frozen_graph


def _eflomal(file, **kwargs):
    path = check_file(
        file=file,
        module='eflomal-alignment',
        keys={'model': 'model.priors'},
        quantized=False,
        **kwargs,
    )
    return Eflomal(priors_filename=path['model'])


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
    from malaya_boilerplate.utils import check_tf2_huggingface

    check_tf2_huggingface()

    try:
        from transformers import TFBertModel, BertTokenizer
    except BaseException:
        raise ModuleNotFoundError(
            'transformers not installed. Please install it by `pip3 install transformers` and try again.'
        )

    device = frozen_graph.get_device(**kwargs)
    with tf.device(device):
        model = BertModel.from_pretrained(model)

    tokenizer = BertTokenizer.from_pretrained(model)
    return HuggingFace(model=model, tokenizer=tokenizer)
