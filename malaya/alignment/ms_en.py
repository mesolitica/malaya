from malaya.alignment.en_ms import _eflomal, huggingface as load_huggingface


def eflomal(**kwargs):
    """
    load eflomal alignment for MS-EN. Model size around ~300MB.
    """
    return _eflomal(file='ms-en', **kwargs)


def huggingface(model: str = 'mesolitica/finetuned-bert-base-multilingual-cased-noisy-en-ms', **kwargs):
    """
    Load huggingface BERTmodel word alignment for MS-EN, Required Tensorflow >= 2.0.

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
    return load_huggingface(model=model, **kwargs)
