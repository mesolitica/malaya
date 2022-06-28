from malaya.alignment.en_ms import _eflomal, huggingface as load_huggingface


def eflomal(**kwargs):
    """
    load eflomal alignment for MS-EN.
    """
    return _eflomal(file='ms-en', **kwargs)


def huggingface(model: str = 'mesolitica/bert-base-multilingual-6layers-ms-en-alignment', **kwargs):
    """
    Load huggingface model word alignment for MS-EN, Required Tensorflow >= 2.0.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'mesolitica/bert-base-multilingual-6layers-ms-en-alignment'`` - finetuned BERT multilanguage on MS-EN.
        * ``'bert-base-multilingual-cased'`` - pretrained BERT multilanguage.

    Returns
    -------
    result: malaya.model.alignment.HuggingFace
    """
    return load_huggingface(model=model, **kwargs)
