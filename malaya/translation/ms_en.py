from malaya.model.tf import TRANSLATION
from malaya.path import PATH_TRANSLATION, S3_PATH_TRANSLATION
from malaya.supervised import transformer as load_transformer
from herpetologist import check_type

_transformer_availability = {
    'small': {'Size (MB)': 42.7, 'Quantized Size (MB)': 13.4, 'BLEU': 0.626},
    'base': {'Size (MB)': 234, 'Quantized Size (MB)': 82.7, 'BLEU': 0.792},
    'large': {'Size (MB)': 815, 'Quantized Size (MB)': 244, 'BLEU': 0.714},
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 100k MY-EN sentences.'
    )


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load Transformer encoder-decoder model to translate MS-to-EN.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'small'`` - Transformer SMALL parameters.
        * ``'base'`` - Transformer BASE parameters.
        * ``'large'`` - Transformer LARGE parameters.
    
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.TRANSLATION class
    """
    model = model.lower()
    if model not in _transformer_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya.translation.ms_en.available_transformer()`.'
        )

    path = PATH_TRANSLATION['ms-en']
    s3_path = S3_PATH_TRANSLATION['ms-en']

    return load_transformer.load(
        path,
        s3_path,
        model,
        'subword',
        TRANSLATION,
        quantized = quantized,
        **kwargs
    )
