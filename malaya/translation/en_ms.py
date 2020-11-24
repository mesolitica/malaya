from malaya.model.tf import TRANSLATION
from malaya.path import PATH_TRANSLATION, S3_PATH_TRANSLATION
from malaya.supervised import transformer as load_transformer
from herpetologist import check_type

_transformer_availability = {
    'small': {'Size (MB)': 42.7, 'Quantized Size (MB)': 13.4, 'BLEU': 0.142},
    'base': {'Size (MB)': 234, 'Quantized Size (MB)': 82.7, 'BLEU': 0.696},
    'large': {'Size (MB)': 817, 'Quantized Size (MB)': 244, 'BLEU': 0.699},
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 77k EN-MY sentences.'
    )


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load transformer encoder-decoder model to translate EN-to-MS.

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
            'model not supported, please check supported models from `malaya.translation.en_ms.available_transformer()`.'
        )

    path = PATH_TRANSLATION['en-ms']
    s3_path = S3_PATH_TRANSLATION['en-ms']

    return load_transformer.load(
        path,
        s3_path,
        model,
        'subword',
        TRANSLATION,
        quantized = quantized,
        **kwargs
    )
