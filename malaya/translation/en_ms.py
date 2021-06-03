from malaya.model.tf import Translation
from malaya.model.bigbird import Translation as BigBird_Translation
from malaya.supervised import transformer as load_transformer
from malaya.supervised import bigbird as load_bigbird
from herpetologist import check_type

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.4,
        'BLEU': 0.512,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 82.7,
        'BLEU': 0.696,
        'Suggested length': 256,
    },
    'large': {
        'Size (MB)': 817,
        'Quantized Size (MB)': 244,
        'BLEU': 0.699,
        'Suggested length': 256,
    },
    'bigbird': {
        'Size (MB)': 246,
        'Quantized Size (MB)': 63.7,
        'BLEU': 0.551,
        'Suggested length': 1024,
    },
    'small-bigbird': {
        'Size (MB)': 50.4,
        'Quantized Size (MB)': 13.1,
        'BLEU': 0.522,
        'Suggested length': 1024,
    },
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 77k EN-MS sentences.'
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
        * ``'bigbird'`` - BigBird BASE parameters.
        * ``'small-bigbird'`` - BigBird SMALL parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `bigbird` in model, return `malaya.model.bigbird.Translation`.
        * else, return `malaya.model.tf.Translation`.
    """
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.en_ms.available_transformer()`.'
        )

    if 'bigbird' in model:
        return load_bigbird.load(
            module='translation-en-ms',
            model=model,
            model_class=BigBird_Translation,
            maxlen=_transformer_availability[model]['Suggested length'],
            quantized=quantized,
            **kwargs
        )

    else:
        return load_transformer.load(
            module='translation-en-ms',
            model=model,
            encoder='subword',
            model_class=Translation,
            quantized=quantized,
            **kwargs
        )

    return load_transformer.load(
        module='translation-en-ms',
        model=model,
        encoder='subword',
        model_class=Translation,
        quantized=quantized,
        **kwargs
    )
