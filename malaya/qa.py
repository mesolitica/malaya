from malaya.supervised import qa
from herpetologist import check_type

_transformer_squad_availability = {
    'tiny-bert': {
        'Size (MB)': 60.9,
        'Quantized Size (MB)': 15.3,
        'exact': 53.45758,
        'f1': 56.79821,
        'total': 11858,
    },
    'bert': {
        'Size (MB)': 452,
        'Quantized Size (MB)': 113,
        'exact': 57.1681,
        'f1': 61.4874,
        'total': 11858,
    },
    'albert': {
        'Size (MB)': 58.1,
        'Quantized Size (MB)': 14.6,
        'exact': 58.97284,
        'f1': 63.12757,
        'total': 11858,
    },
    'tiny-albert': {
        'Size (MB)': 24.8,
        'Quantized Size (MB)': 6.35,
        'exact': 50.00843,
        'f1': 50.00843,
        'total': 11858,
    },
    'xlnet': {
        'Size (MB)': 478,
        'Quantized Size (MB)': 120,
        'exact': 62.74245,
        'f1': 66.56101,
        'total': 11858,
    },
    'alxlnet': {
        'Size (MB)': 58.4,
        'Quantized Size (MB)': 15.6,
        'exact': 61.97503,
        'f1': 65.89765,
        'total': 11858,
    },
}


def available_transformer_squad():
    """
    List available Transformer Span models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_squad_availability, text='tested on SQUAD V2 Dev set.'
    )


@check_type
def transformer_squad(model: str = 'xlnet', quantized: bool = False, **kwargs):
    """
    Load Transformer Span model trained on SQUAD V2 dataset.

    Parameters
    ----------
    model : str, optional (default='xlnet')
        Model architecture supported. Allowed values:

        * ``'bert'`` - Google BERT BASE parameters.
        * ``'tiny-bert'`` - Google BERT TINY parameters.
        * ``'albert'`` - Google ALBERT BASE parameters.
        * ``'tiny-albert'`` - Google ALBERT TINY parameters.
        * ``'xlnet'`` - Google XLNET BASE parameters.
        * ``'alxlnet'`` - Malaya ALXLNET BASE parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.SQUAD class
    """

    model = model.lower()
    if model not in _transformer_squad_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.qa.available_transformer_squad()`.'
        )
    return qa.transformer_squad(
        module='qa-squad', model=model, quantized=quantized, **kwargs
    )
