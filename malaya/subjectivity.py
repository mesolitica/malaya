from malaya.supervised import classification
from malaya.path import PATH_SUBJECTIVE, S3_PATH_SUBJECTIVE
from herpetologist import check_type

label = ['negative', 'positive']

_transformer_availability = {
    'bert': {
        'Size (MB)': 425.6,
        'Quantized Size (MB)': 111,
        'macro precision': 0.92004,
        'macro recall': 0.91748,
        'macro f1-score': 0.91663,
    },
    'tiny-bert': {
        'Size (MB)': 57.4,
        'Quantized Size (MB)': 15.4,
        'macro precision': 0.91023,
        'macro recall': 0.90228,
        'macro f1-score': 0.90301,
    },
    'albert': {
        'Size (MB)': 48.6,
        'Quantized Size (MB)': 12.8,
        'macro precision': 0.90544,
        'macro recall': 0.90299,
        'macro f1-score': 0.90300,
    },
    'tiny-albert': {
        'Size (MB)': 22.4,
        'Quantized Size (MB)': 5.98,
        'macro precision': 0.89457,
        'macro recall': 0.89469,
        'macro f1-score': 0.89461,
    },
    'xlnet': {
        'Size (MB)': 446.6,
        'Quantized Size (MB)': 118,
        'macro precision': 0.91916,
        'macro recall': 0.91753,
        'macro f1-score': 0.91761,
    },
    'alxlnet': {
        'Size (MB)': 46.8,
        'Quantized Size (MB)': 13.3,
        'macro precision': 0.90862,
        'macro recall': 0.90835,
        'macro f1-score': 0.90817,
    },
}


def available_transformer():
    """
    List available transformer subjective analysis models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 20% test set.'
    )


def multinomial(**kwargs):
    """
    Load multinomial subjectivity model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    result : malaya.model.ml.Bayes class
    """
    return classification.multinomial(
        PATH_SUBJECTIVE, S3_PATH_SUBJECTIVE, 'subjective', label, **kwargs
    )


@check_type
def transformer(model: str = 'bert', quantized: bool = False, **kwargs):
    """
    Load Transformer subjectivity model.

    Parameters
    ----------
    model : str, optional (default='bert')
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
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.model.bert.BinaryBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.BinaryXLNET`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.subjectivity.available_transformer()`.'
        )
    return classification.transformer(
        module='subjectivity',
        label=label,
        model=model,
        quantized=quantized,
        **kwargs
    )
