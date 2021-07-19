from malaya.supervised import classification
from malaya.path import PATH_EMOTION, S3_PATH_EMOTION
from herpetologist import check_type

label = ['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']

_transformer_availability = {
    'bert': {
        'Size (MB)': 425.6,
        'Quantized Size (MB)': 111,
        'macro precision': 0.99786,
        'macro recall': 0.99773,
        'macro f1-score': 0.99779,
    },
    'tiny-bert': {
        'Size (MB)': 57.4,
        'Quantized Size (MB)': 15.4,
        'macro precision': 0.99692,
        'macro recall': 0.99696,
        'macro f1-score': 0.99694,
    },
    'albert': {
        'Size (MB)': 48.6,
        'Quantized Size (MB)': 12.8,
        'macro precision': 0.99740,
        'macro recall': 0.99773,
        'macro f1-score': 0.99757,
    },
    'tiny-albert': {
        'Size (MB)': 22.4,
        'Quantized Size (MB)': 5.98,
        'macro precision': 0.99325,
        'macro recall': 0.99378,
        'macro f1-score': 0.99351,
    },
    'xlnet': {
        'Size (MB)': 446.5,
        'Quantized Size (MB)': 118,
        'macro precision': 0.99773,
        'macro recall': 0.99775,
        'macro f1-score': 0.99774,
    },
    'alxlnet': {
        'Size (MB)': 46.8,
        'Quantized Size (MB)': 13.3,
        'macro precision': 0.99663,
        'macro recall': 0.99697,
        'macro f1-score': 0.99680,
    },
}


def available_transformer():
    """
    List available transformer emotion analysis models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 20% test set.'
    )


def multinomial(**kwargs):
    """
    Load multinomial emotion model.

    Returns
    -------
    result : malaya.model.ml.MulticlassBayes class
    """
    return classification.multinomial(
        path=PATH_EMOTION, s3_path=S3_PATH_EMOTION,
        module='emotion', label=label, **kwargs
    )


@check_type
def transformer(model: str = 'xlnet', quantized: bool = False, **kwargs):
    """
    Load Transformer emotion model.

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

        * if `bert` in model, will return `malaya.model.bert.MulticlassBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.MulticlassXLNET`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.emotion.available_transformer()`.'
        )
    return classification.transformer(
        module='emotion',
        label=label,
        model=model,
        quantized=quantized,
        **kwargs
    )
