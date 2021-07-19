from malaya.supervised import classification
from malaya.path import PATH_SENTIMENT, S3_PATH_SENTIMENT
from herpetologist import check_type

label = ['negative', 'positive']

_transformer_availability = {
    'bert': {
        'Size (MB)': 425.6,
        'Quantized Size (MB)': 111,
        'macro precision': 0.99330,
        'macro recall': 0.99330,
        'macro f1-score': 0.99329,
    },
    'tiny-bert': {
        'Size (MB)': 57.4,
        'Quantized Size (MB)': 15.4,
        'macro precision': 0.98774,
        'macro recall': 0.98774,
        'macro f1-score': 0.98774,
    },
    'albert': {
        'Size (MB)': 48.6,
        'Quantized Size (MB)': 12.8,
        'macro precision': 0.99227,
        'macro recall': 0.99226,
        'macro f1-score': 0.99226,
    },
    'tiny-albert': {
        'Size (MB)': 22.4,
        'Quantized Size (MB)': 5.98,
        'macro precision': 0.98554,
        'macro recall': 0.98550,
        'macro f1-score': 0.98551,
    },
    'xlnet': {
        'Size (MB)': 446.6,
        'Quantized Size (MB)': 118,
        'macro precision': 0.99353,
        'macro recall': 0.99353,
        'macro f1-score': 0.99353,
    },
    'alxlnet': {
        'Size (MB)': 46.8,
        'Quantized Size (MB)': 13.3,
        'macro precision': 0.99188,
        'macro recall': 0.99188,
        'macro f1-score': 0.99188,
    },
}


def available_transformer():
    """
    List available transformer sentiment analysis models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 20% test set.'
    )


def multinomial(**kwargs):
    """
    Load multinomial sentiment model.

    Returns
    -------
    result : malaya.model.ml.Bayes class
    """
    return classification.multinomial(
        PATH_SENTIMENT, S3_PATH_SENTIMENT, 'sentiment', label, **kwargs
    )


@check_type
def transformer(model: str = 'bert', quantized: bool = False, **kwargs):
    """
    Load Transformer sentiment model.

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
            'model not supported, please check supported models from `malaya.sentiment.available_transformer()`.'
        )
    return classification.transformer(
        module='sentiment',
        label=label,
        model=model,
        quantized=quantized,
        **kwargs
    )
