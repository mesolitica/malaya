from malaya.supervised import classification
from malaya.path import PATH_SENTIMENT, S3_PATH_SENTIMENT
from herpetologist import check_type

label = ['negative', 'neutral', 'positive']

_transformer_availability = {
    'bert': {
        'Size (MB)': 425.6,
        'Quantized Size (MB)': 111,
        'macro precision': 0.93182,
        'macro recall': 0.93442,
        'macro f1-score': 0.93307,
    },
    'tiny-bert': {
        'Size (MB)': 57.4,
        'Quantized Size (MB)': 15.4,
        'macro precision': 0.93390,
        'macro recall': 0.93141,
        'macro f1-score': 0.93262,
    },
    'albert': {
        'Size (MB)': 48.6,
        'Quantized Size (MB)': 12.8,
        'macro precision': 0.91228,
        'macro recall': 0.91929,
        'macro f1-score': 0.91540,
    },
    'tiny-albert': {
        'Size (MB)': 22.4,
        'Quantized Size (MB)': 5.98,
        'macro precision': 0.91442,
        'macro recall': 0.91646,
        'macro f1-score': 0.91521,
    },
    'xlnet': {
        'Size (MB)': 446.6,
        'Quantized Size (MB)': 118,
        'macro precision': 0.92390,
        'macro recall': 0.92629,
        'macro f1-score': 0.92444,
    },
    'alxlnet': {
        'Size (MB)': 46.8,
        'Quantized Size (MB)': 13.3,
        'macro precision': 0.91896,
        'macro recall': 0.92589,
        'macro f1-score': 0.92198,
    },
    'fastformer': {
        'Size (MB)': 458,
        'Quantized Size (MB)': 116,
        'macro precision': 0.96882,
        'macro recall': 0.96832,
        'macro f1-score': 0.96836,
    },
    'tiny-fastformer': {
        'Size (MB)': 77.3,
        'Quantized Size (MB)': 19.7,
        'macro precision': 0.90655,
        'macro recall': 0.89819,
        'macro f1-score': 0.90196,
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
        path=PATH_SENTIMENT,
        s3_path=S3_PATH_SENTIMENT,
        module='sentiment',
        label=label,
        **kwargs
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
        * ``'fastformer'`` - FastFormer BASE parameters.
        * ``'tiny-fastformer'`` - FastFormer TINY parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.model.bert.MulticlassBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.MulticlassXLNET`.
        * if `fastformer` in model, will return `malaya.model.fastformer.MulticlassFastFormer`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.sentiment.available_transformer()`.'
        )
    return classification.transformer(
        module='sentiment-v2',
        label=label,
        model=model,
        quantized=quantized,
        **kwargs
    )
