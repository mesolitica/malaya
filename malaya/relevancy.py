from malaya.supervised import classification
from herpetologist import check_type

label = ['not relevant', 'relevant']

_transformer_availability = {
    'bert': {
        'Size (MB)': 425.6,
        'Quantized Size (MB)': 111,
        'macro precision': 0.89320,
        'macro recall': 0.89195,
        'macro f1-score': 0.89256,
        'max length': 512,
    },
    'tiny-bert': {
        'Size (MB)': 57.4,
        'Quantized Size (MB)': 15.4,
        'macro precision': 0.87179,
        'macro recall': 0.86324,
        'macro f1-score': 0.86695,
        'max length': 512,
    },
    'albert': {
        'Size (MB)': 48.6,
        'Quantized Size (MB)': 12.8,
        'macro precision': 0.89798,
        'macro recall': 0.86008,
        'macro f1-score': 0.87209,
        'max length': 512,
    },
    'tiny-albert': {
        'Size (MB)': 22.4,
        'Quantized Size (MB)': 5.98,
        'macro precision': 0.82157,
        'macro recall': 0.83410,
        'macro f1-score': 0.82416,
        'max length': 512,
    },
    'xlnet': {
        'Size (MB)': 446.6,
        'Quantized Size (MB)': 118,
        'macro precision': 0.92707,
        'macro recall': 0.92103,
        'macro f1-score': 0.92381,
        'max length': 512,
    },
    'alxlnet': {
        'Size (MB)': 46.8,
        'Quantized Size (MB)': 13.3,
        'macro precision': 0.91135,
        'macro recall': 0.90446,
        'macro f1-score': 0.90758,
        'max length': 512,
    },
    'bigbird': {
        'Size (MB)': 458,
        'Quantized Size (MB)': 116,
        'macro precision': 0.88093,
        'macro recall': 0.86832,
        'macro f1-score': 0.87352,
        'max length': 1024,
    },
    'tiny-bigbird': {
        'Size (MB)': 65,
        'Quantized Size (MB)': 16.9,
        'macro precision': 0.86558,
        'macro recall': 0.85871,
        'macro f1-score': 0.86176,
        'max length': 1024,
    },
}


def available_transformer():
    """
    List available transformer relevancy analysis models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 20% test set.'
    )


@check_type
def transformer(model: str = 'xlnet', quantized: bool = False, **kwargs):
    """
    Load Transformer relevancy model.

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
        * ``'bigbird'`` - Google BigBird BASE parameters.
        * ``'tiny-bigbird'`` - Malaya BigBird BASE parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.model.bert.MulticlassBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.MulticlassXLNET`.
        * if `bigbird` in model, will return `malaya.model.xlnet.MulticlassBigBird`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.relevancy.available_transformer()`.'
        )
    return classification.transformer(
        module='relevancy',
        label=label,
        model=model,
        quantized=quantized,
        **kwargs
    )
