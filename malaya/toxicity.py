from malaya.supervised import classification
from malaya.path import PATH_TOXIC, S3_PATH_TOXIC
from herpetologist import check_type

label = [
    'severe toxic',
    'obscene',
    'identity attack',
    'insult',
    'threat',
    'asian',
    'atheist',
    'bisexual',
    'buddhist',
    'christian',
    'female',
    'heterosexual',
    'indian',
    'homosexual, gay or lesbian',
    'intellectual or learning disability',
    'male',
    'muslim',
    'other disability',
    'other gender',
    'other race or ethnicity',
    'other religion',
    'other sexual orientation',
    'physical disability',
    'psychiatric or mental illness',
    'transgender',
    'malay',
    'chinese',
]

_transformer_availability = {
    'bert': {
        'Size (MB)': 425.6,
        'Quantized Size (MB)': 111,
        'micro precision': 0.86098,
        'micro recall': 0.77313,
        'micro f1-score': 0.81469,
    },
    'tiny-bert': {
        'Size (MB)': 57.4,
        'Quantized Size (MB)': 15.4,
        'micro precision': 0.83535,
        'micro recall': 0.79611,
        'micro f1-score': 0.81526,
    },
    'albert': {
        'Size (MB)': 48.6,
        'Quantized Size (MB)': 12.8,
        'micro precision': 0.86054,
        'micro recall': 0.76973,
        'micro f1-score': 0.81261,
    },
    'tiny-albert': {
        'Size (MB)': 22.4,
        'Quantized Size (MB)': 5.98,
        'micro precision': 0.83535,
        'micro recall': 0.79611,
        'micro f1-score': 0.81526,
    },
    'xlnet': {
        'Size (MB)': 446.6,
        'Quantized Size (MB)': 118,
        'micro precision': 0.77904,
        'micro recall': 0.83829,
        'micro f1-score': 0.80758,
    },
    'alxlnet': {
        'Size (MB)': 46.8,
        'Quantized Size (MB)': 13.3,
        'micro precision': 0.83376,
        'micro recall': 0.80221,
        'micro f1-score': 0.81768,
    },
}


def available_transformer():
    """
    List available transformer toxicity analysis models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 20% test set.'
    )


def multinomial(**kwargs):
    """
    Load multinomial toxicity model.

    Returns
    -------
    result : malaya.model.ml.MultilabelBayes class
    """
    return classification.multinomial(
        PATH_TOXIC, S3_PATH_TOXIC, 'toxicity', label, sigmoid=True, **kwargs
    )


@check_type
def transformer(model: str = 'xlnet', quantized: bool = False, **kwargs):
    """
    Load Transformer toxicity model.

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

        * if `bert` in model, will return `malaya.model.bert.SigmoidBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.SigmoidXLNET`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.toxicity.available_transformer()`.'
        )

    return classification.transformer(
        module='toxicity',
        label=label,
        model=model,
        sigmoid=True,
        quantized=quantized,
        **kwargs
    )
