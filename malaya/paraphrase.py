from malaya.path import PATH_PARAPHRASE, S3_PATH_PARAPHRASE
from malaya.supervised import t5 as t5_load
from malaya.supervised import transformer as transformer_load
from herpetologist import check_type


_t5_availability = {
    'small': {
        'Size (MB)': 122,
        'Uncompressed Size (MB)': 355.6,
        'BLEU': 0.81801,
    },
    'base': {'Size (MB)': 448, 'Uncompressed Size (MB)': 1300, 'BLEU': 0.86698},
}
_transformer_availability = {
    'small': {'Size (MB)': 379, 'BLEU': 0.5534},
    'base': {'Size (MB)': 832, 'BLEU': 0.597},
}


def available_t5():
    """
    List available T5 models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _t5_availability, text = 'tested on 1k paraphrase texts.'
    )


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 1k paraphrase texts.'
    )


@check_type
def t5(model: str = 'base', compressed: bool = True, **kwargs):

    """
    Load T5 model to generate a paraphrase given a string.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'base'`` - T5 BASE parameters.
        * ``'small'`` - T5 SMALL parameters.

    compressed: bool, optional (default=True)
        Load compressed model, but this not able to utilize malaya-gpu function. 
        This only compressed model size, but when loaded into VRAM / RAM, size uncompressed and compressed are the same.
        We prefer un-compressed model due to compressed model prone to error.

    Returns
    -------
    result: malaya.model.t5.PARAPHRASE class
    """

    model = model.lower()
    if model not in _t5_availability:
        raise ValueError(
            'model not supported, please check supported models from malaya.paraphrase.available_t5()'
        )

    from malaya.model.t5 import PARAPHRASE

    return t5_load.load(
        path = PATH_PARAPHRASE,
        s3_path = S3_PATH_PARAPHRASE,
        model = model,
        model_class = PARAPHRASE,
        compressed = compressed,
        **kwargs,
    )


def transformer(model = 'base', **kwargs):
    """
    Load Malaya transformer encoder-decoder model to generate a paraphrase given a string.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'malaya-small'`` - Malaya Transformer SMALL parameters.
        * ``'malaya-base'`` - Malaya Transformer BASE parameters.

    Returns
    -------
    result: malaya.model.tf.PARAPHRASE class
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from malaya.paraphrase.available_transformer()'
        )

    from malaya.model.tf import PARAPHRASE

    return transformer_load.load_lm(
        path = PATH_PARAPHRASE['transformer'],
        s3_path = S3_PATH_PARAPHRASE['transformer'],
        model = model,
        model_class = PARAPHRASE,
        **kwargs,
    )
