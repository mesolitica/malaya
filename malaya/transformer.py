import tensorflow as tf
import logging
from herpetologist import check_type

_transformer_availability = {
    'bert': {'Size (MB)': 425.6, 'Description': 'Google BERT BASE parameters'},
    'tiny-bert': {
        'Size (MB)': 57.4,
        'Description': 'Google BERT TINY parameters',
    },
    'albert': {
        'Size (MB)': 48.6,
        'Description': 'Google ALBERT BASE parameters',
    },
    'tiny-albert': {
        'Size (MB)': 22.4,
        'Description': 'Google ALBERT TINY parameters',
    },
    'xlnet': {
        'Size (MB)': 446.6,
        'Description': 'Google XLNET BASE parameters',
    },
    'alxlnet': {
        'Size (MB)': 46.8,
        'Description': 'Malaya ALXLNET BASE parameters',
    },
    'electra': {
        'Size (MB)': 443,
        'Description': 'Google ELECTRA BASE parameters',
    },
    'small-electra': {
        'Size (MB)': 55,
        'Description': 'Google ELECTRA SMALL parameters',
    },
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(_transformer_availability)


@check_type
def load(model: str = 'electra', pool_mode: str = 'last', **kwargs):
    """
    Load transformer model.

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
        * ``'electra'`` - Google ELECTRA BASE parameters.
        * ``'small-electra'`` - Google ELECTRA SMALL parameters.

    pool_mode : str, optional (default='last')
        Model logits architecture supported. Only usable if model in ['xlnet', 'alxlnet']. Allowed values:

        * ``'last'`` - last of the sequence.
        * ``'first'`` - first of the sequence.
        * ``'mean'`` - mean of the sequence.
        * ``'attn'`` - attention of the sequence.

    Returns
    -------
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.transformers.bert.Model`.
        * if `xlnet` in model, will return `malaya.transformers.xlnet.Model`.
        * if `albert` in model, will return `malaya.transformers.albert.Model`.
        * if `electra` in model, will return `malaya.transformers.electra.Model`.
    """

    model = model.lower()
    pool_mode = pool_mode.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.transformer.available_transformer()`.'
        )

    if tf.executing_eagerly():
        logging.warning(
            'Load pretrained transformer model will disable eager execution.'
        )
        tf.compat.v1.disable_eager_execution()

    if model in ['bert', 'tiny-bert']:
        from malaya.transformers.bert import load

        return load(model=model, **kwargs)
    if model in ['albert', 'tiny-albert']:
        from malaya.transformers.albert import load

        return load(model=model, **kwargs)
    if model in ['xlnet']:
        from malaya.transformers.xlnet import load

        return load(model=model, pool_mode=pool_mode, **kwargs)

    if model in ['alxlnet']:
        from malaya.transformers.alxlnet import load

        return load(model=model, pool_mode=pool_mode, **kwargs)

    if model in ['electra', 'small-electra']:
        from malaya.transformers.electra import load

        return load(model=model, **kwargs)
