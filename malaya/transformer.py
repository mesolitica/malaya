import tensorflow as tf
from herpetologist import check_type
from malaya.function import describe_availability
from malaya.supervised import huggingface as load_huggingface
import logging
import warnings

logger = logging.getLogger(__name__)

_transformer_availability = {
    'bert': {
        'Size (MB)': 425.6,
        'Description': 'Google BERT BASE parameters'
    },
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

_huggingface_availability = {
    'mesolitica/roberta-base-bahasa-cased': {
        'Size (MB)': 443,
    },
    'mesolitica/roberta-tiny-bahasa-cased': {
        'Size (MB)': 66.1,
    },
    'mesolitica/bert-base-standard-bahasa-cased': {
        'Size (MB)': 443,
    },
    'mesolitica/bert-tiny-standard-bahasa-cased': {
        'Size (MB)': 66.1,
    },
    'mesolitica/roberta-base-standard-bahasa-cased': {
        'Size (MB)': 443,
    },
    'mesolitica/roberta-tiny-standard-bahasa-cased': {
        'Size (MB)': 66.1,
    },
    'mesolitica/electra-base-generator-bahasa-cased': {
        'Size (MB)': 140,
    },
    'mesolitica/electra-small-generator-bahasa-cased': {
        'Size (MB)': 19.3,
    },
    'mesolitica/finetune-mnli-t5-super-tiny-standard-bahasa-cased': {
        'Size (MB)': 50.7,
    },
    'mesolitica/finetune-mnli-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
    },
    'mesolitica/finetune-mnli-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
    },
    'mesolitica/finetune-mnli-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
    }
}


def available_transformer():
    """
    List available transformer models.
    """

    warnings.warn(
        '`malaya.transformer.available_transformer` is deprecated, use `malaya.transformer.available_huggingface` instead', DeprecationWarning)

    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available huggingface models.
    """

    return describe_availability(_huggingface_availability)


@check_type
def load(model: str = 'electra', pool_mode: str = 'last', **kwargs):
    """
    Load transformer model.

    Parameters
    ----------
    model: str, optional (default='bert')
        Check available models at `malaya.transformer.available_transformer()`.
    pool_mode: str, optional (default='last')
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

    warnings.warn('`malaya.transformer.load` is deprecated, use `malaya.transformer.huggingface` instead', DeprecationWarning)

    model = model.lower()
    pool_mode = pool_mode.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.transformer.available_transformer()`.'
        )

    if tf.executing_eagerly():
        logger.warning(
            f'Load pretrained transformer {model} model will disable eager execution.'
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


def huggingface(
    model: str = 'mesolitica/electra-base-generator-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load transformer model.

    Parameters
    ----------
    model: str, optional (default='mesolitica/electra-base-generator-bahasa-cased')
        Check available models at `malaya.transformer.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.
    """

    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.transformer.available_huggingface()`.'
        )

    return load_huggingface.load_transformer(model=model, **kwargs)
