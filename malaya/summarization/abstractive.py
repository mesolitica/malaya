from malaya.supervised import t5 as t5_load
from malaya.supervised import bigbird as bigbird_load
from malaya.supervised import pegasus as pegasus_load
from malaya.model.t5 import Summarization as T5_Summarization
from malaya.model.bigbird import Summarization as BigBird_Summarization
from malaya.model.pegasus import Summarization as Pegasus_Summarization
from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
from herpetologist import check_type
import logging
import warnings

logger = logging.getLogger(__name__)

_transformer_availability = {
    'tiny-t5': {
        'Size (MB)': 208,
        'Quantized Size (MB)': 103,
        'ROUGE-1': 0.820415874,
        'ROUGE-2': 0.6233588777,
        'ROUGE-L': 0.418418003,
        'Suggested length': 1024,
    },
    'small-t5': {
        'Size (MB)': 355.6,
        'Quantized Size (MB)': 195,
        'ROUGE-1': 0.8354927651,
        'ROUGE-2': 0.6516714426,
        'ROUGE-L': 0.443235174,
        'Suggested length': 1024,
    },
    't5': {
        'Size (MB)': 1250,
        'Quantized Size (MB)': 481,
        'ROUGE-1': 0.840805448,
        'ROUGE-2': 0.6613555630,
        'ROUGE-L': 0.4529222596,
        'Suggested length': 1024,
    },
}

_huggingface_availability = {
    'mesolitica/finetune-summarization-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'ROUGE-1': 0.75721802,
        'ROUGE-2': 0.496729027,
        'ROUGE-L': 0.304021823,
        'Suggested length': 1024,
    },
    'mesolitica/finetune-summarization-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'ROUGE-1': 0.7132268255,
        'ROUGE-2': 0.470135011,
        'ROUGE-L': 0.366797009,
        'Suggested length': 1024,
    },
    'mesolitica/finetune-summarization-ms-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'ROUGE-1': 0.742572468,
        'ROUGE-2': 0.50196339,
        'ROUGE-L': 0.3741226432,
        'Suggested length': 1024,
    },
    'mesolitica/finetune-summarization-ms-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'ROUGE-1': 0.728116529,
        'ROUGE-2': 0.49656772621,
        'ROUGE-L': 0.376577199,
        'Suggested length': 1024,
    },
}


def _describe():
    logger.info('tested on translated validation set CNN Daily Mail, https://huggingface.co/datasets/mesolitica/translated-cnn-dailymail')
    logger.info('tested on translated test set Xwikis, https://huggingface.co/datasets/mesolitica/translated-xwikis')


def available_transformer():
    """
    List available transformer models.
    """
    warnings.warn(
        '`malaya.summarization.abstractive.available_transformer` is deprecated, use `malaya.summarization.abstractive.available_huggingface` instead', DeprecationWarning)

    _describe()
    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available huggingface models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


@check_type
def transformer(model: str = 'small-t5', quantized: bool = False, **kwargs):
    """
    Load Malaya transformer encoder-decoder model to generate a summary given a string.

    Parameters
    ----------
    model: str, optional (default='small-t5')
        Check available models at `malaya.summarization.abstractive.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `t5` in model, will return `malaya.model.t5.Summarization`.
        * if `bigbird` in model, will return `malaya.model.bigbird.Summarization`.
        * if `pegasus` in model, will return `malaya.model.pegasus.Summarization`.
    """
    warnings.warn(
        '`malaya.summarization.abstractive.transformer` is deprecated, use `malaya.summarization.abstractive.huggingface` instead', DeprecationWarning)

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.summarization.abstractive.available_transformer()`.'
        )

    if 't5' in model:
        return t5_load.load(
            module='abstractive-summarization-v2',
            model=model,
            model_class=T5_Summarization,
            quantized=quantized,
            **kwargs,
        )

    if 'bigbird' in model:
        return bigbird_load.load_pegasus(
            module='abstractive-summarization-v2',
            model=model,
            model_class=BigBird_Summarization,
            maxlen=_transformer_availability[model]['Suggested length'],
            quantized=quantized,
            **kwargs,
        )

    if 'pegasus' in model:
        return pegasus_load.load(
            module='abstractive-summarization-v2',
            model=model,
            model_class=Pegasus_Summarization,
            quantized=quantized,
            **kwargs,
        )


def huggingface(
    model: str = 'mesolitica/finetune-summarization-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to abstractive summarization.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-summarization-t5-small-standard-bahasa-cased')
        Check available models at `malaya.summarization.abstractive.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Summarization
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.summarization.abstractive.available_huggingface()`.'
        )
    return load_huggingface.load_summarization(model=model, initial_text='ringkasan: ', **kwargs)
