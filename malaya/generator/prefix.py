import tensorflow as tf
from malaya.function import describe_availability
from herpetologist import check_type
from malaya.supervised import gpt2 as gpt2_load
from malaya.supervised import huggingface as load_huggingface
import logging
import warnings

logger = logging.getLogger(__name__)


@check_type
def babble_tf(
    string: str,
    model,
    generate_length: int = 30,
    leed_out_len: int = 1,
    temperature: float = 1.0,
    top_k: int = 100,
    burnin: int = 15,
    batch_size: int = 5,
):
    """
    Use pretrained malaya transformer models to generate a string given a prefix string.
    https://github.com/nyu-dl/bert-gen, https://arxiv.org/abs/1902.04094

    Parameters
    ----------
    string: str
    model: object
        transformer interface object. Right now only supported BERT, ALBERT and ELECTRA.
    generate_length: int, optional (default=256)
        length of sentence to generate.
    leed_out_len: int, optional (default=1)
        length of extra masks for each iteration.
    temperature: float, optional (default=1.0)
        logits * temperature.
    top_k: int, optional (default=100)
        k for top-k sampling.
    burnin: int, optional (default=15)
        for the first burnin steps, sample from the entire next word distribution, instead of top_k.
    batch_size: int, optional (default=5)
        generate sentences size of batch_size.

    Returns
    -------
    result: List[str]
    """

    if not hasattr(model, 'samples'):
        raise ValueError('model must have `samples` attribute')
    if generate_length < 10:
        raise ValueError('generate_length must bigger than 10')
    if not 0 < temperature <= 1.0:
        raise ValueError('temperature must, 0 < temperature <= 1.0')
    if not top_k > 0:
        raise ValueError('top_k must be bigger than 0')
    if not burnin > 0:
        raise ValueError('burnin must be bigger than 0')
    if leed_out_len >= generate_length:
        raise ValueError('leed_out_len must be smaller than generate_length')
    if burnin >= generate_length:
        raise ValueError('burnin must be smaller than generate_length')

    from malaya.transformers.babble import sequential_generation

    if tf.executing_eagerly():
        logger.warning(
            'malaya.generator.babble will disable eager execution.'
        )
        tf.compat.v1.disable_eager_execution()

    return sequential_generation(
        string,
        model,
        batch_size=batch_size,
        max_len=generate_length,
        leed_out_len=leed_out_len,
        temperature=temperature,
        top_k=top_k,
        burnin=burnin,
    )


_transformer_availability = {
    '117M': {
        'Size (MB)': 499,
        'Quantized Size (MB)': 126,
        'Perplexity': 6.232461
    },
    '345M': {
        'Size (MB)': 1420,
        'Quantized Size (MB)': 357,
        'Perplexity': 6.1040115
    },
}

_huggingface_availability = {
    'mesolitica/gpt2-117m-bahasa-cased-v2': {
        'Size (MB)': 454,
    },
    'mesolitica/gpt2-355m-bahasa-cased': {
        'Size (MB)': 454,
    },
}


def _describe():
    logger.info('calculate perplexity on never seen malay karangan, ')


def available_transformer():
    """
    List available gpt2 generator models.
    """
    warnings.warn(
        '`malaya.generator.prefix.available_transformer` is deprecated, use `malaya.generator.prefix.available_huggingface` instead', DeprecationWarning)

    _describe()
    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available gpt2 generator models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


@check_type
def transformer(model: str = '345M', quantized: bool = False, **kwargs):
    """
    Load GPT2 model to generate a string given a prefix string.

    Parameters
    ----------
    model: str, optional (default='345M')
        Check available models at `malaya.generator.prefix.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.GPT2 class
    """

    warnings.warn(
        '`malaya.generator.prefix.transformer` is deprecated, use `malaya.generator.prefix.huggingface` instead', DeprecationWarning)

    model = model.upper()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.generator.prefix.available_transformer()`.'
        )

    return gpt2_load.load(
        model=model,
        quantized=quantized,
        **kwargs,
    )


def huggingface(
    model: str = 'mesolitica/gpt2-117m-bahasa-cased-v2',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Prefix language model.

    Parameters
    ----------
    model: str, optional (default='mesolitica/gpt2-117m-bahasa-cased-v2')
        Check available models at `malaya.generator.prefix.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Prefix class
    """

    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.generator.prefix.available_huggingface()`.'
        )

    return load_huggingface.load_prefix(model=model, **kwargs)
