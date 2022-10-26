from malaya.supervised import t5 as t5_load
from malaya.model.t5 import Generator
from malaya.function import describe_availability

_transformer_availability = {
    't5': {'Size (MB)': 1250, 'Quantized Size (MB)': 481, 'Maximum Length': 1024},
    'small-t5': {'Size (MB)': 355.6, 'Quantized Size (MB)': 195, 'Maximum Length': 1024},
}


def available_transformer():
    """
    List available transformer models.
    """

    return describe_availability(_transformer_availability)


def transformer(model: str = 't5', quantized: bool = False, **kwargs):
    """
    Load Transformer model to generate a string given a isu penting.

    Parameters
    ----------
    model: str, optional (default='base')
        Check available models at `malaya.generator.isi_penting.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.t5.Generator class
    """

    model = model.lower()
    if model not in _isi_penting_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.generator.isi_penting.available_transformer()`.'
        )

    return t5_load.load(
        module='generator',
        model=model,
        model_class=Generator,
        quantized=quantized,
        **kwargs,
    )
