from herpetologist import check_type
from typing import List


@check_type
def deep_model(quantized=False, **kwargs):
    """
    Load LSTM + Bahdanau Attention Rumi to Jawi model.
    Original size 41.6MB, quantized size 10.6MB .

    Parameters
    ----------
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.jawi.DeepJawi class
    """


def transformer(model='small', quantized=False, **kwargs):
    pass
