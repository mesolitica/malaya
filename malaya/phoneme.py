from malaya.supervised import t2t
from malaya.supervised.settings import phoneme_left, phoneme_right
from malaya.text.function import phoneme_textcleaning
from herpetologist import check_type
from typing import List

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.1,
    },
    'tiny': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.1,
    },
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(_transformer_availability)


@check_type
def deep_model(quantized: bool = False, **kwargs):
    """
    Load LSTM + Bahdanau Attention phonetic model, 
    originally from https://prpm.dbp.gov.my/ Glosari Dialek.

    Original size 10.4MB, quantized size 2.77MB .

    Parameters
    ----------
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.Seq2SeqLSTM class
    """
    return t2t.load_lstm(
        module='phoneme',
        left_dict=phoneme_left,
        right_dict=phoneme_right,
        cleaning=phoneme_textcleaning,
        quantized=quantized,
        **kwargs,
    )


def transformer(model='small', quantized=False, **kwargs):
    """
    Load transformer encoder-decoder phonetic model, 
    originally from https://prpm.dbp.gov.my/ Glosari Dialek.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'small'`` - Transformer SMALL parameters.
        * ``'tiny'`` - Transformer TINY parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.TransformerChar class
    """
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.phoneme.available_transformer()`.'
        )
    return load_transformer.load_char(
        module='phoneme',
        model=model,
        encoder='yttm',
        left_dict=phoneme_left,
        right_dict=phoneme_right,
        cleaning=phoneme_textcleaning,
        quantized=quantized,
        **kwargs,
    )
