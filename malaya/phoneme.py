from malaya.supervised import t2t
from malaya.supervised.settings import phoneme_left, phoneme_right
from malaya.supervised.settings import phoneme_ipa_left, phoneme_ipa_right
from malaya.text.function import phoneme_textcleaning
from herpetologist import check_type
from typing import List


@check_type
def deep_model_dbp(quantized: bool = False, **kwargs):
    """
    Load LSTM + Bahdanau Attention phonetic model,
    256 filter size, 2 layers, character level.
    original data from https://prpm.dbp.gov.my/ Glosari Dialek.

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


def deep_model_ipa(quantized: bool = False, **kwargs):
    """
    Load LSTM + Bahdanau Attention phonetic model,
    256 filter size, 2 layers, character level. 
    Original data from https://github.com/open-dict-data/ipa-dict/blob/master/data/ma.txt

    Original size 10.4MB, quantized size 2.77MB .

    Parameters
    ----------
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.Seq2SeqLSTM_Split class
    """
    return t2t.load_lstm(
        module='phoneme-ipa',
        left_dict=phoneme_ipa_left,
        right_dict=phoneme_ipa_right,
        cleaning=phoneme_textcleaning,
        split=True,
        quantized=quantized,
        **kwargs,
    )
