from malaya.supervised import t2t
from malaya.supervised.settings import jawi_left, jawi_right
from malaya.text.function import rumi_jawi_textcleaning
from herpetologist import check_type
from typing import List


@check_type
def deep_model(quantized: bool = False, **kwargs):
    """
    Load LSTM + Bahdanau Attention Rumi to Jawi model,
    256 filter size, 2 layers, character level.
    Original size 11MB, quantized size 2.92MB .

    CER on test set: 0.014847105998349451
    WER on test set: 0.06737832963079593

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
        module='rumi-jawi',
        left_dict=jawi_left,
        right_dict=jawi_right,
        cleaning=rumi_jawi_textcleaning,
        quantized=quantized,
        **kwargs,
    )
