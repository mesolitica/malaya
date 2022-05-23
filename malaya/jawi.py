from herpetologist import check_type
from typing import List


class DeepJawi(Abstract):
    def __init__(self, input_nodes, output_nodes, sess):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess

    def greedy_decoder(self, string, window: int = 4):
        """
        Convert to target string using greedy decoder.

        Parameters
        ----------
        string : str
        window: int, optional (default=4)
            maximum words to process for one sentence.
        beam_search : bool, (optional=False)
            If True, use beam search decoder, else use greedy decoder.

        Returns
        -------
        result: str
        """
        return self.convert(string, beam_search=False)

    def beam_decoder(self, string, window: int = 4):
        """
        Convert to target string using beam decoder

        Parameters
        ----------
        string : str
        window: int, optional (default=4)
            maximum words to process for one sentence.
        beam_search : bool, (optional=False)
            If True, use beam search decoder, else use greedy decoder.

        Returns
        -------
        result: str
        """
        return self.convert(string, beam_search=True)

    @check_type
    def convert(self, strings: str, window: int = 4, beam_search: bool = False):
        """
        Convert to target string.

        Parameters
        ----------
        string : str
        window: int, optional (default=4)
            maximum words to process for one sentence.
        beam_search : bool, (optional=False)
            If True, use beam search decoder, else use greedy decoder.

        Returns
        -------
        result: str
        """


@check_type
def rumi_to_jawi(quantized=False, **kwargs):
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


@check_type
def jawi_to_rumi(quantized=False, **kwargs):
    """
    Load LSTM + Bahdanau Attention Jawi to Rumi model.
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
