from malaya.model.tf import TrueCase
from malaya.supervised import transformer as load_transformer
from malaya.supervised import t5 as t5_load
from malaya.model.t5 import TrueCase as T5_TrueCase
from herpetologist import check_type
from malaya.function import describe_availability
import numpy as np
import logging

logger = logging.getLogger(__name__)

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.1,
        'CER': 0.0246012,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 63.8,
        'CER': 0.0146193,
        'Suggested length': 256,
    },
    'super-tiny-t5': {
        'Size (MB)': 81.8,
        'Quantized Size (MB)': 27.1,
        'CER': 0.0254679,
        'Suggested length': 256,
    },
    'super-super-tiny-t5': {
        'Size (MB)': 39.6,
        'Quantized Size (MB)': 12,
        'CER': 0.02533658,
        'Suggested length': 256,
    },
    '3x-super-tiny-t5': {
        'Size (MB)': 18.3,
        'Quantized Size (MB)': 4.46,
        'CER': 0.0487372,
        'Suggested length': 256,
    },
    '3x-super-tiny-t5-4k': {
        'Size (MB)': 5.03,
        'Quantized Size (MB)': 2.99,
        'CER': 0.0798906,
        'Suggested length': 256,
    }
}


class TrueCase_LM:
    def __init__(self, language_model):
        self._language_model = language_model

    def true_case(
        self,
        string: str,
        lookback: int = 3,
        lookforward: int = 3,
    ):
        """
        True case string input.

        Parameters
        ----------
        string: str
            Entire string, `word` must a word inside `string`.
        lookback: int, optional (default=3)
            N words on the left hand side.
            if put -1, will take all words on the left hand side.
            longer left hand side will take longer to compute.
        lookforward: int, optional (default=3)
            N words on the right hand side.
            if put -1, will take all words on the right hand side.
            longer right hand side will take longer to compute.

        Returns
        -------
        result: str
        """

        splitted = string.split()
        for index, word in enumerate(splitted):
            if lookback == -1:
                lookback_ = index
            elif lookback > index:
                lookback_ = index
            else:
                lookback_ = lookback

            if lookforward == -1:
                lookforward_ = 9999999
            else:
                lookforward_ = lookforward

            left_hand = splitted[index - lookback_: index]
            right_hand = splitted[index + 1: index + 1 + lookforward_]

            words = [word, word.lower(), word.upper(), word.title()]
            scores, strings = [], []
            for w in words:
                string_ = left_hand + [w] + right_hand
                score = self._language_model.score(' '.join(string_), bos=index == 0, eos=index == (len(splitted) - 1))
                scores.append(score)
                strings.append(string_)

            s = f'index: {index}, word: {word}, words: {words}, strings: {strings}, scores: {scores}'
            logger.debug(s)

            splitted[index] = words[np.argmin(scores)]

        return ' '.join(splitted)


def available_transformer():
    """
    List available transformer models.
    """
    logger.info('tested on generated dataset at https://github.com/huseinzol05/malaya/tree/master/session/true-case')

    return describe_availability(_transformer_availability)


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load transformer encoder-decoder model to True Case.

    Parameters
    ----------
    model: str, optional (default='base')
        Check available models at `malaya.true_case.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.TrueCase class
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.true_case.available_transformer()`.'
        )

    if 't5' in model:
        return t5_load.load(
            module='true-case',
            model=model,
            model_class=T5_TrueCase,
            quantized=quantized,
            **kwargs,
        )
    else:
        return load_transformer.load(
            module='true-case',
            model=model,
            encoder='yttm',
            model_class=TrueCase,
            quantized=quantized,
            **kwargs,
        )


def probability(language_model):
    """
    Use language model to True Case.

    Parameters
    ----------
    language_model: Callable
        must an object with `score` method.

    Returns
    -------
    result: malaya.true_case.TrueCase_LM class
    """
    return TrueCase_LM(language_model=language_model)
