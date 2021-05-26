from malaya.function import check_file
from malaya.model.tf import TrueCase
from malaya.path import PATH_TRUE_CASE, S3_PATH_TRUE_CASE
from malaya.supervised import transformer as load_transformer
from herpetologist import check_type

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.1,
        'Sequence Accuracy': 0.347,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 63.8,
        'Sequence Accuracy': 0.696,
    },
}


class Sacremoses:
    def __init__(self, mtr):
        self._mtr = mtr

    def true_case(self, strings):
        """
        True case strings.
        Example, "jom makan di us makanan di sana sedap" -> "jom makan di US makanan di sana sedap"

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[str]
        """
        results = []
        for string in strings:
            results.append(' '.join(self._mtr.truecase(string)))
        return results


def sacremoses(**kwargs):
    """
    Load True Case class using sacremoses library.

    Returns
    -------
    result : malaya.true_case.Sacremoses class
    """
    try:
        from sacremoses import MosesTruecaser
    except:
        raise ModuleNotFoundError(
            'sacremoses not installed. Please install it by `pip install sacremoses` and try again.'
        )
    check_file(
        PATH_TRUE_CASE['sacremoses'], S3_PATH_TRUE_CASE['sacremoses'], **kwargs
    )
    mtr = MosesTruecaser(PATH_TRUE_CASE['sacremoses']['model'])

    return Sacremoses(mtr = mtr)


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(_transformer_availability)


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load transformer encoder-decoder model to True Case.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'small'`` - Transformer SMALL parameters.
        * ``'base'`` - Transformer BASE parameters.

    quantized : bool, optional (default=False)
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

    return load_transformer.load(
        module = 'true-case',
        model = model,
        encoder = 'yttm',
        model_class = TrueCase,
        quantized = quantized,
        **kwargs,
    )
