from malaya.function import check_file
from malaya.model.alignment import Eflomal
from typing import Callable


def _eflomal(preprocessing_func, file, **kwargs):
    path = check_file(
        file=file,
        module='eflomal-alignment',
        keys={'model': 'model.priors'},
        quantized=False,
        **kwargs,
    )
    return Eflomal(preprocessing_func=preprocessing_func, priors_filename=path['model'])


def eflomal(alignment: str = 'ms-en', preprocessing_func: Callable = None, **kwargs):
    """
    load https://github.com/robertostling/eflomal word alignment.
    
    Parameters
    ----------
    alignment: str, optional (default='ms-en')
        Alignment type, only accept:
            * `ms-en`, size around ~240MB.
            * `en-ms`, size around ~301MB.
    preprocessing_func: Callable, optional (default=None)
        preprocessing function to call during loading prior file.
        Using `malaya.text.function.replace_punct` able to reduce ~30% of memory usage.

    Returns
    -------
    result: malaya.model.alignment.Eflomal
    """

    acceptable = ['ms-en', 'en-ms']
    alignment = alignment.lower()
    if alignment not in acceptable:
        raise ValueError(f'`aligment` only accept {acceptable}')

    return _eflomal(preprocessing_func=preprocessing_func, file=alignment, **kwargs)
