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


def eflomal(preprocessing_func: Callable = None, **kwargs):
    """
    load eflomal word alignment for EN-MS. Model size around ~300MB.
    Parameters
    ----------
    preprocessing_func: Callable, optional (default=None)
        preprocessing function to call during loading prior file.
        Using `malaya.text.function.replace_punct` able to reduce ~30% of memory usage.
    Returns
    -------
    result: malaya.model.alignment.Eflomal
    """

    try:
        from eflomal import read_text, write_text, align
    except BaseException:
        raise ModuleNotFoundError(
            'eflomal not installed. Please install it from https://github.com/robertostling/eflomal for Linux / Windows or https://github.com/huseinzol05/maceflomal for Mac and try again.'
        )
    return _eflomal(preprocessing_func=preprocessing_func, file='en-ms', **kwargs)
