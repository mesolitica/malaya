from malaya.alignment.en_ms import _eflomal
from typing import Callable


def eflomal(preprocessing_func: Callable = None, **kwargs):
    """
    load eflomal word alignment for MS-EN. Model size around ~300MB.

    Parameters
    ----------
    preprocessing_func: Callable, optional (default=None)
        preprocessing function to call during loading prior file.
        Using `malaya.text.function.replace_punct` able to reduce ~30% of memory usage.

    Returns
    -------
    result: malaya.model.alignment.Eflomal
    """

    return _eflomal(preprocessing_func=preprocessing_func, file='ms-en', **kwargs)
