from typing import List, Tuple, Callable


def validate_object_methods(object, methods, variable):
    if object is not None:
        if all([not hasattr(object, method) for method in methods]):
            s = ' or '.join([f'`{method}`' for method in methods])
            raise ValueError(f'{variable} must has {s} method')


def validate_object(object, method, variable):
    if object is not None:
        if not hasattr(object, method):
            raise ValueError(f'{variable} must has `{method}` method')


def validate_function(function, variable):
    if not isinstance(function, Callable) and function is not None:
        raise ValueError(f'{variable} must be a callable type or None')


def validate_stopwords(stopwords):
    if isinstance(stopwords, Callable):
        s = stopwords()
    else:
        s = stopwords

    return set(s)
