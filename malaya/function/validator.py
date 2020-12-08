from typing import List, Tuple, Callable
from herpetologist import recursive_check


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
    if (
        not isinstance(stopwords, Callable)
        and not recursive_check(stopwords, List[str])
        and not recursive_check(stopwords, Tuple[str])
    ):
        raise ValueError(
            'stopwords must be a callable type or a List[str] or Tuple[str]'
        )
    if isinstance(stopwords, Callable):
        s = stopwords()
        if not recursive_check(s, List[str]) and not recursive_check(
            s, Tuple[str]
        ):
            raise ValueError(
                'stopwords must returned a List[str] or Tuple[str]'
            )
    else:
        s = stopwords

    return set(s)
