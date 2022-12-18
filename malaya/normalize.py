from malaya.normalizer.rules import load
import warnings


def normalizer(*args, **kwargs):
    warnings.warn(
        '`malaya.normalize.normalizer` is deprecated, use `malaya.normalizer.rules.load` instead', DeprecationWarning)
    return load(*args, **kwargs)
