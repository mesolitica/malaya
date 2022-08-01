from packaging import version

try:
    import malaya_boilerplate
except Exception as e:
    raise ModuleNotFoundError(
        'malaya-boilerplate not installed. Please install it by `pip3 install malaya-boilerplate` and try again.')

if version.parse(malaya_boilerplate.__version__) < version.parse('0.0.21'):
    raise ModuleNotFoundError(
        'malaya-boilerplate must version >= 0.0.21, please install it by `pip3 install malaya-boilerplate -U` and try again')

import malaya_boilerplate.train


def __getattr__(value):
    return getattr(malaya_boilerplate.train, value)
