from malaya.function import check_file
from malaya.model.alignment import Eflomal


def _eflomal(file, **kwargs):
    path = check_file(
        file=file,
        module='eflomal-alignment',
        keys={'model': 'model.priors'},
        quantized=False,
        **kwargs,
    )
    return Eflomal(priors_filename=path['model'])


def eflomal(**kwargs):
    """
    """
    return _eflomal(file='en-ms', **kwargs)


def transformer():
    """
    """
