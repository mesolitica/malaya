from herpetologist import check_type

_availability = {
    'bert': ['base', 'small'],
    'xlnet': ['base'],
    'alxlnet': ['base'],
    'albert': ['base', 'large'],
}


def available_model():
    """
    List available transformer models.
    """
    return _availability


@check_type
def load(
    size: int = 'base',
    model: int = 'xlnet',
    pool_mode: int = 'last',
    validate: bool = True,
):

    """
    Load transformer model.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - BERT architecture from google.
        * ``'xlnet'`` - XLNET architecture from google.
        * ``'alxlnet'`` - XLNET architecture from google + Malaya.
        * ``'albert'`` - ALBERT architecture from google.
    size : str, optional (default='base')
        Model size supported. Allowed values:

        * ``'base'`` - BASE size.
        * ``'small'`` - SMALL size.
        * ``'large'`` - LARGE size.
    pool_mode : str, optional (default='last')
        Model logits architecture supported. Only usable if model = 'xlnet'. Allowed values:

        * ``'last'`` - last of the sequence.
        * ``'first'`` - first of the sequence.
        * ``'mean'`` - mean of the sequence.
        * ``'attn'`` - attention of the sequence.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    TRANSFORMER: malaya.transformer class
    """

    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(size, str):
        raise ValueError('size must be a string')
    if not isinstance(pool_mode, str):
        raise ValueError('pool_mode must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

    model = model.lower()
    size = size.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.transformer.available_model()'
        )
    if size not in _availability[model]:
        raise Exception(
            'size not supported, please check supported models from malaya.transformer.available_model()'
        )
    if model == 'bert':
        from ._transformer import _bert

        return _bert.bert(model = size, validate = validate)
    if model == 'albert':
        from ._transformer import _albert

        return _albert.albert(model = size, validate = validate)
    if model == 'xlnet':
        from ._transformer import _xlnet

        return _xlnet.xlnet(
            model = size, pool_mode = pool_mode, validate = validate
        )
    if model == 'alxlnet':
        from ._transformer import _alxlnet

        return _alxlnet.alxlnet(
            model = size, pool_mode = pool_mode, validate = validate
        )
