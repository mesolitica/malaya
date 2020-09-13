from herpetologist import check_type

_transformer_availability = {
    'small': ['42.7MB', 'sequence accuracy: 0.142'],
    'base': ['234MB', 'sequence accuracy: 0.696'],
}


def available_transformer():
    """
    List available transformer models.
    """
    return _transformer_availability


@check_type
def transformer(model: str = 'base', **kwargs):
    model = model.lower()
    if model not in _transformer_availability:
        raise Exception(
            'model not supported, please check supported models from malaya.true_case.available_transformer()'
        )
