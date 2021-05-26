from herpetologist import check_type

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.4,
        'BLEU': 0.512,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 82.7,
        'BLEU': 0.696,
        'Suggested length': 256,
    },
    'large': {
        'Size (MB)': 817,
        'Quantized Size (MB)': 244,
        'BLEU': 0.699,
        'Suggested length': 256,
    },
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 200k test set.'
    )


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load transformer to generate knowledge graph from graphs.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'small'`` - Transformer SMALL parameters.
        * ``'base'`` - Transformer BASE parameters.
        * ``'large'`` - Transformer LARGE parameters.
        * ``'bigbird'`` - BigBird BASE parameters.
        * ``'small-bigbird'`` - BigBird SMALL parameters.
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.KnowledgeGraph class
    """
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.knowledge_graph.available_transformer()`.'
        )
