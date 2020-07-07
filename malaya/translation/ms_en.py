from malaya.path import PATH_TRANSLATION, S3_PATH_TRANSLATION
from malaya.function import check_file, load_graph, generate_session

_transformer_availability = {
    'small': ['42.5MB', 'BLEU: 0.626'],
    'base': ['234MB', 'BLEU: 0.792'],
    'large': ['815MB', 'BLEU: 0.714'],
}


def available_transformer():
    """
    List available transformer models.
    """
    return _transformer_availability


def transformer(model = 'base', **kwargs):
    """
    Load transformer encoder-decoder model to translate MS-to-EN.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'small'`` - transformer Small parameters.
        * ``'base'`` - transformer Base parameters.
        * ``'large'`` - transformer Large parameters.

    Returns
    -------
    result: malaya.model.tf.TRANSLATION class
    """
    model = model.lower()
    if model not in _transformer_availability:
        raise Exception(
            'model not supported, please check supported models from malaya.translation.ms_en.available_transformer()'
        )

    path = PATH_TRANSLATION['ms-en']
    s3_path = S3_PATH_TRANSLATION['ms-en']

    check_file(path[model], s3_path[model], **kwargs)
    g = load_graph(path[model]['model'])

    from malaya.text.t2t import text_encoder
    from malaya.model.tf import TRANSLATION

    encoder = text_encoder.SubwordTextEncoder(path[model]['vocab'])
    return TRANSLATION(
        g.get_tensor_by_name('import/Placeholder:0'),
        g.get_tensor_by_name('import/greedy:0'),
        g.get_tensor_by_name('import/beam:0'),
        generate_session(graph = g, **kwargs),
        encoder,
    )
