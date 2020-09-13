from malaya.path import PATH_PARAPHRASE, S3_PATH_PARAPHRASE
from malaya.function import check_file, load_graph, generate_session
from malaya.supervised import t5 as t5_load
import os
import tensorflow as tf
from herpetologist import check_type


_t5_availability = {
    'small': ['122MB', '*BLEU: 0.953'],
    'base': ['448MB', '*BLEU: 0.953'],
}
_transformer_availability = {
    'tiny': ['18.4MB', 'BLEU: 0.594'],
    'small': ['43MB', 'BLEU: 0.737'],
    'base': ['234MB', 'BLEU: 0.792'],
    'tiny-bert': ['60.6MB', 'BLEU: 0.609'],
    'bert': ['449MB', 'BLUE: 0.696'],
}


def available_t5():
    """
    List available T5 models.
    """
    return _t5_availability


def available_transformer():
    """
    List available transformer models.
    """
    return _transformer_availability


@check_type
def t5(model: str = 'base', compressed: bool = True, **kwargs):

    """
    Load T5 model to generate a paraphrase given a string.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'base'`` - T5 BASE parameters.
        * ``'small'`` - T5 SMALL parameters.

    compressed: bool, optional (default=True)
        Load compressed model, but this not able to utilize malaya-gpu function. 
        This only compressed model size, but when loaded into VRAM / RAM, size uncompressed and compressed are the same.
        We prefer un-compressed model due to compressed model prone to error.

    Returns
    -------
    result: malaya.model.t5.PARAPHRASE class
    """

    model = model.lower()
    if model not in _t5_availability:
        raise Exception(
            'model not supported, please check supported models from malaya.paraphrase.available_t5()'
        )

    from malaya.model.t5 import PARAPHRASE

    return t5_load.load(
        path = PATH_PARAPHRASE,
        s3_path = S3_PATH_PARAPHRASE,
        model = model,
        model_class = PARAPHRASE,
        compressed = compressed,
        **kwargs,
    )


def transformer(model = 'base', **kwargs):
    """
    Load Malaya transformer encoder-decoder model to generate a paraphrase given a string.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'malaya-small'`` - Malaya Transformer SMALL parameters.
        * ``'malaya-base'`` - Malaya Transformer BASE parameters.

    Returns
    -------
    result: malaya.model.tf.PARAPHRASE class
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise Exception(
            'model not supported, please check supported models from malaya.paraphrase.available_transformer()'
        )

    path = PATH_PARAPHRASE['transformer']
    s3_path = S3_PATH_PARAPHRASE['transformer']

    check_file(path[model], s3_path[model], **kwargs)
    g = load_graph(path[model]['model'], **kwargs)

    from malaya.text.t2t import text_encoder
    from malaya.model.tf import PARAPHRASE

    encoder = text_encoder.SubwordTextEncoder(path[model]['vocab'])
    return PARAPHRASE(
        g.get_tensor_by_name('import/Placeholder:0'),
        g.get_tensor_by_name('import/greedy:0'),
        g.get_tensor_by_name('import/beam:0'),
        generate_session(graph = g, **kwargs),
        encoder,
    )
