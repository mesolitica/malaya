from malaya.path import PATH_PARAPHRASE, S3_PATH_PARAPHRASE
from malaya.function import check_file, load_graph, generate_session
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
def t5(model: str = 'base', **kwargs):

    """
    Load T5 model to generate a paraphrase given a string.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'base'`` - T5 Base parameters.
        * ``'small'`` - T5 Small parameters.

    Returns
    -------
    result: malaya.model.t5.PARAPHRASE class
    """

    model = model.lower()
    if model not in _t5_availability:
        raise Exception(
            'model not supported, please check supported models from malaya.paraphrase.available_t5()'
        )

    path = PATH_PARAPHRASE['t5']
    s3_path = S3_PATH_PARAPHRASE['t5']

    from malaya.function import check_file

    try:
        import tensorflow_text
        import tf_sentencepiece
        import tensorflow as tf
    except:
        raise Exception(
            'tensorflow-text and tf-sentencepiece not installed. Please install it by `pip install tensorflow-text tf-sentencepiece` and try again. Also, make sure tensorflow-text version same as tensorflow version.'
        )

    check_file(path[model]['model'], s3_path[model], **kwargs)

    if not os.path.exists(path[model]['directory'] + 'saved_model.pb'):
        import tarfile

        with tarfile.open(path[model]['model']['model']) as tar:
            tar.extractall(path = path[model]['path'])

    sess = tf.InteractiveSession()
    meta_graph_def = tf.compat.v1.saved_model.load(
        sess, ['serve'], path[model]['directory']
    )
    signature_def = meta_graph_def.signature_def['serving_default']
    pred = lambda x: sess.run(
        fetches = signature_def.outputs['outputs'].name,
        feed_dict = {signature_def.inputs['input'].name: x},
    )

    from malaya.model.t5 import PARAPHRASE

    return PARAPHRASE(pred)


def transformer(model = 'base', **kwargs):
    """
    Load transformer encoder-decoder model to generate a paraphrase given a string.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'tiny'`` - transformer Tiny parameters.
        * ``'small'`` - transformer Small parameters.
        * ``'base'`` - transformer Base parameters.
        * ``'tiny-bert'`` - BERT-BERT Tiny parameters.
        * ``'bert'`` - BERT-BERT Base parameters.

    Returns
    -------
    result: malaya.model.tf.PARAPHRASE class
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise Exception(
            'model not supported, please check supported models from malaya.paraphrase.available_transformer()'
        )

    if 'bert' in model:

        path = PATH_PARAPHRASE[model]
        s3_path = S3_PATH_PARAPHRASE[model]

        check_file(path, s3_path, **kwargs)
        g = load_graph(path['model'])

        if model in ['bert', 'tiny-bert']:
            from malaya.text.bpe import sentencepiece_tokenizer_bert

            tokenizer = sentencepiece_tokenizer_bert(
                path['tokenizer'], path['vocab']
            )

        from malaya.model.bert import PARAPHRASE_BERT

        return PARAPHRASE_BERT(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
            input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
            logits = g.get_tensor_by_name('import/greedy:0'),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
        )

    else:
        path = PATH_PARAPHRASE['transformer']
        s3_path = S3_PATH_PARAPHRASE['transformer']

        check_file(path[model], s3_path[model], **kwargs)
        g = load_graph(path[model]['model'])

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
