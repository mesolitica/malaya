from malaya.path import PATH_SUMMARIZE, S3_PATH_SUMMARIZE
from herpetologist import check_type
import os

_t5_availability = {
    'small': [
        '122MB',
        'ROUGE-1: 0.33854',
        'ROUGE-2: 0.14588',
        'ROUGE-L: 0.23528',
    ],
    'base': [
        '448MB',
        'ROUGE-1: 0.34103',
        'ROUGE-2: 0.14994',
        'ROUGE-L: 0.23655',
    ],
}


def available_t5():
    """
    List available T5 models.
    """
    return _t5_availability


@check_type
def t5(model: str = 'base', **kwargs):

    """
    Load T5 model to generate a summarization given a string.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'base'`` - T5 Base parameters.
        * ``'small'`` - T5 Small parameters.

    Returns
    -------
    result: malaya.model.t5.SUMMARIZATION class
    """

    model = model.lower()
    if model not in _t5_availability:
        raise Exception(
            'model not supported, please check supported models from malaya.summarize.available_t5()'
        )
    path = PATH_SUMMARIZE['argmax']
    s3_path = S3_PATH_SUMMARIZE['argmax']

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

    from malaya.model.t5 import SUMMARIZATION

    return SUMMARIZATION(pred)
