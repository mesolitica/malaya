import pickle
from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_boilerplate.huggingface import download_files
from malaya.function import describe_availability
from malaya.model.ml import LanguageDetection
from malaya.model.tf import DeepLang
from malaya.model.rules import LanguageDict
from malaya.text.bpe import YTTMEncoder
from malaya.path import (
    LANGUAGE_DETECTION_BOW,
    LANGUAGE_DETECTION_VOCAB,
)
import logging

logger = logging.getLogger(__name__)

lang_labels_v1 = {
    0: 'eng',
    1: 'ind',
    2: 'malay',
    3: 'manglish',
    4: 'other',
    5: 'rojak',
}

lang_labels_v2 = {
    0: 'eng',
    1: 'ind',
    2: 'malay',
    3: 'mandarin',
    4: 'manglish',
    5: 'other',
    6: 'rojak',
}

label_v1 = list(lang_labels_v1.values())
label_v2 = list(lang_labels_v2.values())

_fasttext_availability = {
    'mesolitica/fasttext-language-detection-v1': {
        'Size (MB)': 353,
        'Quantized Size (MB)': 31.1,
        'Label': lang_labels_v1,
    },
    'mesolitica/fasttext-language-detection-v2': {
        'Size (MB)': 425.6,
        'Quantized Size (MB)': 111,
        'Label': lang_labels_v2,
    }
}


def available_fasttext():
    """
    List available fasttext language detection..
    """

    logger.info('trained on 90% dataset, tested on another 10% test set, dataset at https://github.com/huseinzol05/malaya/blob/master/session/relevancy/download-data.ipynb')

    return describe_availability(_fasttext_availability)


def fasttext(
    model: str = 'mesolitica/fasttext-language-detection-v2',
    quantized: bool = True,
    **kwargs,
):
    """
    Load Fasttext language detection model.

    Parameters
    ----------
    model: str, optional (default='mesolitica/fasttext-language-detection-v2')
    quantized: bool, optional (default=True)
        if True, load quantized fasttext model. Else, load original fasttext model.

    Returns
    -------
    result : malaya.model.ml.LanguageDetection class
    """

    try:
        import fasttext
    except BaseException:
        raise ModuleNotFoundError(
            'fasttext not installed. Please install it by `pip install fasttext` and try again.'
        )

    if model not in _fasttext_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.language_detection.available_fasttext()`.'
        )

    if quantized:
        filename = 'fasttext.ftz'
    else:
        filename = 'fasttext.bin'

    s3_file = {'model': filename}
    path = download_files(model, s3_file, **kwargs)

    try:
        model_fasttext = fasttext.load_model(path['model'])
    except BaseException:
        raise Exception(f'failed to load fasttext model, please try clear the cache and try again')

    return LanguageDetection(model_fasttext, _fasttext_availability[model]['Label'])


def deep_model(quantized: bool = False, **kwargs):
    """
    Load deep learning language detection model.
    Original size is 51.2MB, Quantized size 12.8MB.

    Parameters
    ----------
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya.model.tf.DeepLang class
    """

    path = check_file(
        file='lang-32',
        module='language-detection',
        keys={
            'model': 'model.pb',
            'vector': LANGUAGE_DETECTION_BOW,
            'bpe': LANGUAGE_DETECTION_VOCAB,
        },
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    bpe = YTTMEncoder(vocab_file=path['bpe'])

    with open(path['vector'], 'rb') as fopen:
        vector = pickle.load(fopen)

    inputs = [
        'X_Placeholder/shape',
        'X_Placeholder/values',
        'X_Placeholder/indices',
        'W_Placeholder/shape',
        'W_Placeholder/values',
        'W_Placeholder/indices',
    ]
    outputs = ['logits']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return DeepLang(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        vectorizer=vector,
        bpe=bpe,
        label=lang_labels_v1,
    )


def substring_rules(model, **kwargs):
    """
    detect EN, MS, MANDARIN and OTHER languages in a string.

    EN words detection are using `pyenchant` from https://pyenchant.github.io/pyenchant/ and
    user language detection model.

    MS words detection are using `malaya.text.function.is_malay` and
    user language detection model.

    OTHER words detection are using any language detection classification model, such as,
    `malaya.language_detection.fasttext` or `malaya.language_detection.deep_model`.

    Parameters
    ----------
    model : Callable
        Callable model, must have `predict` method.

    Returns
    -------
    result : malaya.model.rules.LanguageDict class
    """

    if not hasattr(model, 'predict'):
        raise ValueError('model must have `predict` method')

    return LanguageDict(model=model, **kwargs)
