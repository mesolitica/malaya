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

metrics = {
    'deep-model': """
              precision    recall  f1-score   support

         eng    0.96760   0.97401   0.97080    553739
         ind    0.97635   0.96131   0.96877    576059
       malay    0.96985   0.98498   0.97736   1800649
    manglish    0.98036   0.96569   0.97297    181442
       other    0.99641   0.99627   0.99634   1428083
       rojak    0.94221   0.84302   0.88986    189678

    accuracy                        0.97779   4729650
   macro avg    0.97213   0.95421   0.96268   4729650
weighted avg    0.97769   0.97779   0.97760   4729650
""",
    'mesolitica/fasttext-language-detection-v1': """
              precision    recall  f1-score   support

         eng    0.94014   0.96750   0.95362    553739
         ind    0.97290   0.97316   0.97303    576059
       malay    0.98674   0.95262   0.96938   1800649
    manglish    0.96595   0.98417   0.97498    181442
       other    0.98454   0.99698   0.99072   1428083
       rojak    0.81149   0.91650   0.86080    189678

    accuracy                        0.97002   4729650
   macro avg    0.94363   0.96515   0.95375   4729650
weighted avg    0.97111   0.97002   0.97028   4729650
""",
    'mesolitica/fasttext-language-detection-ms-id': """
                        precision    recall  f1-score   support

           local-malay    0.95063   0.93858   0.94457    199961
                 other    0.97145   0.98889   0.98009    125920
socialmedia-indonesian    0.97923   0.96303   0.97106    213486
   standard-indonesian    0.99119   0.99610   0.99364    149055
        standard-malay    0.93743   0.95669   0.94696    149336

              accuracy                        0.96584    837758
             macro avg    0.96599   0.96866   0.96727    837758
          weighted avg    0.96591   0.96584   0.96582    837758
""",
    'mesolitica/fasttext-language-detection-en': """
                  precision    recall  f1-score   support

   local-english    0.88991   0.89457   0.89223    149823
        manglish    0.98619   0.98479   0.98549    149535
           other    0.99439   0.99268   0.99354    140651
standard-english    0.89162   0.88967   0.89064    150703

        accuracy                        0.93952    590712
       macro avg    0.94053   0.94043   0.94047    590712
    weighted avg    0.93960   0.93952   0.93955    590712
""",
}

lang_labels_v1 = {
    0: 'eng',
    1: 'ind',
    2: 'malay',
    3: 'manglish',
    4: 'other',
    5: 'rojak',
}

lang_labels_v2 = {
    0: 'standard-english',
    1: 'local-english',
    2: 'manglish',
    3: 'standard-indonesian',
    4: 'socialmedia-indonesian',
    5: 'standard-malay',
    6: 'local-malay',
    7: 'standard-mandarin',
    8: 'local-mandarin',
    9: 'other',
}

lang_labels_ms_id = {
    0: 'standard-indonesian',
    1: 'socialmedia-indonesian',
    2: 'standard-malay',
    3: 'local-malay',
    4: 'other',
}

lang_labels_en = {
    0: 'standard-english',
    1: 'local-english',
    2: 'manglish',
    3: 'other',
}

label_v1 = list(lang_labels_v1.values())
label_v2 = list(lang_labels_v2.values())
label_ms_id = list(lang_labels_ms_id.values())
label_en = list(lang_labels_en.values())

_fasttext_availability = {
    'mesolitica/fasttext-language-detection-v1': {
        'Size (MB)': 353,
        'Quantized Size (MB)': 31.1,
        'dim': 16,
        'Label': lang_labels_v1,
    },
    'mesolitica/fasttext-language-detection-v2': {
        'Size (MB)': 1840,
        'Quantized Size (MB)': 227,
        'dim': 16,
        'Label': lang_labels_v2,
    },
    'mesolitica/fasttext-language-detection-ms-id': {
        'Size (MB)': 537,
        'Quantized Size (MB)': 62.5,
        'dim': 16,
        'Label': label_ms_id,
    },
    'mesolitica/fasttext-language-detection-en': {
        'Size (MB)': 383,
        'Quantized Size (MB)': 42.3,
        'dim': 16,
        'Label': label_ms_id,
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
