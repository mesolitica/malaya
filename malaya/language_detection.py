import pickle
from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.model.ml import LanguageDetection
from malaya.model.tf import DeepLang
from malaya.text.bpe import YTTMEncoder
from malaya.path import (
    PATH_LANG_DETECTION,
    S3_PATH_LANG_DETECTION,
    LANGUAGE_DETECTION_BOW,
    LANGUAGE_DETECTION_VOCAB,
)
from herpetologist import check_type

lang_labels = {
    0: 'eng',
    1: 'ind',
    2: 'malay',
    3: 'manglish',
    4: 'other',
    5: 'rojak',
}

label = list(lang_labels.values())


@check_type
def fasttext(quantized: bool = True, **kwargs):
    """
    Load Fasttext language detection model.
    Original size is 353MB, Quantized size 31.1MB.

    Parameters
    ----------
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
    if quantized:
        model = 'fasttext-quantized'
    else:
        model = 'fasttext-original'

    check_file(
        PATH_LANG_DETECTION[model], S3_PATH_LANG_DETECTION[model], **kwargs
    )
    try:
        model_fasttext = fasttext.load_model(
            PATH_LANG_DETECTION[model]['model']
        )
    except:
        raise Exception(
            f"failed to load fasttext model, please run `malaya.utils.delete_cache('language-detection/{model}")
    return LanguageDetection(model_fasttext, lang_labels)


def deep_model(quantized: bool = False, **kwargs):
    """
    Load deep learning language detection model.
    Original size is 51.2MB, Quantized size 12.8MB.

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
        label=lang_labels,
    )
