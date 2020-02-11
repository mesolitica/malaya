import pickle
from ._utils._utils import check_file, check_available
from ._models._sklearn_model import LANGUAGE_DETECTION
from ._models._tensorflow_model import DEEP_LANG
from ._utils._paths import PATH_LANG_DETECTION, S3_PATH_LANG_DETECTION
from herpetologist import check_type

lang_labels = {
    0: 'eng',
    1: 'ind',
    2: 'malay',
    3: 'manglish',
    4: 'other',
    5: 'rojak',
}


def label():
    return lang_labels


@check_type
def fasttext(quantization: bool = True, validate: bool = True):

    """
    Load Fasttext language detection model.
    Parameters
    ----------
    quantization: bool, optional (default=True)
        if True, load quantized fasttext model. Else, load original fasttext model.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    LANGUAGE_DETECTION : malaya._models._sklearn_model.FASTTEXT_LANGUAGE_DETECTION class
    """

    try:
        import fasttext
    except:
        raise Exception(
            'fasttext not installed. Please install it by `pip install fasttext` and try again.'
        )
    if quantization:
        model = 'fasttext-quantized'
    else:
        model = 'fasttext-original'
    if validate:
        check_file(PATH_LANG_DETECTION[model], S3_PATH_LANG_DETECTION[model])
    else:
        if not check_available(PATH_LANG_DETECTION[model]):
            raise Exception(
                'language-detection/%s is not available, please `validate = True`'
                % (model)
            )
    try:
        model_fasttext = fasttext.load_model(
            PATH_LANG_DETECTION[model]['model']
        )
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('language-detection/%s') and try again"
            % (model)
        )
    return LANGUAGE_DETECTION(model_fasttext, lang_labels)


@check_type
def deep_model(validate: bool = True):
    """
    Load deep learning language detection model.
    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    DEEP_LANG : malaya._models._tensorflow_model.DEEP_LANG class
    """
    try:
        import youtokentome as yttm
    except:
        raise Exception(
            'youtokentome not installed. Please install it by `pip install youtokentome` and try again.'
        )
    import os

    if validate:
        check_file(PATH_LANG_DETECTION['deep'], S3_PATH_LANG_DETECTION['deep'])
    else:
        if not check_available(PATH_LANG_DETECTION['deep']):
            raise Exception(
                'language-detection/deep is not available, please `validate = True`'
            )
    try:
        with open(PATH_LANG_DETECTION['deep']['vector'], 'rb') as fopen:
            vector = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('language-detection/deep') and try again"
        )
    try:
        bpe = yttm.BPE(model = PATH_LANG_DETECTION['deep']['bpe'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('language-detection/deep') and try again"
        )
    return DEEP_LANG(
        os.path.dirname(PATH_LANG_DETECTION['deep']['model']),
        vector,
        lang_labels,
        bpe,
        yttm.OutputType.SUBWORD,
    )
