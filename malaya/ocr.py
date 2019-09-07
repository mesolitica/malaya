from ._ocr._model import DETECT_JAWI, JAWI_TO_MALAY
from ._utils._paths import PATH_OCR, S3_PATH_OCR
from ._utils._utils import (
    check_file,
    load_graph,
    check_available,
    generate_session,
)
import pickle


def available_detect_jawi_model():
    """
    List available detect jawi models.
    """
    return ['logistic', 'svm']


def detect_jawi(model = 'svm', validate = True):
    """
    Load model to detect a handwriting is a jawi or not.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    DETECT_JAWI: malaya._ocr._model.DETECT_JAWI class
    """
    if not isinstance(model, str):
        raise ValueError('validate must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')
    model = model.lower()

    if model not in available_detect_jawi_model():
        raise Exception(
            'model is not supported, please check supported models from malaya.ocr.available_detect_jawi_model()'
        )

    if validate:
        check_file(PATH_OCR[model], S3_PATH_OCR[model])
    else:
        if not check_available(PATH_OCR[model]):
            raise Exception(
                'ocr/%s is not available, please `validate = True`' % (model)
            )

    try:
        with open(PATH_OCR[model]['model'], 'rb') as fopen:
            model = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('ocr/%s') and try again"
            % (model)
        )
    return DETECT_JAWI(model)


def jawi_to_malay(validate = True, jawi_detector = None):
    """
    Load Im2Latex model to do OCR Jawi to Malay.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.
    jawi_detector: object, optional (default=None)
        detector object to detect whether a handwriting is jawi or not. Must have method `predict_batch`. Suggest to use malaya.ocr.detect_jawi .

    Returns
    -------
    JAWI_TO_MALAY: malaya._ocr._model.JAWI_TO_MALAY class
    """

    if validate:
        check_file(PATH_OCR['jawi-to-malay'], S3_PATH_OCR['jawi-to-malay'])
    else:
        if not check_available(PATH_OCR['jawi-to-malay']):
            raise Exception(
                'ocr/jawi-to-malay is not available, please `validate = True`'
            )
    try:
        g = load_graph(PATH_OCR['jawi-to-malay']['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('ocr/jawi-to-malay') and try again"
        )

    return JAWI_TO_MALAY(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        sess = generate_session(graph = g),
        jawi_detector = jawi_detector,
    )
