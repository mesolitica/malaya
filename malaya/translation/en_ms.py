from malaya.model.tf import Translation
from malaya.model.huggingface import Generator
from malaya.model.bigbird import Translation as BigBird_Translation
from malaya.supervised import transformer as load_transformer
from malaya.supervised import bigbird as load_bigbird
from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability, check_file
from herpetologist import check_type
from malaya.path import PATH_PREPROCESSING, S3_PATH_PREPROCESSING
import json
import logging

logger = logging.getLogger(__name__)

"""
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, eng_Latn-zsm_Latn,66.5
2. NLLB-200, Dense, 3.3B, https://tinyurl.com/nllb200dense3bmetrics, eng_Latn-zsm_Latn,66.3
3. NLLB-200, Dense, 1.3B, https://tinyurl.com/nllb200dense1bmetrics, eng_Latn-zsm_Latn,65.2
4. NLLB-200-Distilled, Dense, 1.3B, https://tinyurl.com/nllb200densedst1bmetrics, eng_Latn-zsm_Latn,65.5
5. NLLB-200-Distilled, Dense, 600M, https://tinyurl.com/nllb200densedst600mmetrics, eng_Latn-zsm_Latn,63.5
"""

"""
Google Translation metrics (2022-07-23) on FLORES200, https://github.com/huseinzol05/malay-dataset/blob/master/translation/malay-english/flores200-en-ms-google-translate.ipynb:
{'name': 'BLEU',
 'score': 39.12728212969207,
 '_mean': -1.0,
 '_ci': -1.0,
 '_verbose': '71.1/47.2/32.7/22.8 (BP = 0.984 ratio = 0.984 hyp_len = 21679 ref_len = 22027)',
 'bp': 0.9840757522087613,
 'counts': [15406, 9770, 6435, 4256],
 'totals': [21679, 20682, 19685, 18688],
 'sys_len': 21679,
 'ref_len': 22027,
 'precisions': [71.0641634761751,
  47.2391451503723,
  32.68986537973076,
  22.773972602739725],
 'prec_str': '71.1/47.2/32.7/22.8',
 'ratio': 0.9842012076088437}
chrF2++ = 64.45
"""

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.4,
        'BLEU': 58.67129043177485,
        'SacreBLEU Verbose': '80.2/63.8/52.8/44.4 (BP = 0.997 ratio = 0.997 hyp_len = 2621510 ref_len = 2630014)',
        'SacreBLEU-chrF++-FLORES200': 64.46,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 82.7,
        'BLEU': 68.25956937012508,
        'SacreBLEU Verbose': '86.3/73.3/64.1/56.8 (BP = 0.985 ratio = 0.985 hyp_len = 2591093 ref_len = 2630014)',
        'SacreBLEU-chrF++-FLORES200': 66.28,
        'Suggested length': 256,
    },
    'bigbird': {
        'Size (MB)': 246,
        'Quantized Size (MB)': 63.7,
        'BLEU': 59.86353498474623,
        'SacreBLEU Verbose': '82.2/65.9/54.9/46.4 (BP = 0.982 ratio = 0.982 hyp_len = 2583848 ref_len = 2630014)',
        'SacreBLEU-chrF++-FLORES200': 59.64,
        'Suggested length': 1024,
    },
    'small-bigbird': {
        'Size (MB)': 50.4,
        'Quantized Size (MB)': 13.1,
        'BLEU': 56.70133817548828,
        'SacreBLEU Verbose': '80.7/63.2/51.6/42.8 (BP = 0.979 ratio = 0.979 hyp_len = 2575384 ref_len = 2630014)',
        'SacreBLEU-chrF++-FLORES200': 59.64,
        'Suggested length': 1024,
    },
    'noisy-base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 82.7,
        'BLEU': 67.28571638641796,
        'SacreBLEU Verbose': '86.1/72.7/63.3/55.8 (BP = 0.981 ratio = 0.981 hyp_len = 2580126 ref_len = 2630014)',
        'SacreBLEU-chrF++-FLORES200': 66.20,
        'Suggested length': 256,
    },
}

_huggingface_availability = {
    'mesolitica/t5-super-tiny-finetuned-noisy-en-ms': {
        'Size (MB)': 50.8,
        'BLEU': 58.72114029430599,
        'SacreBLEU Verbose': '80.4/64.0/53.0/44.7 (BP = 0.994 ratio = 0.994 hyp_len = 2614280 ref_len = 2630014)',
        'SacreBLEU-chrF++-FLORES200': 64.22,
        'Suggested length': 256,
    },
    'mesolitica/t5-tiny-finetuned-noisy-en-ms': {
        'Size (MB)': 139,
        'BLEU': 62.34308405954152,
        'SacreBLEU Verbose': '82.6/67.5/57.2/49.3 (BP = 0.990 ratio = 0.990 hyp_len = 2604652 ref_len = 2630014)',
        'SacreBLEU-chrF++-FLORES200': 64.26,
        'Suggested length': 256,
    },
    'mesolitica/t5-small-finetuned-noisy-en-ms': {
        'Size (MB)': 242,
        'BLEU': 65.00070822235693,
        'SacreBLEU Verbose': '84.8/70.5/60.6/52.9 (BP = 0.983 ratio = 0.983 hyp_len = 2585122 ref_len = 2630014)',
        'SacreBLEU-chrF++-FLORES200': 66.31,
        'Suggested length': 256,
    },
}


def available_transformer():
    """
    List available transformer models.
    """

    logger.info('tested on 77k EN-MS test set generated from teacher semisupervised model, https://huggingface.co/datasets/mesolitica/en-ms')
    logger.info('tested on FLORES200 EN-MS (eng_Latn-zsm_Latn) pair `dev` set, https://github.com/facebookresearch/flores/tree/main/flores200')

    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available HuggingFace models.
    """

    logger.info('tested on 77k EN-MS test set generated from teacher semisupervised model, https://huggingface.co/datasets/mesolitica/en-ms')
    logger.info('tested on FLORES200 EN-MS (eng_Latn-zsm_Latn) pair `dev` set, https://github.com/facebookresearch/flores/tree/main/flores200')
    logger.warning(
        '77k EN-MS test set generated from teacher semisupervised model, the models might generate better results compared to '
        'to the teacher semisupervised model, thus lower BLEU score.'
    )
    return describe_availability(_huggingface_availability)


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load transformer encoder-decoder model to translate EN-to-MS.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'small'`` - Transformer SMALL parameters.
        * ``'base'`` - Transformer BASE parameters.
        * ``'bigbird'`` - BigBird BASE parameters.
        * ``'small-bigbird'`` - BigBird SMALL parameters.
        * ``'noisy-base'`` - Transformer BASE parameters trained on noisy dataset.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `bigbird` in model, return `malaya.model.bigbird.Translation`.
        * else, return `malaya.model.tf.Translation`.
    """
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.en_ms.available_transformer()`.'
        )

    if 'bigbird' in model:
        return load_bigbird.load(
            module='translation-en-ms',
            model=model,
            model_class=BigBird_Translation,
            maxlen=_transformer_availability[model]['Suggested length'],
            quantized=quantized,
            **kwargs
        )

    else:
        return load_transformer.load(
            module='translation-en-ms',
            model=model,
            encoder='subword',
            model_class=Translation,
            quantized=quantized,
            **kwargs
        )


def dictionary(**kwargs):
    """
    Load dictionary {EN: MS} .

    Returns
    -------
    result: Dict[str, str]
    """
    path = check_file(
        PATH_PREPROCESSING['english-malay'],
        S3_PATH_PREPROCESSING['english-malay'],
        **kwargs,
    )
    try:
        with open(path['model']) as fopen:
            translator = json.load(fopen)
    except BaseException:
        raise Exception('failed to load EN-MS vocab, please try clear cache or rerun again.')
    return translator


def _huggingface(model, initial_text, **kwargs):

    try:
        from transformers import TFT5ForConditionalGeneration
    except BaseException:
        raise ModuleNotFoundError(
            'transformers not installed. Please install it by `pip3 install transformers` and try again.'
        )

    if 't5' in model:
        huggingface_class = TFT5ForConditionalGeneration

    return load_huggingface.load_automodel(
        model=model,
        model_class=Generator,
        huggingface_class=huggingface_class,
        initial_text=initial_text,
        **kwargs
    )


@check_type
def huggingface(model: str = 'mesolitica/t5-tiny-finetuned-noisy-en-ms', **kwargs):
    """
    Load HuggingFace model to translate EN-to-MS.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'mesolitica/t5-super-tiny-finetuned-noisy-en-ms'`` - https://huggingface.co/mesolitica/t5-super-tiny-finetuned-noisy-en-ms
        * ``'mesolitica/t5-tiny-finetuned-noisy-en-ms'`` - https://huggingface.co/mesolitica/t5-tiny-finetuned-noisy-en-ms
        * ``'mesolitica/t5-small-finetuned-noisy-en-ms'`` - https://huggingface.co/mesolitica/t5-small-finetuned-noisy-en-ms

    Returns
    -------
    result: malaya.model.huggingface.Generator
    """
    model = model.lower()
    if model not in _huggingface_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.en_ms.available_huggingface()`.'
        )
    return _huggingface(model=model, initial_text='terjemah Inggeris ke Melayu: ', **kwargs)
