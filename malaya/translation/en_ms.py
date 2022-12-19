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
import warnings

logger = logging.getLogger(__name__)

nllb_metrics = """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, eng_Latn-zsm_Latn, 66.5
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, eng_Latn-zsm_Latn, 66.3
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, eng_Latn-zsm_Latn, 65.2
4. NLLB-200-Distilled, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200densedst1bmetrics, eng_Latn-zsm_Latn, 65.5
5. NLLB-200-Distilled, Dense, 600M, 2.46 GB, https://tinyurl.com/nllb200densedst600mmetrics, eng_Latn-zsm_Latn, 63.5
"""

google_translate_metrics = """
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
        'BLEU': 39.80538744027295,
        'SacreBLEU Verbose': '80.2/63.8/52.8/44.4 (BP = 0.997 ratio = 0.997 hyp_len = 2621510 ref_len = 2630014)',
        'SacreBLEU-chrF++-FLORES200': 64.46,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 82.7,
        'BLEU': 42.21071347388556,
        'SacreBLEU Verbose': '86.3/73.3/64.1/56.8 (BP = 0.985 ratio = 0.985 hyp_len = 2591093 ref_len = 2630014)',
        'SacreBLEU-chrF++-FLORES200': 66.28,
        'Suggested length': 256,
    },
    'bigbird': {
        'Size (MB)': 246,
        'Quantized Size (MB)': 63.7,
        'BLEU': 39.09071749208737,
        'SacreBLEU Verbose': '70.5/46.7/32.4/22.9 (BP = 0.989 ratio = 0.989 hyp_len = 21782 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 63.96,
        'Suggested length': 1024,
    },
    'small-bigbird': {
        'Size (MB)': 50.4,
        'Quantized Size (MB)': 13.1,
        'BLEU': 36.90195033318057,
        'SacreBLEU Verbose': '67.0/43.8/30.1/21.0 (BP = 1.000 ratio = 1.028 hyp_len = 22637 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 62.85,
        'Suggested length': 1024,
    },
    'noisy-base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 82.7,
        'BLEU': 41.8278308435666,
        'SacreBLEU Verbose': '73.1/49.7/35.3/25.4 (BP = 0.985 ratio = 0.985 hyp_len = 63506 ref_len = 64473)',
        'SacreBLEU-chrF++-FLORES200': 66.46,
        'Suggested length': 256,
    },
}

_huggingface_availability = {
    'mesolitica/finetune-translation-t5-super-super-tiny-standard-bahasa-cased': {
        'Size (MB)': 23.3,
        'BLEU': 36.29074311583665,
        'SacreBLEU Verbose': '71.2/46.0/30.9/21.0 (BP = 0.950 ratio = 0.951 hyp_len = 20958 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 61.89,
        'Suggested length': 256,
    },
    'mesolitica/finetune-translation-t5-super-tiny-standard-bahasa-cased': {
        'Size (MB)': 50.7,
        'BLEU': 39.18834189893951,
        'SacreBLEU Verbose': '72.6/48.3/33.5/23.6 (BP = 0.960 ratio = 0.961 hyp_len = 21172 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 64.03,
        'Suggested length': 256,
    },
    'mesolitica/finetune-translation-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 41.625536185056305,
        'SacreBLEU Verbose': '73.4/50.1/35.7/25.7 (BP = 0.971 ratio = 0.972 hyp_len = 21400 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 65.70,
        'Suggested length': 256,
    },
    'mesolitica/finetune-translation-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 43.93729753370648,
        'SacreBLEU Verbose': '74.9/52.2/37.9/27.7 (BP = 0.976 ratio = 0.977 hyp_len = 21510 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 67.43,
        'Suggested length': 256,
    },
    'mesolitica/finetune-translation-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 44.17355862158963,
        'SacreBLEU Verbose': '74.7/52.3/38.0/28.0 (BP = 0.979 ratio = 0.979 hyp_len = 21569 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 67.60,
        'Suggested length': 256,
    },
    'mesolitica/finetune-noisy-translation-t5-tiny-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 41.03641425544081,
        'SacreBLEU Verbose': '72.9/49.2/34.8/25.0 (BP = 0.977 ratio = 0.977 hyp_len = 63005 ref_len = 64473)',
        'SacreBLEU-chrF++-FLORES200': 65.58,
        'Suggested length': 256,
    },
    'mesolitica/finetune-noisy-translation-t5-small-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 41.15794003172596,
        'SacreBLEU Verbose': '72.2/48.8/34.5/24.8 (BP = 0.988 ratio = 0.988 hyp_len = 63689 ref_len = 64473)',
        'SacreBLEU-chrF++-FLORES200': 65.51,
        'Suggested length': 256,
    },
    'mesolitica/finetune-noisy-translation-t5-base-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 41.8278308435666,
        'SacreBLEU Verbose': '73.4/50.1/35.7/25.8 (BP = 0.982 ratio = 0.982 hyp_len = 63335 ref_len = 64473)',
        'SacreBLEU-chrF++-FLORES200': 66.51,
        'Suggested length': 256,
    },
    'mesolitica/finetune-noisy-translation-t5-tiny-bahasa-cased-v2': {
        'Size (MB)': 139,
        'BLEU': 60.0009672168891,
        'SacreBLEU Verbose': '77.9/63.9/54.6/47.7 (BP = 1.000 ratio = 1.036 hyp_len = 110970 ref_len = 107150)',
        'SacreBLEU-chrF++-FLORES200': None,
        'Suggested length': 256,
    },
    'mesolitica/finetune-noisy-translation-t5-small-bahasa-cased-v4': {
        'Size (MB)': 242,
        'BLEU': 64.06258219941243,
        'SacreBLEU Verbose': '80.1/67.7/59.1/52.5 (BP = 1.000 ratio = 1.042 hyp_len = 111635 ref_len = 107150)',
        'SacreBLEU-chrF++-FLORES200': None,
        'Suggested length': 256,
    },
    'mesolitica/finetune-noisy-translation-t5-base-bahasa-cased-v2': {
        'Size (MB)': 892,
        'BLEU': 64.583819005204,
        'SacreBLEU Verbose': '80.2/68.1/59.8/53.2 (BP = 1.000 ratio = 1.048 hyp_len = 112260 ref_len = 107150)',
        'SacreBLEU-chrF++-FLORES200': None,
        'Suggested length': 256,
    },
}


def _describe():
    logger.info('tested on FLORES200 EN-MS (eng_Latn-zsm_Latn) pair `dev` set, https://github.com/facebookresearch/flores/tree/main/flores200')
    logger.info('for noisy, tested on noisy twitter google translation, https://huggingface.co/datasets/mesolitica/augmentation-test-set')


def available_transformer():
    """
    List available transformer models.
    """

    warnings.warn('`malaya.translation.en_ms.available_transformer` is deprecated, use `malaya.translation.en_ms.available_huggingface` instead', DeprecationWarning)

    _describe()
    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available HuggingFace models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


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


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load transformer encoder-decoder model to translate EN-to-MS.

    Parameters
    ----------
    model: str, optional (default='base')
        Check available models at `malaya.translation.en_ms.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `bigbird` in model, return `malaya.model.bigbird.Translation`.
        * else, return `malaya.model.tf.Translation`.
    """
    warnings.warn(
        '`malaya.translation.en_ms.transformer` is deprecated, use `malaya.translation.en_ms.huggingface` instead', DeprecationWarning)

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


@check_type
def huggingface(
    model: str = 'mesolitica/finetune-translation-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to translate EN-to-MS.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-translation-t5-small-standard-bahasa-cased')
        Check available models at `malaya.translation.en_ms.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Generator
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.en_ms.available_huggingface()`.'
        )
    return load_huggingface.load_generator(model=model, initial_text='terjemah Inggeris ke Melayu: ', **kwargs)
