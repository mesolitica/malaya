from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability, check_file
from malaya.path import PATH_PREPROCESSING, S3_PATH_PREPROCESSING
import json
import logging

logger = logging.getLogger(__name__)

nllb_metrics = {
    'en-ms': """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, eng_Latn-zsm_Latn, 66.5
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, eng_Latn-zsm_Latn, 66.3
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, eng_Latn-zsm_Latn, 65.2
4. NLLB-200-Distilled, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200densedst1bmetrics, eng_Latn-zsm_Latn, 65.5
5. NLLB-200-Distilled, Dense, 600M, 2.46 GB, https://tinyurl.com/nllb200densedst600mmetrics, eng_Latn-zsm_Latn, 63.5
""",
    'ind-ms': """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, ind_Latn-zsm_Latn, 60.2
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, ind_Latn-zsm_Latn, None
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, ind_Latn-zsm_Latn, None
""",
    'jav-ms': """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, jav_Latn-zsm_Latn, 56.5
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, jav_Latn-zsm_Latn, None
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, jav_Latn-zsm_Latn, None
""",
    'ms-en': """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, zsm_Latn-eng_Latn,68
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, zsm_Latn-eng_Latn,67.8
3. NLLB-200, Dense, 1.3B,  5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, zsm_Latn-eng_Latn,66.4
4. NLLB-200-Distilled, Dense, 1.3B,  5.48 GB, https://tinyurl.com/nllb200densedst1bmetrics, zsm_Latn-eng_Latn,66.2
5. NLLB-200-Distilled, Dense, 600M, 2.46 GB, https://tinyurl.com/nllb200densedst600mmetrics, zsm_Latn-eng_Latn,64.3
""",
    'ms-ind': """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, zsm_Latn-ind_Latn, 62.4
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, zsm_Latn-ind_Latn, None
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, zsm_Latn-ind_Latn, None
""",
    'ms-jav': """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, zsm_Latn-jav_Latn, 49.5
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, zsm_Latn-jav_Latn, None
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, zsm_Latn-jav_Latn, None
"""
}

google_translate_metrics = {
    'en-ms': """
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
""",
    'ms-en': """
Google Translation metrics (2022-07-23) on FLORES200, https://github.com/huseinzol05/malay-dataset/blob/master/translation/malay-english/flores200-ms-en-google-translate.ipynb:
{'name': 'BLEU',
 'score': 36.152220848177286,
 '_mean': -1.0,
 '_ci': -1.0,
 '_verbose': '68.2/43.5/29.7/20.5 (BP = 0.986 ratio = 0.986 hyp_len = 23243 ref_len = 23570)',
 'bp': 0.9860297505310752,
 'counts': [15841, 9688, 6318, 4147],
 'totals': [23243, 22246, 21249, 20252],
 'sys_len': 23243,
 'ref_len': 23570,
 'precisions': [68.15385277287785,
  43.54940213971051,
  29.733163913595934,
  20.476989926920798],
 'prec_str': '68.2/43.5/29.7/20.5',
 'ratio': 0.986126431904964}
chrF2++ = 60.27
"""
}

_huggingface_availability = {
    'mesolitica/finetune-translation-t5-tiny-standard-bahasa-cased-v2': {
        'Size (MB)': 139,
        'BLEU': 41.625536185056305,
        'SacreBLEU Verbose': '73.4/50.1/35.7/25.7 (BP = 0.971 ratio = 0.972 hyp_len = 21400 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 65.70,
        'Suggested length': 256,
        'from lang': ['en', 'ms'],
        'to lang': ['ms', 'ms'],
    },
    'mesolitica/finetune-translation-t5-small-standard-bahasa-cased-v2': {
        'Size (MB)': 242,
        'BLEU': 43.93729753370648,
        'SacreBLEU Verbose': '74.9/52.2/37.9/27.7 (BP = 0.976 ratio = 0.977 hyp_len = 21510 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 67.43,
        'Suggested length': 256,
        'from lang': ['en', 'ms'],
        'to lang': ['ms', 'ms'],
    },
    'mesolitica/finetune-translation-t5-base-standard-bahasa-cased-v2': {
        'Size (MB)': 892,
        'BLEU': 44.17355862158963,
        'SacreBLEU Verbose': '74.7/52.3/38.0/28.0 (BP = 0.979 ratio = 0.979 hyp_len = 21569 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 67.60,
        'Suggested length': 256,
        'from lang': ['en', 'ms'],
        'to lang': ['ms', 'ms'],
    },
    'mesolitica/finetune-noisy-translation-t5-tiny-bahasa-cased-v3': {
        'Size (MB)': 139,
        'BLEU': 60.0009672168891,
        'SacreBLEU Verbose': '77.9/63.9/54.6/47.7 (BP = 1.000 ratio = 1.036 hyp_len = 110970 ref_len = 107150)',
        'SacreBLEU-chrF++-FLORES200': None,
        'Suggested length': 256,
        'from lang': ['en', 'ms'],
        'to lang': ['ms', 'ms'],
    },
    'mesolitica/finetune-noisy-translation-t5-small-bahasa-cased-v3': {
        'Size (MB)': 242,
        'BLEU': 64.06258219941243,
        'SacreBLEU Verbose': '80.1/67.7/59.1/52.5 (BP = 1.000 ratio = 1.042 hyp_len = 111635 ref_len = 107150)',
        'SacreBLEU-chrF++-FLORES200': None,
        'Suggested length': 256,
        'from lang': ['en', 'ms'],
        'to lang': ['ms', 'ms'],
    },
    'mesolitica/finetune-noisy-translation-t5-base-bahasa-cased-v2': {
        'Size (MB)': 892,
        'BLEU': 64.583819005204,
        'SacreBLEU Verbose': '80.2/68.1/59.8/53.2 (BP = 1.000 ratio = 1.048 hyp_len = 112260 ref_len = 107150)',
        'SacreBLEU-chrF++-FLORES200': None,
        'Suggested length': 256,
        'from lang': ['en', 'ms'],
        'to lang': ['ms', 'ms'],
    },
    'mesolitica/finetune-translation-austronesian-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 30.277470707798773,
        'SacreBLEU Verbose': '64.2/38.0/24.1/15.6 (BP = 0.978 ratio = 0.978 hyp_len = 21542 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 57.38,
        'Suggested length': 256,
        'from lang': ['ind', 'jav', 'ms'],
        'to lang': ['ind', 'jav', 'ms'],
    },
    'mesolitica/finetune-translation-austronesian-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 30.24358980824753,
        'SacreBLEU Verbose': '61.1/36.9/23.8/15.6 (BP = 1.000 ratio = 1.052 hyp_len = 23174 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 58.43,
        'Suggested length': 512,
        'from lang': ['ind', 'jav', 'ms'],
        'to lang': ['ind', 'jav', 'ms'],
    },
    'mesolitica/finetune-translation-austronesian-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 31.494673032706213,
        'SacreBLEU Verbose': '64.1/38.8/25.1/16.5 (BP = 0.989 ratio = 0.990 hyp_len = 21796 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 58.10,
        'Suggested length': 512,
        'from lang': ['ind', 'jav', 'ms'],
        'to lang': ['ind', 'jav', 'ms'],
    },
}


def available_huggingface():
    """
    List available HuggingFace models.
    """

    logger.info(
        'tested on FLORES200 EN-MS pair `dev` set, https://github.com/facebookresearch/flores/tree/main/flores200')
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


def huggingface(
    model: str = 'mesolitica/finetune-translation-t5-small-standard-bahasa-cased-v2',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to translate.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-translation-t5-small-standard-bahasa-cased-v2')
        Check available models at `malaya.translation.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Translation
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.available_huggingface()`.'
        )
    return load_huggingface.load_translation(
        model=model,
        from_lang=_huggingface_availability[model]['from lang'],
        to_lang=_huggingface_availability[model]['to lang'],
        **kwargs,
    )
