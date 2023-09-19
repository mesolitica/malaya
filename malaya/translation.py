from malaya.function import describe_availability
from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Translation
from malaya_boilerplate.huggingface import download_files
from malaya.model.alignment import Eflomal
from typing import Callable, List
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

_eflomal_availability = {
    'mesolitica/eflomal-ms-en': {
        'Size (MB)': 240,
    },
    'mesolitica/eflomal-en-ms': {
        'Size (MB)': 301,
    },
}

_word_availability = {
    'mesolitica/en-ms': {
        'Size (MB)': 1,
        'total words': 1,
    },
    'mesolitica/id-ms': {
        'Size (MB)': 1,
        'total words': 1,
    }
}

_huggingface_availability = {
    'mesolitica/translation-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'Suggested length': 1536,
        'en-ms chrF2++': 65.91,
        'ms-en chrF2++': 61.30,
        'ind-ms chrF2++': 58.15,
        'jav-ms chrF2++': 49.33,
        'pasar ms-ms chrF2++': 58.46,
        'pasar ms-en chrF2++': 55.76,
        'manglish-ms chrF2++': 51.04,
        'manglish-en chrF2++': 52.20,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn', 'manglish', 'pasar ms'],
        'to lang': ['en', 'ms'],
    },
    'mesolitica/translation-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'Suggested length': 1536,
        'en-ms chrF2++': 67.37,
        'ms-en chrF2++': 63.79,
        'ind-ms chrF2++': 58.09,
        'jav-ms chrF2++': 52.11,
        'pasar ms-ms chrF2++': 62.49,
        'pasar ms-en chrF2++': 60.77,
        'manglish-ms chrF2++': 52.84,
        'manglish-en chrF2++': 53.65,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn', 'manglish', 'pasar ms'],
        'to lang': ['en', 'ms'],
    },
    'mesolitica/translation-nanot5-tiny-code-cased': {
        'Size (MB)': 358,
        'Suggested length': 2048,
        'from lang': ['en', 'ms'],
        'en-ms chrF2++': 67.62,
        'ms-en chrF2++': 64.41,
        'ind-ms chrF2++': 59.25,
        'jav-ms chrF2++': 52.86,
        'pasar ms-ms chrF2++': 62.99,
        'pasar ms-en chrF2++': 62.06,
        'manglish-ms chrF2++': 54.40,
        'manglish-en chrF2++': 54.14,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn', 'manglish', 'pasar ms', 'mandarin', 'pasar mandarin'],
        'to lang': ['en', 'ms'],
    },
    'mesolitica/translation-nanot5-small-code-cased': {
        'Size (MB)': 358,
        'Suggested length': 2048,
        'from lang': ['en', 'ms'],
        'en-ms chrF2++': 67.62,
        'ms-en chrF2++': 64.41,
        'ind-ms chrF2++': 59.25,
        'jav-ms chrF2++': 52.86,
        'pasar ms-ms chrF2++': 62.99,
        'pasar ms-en chrF2++': 62.06,
        'manglish-ms chrF2++': 54.40,
        'manglish-en chrF2++': 54.14,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn', 'manglish', 'pasar ms', 'mandarin', 'pasar mandarin'],
        'to lang': ['en', 'ms'],
    },
    'mesolitica/translation-nanot5-base-code-cased': {
        'Size (MB)': 892,
        'Suggested length': 1536,
        'en-ms chrF2++': 67.62,
        'ms-en chrF2++': 64.41,
        'ind-ms chrF2++': 59.25,
        'jav-ms chrF2++': 52.86,
        'pasar ms-ms chrF2++': 62.99,
        'pasar ms-en chrF2++': 62.06,
        'manglish-ms chrF2++': 54.40,
        'manglish-en chrF2++': 54.14,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn', 'manglish', 'pasar ms'],
        'to lang': ['en', 'ms'],
    },

}


def available_word():
    return describe_availability(_word_availability)


def available_huggingface():
    """
    List available HuggingFace models.
    """

    logger.info(
        'tested on FLORES200 pair `dev` set, https://github.com/huseinzol05/malay-dataset/tree/master/translation/flores200-eval')
    logger.info(
        'tested on noisy test set, https://github.com/huseinzol05/malay-dataset/tree/master/translation/noisy-eval')
    logger.info('check out NLLB 200 metrics from `malaya.translation.nllb_metrics`.')
    logger.info('check out Google Translate metrics from `malaya.translation.google_translate_metrics`.')
    return describe_availability(_huggingface_availability)


def eflomal(
    model: str = 'mesolitica/eflomal-ms-en',
    preprocessing_func: Callable = None,
    **kwargs,
):
    """
    load https://github.com/robertostling/eflomal word alignment.

    Parameters
    ----------
    model, optional (default='mesolitica/eflomal-ms-en')
        Check available models at `malaya.translation.available_eflomal()`.
    preprocessing_func: Callable, optional (default=None)
        preprocessing function to call during loading prior file.
        Using `malaya.text.function.replace_punct` able to reduce ~30% of memory usage.

    Returns
    -------
    result: malaya.model.alignment.Eflomal
    """

    if model not in _eflomal_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.available_eflomal()`.'
        )

    s3_file = {'model': 'model.priors'}
    path = download_files(model, s3_file, **kwargs)

    return Eflomal(preprocessing_func=preprocessing_func, priors_filename=path['model'])


def word(model: str = 'mesolitica/en-ms', **kwargs):
    """
    Load word dictionary, based on google translate.

    Parameters
    ----------
    model, optional (default='mesolitica/en-ms')
        Check available models at `malaya.translation.available_word()`.

    Returns
    -------
    result: Dict[str, str]
    """
    if model not in _word_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.available_word()`.'
        )

    s3_file = {'model': 'dictionary.json'}
    path = download_files(model, s3_file, **kwargs)

    with open(path['model']) as fopen:
        translator = json.load(fopen)
    return translator


def huggingface(
    model: str = 'mesolitica/translation-nanot5-small-code-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to translate.

    Parameters
    ----------
    model: str, optional (default='mesolitica/translation-nanot5-small-code-cased')
        Check available models at `malaya.translation.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Translation
    """
    return load(
        model=model,
        class_model=Translation,
        availability=_huggingface_availability,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
