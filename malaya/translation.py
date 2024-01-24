from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Translation
from malaya_boilerplate.huggingface import download_files
from typing import Callable, List
import json

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
""",
    'en-zho_Hans': """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, eng_Latn-zho_Hans,22.8
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, eng_Latn-zho_Hans,22.3
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, eng_Latn-zho_Hans,21.3
""",
    'zho_Hans-en': """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, zho_Hans-eng_Latn,54.7
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, zho_Hans-eng_Latn,56.2
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, zho_Hans-eng_Latn,54.7
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

available_word = {
    'mesolitica/word-en-ms': {
        'Size (MB)': 42.6,
        'total words': 1599797,
    },
    'mesolitica/word-id-ms': {
        'Size (MB)': 53,
        'total words': 1902607,
    }
}

available_huggingface = {
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
    'mesolitica/translation-t5-base-standard-bahasa-cased': {
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
    'mesolitica/translation-t5-small-standard-bahasa-cased-v2': {
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
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn'],
        'to lang': ['en', 'ms'],
    },
    'mesolitica/translation-t5-small-standard-bahasa-cased-code': {
        'Size (MB)': 242,
        'Suggested length': 2048,
        'en-ms chrF2++': 67.37,
        'ms-en chrF2++': 63.79,
        'ind-ms chrF2++': 58.09,
        'jav-ms chrF2++': 52.11,
        'pasar ms-ms chrF2++': 62.49,
        'pasar ms-en chrF2++': 60.77,
        'manglish-ms chrF2++': 52.84,
        'manglish-en chrF2++': 53.65,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn'],
        'to lang': ['en', 'ms'],
    },
    'mesolitica/translation-nanot5-tiny-malaysian-cased': {
        'Size (MB)': 205,
        'Suggested length': 2048,
        'en-ms chrF2++': 63.61,
        'ms-en chrF2++': 59.55,
        'ind-ms chrF2++': 56.38,
        'jav-ms chrF2++': 47.68,
        'mandarin-ms chrF2++': 36.61,
        'mandarin-en chrF2++': 39.78,
        'pasar ms-ms chrF2++': 58.74,
        'pasar ms-en chrF2++': 54.87,
        'manglish-ms chrF2++': 50.76,
        'manglish-en chrF2++': 53.16,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn', 'manglish', 'pasar ms', 'mandarin', 'pasar mandarin'],
        'to lang': ['en', 'ms'],
    },
    'mesolitica/translation-nanot5-small-malaysian-cased': {
        'Size (MB)': 358,
        'Suggested length': 2048,
        'en-ms chrF2++': 66.98,
        'ms-en chrF2++': 63.52,
        'ind-ms chrF2++': 58.10,
        'jav-ms chrF2++': 51.55,
        'mandarin-ms chrF2++': 46.09,
        'mandarin-en chrF2++': 44.13,
        'pasar ms-ms chrF2++': 63.20,
        'pasar ms-en chrF2++': 59.78,
        'manglish-ms chrF2++': 54.09,
        'manglish-en chrF2++': 55.27,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn', 'manglish', 'pasar ms', 'mandarin', 'pasar mandarin'],
        'to lang': ['en', 'ms'],
    },
    'mesolitica/translation-nanot5-base-malaysian-cased': {
        'Size (MB)': 990,
        'Suggested length': 2048,
        'en-ms chrF2++': 67.87,
        'ms-en chrF2++': 64.79,
        'ind-ms chrF2++': 56.98,
        'jav-ms chrF2++': 51.21,
        'mandarin-ms chrF2++': 47.39,
        'mandarin-en chrF2++': 48.78,
        'pasar ms-ms chrF2++': 65.06,
        'pasar ms-en chrF2++': 64.03,
        'manglish-ms chrF2++': 57.91,
        'manglish-en chrF2++': 55.66,
        'from lang': ['en', 'ms', 'ind', 'jav', 'bjn', 'manglish', 'pasar ms', 'mandarin', 'pasar mandarin'],
        'to lang': ['en', 'ms'],
    },

}

info = """
1. tested on FLORES200 pair `dev` set, https://github.com/huseinzol05/malay-dataset/tree/master/translation/flores200-eval
2. tested on noisy test set, https://github.com/huseinzol05/malay-dataset/tree/master/translation/noisy-eval
3. check out NLLB 200 metrics from `malaya.translation.nllb_metrics`.
4. check out Google Translate metrics from `malaya.translation.google_translate_metrics`.
""".strip()


def word(model: str = 'mesolitica/word-en-ms', **kwargs):
    """
    Load word dictionary, based on google translate.

    Parameters
    ----------
    model, optional (default='mesolitica/word-en-ms')
        Check available models at `malaya.translation.available_word`.

    Returns
    -------
    result: Dict[str, str]
    """
    if model not in available_word:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.available_word`.'
        )

    s3_file = {'model': 'dictionary.json'}
    path = download_files(model, s3_file, **kwargs)

    with open(path['model']) as fopen:
        translator = json.load(fopen)
    return translator


def huggingface(
    model: str = 'mesolitica/translation-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to translate.

    Parameters
    ----------
    model: str, optional (default='mesolitica/translation-t5-small-standard-bahasa-cased')
        Check available models at `malaya.translation.available_huggingface`.
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
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
