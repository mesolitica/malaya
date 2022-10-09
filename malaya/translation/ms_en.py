from malaya.model.tf import Translation
from malaya.model.bigbird import Translation as BigBird_Translation
from malaya.supervised import transformer as load_transformer
from malaya.supervised import bigbird as load_bigbird
from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
from herpetologist import check_type
from malaya.translation.en_ms import dictionary as load_dictionary
import logging

logger = logging.getLogger(__name__)

nllb_metrics = """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, zsm_Latn-eng_Latn,68
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, zsm_Latn-eng_Latn,67.8
3. NLLB-200, Dense, 1.3B,  5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, zsm_Latn-eng_Latn,66.4
4. NLLB-200-Distilled, Dense, 1.3B,  5.48 GB, https://tinyurl.com/nllb200densedst1bmetrics, zsm_Latn-eng_Latn,66.2
5. NLLB-200-Distilled, Dense, 600M, 2.46 GB, https://tinyurl.com/nllb200densedst600mmetrics, zsm_Latn-eng_Latn,64.3
"""

google_translate_metrics = """
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

_transformer_availability = {
    'small': {
        'Size (MB)': 42.7,
        'Quantized Size (MB)': 13.4,
        'BLEU': 35.392560621251945,
        'SacreBLEU Verbose': '68.4/43.4/29.5/20.5 (BP = 0.967 ratio = 0.967 hyp_len = 22798 ref_len = 23570)',
        'SacreBLEU-chrF++-FLORES200': 59.64,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 82.7,
        'BLEU': 40.35382320721753,
        'SacreBLEU Verbose': '71.4/48.1/34.5/25.1 (BP = 0.971 ratio = 0.972 hyp_len = 22907 ref_len = 23570)',
        'SacreBLEU-chrF++-FLORES200': 63.24,
        'Suggested length': 256,
    },
    'bigbird': {
        'Size (MB)': 246,
        'Quantized Size (MB)': 63.7,
        'BLEU': 38.87758831959203,
        'SacreBLEU Verbose': '70.2/46.3/32.5/23.2 (BP = 0.983 ratio = 0.983 hyp_len = 23166 ref_len = 23570)',
        'SacreBLEU-chrF++-FLORES200': 62.49,
        'Suggested length': 1024,
    },
    'small-bigbird': {
        'Size (MB)': 50.4,
        'Quantized Size (MB)': 13.1,
        'BLEU': 36.33139297963485,
        'SacreBLEU Verbose': '68.9/44.1/30.3/21.2 (BP = 0.973 ratio = 0.973 hyp_len = 22937 ref_len = 23570)',
        'SacreBLEU-chrF++-FLORES200': 60.57,
        'Suggested length': 1024,
    },
    'noisy-base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 82.7,
        'BLEU': 40.38461101332913,
        'SacreBLEU Verbose': '71.4/48.2/34.6/25.1 (BP = 0.971 ratio = 0.971 hyp_len = 22889 ref_len = 23570)',
        'SacreBLEU-chrF++-FLORES200': 63.31,
        'Suggested length': 256,
    },
}

_huggingface_availability = {
    'mesolitica/finetune-translation-t5-super-super-tiny-standard-bahasa-cased': {
        'Size (MB)': 23.3,
        'BLEU': 30.216143755278946,
        'SacreBLEU Verbose': '64.9/38.1/24.1/15.3 (BP = 0.978 ratio = 0.978 hyp_len = 23057 ref_len = 23570)',
        'SacreBLEU-chrF++-FLORES200': 56.46,
        'Suggested length': 256,
    },
    'mesolitica/finetune-translation-t5-super-tiny-standard-bahasa-cased': {
        'Size (MB)': 50.7,
        'BLEU': 34.10561487832948,
        'SacreBLEU Verbose': '67.3/41.6/27.8/18.7 (BP = 0.982 ratio = 0.982 hyp_len = 23139 ref_len = 23570)',
        'SacreBLEU-chrF++-FLORES200': 59.18,
        'Suggested length': 256,
    },
    'mesolitica/finetune-translation-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 37.26048464066508,
        'SacreBLEU Verbose': '68.3/44.1/30.5/21.4 (BP = 0.995 ratio = 0.995 hyp_len = 23457 ref_len = 23570)',
        'SacreBLEU-chrF++-FLORES200': 61.29,
        'Suggested length': 256,
    },
    'mesolitica/finetune-translation-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 42.01021763049599,
        'SacreBLEU Verbose': '71.7/49.0/35.6/26.1 (BP = 0.989 ratio = 0.989 hyp_len = 23302 ref_len = 23570)',
        'SacreBLEU-chrF++-FLORES200': 64.67,
        'Suggested length': 256,
    },
    'mesolitica/finetune-translation-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 43.40885318934906,
        'SacreBLEU Verbose': '72.3/50.5/37.1/27.7 (BP = 0.987 ratio = 0.987 hyp_len = 23258 ref_len = 23570)',
        'SacreBLEU-chrF++-FLORES200': 65.44,
        'Suggested length': 256,
    }
}


def available_transformer():
    """
    List available transformer models.
    """

    logger.info('tested on FLORES200 MS-EN (zsm_Latn-eng_Latn) pair `dev` set, https://github.com/facebookresearch/flores/tree/main/flores200')
    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available HuggingFace models.
    """

    logger.info('tested on FLORES200 MS-EN (zsm_Latn-eng_Latn) pair, https://github.com/facebookresearch/flores/tree/main/flores200')
    return describe_availability(_huggingface_availability)


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load Transformer encoder-decoder model to translate MS-to-EN.

    Parameters
    ----------
    model: str, optional (default='base')
        Check available models at `malaya.translation.ms_en.available_transformer()`.
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
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.ms_en.available_transformer()`.'
        )

    if 'bigbird' in model:
        return load_bigbird.load(
            module='translation-ms-en',
            model=model,
            model_class=BigBird_Translation,
            maxlen=_transformer_availability[model]['Suggested length'],
            quantized=quantized,
            **kwargs
        )

    else:
        return load_transformer.load(
            module='translation-ms-en',
            model=model,
            encoder='subword',
            model_class=Translation,
            quantized=quantized,
            **kwargs
        )


@check_type
def huggingface(model: str = 'mesolitica/finetune-translation-t5-small-standard-bahasa-cased', **kwargs):
    """
    Load HuggingFace model to translate MS-to-EN.

    Parameters
    ----------
    model: str, optional (default=''mesolitica/finetune-translation-t5-small-standard-bahasa-cased')
        Check available models at `malaya.translation.ms_en.available_huggingface()`.

    Returns
    -------
    result: malaya.torch_model.huggingface.Generator
    """
    if model not in _huggingface_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.ms_en.available_huggingface()`.'
        )
    return load_huggingface.load_generator(model=model, initial_text='terjemah Melayu ke Inggeris: ', **kwargs)


def dictionary(**kwargs):
    """
    Load dictionary {MS: EN} .

    Returns
    -------
    result: Dict[str, str]
    """
    translator = load_dictionary()
    translator = {v: k for k, v in translator.items()}
    return translator
