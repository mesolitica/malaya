from malaya.model.tf import Translation
from malaya.model.bigbird import Translation as BigBird_Translation
from malaya.supervised import transformer as load_transformer
from malaya.supervised import bigbird as load_bigbird
from malaya.function import describe_availability
from herpetologist import check_type
from malaya.translation.en_ms import dictionary as load_dictionary
from malaya.translation.en_ms import _huggingface
import logging

logger = logging.getLogger(__name__)

"""
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, zsm_Latn-eng_Latn,68
2. NLLB-200, Dense, 3.3B, https://tinyurl.com/nllb200dense3bmetrics, zsm_Latn-eng_Latn,67.8
3. NLLB-200, Dense, 1.3B, https://tinyurl.com/nllb200dense1bmetrics, zsm_Latn-eng_Latn,66.4
4. NLLB-200-Distilled, Dense, 1.3B, https://tinyurl.com/nllb200densedst1bmetrics, zsm_Latn-eng_Latn,66.2
5. NLLB-200-Distilled, Dense, 600M, https://tinyurl.com/nllb200densedst600mmetrics, zsm_Latn-eng_Latn,64.3
"""

"""
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
        'BLEU': 59.87473086087865,
        'SacreBLEU Verbose': '80.6/64.3/54.1/46.3 (BP = 0.998 ratio = 0.998 hyp_len = 1996245 ref_len = 2001100)',
        'SacreBLEU-chrF++-FLORES200': 59.64,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 82.7,
        'BLEU': 71.68758277346333,
        'SacreBLEU Verbose': '86.2/74.8/67.2/61.0 (BP = 1.000 ratio = 1.005 hyp_len = 2010492 ref_len = 2001100)',
        'SacreBLEU-chrF++-FLORES200': 63.24,
        'Suggested length': 256,
    },
    'bigbird': {
        'Size (MB)': 246,
        'Quantized Size (MB)': 63.7,
        'BLEU': 59.54825688811841,
        'SacreBLEU Verbose': '79.6/63.8/53.8/46.0 (BP = 1.000 ratio = 1.026 hyp_len = 332247 ref_len = 323967)',
        'SacreBLEU-chrF++-FLORES200': 62.49,
        'Suggested length': 1024,
    },
    'small-bigbird': {
        'Size (MB)': 50.4,
        'Quantized Size (MB)': 13.1,
        'BLEU': 55.967144835516855,
        'SacreBLEU Verbose': '77.4/60.5/49.9/41.9 (BP = 1.000 ratio = 1.026 hyp_len = 2053414 ref_len = 2001100)',
        'SacreBLEU-chrF++-FLORES200': 60.57,
        'Suggested length': 1024,
    },
    'noisy-base': {
        'Size (MB)': 234,
        'Quantized Size (MB)': 82.7,
        'BLEU': 71.72549311084798,
        'SacreBLEU Verbose': '86.3/74.8/67.2/61.0 (BP = 1.000 ratio = 1.002 hyp_len = 2004164 ref_len = 2001100)',
        'SacreBLEU-chrF++-FLORES200': 63.31,
        'Suggested length': 256,
    },
}

_huggingface_availability = {
    'mesolitica/t5-super-tiny-finetuned-noisy-ms-en': {
        'Size (MB)': 50.8,
        'BLEU': 59.92897086989418,
        'SacreBLEU Verbose': '79.8/64.0/54.1/46.6 (BP = 1.000 ratio = 1.008 hyp_len = 2017101 ref_len = 2001100)',
        'SacreBLEU-chrF++-FLORES200': 59.12,
        'Suggested length': 256,
    },
    'mesolitica/t5-tiny-finetuned-noisy-ms-en': {
        'Size (MB)': 139,
        'BLEU': 65.9069151371865,
        'SacreBLEU Verbose': '83.0/69.3/60.7/54.1 (BP = 1.000 ratio = 1.001 hyp_len = 2003273 ref_len = 2001100)',
        'SacreBLEU-chrF++-FLORES200': 59.91,
        'Suggested length': 256,
    },
    'mesolitica/t5-small-finetuned-noisy-ms-en': {
        'Size (MB)': 242,
        'BLEU': 63.806656594496836,
        'SacreBLEU Verbose': '82.1/67.5/58.3/51.3 (BP = 1.000 ratio = 1.001 hyp_len = 2002291 ref_len = 2001100)',
        'SacreBLEU-chrF++-FLORES200': 62.60,
        'Suggested length': 256,
    }
}


def available_transformer():
    """
    List available transformer models.
    """

    logger.info('tested on 100k MS-EN test set generated from teacher semisupervised model, https://huggingface.co/datasets/mesolitica/ms-en')
    logger.info('tested on FLORES200 MS-EN (zsm_Latn-eng_Latn) pair `dev` set, https://github.com/facebookresearch/flores/tree/main/flores200')
    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available HuggingFace models.
    """

    logger.info('tested on 100k MS-EN test set generated from teacher semisupervised model, https://huggingface.co/datasets/mesolitica/ms-en')
    logger.info('tested on FLORES200 MS-EN (zsm_Latn-eng_Latn) pair, https://github.com/facebookresearch/flores/tree/main/flores200')
    logger.warning(
        '100k MS-EN test set generated from teacher semisupervised model, the models might generate better results compared to '
        'to the teacher semisupervised model, thus lower BLEU score.'
    )
    return describe_availability(_huggingface_availability)


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load Transformer encoder-decoder model to translate MS-to-EN.

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
def huggingface(model: str = 'mesolitica/t5-tiny-finetuned-noisy-ms-en', **kwargs):
    """
    Load HuggingFace model to translate MS-to-EN.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'mesolitica/t5-super-tiny-finetuned-noisy-ms-en'`` - https://huggingface.co/mesolitica/t5-super-tiny-finetuned-noisy-ms-en
        * ``'mesolitica/t5-tiny-finetuned-noisy-ms-en'`` - https://huggingface.co/mesolitica/t5-tiny-finetuned-noisy-ms-en
        * ``'mesolitica/t5-small-finetuned-noisy-ms-en'`` - https://huggingface.co/mesolitica/t5-small-finetuned-noisy-ms-en

    Returns
    -------
    result: malaya.model.huggingface.Generator
    """
    model = model.lower()
    if model not in _huggingface_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.ms_en.available_huggingface()`.'
        )
    return _huggingface(model=model, initial_text='terjemah Melayu ke Inggeris: ', **kwargs)


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
