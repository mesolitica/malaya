from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
from herpetologist import check_type
import logging

logger = logging.getLogger(__name__)

nllb_metrics = """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, zsm_Latn-bjn_Latn, 45.2
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, zsm_Latn-bjn_Latn, None
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, zsm_Latn-bjn_Latn, None
"""

_huggingface_availability = {
    'mesolitica/finetune-translation-austronesian-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 41.625536185056305,
        'SacreBLEU Verbose': '73.4/50.1/35.7/25.7 (BP = 0.971 ratio = 0.972 hyp_len = 21400 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 65.70,
        'Suggested length': 256,
    },
    'mesolitica/finetune-translation-austronesian-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 43.93729753370648,
        'SacreBLEU Verbose': '74.9/52.2/37.9/27.7 (BP = 0.976 ratio = 0.977 hyp_len = 21510 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 67.43,
        'Suggested length': 256,
    },
    'mesolitica/finetune-translation-austronesian-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 44.17355862158963,
        'SacreBLEU Verbose': '74.7/52.3/38.0/28.0 (BP = 0.979 ratio = 0.979 hyp_len = 21569 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 67.60,
        'Suggested length': 256,
    },
}


def available_huggingface():
    """
    List available HuggingFace models.
    """

    logger.info('tested on FLORES200 MS-JAV (zsm_Latn-bjn_Latn) pair `dev` set, https://github.com/facebookresearch/flores/tree/main/flores200')
    return describe_availability(_huggingface_availability)


@check_type
def huggingface(model: str = 'mesolitica/finetune-translation-austronesian-t5-small-standard-bahasa-cased', **kwargs):
    """
    Load HuggingFace model to translate MS-to-JAV.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-translation-t5-small-standard-bahasa-cased')
        Check available models at `malaya.translation.ms_bjn.available_huggingface()`.

    Returns
    -------
    result: malaya.torch_model.huggingface.Generator
    """
    if model not in _huggingface_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.ms_bjn.available_huggingface()`.'
        )
    return load_huggingface.load_generator(model=model, initial_text='terjemah Melayu ke Banjar: ', **kwargs)
