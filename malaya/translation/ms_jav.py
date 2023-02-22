from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
from herpetologist import check_type
import logging

logger = logging.getLogger(__name__)

nllb_metrics = """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, zsm_Latn-jav_Latn, 49.5
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, zsm_Latn-jav_Latn, None
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, zsm_Latn-jav_Latn, None
"""

_huggingface_availability = {
    'mesolitica/finetune-translation-austronesian-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 23.79649854962551,
        'SacreBLEU Verbose': '59.2/31.6/18.2/10.8 (BP = 0.966 ratio = 0.967 hyp_len = 20886 ref_len = 21609)',
        'SacreBLEU-chrF++-FLORES200': 51.21,
        'Suggested length': 512,
    },
    'mesolitica/finetune-translation-austronesian-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 24.599989427145964,
        'SacreBLEU Verbose': '58.3/31.6/18.5/11.2 (BP = 0.990 ratio = 0.990 hyp_len = 21391 ref_len = 21609)',
        'SacreBLEU-chrF++-FLORES200': 51.65,
        'Suggested length': 512,
    },
    'mesolitica/finetune-translation-austronesian-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 24.642363178393833,
        'SacreBLEU Verbose': '60.1/32.7/19.1/11.5 (BP = 0.961 ratio = 0.961 hyp_len = 20774 ref_len = 21609)',
        'SacreBLEU-chrF++-FLORES200': 51.91,
        'Suggested length': 512,
    },
}


def available_huggingface():
    """
    List available HuggingFace models.
    """

    logger.info('tested on FLORES200 MS-JAV (zsm_Latn-jav_Latn) pair `dev` set, https://github.com/facebookresearch/flores/tree/main/flores200')
    return describe_availability(_huggingface_availability)


@check_type
def huggingface(
    model: str = 'mesolitica/finetune-translation-austronesian-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to translate MS-to-JAV.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-translation-t5-small-standard-bahasa-cased')
        Check available models at `malaya.translation.ms_jav.available_huggingface()`.

    Returns
    -------
    result: malaya.torch_model.huggingface.Generator
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.ms_jav.available_huggingface()`.'
        )
    return load_huggingface.load_generator(
        model=model, initial_text='terjemah Melayu ke Jawa: ', **kwargs)
