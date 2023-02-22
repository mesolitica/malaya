from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
from herpetologist import check_type
import logging

logger = logging.getLogger(__name__)

nllb_metrics = """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, jav_Latn-zsm_Latn, 56.5
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, jav_Latn-zsm_Latn, None
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, jav_Latn-zsm_Latn, None
"""

_huggingface_availability = {
    'mesolitica/finetune-translation-austronesian-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 23.797627841793492,
        'SacreBLEU Verbose': '58.2/31.1/18.1/10.8 (BP = 0.977 ratio = 0.977 hyp_len = 21521 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 50.65,
        'Suggested length': 512,
    },
    'mesolitica/finetune-translation-austronesian-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 25.24437731940083,
        'SacreBLEU Verbose': '57.7/31.9/19.0/11.6 (BP = 1.000 ratio = 1.022 hyp_len = 22516 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 52.58,
        'Suggested length': 512,
    },
    'mesolitica/finetune-translation-austronesian-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 25.772896570805038,
        'SacreBLEU Verbose': '58.9/32.6/19.6/12.1 (BP = 0.992 ratio = 0.992 hyp_len = 21851 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 52.21,
        'Suggested length': 512,
    },
}


def available_huggingface():
    """
    List available HuggingFace models.
    """

    logger.info('tested on FLORES200 JAV-MS (jav_Latn-zsm_Latn) pair `dev` set, https://github.com/facebookresearch/flores/tree/main/flores200')
    return describe_availability(_huggingface_availability)


@check_type
def huggingface(
    model: str = 'mesolitica/finetune-translation-austronesian-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to translate JAV-to-MS.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-translation-t5-small-standard-bahasa-cased')
        Check available models at `malaya.translation.jav_ms.available_huggingface()`.

    Returns
    -------
    result: malaya.torch_model.huggingface.Generator
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.jav_ms.available_huggingface()`.'
        )
    return load_huggingface.load_generator(
        model=model, initial_text='terjemah Jawa ke Melayu: ', **kwargs)
