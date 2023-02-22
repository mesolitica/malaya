from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
from herpetologist import check_type
import logging

logger = logging.getLogger(__name__)

nllb_metrics = """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, ind_Latn-zsm_Latn, 60.2
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, ind_Latn-zsm_Latn, None
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, ind_Latn-zsm_Latn, None
"""

_huggingface_availability = {
    'mesolitica/finetune-translation-austronesian-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 30.277470707798773,
        'SacreBLEU Verbose': '64.2/38.0/24.1/15.6 (BP = 0.978 ratio = 0.978 hyp_len = 21542 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 57.38,
        'Suggested length': 256,
    },
    'mesolitica/finetune-translation-austronesian-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 30.24358980824753,
        'SacreBLEU Verbose': '61.1/36.9/23.8/15.6 (BP = 1.000 ratio = 1.052 hyp_len = 23174 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 58.43,
        'Suggested length': 512,
    },
    'mesolitica/finetune-translation-austronesian-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 31.494673032706213,
        'SacreBLEU Verbose': '64.1/38.8/25.1/16.5 (BP = 0.989 ratio = 0.990 hyp_len = 21796 ref_len = 22027)',
        'SacreBLEU-chrF++-FLORES200': 58.10,
        'Suggested length': 512,
    },
}


def available_huggingface():
    """
    List available HuggingFace models.
    """

    logger.info('tested on FLORES200 IND-MS (ind_Latn-zsm_Latn) pair `dev` set, https://github.com/facebookresearch/flores/tree/main/flores200')
    return describe_availability(_huggingface_availability)


@check_type
def huggingface(
    model: str = 'mesolitica/finetune-translation-austronesian-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to translate IND-to-MS.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-translation-t5-small-standard-bahasa-cased')
        Check available models at `malaya.translation.ind_ms.available_huggingface()`.

    Returns
    -------
    result: malaya.torch_model.huggingface.Generator
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.ind_ms.available_huggingface()`.'
        )
    return load_huggingface.load_generator(
        model=model, initial_text='terjemah Indonesia ke Melayu: ', **kwargs)
