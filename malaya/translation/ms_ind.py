from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
from herpetologist import check_type
import logging

logger = logging.getLogger(__name__)

nllb_metrics = """
NLLB Metrics, https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models:
1. NLLB-200, MOE, 54.5B, https://tinyurl.com/nllb200moe54bmetrics, zsm_Latn-ind_Latn, 62.4
2. NLLB-200, Dense, 3.3B, 17.58 GB, https://tinyurl.com/nllb200dense3bmetrics, zsm_Latn-ind_Latn, None
3. NLLB-200, Dense, 1.3B, 5.48 GB, https://tinyurl.com/nllb200dense1bmetrics, zsm_Latn-ind_Latn, None
"""

_huggingface_availability = {
    'mesolitica/finetune-translation-austronesian-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 33.88207737320432,
        'SacreBLEU Verbose': '67.7/42.2/28.0/18.9 (BP = 0.966 ratio = 0.966 hyp_len = 21116 ref_len = 21856)',
        'SacreBLEU-chrF++-FLORES200': 59.46,
        'Suggested length': 512,
    },
    'mesolitica/finetune-translation-austronesian-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 35.95448072225675,
        'SacreBLEU Verbose': '66.3/42.6/29.1/20.3 (BP = 1.000 ratio = 1.014 hyp_len = 22164 ref_len = 21856)',
        'SacreBLEU-chrF++-FLORES200': 61.02,
        'Suggested length': 512,
    },
    'mesolitica/finetune-translation-austronesian-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 37.62068001899787,
        'SacreBLEU Verbose': '70.0/45.8/31.7/22.5 (BP = 0.967 ratio = 0.968 hyp_len = 21152 ref_len = 21856)',
        'SacreBLEU-chrF++-FLORES200': 62.10,
        'Suggested length': 512,
    },
}


def available_huggingface():
    """
    List available HuggingFace models.
    """

    logger.info('tested on FLORES200 MS-IND (zsm_Latn-ind_Latn) pair `dev` set, https://github.com/facebookresearch/flores/tree/main/flores200')
    return describe_availability(_huggingface_availability)


@check_type
def huggingface(
    model: str = 'mesolitica/finetune-translation-austronesian-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to translate MS-to-IND.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-translation-t5-small-standard-bahasa-cased')
        Check available models at `malaya.translation.ms_ind.available_huggingface()`.

    Returns
    -------
    result: malaya.torch_model.huggingface.Generator
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.translation.ms_ind.available_huggingface()`.'
        )
    return load_huggingface.load_generator(
        model=model, initial_text='terjemah Melayu ke Indonesia: ', **kwargs)
