from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/malaya-one-for-all-pythia-160m': {
        'Size (MB)': 242,
        'BLEU': 37.598729045833316,
        'SacreBLEU Verbose': '62.6/42.5/33.2/27.0 (BP = 0.957 ratio = 0.958 hyp_len = 96781 ref_len = 101064)',
        'Suggested length': 256,
    },
    'mesolitica/malaya-one-for-all-pythia-410m': {
        'Size (MB)': 892,
        'BLEU': 35.95965899952292,
        'SacreBLEU Verbose': '61.7/41.3/32.0/25.8 (BP = 0.944 ratio = 0.946 hyp_len = 95593 ref_len = 101064)',
        'Suggested length': 256,
    },
}


def available_huggingface():
    """
    List available huggingface models.
    """
    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/finetune-one-for-all-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to paraphrase.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-paraphrase-t5-small-standard-bahasa-cased')
        Check available models at `malaya.paraphrase.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Paraphrase
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.paraphrase.available_huggingface()`.'
        )
    return load_huggingface.load_paraphrase(model=model, initial_text='parafrasa: ', **kwargs)
