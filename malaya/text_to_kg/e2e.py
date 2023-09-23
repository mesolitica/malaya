import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/finetune-ttkg-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 61.06784273649806,
        'SacreBLEU Verbose': '86.1/68.4/55.8/45.9 (BP = 0.980 ratio = 0.980 hyp_len = 138209 ref_len = 141004)',
        'Suggested length': 256,
    },
    'mesolitica/finetune-ttkg-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 61.559202822392486,
        'SacreBLEU Verbose': '86.0/68.4/56.1/46.3 (BP = 0.984 ratio = 0.984 hyp_len = 138806 ref_len = 141004)',
        'Suggested length': 256,
    },
    'mesolitica/finetune-ttkg-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 58.764876478744064,
        'SacreBLEU Verbose': '84.5/65.8/53.0/43.1 (BP = 0.984 ratio = 0.985 hyp_len = 138828 ref_len = 141004)',
        'Suggested length': 256,
    },
}


def available_huggingface():
    """
    List available huggingface models.
    """

    logger.info(
        'tested on test set 02 part translated KELM, https://huggingface.co/datasets/mesolitica/translated-REBEL')

    return describe_availability(_huggingface_availability)


def huggingface(model: str = 'mesolitica/finetune-ttkg-t5-small-standard-bahasa-cased', **kwargs):
    """
    Load HuggingFace model to End-to-End text to knowledge graph.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-ttkg-t5-small-standard-bahasa-cased')
        Check available models at `malaya.text_to_kg.e2e.available_huggingface()`.

    Returns
    -------
    result: malaya.torch_model.huggingface.TexttoKG
    """

    if model not in _huggingface_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.text_to_kg.e2e.available_huggingface()`.'
        )
    return load_ttkg(model=model, **kwargs)
