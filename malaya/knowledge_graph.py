from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import TexttoKG

available_huggingface = {
    'mesolitica/finetune-ttkg-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 61.06784273649806,
        'SacreBLEU Verbose': '86.1/68.4/55.8/45.9 (BP = 0.980 ratio = 0.980 hyp_len = 138209 ref_len = 141004)',
        'Suggested length': 512,
    },
    'mesolitica/finetune-ttkg-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 61.559202822392486,
        'SacreBLEU Verbose': '86.0/68.4/56.1/46.3 (BP = 0.984 ratio = 0.984 hyp_len = 138806 ref_len = 141004)',
        'Suggested length': 512,
    },
    'mesolitica/finetune-ttkg-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 58.764876478744064,
        'SacreBLEU Verbose': '84.5/65.8/53.0/43.1 (BP = 0.984 ratio = 0.985 hyp_len = 138828 ref_len = 141004)',
        'Suggested length': 512,
    },
}

info = """
tested on test set 02 part translated KELM, https://huggingface.co/datasets/mesolitica/translated-KELM
""".strip()


def huggingface(
    model: str = 'mesolitica/finetune-ttkg-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to convert text to triplet format knowledge graph.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-ttkg-t5-small-standard-bahasa-cased')
        Check available models at `malaya.knowledge_graph.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.TexttoKG
    """

    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.knowledge_graph.available_huggingface`.'
        )
    return load(
        model=model,
        class_model=TexttoKG,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
