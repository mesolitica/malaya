from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Paraphrase

available_huggingface = {
    'mesolitica/finetune-paraphrase-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'BLEU': 36.92696648298233,
        'SacreBLEU Verbose': '62.5/42.3/33.0/26.9 (BP = 0.943 ratio = 0.945 hyp_len = 95496 ref_len = 101064)',
        'Suggested length': 256,
    },
    'mesolitica/finetune-paraphrase-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'BLEU': 37.598729045833316,
        'SacreBLEU Verbose': '62.6/42.5/33.2/27.0 (BP = 0.957 ratio = 0.958 hyp_len = 96781 ref_len = 101064)',
        'Suggested length': 256,
    },
    'mesolitica/finetune-paraphrase-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'BLEU': 35.95965899952292,
        'SacreBLEU Verbose': '61.7/41.3/32.0/25.8 (BP = 0.944 ratio = 0.946 hyp_len = 95593 ref_len = 101064)',
        'Suggested length': 256,
    },
}

info = """
tested on MRPC validation set, https://huggingface.co/datasets/mesolitica/translated-MRPC
tested on ParaSCI ARXIV test set, https://huggingface.co/datasets/mesolitica/translated-paraSCI
""".strip()


def huggingface(
    model: str = 'mesolitica/finetune-paraphrase-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to paraphrase.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-paraphrase-t5-small-standard-bahasa-cased')
        Check available models at `malaya.paraphrase.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Paraphrase
    """
    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.paraphrase.available_huggingface`.'
        )

    return load(
        model=model,
        class_model=Paraphrase,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
