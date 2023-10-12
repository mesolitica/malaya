from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Generator

available_huggingface = {
    'mesolitica/finetune-true-case-t5-super-tiny-standard-bahasa-cased': {
        'Size (MB)': 51,
        'WER': 0.105094863,
        'CER': 0.02163576,
        'Suggested length': 256,
    },
    'mesolitica/finetune-true-case-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'WER': 0.0967551738,
        'CER': 0.0201099683,
        'Suggested length': 256,
    },
    'mesolitica/finetune-true-case-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'WER': 0.081104625471,
        'CER': 0.0163838230,
        'Suggested length': 256,
    },
}

info = """
tested on generated dataset at https://f000.backblazeb2.com/file/malay-dataset/true-case/test-set-true-case.json
""".strip()


def huggingface(
    model: str = 'mesolitica/finetune-true-case-t5-tiny-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to true case.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-true-case-t5-tiny-standard-bahasa-cased')
        Check available models at `malaya.true_case.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Generator
    """
    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.true_case.available_huggingface`.'
        )
    return load(
        model=model,
        class_model=Generator,
        available_huggingface=available_huggingface,
        path=__name__,
        initial_text='kes benar: ',
        **kwargs,
    )
