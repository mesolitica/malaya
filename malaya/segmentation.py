from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Generator


available_huggingface = {
    'mesolitica/finetune-segmentation-t5-super-tiny-standard-bahasa-cased': {
        'Size (MB)': 51,
        'WER': 0.030962535,
        'CER': 0.0041129253,
        'Suggested length': 256,
    },
    'mesolitica/finetune-segmentation-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'WER': 0.0207876127,
        'CER': 0.002146691161,
        'Suggested length': 256,
    },
    'mesolitica/finetune-segmentation-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'WER': 0.0202468274,
        'CER': 0.0024325431,
        'Suggested length': 256,
    },
}

info = """
tested on random generated dataset at https://f000.backblazeb2.com/file/malay-dataset/segmentation/test-set-segmentation.json
""".strip()


def huggingface(
    model: str = 'mesolitica/finetune-segmentation-t5-tiny-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to segmentation.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-segmentation-t5-tiny-standard-bahasa-cased')
        Check available models at `malaya.segmentation.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Generator
    """
    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.segmentation.available_huggingface`.'
        )
    return load(
        model=model,
        class_model=Generator,
        available_huggingface=available_huggingface,
        path=__name__,
        initial_text='segmentasi: ',
        **kwargs,
    )
