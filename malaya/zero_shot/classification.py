from malaya.similarity.semantic import available_huggingface
from malaya.supervised import huggingface as load_huggingface


def huggingface(
    model: str = 'mesolitica/finetune-mnli-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to zeroshot text classification.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-mnli-t5-small-standard-bahasa-cased')
        Check available models at `malaya.zero_shot.classification.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.ZeroShotClassification
    """

    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.zero_shot.classification.available_huggingface`.'
        )
    return load_huggingface.load_zeroshot_classification(model=model, **kwargs)
