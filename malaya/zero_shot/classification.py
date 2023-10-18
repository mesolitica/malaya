from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import ZeroShotClassification
from malaya.similarity.semantic import available_huggingface


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

    return load(
        model=model,
        class_model=ZeroShotClassification,
        available_huggingface=available_huggingface,
        path=__name__,
        **kwargs,
    )
