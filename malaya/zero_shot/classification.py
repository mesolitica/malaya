from malaya.model.bert import ZeroshotBERT
from malaya.model.xlnet import ZeroshotXLNET
from herpetologist import check_type
from malaya.similarity.semantic import (
    _transformer_availability,
    _huggingface_availability,
    _describe,
    _transformer,
)
from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
import warnings


def available_transformer():
    """
    List available transformer zero-shot models.
    """

    warnings.warn(
        '`malaya.zero_shot.classification.available_transformer` is deprecated, use `malaya.zero_shot.classification.available_huggingface` instead', DeprecationWarning)

    _describe()
    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available huggingface zero-shot models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


@check_type
def transformer(model: str = 'bert', quantized: bool = False, **kwargs):
    """
    Load Transformer zero-shot model.

    Parameters
    ----------
    model: str, optional (default='bert')
        Check available models at `malaya.zero_shot.classification.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.model.bert.ZeroshotBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.ZeroshotXLNET`.
    """

    warnings.warn(
        '`malaya.zero_shot.classification.transformer` is deprecated, use `malaya.zero_shot.classification.huggingface` instead', DeprecationWarning)

    return _transformer(
        model=model,
        bert_model=ZeroshotBERT,
        xlnet_model=ZeroshotXLNET,
        quantized=quantized,
        siamese=False,
        **kwargs
    )


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
            'model not supported, please check supported models from `malaya.zero_shot.classification.available_huggingface()`.'
        )
    return load_huggingface.load_zeroshot_classification(model=model, **kwargs)
