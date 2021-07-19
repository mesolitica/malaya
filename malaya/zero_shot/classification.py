from malaya.model.bert import ZeroshotBERT
from malaya.model.xlnet import ZeroshotXLNET
from herpetologist import check_type
from malaya.similarity import _transformer_availability, _transformer


def available_transformer():
    """
    List available transformer zero-shot models.
    """
    from malaya.function import describe_availability

    return describe_availability(_transformer_availability)


@check_type
def transformer(model: str = 'bert', quantized: bool = False, **kwargs):
    """
    Load Transformer zero-shot model.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - Google BERT BASE parameters.
        * ``'tiny-bert'`` - Google BERT TINY parameters.
        * ``'albert'`` - Google ALBERT BASE parameters.
        * ``'tiny-albert'`` - Google ALBERT TINY parameters.
        * ``'xlnet'`` - Google XLNET BASE parameters.
        * ``'alxlnet'`` - Malaya ALXLNET BASE parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.model.bert.ZeroshotBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.ZeroshotXLNET`.
    """

    return _transformer(
        model=model,
        bert_model=ZeroshotBERT,
        xlnet_model=ZeroshotXLNET,
        quantized=quantized,
        siamese=False,
        **kwargs
    )
