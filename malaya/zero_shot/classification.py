from malaya.model.bert import ZEROSHOT_BERT
from malaya.model.xlnet import ZEROSHOT_XLNET
from herpetologist import check_type
from malaya.similarity import _availability, _transformer


def available_transformer():
    """
    List available transformer zero-shot models.
    """
    return _availability


@check_type
def transformer(model: str = 'bert', **kwargs):
    """
    Load Transformer zero-shot model.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - BERT architecture from google.
        * ``'tiny-bert'`` - BERT architecture from google with smaller parameters.
        * ``'albert'`` - ALBERT architecture from google.
        * ``'tiny-albert'`` - ALBERT architecture from google with smaller parameters.
        * ``'xlnet'`` - XLNET architecture from google.
        * ``'alxlnet'`` - XLNET architecture from google + Malaya.

    Returns
    -------
    result : malaya.model.bert.ZEROSHOT_BERT class
    """

    return _transformer(
        model = model,
        bert_class = ZEROSHOT_BERT,
        xlnet_class = ZEROSHOT_XLNET,
        **kwargs
    )
