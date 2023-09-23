from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Constituency
import logging

logger = logging.getLogger(__name__)

available_huggingface = {
    'mesolitica/constituency-parsing-nanot5-small-malaysian-cased': {
        'Size (MB)': 273,
        'Recall': 79.10,
        'Precision': 80.63,
        'FScore': 79.86,
        'CompleteMatch': 20.10,
        'TaggingAccuracy': 93.96,
    },
    'mesolitica/constituency-parsing-nanot5-base-malaysian-cased': {
        'Size (MB)': 545,
        'Recall': 80.89,
        'Precision': 81.87,
        'FScore': 81.38,
        'CompleteMatch': 23.90,
        'TaggingAccuracy': 94.45,
    }
}

info = """
Tested on https://github.com/aisingapore/seacorenlp-data/tree/main/id/constituency test set.
""".strip()


def huggingface(
    model: str = 'mesolitica/constituency-parsing-nanot5-base-malaysian-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to Constituency parsing.

    Parameters
    ----------
    model: str, optional (default='mmesolitica/constituency-parsing-nanot5-base-malaysian-cased')
        Check available models at `malaya.constituency.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Constituency
    """
    logger.warning(
        '`malaya.constituency.huggingface` trained on indonesian dataset, not an actual malay dataset.')

    return load(
        model=model,
        class_model=Constituency,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
