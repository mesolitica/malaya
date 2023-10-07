from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Constituency
import logging

logger = logging.getLogger(__name__)

available_huggingface = {
    'mesolitica/constituency-parsing-t5-small-standard-bahasa-cased': {
        'Size (MB)': 247,
        'Recall': 81.62,
        'Precision': 83.32,
        'FScore': 82.46,
        'CompleteMatch': 22.40,
        'TaggingAccuracy': 94.95,
    },
    'mesolitica/constituency-parsing-t5-base-standard-bahasa-cased': {
        'Size (MB)': 545,
        'Recall': 82.23,
        'Precision': 82.12,
        'FScore': 82.18,
        'CompleteMatch': 23.50,
        'TaggingAccuracy': 94.69,
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
