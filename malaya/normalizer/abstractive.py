from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
from malaya.language_model import kenlm
from malaya.stem import deep_model
from typing import Callable
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
}


def available_huggingface():
    """
    List available huggingface models.
    """

    logger.info('tested on test set IIUM confession normalization dataset, https://huggingface.co/datasets/mesolitica/IIUM-Confession-abstractive-normalization')
    logger.info('tested on test set Twitter normalization dataset, https://huggingface.co/datasets/mesolitica/twitter-abstractive-normalization')

    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/finetune-normalizer-t5-small-standard-bahasa-cased',
    kenlm_model: str = 'bahasa-wiki-news',
    stem_model: str = 'noisy',
    segmenter: Callable = None,
    text_scorer: Callable = None,
    replace_augmentation: bool = True,
    minlen_speller: int = 2,
    maxlen_speller: int = 12,
    **kwargs,
):
    """
    Load HuggingFace model to abstractive text normalizer.
    text -> rules based text normalizer -> abstractive.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-normalizer-t5-small-standard-bahasa-cased')
        Check available models at `malaya.normalizer.abstractive.available_huggingface()`.
    kenlm_model: str, optional (default='bahasa-wiki-news')
        the model trained on `malaya.language_model.kenlm(model = 'bahasa-wiki-news')`,
        but you can use any kenlm model from `malaya.language_model.available_kenlm`.
    stem_model: str, optional (default='noisy')
        the model trained on `malaya.stem.deep_model(model = 'noisy'),
        but you can use any stemmer model from `malaya.stem.available_model`.

    Returns
    -------
    result: malaya.torch_model.huggingface.Normalizer
    """
